#!/usr/bin/env bash
# TEMPO-RD G1 tier-attribution allocation.  This is separate from the
# historical v4_open/tempo_v4 functional screen in run_g1_1node.slurm.
# It executes the five unpaced tier modes only after explicit approval.
#SBATCH --job-name=tempo_g1_tier
#SBATCH --account=m5320_g
#SBATCH --qos=debug_preempt
#SBATCH --constraint=gpu&hbm40g
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --no-requeue
#SBATCH --mail-type=NONE
#SBATCH --output=logs/slurm/tempo_g1_tier_%j.out
#SBATCH --error=logs/slurm/tempo_g1_tier_%j.err
#SBATCH --signal=B:USR1@45

set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:?G1 tier requires the repository root}
REPO_ROOT=$(realpath -e -- "${REPO_ROOT}")
[[ "${TEMPO_RD_APPROVE_G1:-}" == "YES" ]] || {
    echo "ERROR: G1 requires explicit TEMPO_RD_APPROVE_G1=YES" >&2
    exit 2
}
rd_source_bundle_sha256() {
    (
        cd -- "${REPO_ROOT}"
        {
                sha256sum \
                eval/sota_4node/run_g1_tier_1node.slurm \
                eval/sota_4node/tier_attribution_runner.py \
                eval/sota_4node/train.py \
                eval/sota_4node/validate_g1_tier_raw.py \
                eval/sota_4node/validate_g1_result.py \
                eval/sota_4node/compose_g1_result.py \
                eval/sota_4node/build_g1_causal_readiness.py \
                eval/sota_4node/host_pressure_train_wrapper.py \
                eval/sota_4node/host_pressure_placebo.py \
                eval/sota_4node/capture_g1_domain_counters.py \
                eval/sota_4node/prepare_foreground_path.py \
                eval/sota_4node/capture_nvml_pcie_observation.py \
                eval/sota_4node/capture_lustre_rpc_observation.py
            find tempo -type f -name '*.py' -print | sort | xargs sha256sum
        } | sha256sum | awk '{print $1}'
    )
}
G1_SOURCE_BUNDLE_SHA256="$(rd_source_bundle_sha256)"
[[ "${TEMPO_RD_G1_EXPECTED_SOURCE_BUNDLE_SHA256:-}" == "${G1_SOURCE_BUNDLE_SHA256}" ]] || {
    echo "ERROR: G1 source bundle hash is missing or changed after approval" >&2
    echo "expected=${TEMPO_RD_G1_EXPECTED_SOURCE_BUNDLE_SHA256:-unset} actual=${G1_SOURCE_BUNDLE_SHA256}" >&2
    exit 2
}
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 && "${SLURM_NTASKS:-}" == 4 ]] || {
    echo "ERROR: G1 tier requires one node and four ranks" >&2
    exit 2
}
case "${SLURM_JOB_QOS:-}" in
    debug_preempt) ;;
    interactive|gpu_interactive)
        [[ "${TEMPO_RD_ALLOW_INTERACTIVE:-}" == YES ]] || {
            echo "ERROR: interactive G1 requires TEMPO_RD_ALLOW_INTERACTIVE=YES" >&2
            exit 2
        }
        ;;
    *)
        echo "ERROR: G1 tier requires debug_preempt or explicitly approved interactive QOS" >&2
        exit 2
        ;;
esac
[[ "${PSCRATCH:-}" == /pscratch/sd/s/sgkim ]] || {
    echo "ERROR: unexpected PSCRATCH" >&2
    exit 2
}

module reset
module load pytorch/2.8.0
python -c 'import msgpack; assert msgpack.__version__ == "1.1.1"'
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 WORLD_SIZE=4
export PERLMUTTER_CPU_LDOM_MAP=3,2,1,0
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=$((25000 + (SLURM_JOB_ID % 10000)))
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_IB_DISABLE NCCL_CROSS_NIC NCCL_ALGO NCCL_PROTO

RESULT_DIR="${REPO_ROOT}/results/sota_4node/g1_tier_job_${SLURM_JOB_ID}"
CHECKPOINT_BASE="${PSCRATCH}/tempo_rd_g1_tier_checkpoints/job_${SLURM_JOB_ID}"
D2H_SINK_ROOT="${TMPDIR:-/tmp}/tempo_rd_g1_${SLURM_JOB_ID}"
[[ ! -e "${RESULT_DIR}" && ! -e "${CHECKPOINT_BASE}" ]] || {
    echo "ERROR: result/checkpoint path already exists" >&2
    exit 2
}
mkdir -p -- "${RESULT_DIR}" "${CHECKPOINT_BASE}"

if [[ "${TEMPO_RD_CAPTURE_D2H_NSYS:-}" == YES ]]; then
    NSYS_BIN=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Systems/bin/nsys
    [[ -x "${NSYS_BIN}" ]] || { echo "ERROR: requested Nsight diagnostic is unavailable" >&2; exit 2; }
    NSYS_ROOT="${RESULT_DIR}/nsys_d2h"
    mkdir -p -- "${NSYS_ROOT}"
    NSYS_WRAPPER="${RESULT_DIR}/nsys_d2h_wrapper.sh"
    cat > "${NSYS_WRAPPER}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exec "${NSYS_BIN}" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --cuda-event-trace=false --force-overwrite=true \
    --output="${NSYS_ROOT}/${NSYS_MODE}_rank_${SLURM_PROCID}" "$@"
EOF
    chmod +x "${NSYS_WRAPPER}"
    export NSYS_BIN NSYS_ROOT NSYS_WRAPPER
fi

cleanup_rc=0
cleanup() {
    local rc=$?
    trap - EXIT TERM INT USR1
    if [[ -d "${CHECKPOINT_BASE}" ]]; then
        find "${CHECKPOINT_BASE}" -mindepth 1 -depth -delete || cleanup_rc=$?
        rmdir -- "${CHECKPOINT_BASE}" || cleanup_rc=$?
    fi
    if [[ -d "${D2H_SINK_ROOT}" ]]; then
        find "${D2H_SINK_ROOT}" -mindepth 1 -depth -delete || cleanup_rc=$?
        rmdir -- "${D2H_SINK_ROOT}" || cleanup_rc=$?
    fi
    printf 'schema_version=tempo-rd-g1-tier-cleanup-1\njob_id=%s\ncleanup_rc=%s\nexit_code=%s\n' \
        "${SLURM_JOB_ID}" "${cleanup_rc}" "${rc}" > "${RESULT_DIR}/cleanup_status.env"
    if (( cleanup_rc != 0 && rc == 0 )); then rc=${cleanup_rc}; fi
    exit "${rc}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# The debug-preempt warning is an early termination signal.  Exit through the
# normal cleanup trap so checkpoint/sink scratch is removed and the raw result
# is marked incomplete rather than leaving a partial allocation behind.
trap 'exit 143' USR1

PYTHON_SNAPSHOT_ROOT="${RESULT_DIR}/python_snapshot"
mkdir -p -- "${PYTHON_SNAPSHOT_ROOT}"
cp -a -- "${REPO_ROOT}/tempo" "${PYTHON_SNAPSHOT_ROOT}/tempo"
cp -- "${REPO_ROOT}/eval/sota_4node/train.py" "${RESULT_DIR}/train_executed.py"
cp -- "${REPO_ROOT}/eval/sota_4node/tier_attribution_runner.py" \
    "${RESULT_DIR}/tier_attribution_runner_executed.py"
cp -- "${REPO_ROOT}/eval/sota_4node/validate_g1_tier_raw.py" \
    "${RESULT_DIR}/validate_g1_tier_raw_executed.py"
cp -- "${REPO_ROOT}/eval/sota_4node/validate_g1_result.py" \
    "${RESULT_DIR}/validate_g1_result_executed.py"
cp -- "${REPO_ROOT}/eval/sota_4node/compose_g1_result.py" \
    "${RESULT_DIR}/compose_g1_result_executed.py"
cp -- "${REPO_ROOT}/eval/sota_4node/build_g1_causal_readiness.py" \
    "${RESULT_DIR}/build_g1_causal_readiness_executed.py"
cp -- "${REPO_ROOT}/eval/sota_4node/host_pressure_train_wrapper.py" \
    "${RESULT_DIR}/host_pressure_train_wrapper_executed.py"
cp -- "${REPO_ROOT}/eval/sota_4node/host_pressure_placebo.py" \
    "${RESULT_DIR}/host_pressure_placebo.py"
cp -- "${REPO_ROOT}/eval/sota_4node/capture_g1_domain_counters.py" \
    "${RESULT_DIR}/capture_g1_domain_counters_executed.py"
cp -- "${REPO_ROOT}/eval/sota_4node/prepare_foreground_path.py" \
    "${RESULT_DIR}/prepare_foreground_path_executed.py"
cp -- "${REPO_ROOT}/eval/sota_4node/capture_nvml_pcie_observation.py" \
    "${RESULT_DIR}/capture_nvml_pcie_observation_executed.py"
cp -- "${REPO_ROOT}/eval/sota_4node/capture_lustre_rpc_observation.py" \
    "${RESULT_DIR}/capture_lustre_rpc_observation_executed.py"
export TEMPO_RD_REPO_ROOT="${REPO_ROOT}"
export PYTHONPATH="${PYTHON_SNAPSHOT_ROOT}:${REPO_ROOT}/.baseline_deps/runtime"
export PYTHONSAFEPATH=1 LD_LIBRARY_PATH="${REPO_ROOT}/.baseline_deps/runtime/datastates/lib:${REPO_ROOT}/.baseline_deps/liburing/lib:/opt/cray/libfabric/1.22.0/lib64:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export RESULT_DIR CHECKPOINT_BASE PYTHON_SNAPSHOT_ROOT

# Compute-node capability/path evidence is recorded separately from causal
# promotion.  Tool output and topology labels never become counters here.
python "${RESULT_DIR}/capture_g1_domain_counters_executed.py" \
    --capabilities "${RESULT_DIR}/domain_capabilities.json"

# Placement is part of the causal record: without a fixed local-rank/NUMA
# mapping, a host-pressure or PCIe result cannot be attributed to a domain.
srun --overlap --time=00:00:20 --ntasks=4 --ntasks-per-node=4 \
    --distribution=block:block --gpus-per-node=4 --gpu-bind=none \
    --cpu-bind="verbose,map_ldom:${PERLMUTTER_CPU_LDOM_MAP}" \
    --kill-on-bad-exit=1 --wait=3 --export=ALL \
    bash -euo pipefail -c '
        expected_local=$((SLURM_PROCID % 4))
        [[ "${SLURM_LOCALID}" == "${expected_local}" ]]
        printf "rank=%s local_rank=%s host=%s numa=%s cuda=%s\\n" \
            "${SLURM_PROCID}" "${SLURM_LOCALID}" "${HOSTNAME}" \
            "$(numactl --show 2>/dev/null | sed -n "s/^cpubind: //p")" \
            "${CUDA_VISIBLE_DEVICES:-unset}"
    ' > "${RESULT_DIR}/placement.txt"

python "${RESULT_DIR}/tier_attribution_runner_executed.py" \
    --command-plan-output "${RESULT_DIR}/g1_command_plan.json" \
    --repo-root "${REPO_ROOT}" --result-root "${RESULT_DIR}" \
    --checkpoint-root "${CHECKPOINT_BASE}" --steps 72 --warmup-steps 12 \
    --checkpoint-steps 16,52 --layers 2 --hidden-size 2048 --ffn-size 8192 \
    --heads 16 --sequence-length 64 --batch-size 1

# Bind the raw result to the exact executable snapshots and fixed G1 geometry.
# This is written before the first mode so a partial allocation remains
# diagnosable, while the final execution_status.env below is the completion
# marker consumed by validate_g1_tier_raw.py.
python - "${RESULT_DIR}" "${SLURM_JOB_ID}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
job_id = int(sys.argv[2])
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = {
    "schema_version": "tempo-rd-g1-tier-runtime-1",
    "stage": "g1_tier",
    "job_id": job_id,
    "nodes": 1,
    "world_size": 4,
    "state_bytes_per_rank": 402_714_696,
    "logical_file_extent_bytes": 402_890_752,
    "checkpoint_steps": [16, 52],
    "steps": 72,
    "warmup_steps": 12,
    "mode_outer_seconds": 35,
    "restore_outer_seconds": 20,
    "phase_budget_seconds": 235,
    "cleanup_reserve_seconds": 65,
    "geometry": {
        "layers": 2,
        "hidden_size": 2048,
        "ffn_size": 8192,
        "heads": 16,
        "sequence_length": 64,
        "batch_size": 1,
    },
    "host_pressure_placebo": {
        "mode": "fg_only",
        "buffer_bytes": 64 * 1024 * 1024,
        "source": "proc_self_numa_maps_plus_touch_loop",
        "rank_files": [
            "fg_only/host_pressure_rank_0.json",
            "fg_only/host_pressure_rank_1.json",
            "fg_only/host_pressure_rank_2.json",
            "fg_only/host_pressure_rank_3.json",
        ],
    },
    "modes": ["fg_only", "open_combined", "d2h_only", "persist_only", "combined"],
    "source_sha256": {
        "train_executed.py": sha256(root / "train_executed.py"),
        "tier_attribution_runner_executed.py": sha256(root / "tier_attribution_runner_executed.py"),
        "validate_g1_tier_raw_executed.py": sha256(root / "validate_g1_tier_raw_executed.py"),
        "validate_g1_result_executed.py": sha256(root / "validate_g1_result_executed.py"),
        "compose_g1_result_executed.py": sha256(root / "compose_g1_result_executed.py"),
        "build_g1_causal_readiness_executed.py": sha256(root / "build_g1_causal_readiness_executed.py"),
        "host_pressure_train_wrapper_executed.py": sha256(root / "host_pressure_train_wrapper_executed.py"),
        "host_pressure_placebo.py": sha256(root / "host_pressure_placebo.py"),
        "capture_g1_domain_counters_executed.py": sha256(root / "capture_g1_domain_counters_executed.py"),
        "prepare_foreground_path_executed.py": sha256(root / "prepare_foreground_path_executed.py"),
        "capture_nvml_pcie_observation_executed.py": sha256(root / "capture_nvml_pcie_observation_executed.py"),
        "capture_lustre_rpc_observation_executed.py": sha256(root / "capture_lustre_rpc_observation_executed.py"),
    },
    "slurm_submitted": True,
    "slurm_qos": os.environ.get("SLURM_JOB_QOS", ""),
    "no_retry": True,
    "logical_stage_schema": "tempo-rd-logical-stage-timing-1",
    "domain_counter_schema": "tempo-rd-domain-counter-record-2",
    "causal_readiness_builder": "build_g1_causal_readiness_executed.py",
}
(root / "g1_tier_runtime_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

STEPS=72
WARMUP_STEPS=12
CHECKPOINT_STEPS=16,52
WINDOW_STEPS=16
LAYERS=2
HIDDEN_SIZE=2048
FFN_SIZE=8192
HEADS=16
SEQUENCE_LENGTH=64
PROBE_MB=64
DEADLINE_SECONDS=1.0
SEED=20260811
# Five training modes plus three persistent restores must fit inside the
# ten-minute interactive allocation with a cleanup reserve.  These are outer
# caps, not deadline claims; a timeout is a failed raw artifact, never a
# partial causal result.
MODE_OUTER_SECONDS=35
RESTORE_OUTER_SECONDS=20
PHASE_BUDGET_SECONDS=$((5 * MODE_OUTER_SECONDS + 3 * RESTORE_OUTER_SECONDS))
if [[ "${SLURM_JOB_QOS:-}" == gpu_interactive || "${SLURM_JOB_QOS:-}" == interactive ]]; then
    ALLOCATION_SECONDS=600
else
    ALLOCATION_SECONDS=300
fi
[[ $((ALLOCATION_SECONDS - PHASE_BUDGET_SECONDS)) -ge 45 ]] || {
    echo "ERROR: G1 tier phase budget leaves less than 45s for cleanup" >&2
    exit 2
}
export STEPS WARMUP_STEPS CHECKPOINT_STEPS WINDOW_STEPS LAYERS HIDDEN_SIZE
export FFN_SIZE HEADS SEQUENCE_LENGTH PROBE_MB DEADLINE_SECONDS SEED

run_mode() {
    local mode=$1 policy=$2 endpoint=$3
    local mode_result="${RESULT_DIR}/${mode}"
    local mode_checkpoint="${CHECKPOINT_BASE}/${mode}"
    mkdir -p -- "${mode_result}" "${mode_checkpoint}"
    export TEMPO_RD_TIER_MODE="${mode}" TEMPO_RD_ENDPOINT="${endpoint}"
    if [[ "${mode}" == d2h_only ]]; then
        export TEMPO_RD_LOCAL_SINK_ROOT="${D2H_SINK_ROOT}"
        mkdir -p -- "${TEMPO_RD_LOCAL_SINK_ROOT}"
    else
        unset TEMPO_RD_LOCAL_SINK_ROOT
    fi
    local -a foreground_argv=(
        --policy "${policy}" --tier-mode "${mode}"
        --output-dir "${mode_result}" --checkpoint-dir "${mode_checkpoint}"
        --steps "${STEPS}" --warmup-steps "${WARMUP_STEPS}"
        --checkpoint-steps "${CHECKPOINT_STEPS}" --window-steps "${WINDOW_STEPS}"
        --layers "${LAYERS}" --hidden-size "${HIDDEN_SIZE}" --ffn-size "${FFN_SIZE}"
        --heads "${HEADS}" --sequence-length "${SEQUENCE_LENGTH}" --batch-size 1
        --probe-mb "${PROBE_MB}" --deadline-seconds "${DEADLINE_SECONDS}"
        --datastates-cache-gb 1 --seed "${SEED}"
    )
    local -a command=(python "${RESULT_DIR}/train_executed.py")
    if [[ "${TEMPO_RD_CAPTURE_NVML_PCIE:-}" == YES ]]; then
        mkdir -p -- "${RESULT_DIR}/domain_observations/${mode}/pcie"
        srun --overlap --time=00:00:20 --ntasks=4 --ntasks-per-node=4 \
            --distribution=block:block --gpus-per-node=4 --gpu-bind=none \
            --cpu-bind="verbose,map_ldom:${PERLMUTTER_CPU_LDOM_MAP}" \
            --kill-on-bad-exit=1 --wait=3 --export=ALL \
            python "${RESULT_DIR}/capture_nvml_pcie_observation_executed.py" \
            --mode "${mode}" --phase start \
            --output-root "${RESULT_DIR}/domain_observations/${mode}/pcie"
    fi
    if [[ "${mode}" == open_combined || "${mode}" == persist_only || "${mode}" == combined ]]; then
        mkdir -p -- "${RESULT_DIR}/domain_observations/${mode}/lustre_rpc"
        python "${RESULT_DIR}/capture_lustre_rpc_observation_executed.py" \
            --mode "${mode}" --phase start \
            --output-root "${RESULT_DIR}/domain_observations/${mode}/lustre_rpc" \
            || printf 'capture_failed=1\nphase=start\n' > "${RESULT_DIR}/domain_observations/${mode}/lustre_rpc/capture_error.env"
    fi
    if [[ "${mode}" == fg_only ]]; then
        # The foreground-only mode doubles as the independent host-pressure
        # placebo.  %r is expanded inside the rank-local wrapper, so no rank
        # writes another rank's evidence file.
        command=(
            python "${RESULT_DIR}/host_pressure_train_wrapper_executed.py"
            --train-script "${RESULT_DIR}/train_executed.py"
            --pressure-output "${mode_result}/host_pressure_rank_%r.json"
            --world-size 4 --buffer-mib 64
            --pressure-duration-ms 30000 --sample-ms 20 --
        )
    fi
    if [[ ( "${mode}" == fg_only || "${mode}" == open_combined || "${mode}" == d2h_only || "${mode}" == combined ) && "${TEMPO_RD_CAPTURE_D2H_NSYS:-}" == YES ]]; then
        export NSYS_MODE="${mode}"
        command=(bash "${NSYS_WRAPPER}" "${command[@]}")
    fi
    if [[ "${mode}" == fg_only || "${mode}" == open_combined || "${mode}" == d2h_only || "${mode}" == persist_only || "${mode}" == combined ]]; then
        python "${RESULT_DIR}/capture_g1_domain_counters_executed.py" \
            --mode "${mode}" --phase start \
            --output "${RESULT_DIR}/domain_counters/${mode}/nic_fabric.start.json"
    fi
    timeout --foreground --signal=TERM --kill-after=2s "${MODE_OUTER_SECONDS}s" \
        srun --overlap --time=00:01:00 --ntasks=4 --ntasks-per-node=4 \
        --distribution=block:block --gpus-per-node=4 --gpu-bind=none \
        --cpu-bind="verbose,map_ldom:${PERLMUTTER_CPU_LDOM_MAP}" \
        --kill-on-bad-exit=1 --wait=3 --export=ALL \
        "${command[@]}" "${foreground_argv[@]}"
    if [[ "${TEMPO_RD_CAPTURE_NVML_PCIE:-}" == YES ]]; then
        srun --overlap --time=00:00:20 --ntasks=4 --ntasks-per-node=4 \
            --distribution=block:block --gpus-per-node=4 --gpu-bind=none \
            --cpu-bind="verbose,map_ldom:${PERLMUTTER_CPU_LDOM_MAP}" \
            --kill-on-bad-exit=1 --wait=3 --export=ALL \
            python "${RESULT_DIR}/capture_nvml_pcie_observation_executed.py" \
            --mode "${mode}" --phase end \
            --output-root "${RESULT_DIR}/domain_observations/${mode}/pcie"
    fi
    if [[ "${mode}" == open_combined || "${mode}" == persist_only || "${mode}" == combined ]]; then
        mkdir -p -- "${RESULT_DIR}/domain_observations/${mode}/lustre_rpc"
        python "${RESULT_DIR}/capture_lustre_rpc_observation_executed.py" \
            --mode "${mode}" --phase end \
            --output-root "${RESULT_DIR}/domain_observations/${mode}/lustre_rpc" \
            || printf 'capture_failed=1\nphase=end\n' > "${RESULT_DIR}/domain_observations/${mode}/lustre_rpc/capture_error.env"
    fi
    if [[ ( "${mode}" == fg_only || "${mode}" == open_combined || "${mode}" == d2h_only || "${mode}" == combined ) && "${TEMPO_RD_CAPTURE_D2H_NSYS:-}" == YES ]]; then
        for rank in 0 1 2 3; do
            report="${NSYS_ROOT}/${mode}_rank_${rank}.nsys-rep"
            stats="${NSYS_ROOT}/${mode}_rank_${rank}.json_cuda_gpu_mem_size_sum.json"
            time_stats="${NSYS_ROOT}/${mode}_rank_${rank}.time_cuda_gpu_mem_time_sum.json"
            "${NSYS_BIN}" stats -r cuda_gpu_mem_size_sum -f json \
                -o "${NSYS_ROOT}/${mode}_rank_${rank}.json" "${report}"
            "${NSYS_BIN}" stats -r cuda_gpu_mem_time_sum -f json \
                -o "${NSYS_ROOT}/${mode}_rank_${rank}.time" "${report}"
            [[ -s "${stats}" ]] || { echo "ERROR: missing Nsight D2H stats ${stats}" >&2; exit 3; }
            [[ -s "${time_stats}" ]] || { echo "ERROR: missing Nsight D2H time stats ${time_stats}" >&2; exit 3; }
            SLURM_PROCID="${rank}" python "${RESULT_DIR}/capture_g1_domain_counters_executed.py" \
                --mode "${mode}" --nsys-report "${stats}" \
                --nsys-time-report "${time_stats}" \
                --output "${RESULT_DIR}/domain_counters/${mode}/gpu_local_rank_${rank}.json"
        done
        cp -- "${NSYS_ROOT}/${mode}_rank_0.json_cuda_gpu_mem_size_sum.json" \
            "${RESULT_DIR}/domain_counters/${mode}/gpu_local_nsys.json"
        if [[ "${mode}" == d2h_only || "${mode}" == open_combined || "${mode}" == combined ]]; then
            cp -- "${RESULT_DIR}/domain_counters/${mode}/gpu_local_rank_0.json" \
                "${RESULT_DIR}/domain_counters/${mode}/gpu_local.json"
        else
            python - "${RESULT_DIR}/domain_counters/fg_only/gpu_local_rank_0.json" \
                "${RESULT_DIR}/foreground_path_input.json" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
samples = record.get("samples")
if not isinstance(samples, list) or len(samples) < 2:
    raise SystemExit("foreground Nsight record has no complete counter series")
scope = record.get("scope")
scope_id = record.get("scope_id")
source = record.get("source")
if scope != "rank" or not isinstance(scope_id, str) or not scope_id or not isinstance(source, str) or not source:
    raise SystemExit("foreground Nsight record is missing rank-bound provenance")
bound_samples = []
for sample in samples:
    bound_samples.append({
        "domain": "gpu_local",
        "sample_id": sample["sample_id"],
        "source": source,
        "timestamp_ns": sample["timestamp_ns"],
        "cumulative_bytes": sample["cumulative_bytes"],
        "cumulative_busy_ns": sample["cumulative_busy_ns"],
        "support": sample["support"],
        "scope": scope,
        "scope_id": scope_id,
        "intervention_id": "fg_only",
    })
payload = {
    "domains": ["gpu_local"],
    "path_status": {"gpu_local": "observed"},
    "counter_support": {"gpu_local": "supported"},
    "path_evidence": {"gpu_local": "gpu_hbm_copy_engine"},
    "counter_family": {"gpu_local": "gpu_copy_engine_bytes"},
    "counters": {"gpu_local": bound_samples},
}
Path(sys.argv[2]).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
PY
            python "${RESULT_DIR}/prepare_foreground_path_executed.py" \
                "${RESULT_DIR}/foreground_path_input.json" \
                "${RESULT_DIR}/foreground_path.json" --intervention-id fg_only
        fi
    fi
    if [[ "${mode}" == fg_only || "${mode}" == open_combined || "${mode}" == d2h_only || "${mode}" == persist_only || "${mode}" == combined ]]; then
        python "${RESULT_DIR}/capture_g1_domain_counters_executed.py" \
            --mode "${mode}" --phase end \
            --output "${RESULT_DIR}/domain_counters/${mode}/nic_fabric.start.json"
        mv -- "${RESULT_DIR}/domain_counters/${mode}/nic_fabric.start.json" \
              "${RESULT_DIR}/domain_counters/${mode}/nic_fabric.json"
    fi
    if [[ "${mode}" == fg_only ]]; then
        for rank in 0 1 2 3; do
            [[ -s "${mode_result}/host_pressure_rank_${rank}.json" ]] || {
                echo "ERROR: host-pressure placebo record missing for rank ${rank}" >&2
                return 3
            }
        done
    fi
    printf 'train=%s\n' "${mode}" >> "${RESULT_DIR}/execution_status.env"
}

run_mode fg_only none none
run_mode open_combined datastates persistent_endpoint
run_mode d2h_only datastates node_local_sink
run_mode persist_only datastates persistent_endpoint
run_mode combined datastates persistent_endpoint

run_restore() {
    local mode=$1
    [[ "${mode}" == d2h_only ]] && return 0
        timeout --foreground --signal=TERM --kill-after=2s "${RESTORE_OUTER_SECONDS}s" \
        srun --overlap --time=00:00:30 --ntasks=4 --ntasks-per-node=4 \
        --distribution=block:block --gpus-per-node=4 --gpu-bind=none \
        --cpu-bind="verbose,map_ldom:${PERLMUTTER_CPU_LDOM_MAP}" \
        --kill-on-bad-exit=1 --wait=3 --export=ALL \
        python "${RESULT_DIR}/train_executed.py" --restore-only \
        --policy datastates --tier-mode "${mode}" \
        --output-dir "${RESULT_DIR}/${mode}" \
        --checkpoint-dir "${CHECKPOINT_BASE}/${mode}" \
        --steps "${STEPS}" --warmup-steps "${WARMUP_STEPS}" \
        --checkpoint-steps "${CHECKPOINT_STEPS}" --window-steps "${WINDOW_STEPS}" \
        --layers "${LAYERS}" --hidden-size "${HIDDEN_SIZE}" --ffn-size "${FFN_SIZE}" \
        --heads "${HEADS}" --sequence-length "${SEQUENCE_LENGTH}" --batch-size 1 \
        --probe-mb "${PROBE_MB}" --deadline-seconds "${DEADLINE_SECONDS}" \
        --datastates-cache-gb 1 --seed "${SEED}"
    printf 'restore=%s\n' "${mode}" >> "${RESULT_DIR}/execution_status.env"
}
run_restore open_combined
run_restore persist_only
run_restore combined
printf 'schema_version=tempo-rd-g1-tier-execution-1\nstatus=raw_complete\n' \
    >> "${RESULT_DIR}/execution_status.env"
# A site-specific GPU/NVLink/PCIe/NUMA collector may publish a raw
# foreground-path record through this opt-in input.  The helper only validates
# and canonicalizes it; absence intentionally leaves readiness not_ready.
if [[ -n "${TEMPO_RD_FOREGROUND_PATH_INPUT:-}" ]]; then
    python "${RESULT_DIR}/prepare_foreground_path_executed.py" \
        "${TEMPO_RD_FOREGROUND_PATH_INPUT}" \
        "${RESULT_DIR}/foreground_path.json" \
        --intervention-id fg_only
fi
# This is a diagnostic boundary, not a promotion step.  The builder exits
# successfully for both ready and not_ready so a partial raw artifact remains
# inspectable; its JSON is the explicit input required before any G2 request.
python "${RESULT_DIR}/build_g1_causal_readiness_executed.py" \
    "${RESULT_DIR}" --output "${RESULT_DIR}/causal_readiness.json"
echo "TEMPO_RD_G1_TIER_RAW_COMPLETE result=${RESULT_DIR}"
