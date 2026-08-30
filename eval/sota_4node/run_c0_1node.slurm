#!/usr/bin/env bash
# Reverse-order one-node replication: fixed-rate C0 followed by matched v4_open.
#SBATCH --job-name=tempo_c0_ab
#SBATCH --account=m5320_g
#SBATCH --qos=debug_preempt
#SBATCH --constraint=gpu&hbm40g
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --no-requeue
#SBATCH --mail-type=NONE
#SBATCH --output=results/c0_live_%j.out
#SBATCH --error=results/c0_live_%j.err

set -euo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:?C0 requires the repository root}
REPO_ROOT=$(realpath -e -- "${REPO_ROOT}")
[[ "${TEMPO_C0_APPROVED:-}" == YES ]] || {
    echo "ERROR: set TEMPO_C0_APPROVED=YES only after explicit approval" >&2
    exit 2
}
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 && "${SLURM_NTASKS:-}" == 4 ]] || {
    echo "ERROR: C0 screen requires one node and four ranks" >&2
    exit 2
}
[[ "${PSCRATCH:-}" == /pscratch/sd/s/sgkim ]] || {
    echo "ERROR: unexpected PSCRATCH" >&2
    exit 2
}

RUNTIME="${REPO_ROOT}/.baseline_deps/runtime"
LIBURING_ROOT="${REPO_ROOT}/.baseline_deps/liburing"
[[ -s "${RUNTIME}/datastates_build_manifest.txt" ]] || {
    echo "ERROR: rebuild .baseline_deps with setup_baselines.sh before allocation" >&2
    exit 2
}

module reset
module load pytorch/2.8.0
export PYTHONPATH="${RUNTIME}:${REPO_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${RUNTIME}/datastates/lib:${LIBURING_ROOT}/lib:${LD_LIBRARY_PATH:-}"
python -c 'import datastates, msgpack; assert msgpack.__version__ == "1.1.1"'

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export WORLD_SIZE=4
export PERLMUTTER_CPU_LDOM_MAP=3,2,1,0
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')
export MASTER_ADDR
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 10000))
export NCCL_DEBUG=WARN
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_IB_DISABLE NCCL_CROSS_NIC NCCL_ALGO NCCL_PROTO

RESULT_DIR="${REPO_ROOT}/results/c0_live_job_${SLURM_JOB_ID}"
CHECKPOINT_BASE="${PSCRATCH}/tempo_c0_checkpoints/job_${SLURM_JOB_ID}"
[[ ! -e "${RESULT_DIR}" && ! -e "${CHECKPOINT_BASE}" ]] || {
    echo "ERROR: C0 result/checkpoint path already exists" >&2
    exit 2
}
mkdir -p -- "${RESULT_DIR}" "${CHECKPOINT_BASE}"

cleanup() {
    local rc=$?
    trap - EXIT TERM INT
    rm -rf -- "${CHECKPOINT_BASE}"
    exit "${rc}"
}
trap cleanup EXIT TERM INT

C0_RATE_GBPS=5.936536675
PFS_RATE_GBPS=2.211348539
COMMON_ARGS=(
    --policy v4_open
    --tier-mode open_combined
    --steps 72
    --warmup-steps 12
    --checkpoint-steps 16,52
    --window-steps 16
    --layers 2
    --hidden-size 2048
    --ffn-size 8192
    --heads 16
    --sequence-length 64
    --batch-size 1
    --probe-mb 64
    --deadline-seconds 1.0
    --datastates-cache-gb 1
    --tempo-v4-d2h-floor-gbps "${C0_RATE_GBPS}"
    --tempo-v4-pfs-floor-gbps "${PFS_RATE_GBPS}"
    --tempo-v4-telemetry off
    --seed 20260811
)

run_case() {
    local mode=$1 expected_c0=$2
    shift 2
    local output_dir="${RESULT_DIR}/${mode}"
    local checkpoint_dir="${CHECKPOINT_BASE}/${mode}"
    mkdir -p -- "${output_dir}" "${checkpoint_dir}"
    timeout --foreground --signal=TERM --kill-after=3s 35s         srun --overlap --time=00:00:45         --ntasks=4 --ntasks-per-node=4         --distribution=block:block         --gpus-per-node=4 --gpu-bind=none         --cpu-bind="map_ldom:${PERLMUTTER_CPU_LDOM_MAP}"         --kill-on-bad-exit=1 --wait=3 --export=ALL         python "${REPO_ROOT}/eval/sota_4node/train.py"         "${COMMON_ARGS[@]}"         --output-dir "${output_dir}"         --checkpoint-dir "${checkpoint_dir}"         "$@"
    python - "${output_dir}" "${expected_c0}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = sys.argv[2] == "1"
for rank in range(4):
    summary = json.loads((root / f"summary_rank{rank}.json").read_text())
    events = json.loads((root / f"checkpoint_events_rank{rank}.json").read_text())
    assert summary["c0_enabled"] is expected
    assert summary["c0_max_inflight_bytes"] == 1 << 20
    assert len(events) == 2
    assert all(
        event["commit_validated"] and event["fsync_evidence_valid"] for event in events
    )
    assert (root / f"collectives_rank{rank}.csv").stat().st_size > 0
    assert (root / f"steps_rank{rank}.csv").stat().st_size > 0
    if expected:
        assert all(event["deadline_met"] for event in events)
        assert summary["c0_d2h_rate_bps"] == 5_936_536_675
PY
}

run_case c0 1 --v4-open-c0
run_case open 0
python "${REPO_ROOT}/eval/sota_4node/analyze_c0_live.py"     "${RESULT_DIR}" --output "${RESULT_DIR}/C0_AB_RESULT.json"
echo "TEMPO_C0_AB_COMPLETE result=${RESULT_DIR}/C0_AB_RESULT.json"
