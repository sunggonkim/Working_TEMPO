#!/usr/bin/env bash
# Two-node raw composite/fabric attribution; no scheduler promotion.
#SBATCH --job-name=tempo_g2_fabric_raw
#SBATCH --account=m5320_g
#SBATCH --qos=debug_preempt
#SBATCH --constraint=gpu&hbm80g
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --no-requeue
#SBATCH --mail-type=NONE
#SBATCH --output=logs/slurm/tempo_g2_fabric_raw_%j.out
#SBATCH --error=logs/slurm/tempo_g2_fabric_raw_%j.err
#SBATCH --signal=B:USR1@45

set -euo pipefail
REPO_ROOT=$(realpath -e -- "${SLURM_SUBMIT_DIR:?submit from repository root}")
[[ "${TEMPO_RD_APPROVE_G2_RAW:-}" == YES ]] || exit 2
[[ "${SLURM_JOB_NUM_NODES:-}" == 2 && "${SLURM_NTASKS:-}" == 8 ]] || exit 2
[[ "${PSCRATCH:-}" == /pscratch/sd/s/sgkim ]] || exit 2

bundle_sha() {
    ( cd "${REPO_ROOT}"; { sha256sum eval/sota_4node/run_g2_fabric_raw_2node.slurm eval/sota_4node/run_g2_fabric_raw.py eval/sota_4node/train.py eval/sota_4node/capture_g1_domain_counters.py eval/sota_4node/build_g2_fabric_observation.py; find tempo -type f -name '*.py' -print | sort | xargs sha256sum; } | sha256sum | awk '{print $1}' )
}
SOURCE_BUNDLE_SHA256=$(bundle_sha)
[[ "${TEMPO_RD_G2_RAW_EXPECTED_SOURCE_BUNDLE_SHA256:-}" == "${SOURCE_BUNDLE_SHA256}" ]] || exit 2

module reset
module load pytorch/2.8.0
python -c 'import msgpack; assert msgpack.__version__ == "1.1.1"'
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 WORLD_SIZE=8
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 10000))
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_IB_DISABLE NCCL_CROSS_NIC NCCL_ALGO NCCL_PROTO

RESULT_DIR="${REPO_ROOT}/results/sota_4node/g2_fabric_raw_job_${SLURM_JOB_ID}"
CHECKPOINT_BASE="${PSCRATCH}/tempo_rd_g2_fabric_checkpoints/job_${SLURM_JOB_ID}"
D2H_SINK_ROOT="${TMPDIR:-/tmp}/tempo_rd_g2_fabric_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}" "${CHECKPOINT_BASE}" "${D2H_SINK_ROOT}"
trap 'rc=$?; find "${CHECKPOINT_BASE}" "${D2H_SINK_ROOT}" -mindepth 1 -depth -delete 2>/dev/null || true; rmdir "${CHECKPOINT_BASE}" "${D2H_SINK_ROOT}" 2>/dev/null || true; printf "schema_version=tempo-rd-g2-fabric-raw-execution-1\nsource_bundle_sha256=%s\nexit_code=%s\n" "${SOURCE_BUNDLE_SHA256}" "${rc}" > "${RESULT_DIR}/cleanup_status.env"; exit "${rc}"' EXIT
trap 'exit 143' TERM USR1

SNAPSHOT="${RESULT_DIR}/python_snapshot"
mkdir -p "${SNAPSHOT}"
cp -a "${REPO_ROOT}/tempo" "${SNAPSHOT}/tempo"
cp "${REPO_ROOT}/eval/sota_4node/train.py" "${RESULT_DIR}/train_executed.py"
cp "${REPO_ROOT}/eval/sota_4node/capture_g1_domain_counters.py" "${RESULT_DIR}/capture_g2_counters_executed.py"
cp "${REPO_ROOT}/eval/sota_4node/build_g2_fabric_observation.py" "${RESULT_DIR}/build_g2_fabric_observation_executed.py"
cp "${REPO_ROOT}/eval/sota_4node/run_g2_fabric_raw.py" "${RESULT_DIR}/run_g2_fabric_raw_executed.py"
export PYTHONPATH="${SNAPSHOT}:${REPO_ROOT}/.baseline_deps/runtime" PYTHONSAFEPATH=1
export LD_LIBRARY_PATH="${REPO_ROOT}/.baseline_deps/runtime/datastates/lib:${REPO_ROOT}/.baseline_deps/liburing/lib:/opt/cray/libfabric/1.22.0/lib64:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export RESULT_DIR D2H_SINK_ROOT

cat > "${RESULT_DIR}/raw_manifest.json" <<EOF
{"schema_version":"tempo-rd-g2-fabric-raw-runtime-1","job_id":${SLURM_JOB_ID},"nodes":2,"world_size":8,"source_bundle_sha256":"${SOURCE_BUNDLE_SHA256}","evidence_state":"raw_observation","promotion_eligible":false}
EOF
python "${RESULT_DIR}/run_g2_fabric_raw_executed.py" --result-dir "${RESULT_DIR}" --checkpoint-root "${CHECKPOINT_BASE}" --train "${RESULT_DIR}/train_executed.py" --capture "${RESULT_DIR}/capture_g2_counters_executed.py" --observer "${RESULT_DIR}/build_g2_fabric_observation_executed.py"
