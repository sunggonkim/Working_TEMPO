#!/bin/bash
#SBATCH --job-name=tempo_g2_p2p_path
#SBATCH --account=m5320_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=debug_preempt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --mail-type=NONE
#SBATCH --output=logs/slurm/tempo_g2_p2p_path_%j.out
#SBATCH --error=logs/slurm/tempo_g2_p2p_path_%j.err
#SBATCH --signal=B:USR1@45

set -euo pipefail

if [[ "${TEMPO_RD_APPROVE_G2_P2P_RAW:-}" != "YES" ]]; then
  echo "refusing path-intervention submission without TEMPO_RD_APPROVE_G2_P2P_RAW=YES" >&2
  exit 2
fi
REPO_ROOT="$(realpath -e -- "${SLURM_SUBMIT_DIR:?submit from repository root}")"
EXPECTED="${TEMPO_RD_G2_P2P_EXPECTED_SOURCE_BUNDLE_SHA256:-}"
SOURCE_BUNDLE_SHA256="$(
  cd "${REPO_ROOT}"
  {
    sha256sum eval/sota_4node/run_g2_p2p_path_raw_2node.slurm \
      eval/sota_4node/run_g2_fabric_raw.py eval/sota_4node/train.py \
      eval/sota_4node/capture_g1_domain_counters.py \
      eval/sota_4node/build_g2_fabric_observation.py
    find tempo -type f -name '*.py' -print | sort | xargs sha256sum
  } | sha256sum | awk '{print $1}'
)"
if [[ -n "${EXPECTED}" && "${SOURCE_BUNDLE_SHA256}" != "${EXPECTED}" ]]; then
  echo "source bundle mismatch expected=${EXPECTED} actual=${SOURCE_BUNDLE_SHA256}" >&2
  exit 3
fi

module load pytorch/2.8.0
RESULT_DIR="${REPO_ROOT}/results/sota_4node/g2_p2p_path_job_${SLURM_JOB_ID}"
CHECKPOINT_ROOT="${SCRATCH:-/tmp}/tempo-rd-g2-p2p-${SLURM_JOB_ID}"
D2H_SINK_ROOT="${SCRATCH:-/tmp}/tempo-rd-g2-p2p-d2h-${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}" "${CHECKPOINT_ROOT}" "${D2H_SINK_ROOT}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 WORLD_SIZE=8
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 10000))
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
unset NCCL_SHM_DISABLE NCCL_IB_DISABLE NCCL_CROSS_NIC NCCL_ALGO NCCL_PROTO
cleanup() {
  rc=$?
  find "${CHECKPOINT_ROOT}" "${D2H_SINK_ROOT}" -mindepth 1 -depth -delete 2>/dev/null || true
  rmdir "${CHECKPOINT_ROOT}" "${D2H_SINK_ROOT}" 2>/dev/null || true
  {
    printf 'schema_version=tempo-rd-g2-p2p-path-execution-1\n'
    printf 'source_bundle_sha256=%s\n' "${SOURCE_BUNDLE_SHA256}"
    printf 'path_intervention=NCCL_P2P_DISABLE=1\n'
    printf 'exit_code=%s\n' "${rc}"
  } > "${RESULT_DIR}/cleanup_status.env"
  exit "${rc}"
}
trap cleanup EXIT

cp "${REPO_ROOT}/eval/sota_4node/run_g2_fabric_raw.py" "${RESULT_DIR}/run_g2_fabric_raw_executed.py"
cp "${REPO_ROOT}/eval/sota_4node/train.py" "${RESULT_DIR}/train_executed.py"
cp "${REPO_ROOT}/eval/sota_4node/capture_g1_domain_counters.py" "${RESULT_DIR}/capture_g1_domain_counters_executed.py"
cp "${REPO_ROOT}/eval/sota_4node/build_g2_fabric_observation.py" "${RESULT_DIR}/build_g2_fabric_observation_executed.py"
mkdir -p "${RESULT_DIR}/tempo"
cp -a "${REPO_ROOT}/tempo/." "${RESULT_DIR}/tempo/"
cat > "${RESULT_DIR}/raw_manifest.json" <<EOF
{"schema_version":"tempo-rd-g2-p2p-path-runtime-1","job_id":${SLURM_JOB_ID},"nodes":2,"world_size":8,"source_bundle_sha256":"${SOURCE_BUNDLE_SHA256}","path_intervention":{"name":"NCCL_P2P_DISABLE","value":"1"},"evidence_state":"raw_path_intervention","promotion_eligible":false}
EOF

# This is a path intervention, not a scheduler promotion.  Keep the rest of
# the raw matrix identical to the source-bound two-node observation.
export NCCL_P2P_DISABLE=1
export TEMPO_RD_PATH_INTERVENTION=NCCL_P2P_DISABLE=1
export RESULT_DIR
export WORLD_SIZE=8
export PYTHONSAFEPATH=1
export PYTHONPATH="${RESULT_DIR}:${REPO_ROOT}/.baseline_deps/runtime"
export LD_LIBRARY_PATH="${REPO_ROOT}/.baseline_deps/runtime/datastates/lib:${REPO_ROOT}/.baseline_deps/liburing/lib:/opt/cray/libfabric/1.22.0/lib64:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
python "${RESULT_DIR}/run_g2_fabric_raw_executed.py" \
  --result-dir "${RESULT_DIR}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --train "${RESULT_DIR}/train_executed.py" \
  --capture "${RESULT_DIR}/capture_g1_domain_counters_executed.py" \
  --observer "${RESULT_DIR}/build_g2_fabric_observation_executed.py"
printf 'schema_version=tempo-rd-g2-p2p-path-execution-1\nstatus=raw_complete\n' > "${RESULT_DIR}/execution_status.env"
