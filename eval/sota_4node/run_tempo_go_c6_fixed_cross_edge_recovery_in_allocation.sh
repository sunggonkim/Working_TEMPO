#!/usr/bin/env bash
set -euo pipefail

[[ "${TEMPO_GO_C6_FIXED_CROSS_APPROVED:-}" == YES ]] || exit 2
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 2
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_JOB_NODES:-}}" == 4 ]] || exit 2
[[ -z "${SHIFTER_RUNTIME:-}${SHIFTER_IMAGE:-}${UDI:-}${CRAY_ROOTFS:-}${SLURM_CONTAINER:-}" ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONTRACT="${REPO_ROOT}/eval/sota_4node/tempo_go_c6_fixed_cross_edge_recovery_contract_v1.json"
[[ -s "${CONTRACT}" ]]
[[ "$(jq -er '.schema' "${CONTRACT}")" == tempo-go-c6-qualification-contract-v1 ]]
[[ "$(jq -er '.qualification_kind' "${CONTRACT}")" == fixed_cross_edge_recovery ]]
[[ "$(jq -er '.fixed_cross_edge_recovery.remote_decode_placement' "${CONTRACT}")" == cross ]]

SOURCE_REL=$(jq -er '.fixed_cross_edge_recovery.source_workload.path' "${CONTRACT}")
PROFILE_REL=$(jq -er '.fixed_cross_edge_recovery.profile.path' "${CONTRACT}")
SOURCE_WORKLOAD=$(realpath -e -- "${REPO_ROOT}/${SOURCE_REL}")
PROFILE=$(realpath -e -- "${REPO_ROOT}/${PROFILE_REL}")
[[ "$(sha256sum "${SOURCE_WORKLOAD}" | awk '{print $1}')" == "$(jq -er '.fixed_cross_edge_recovery.source_workload.sha256' "${CONTRACT}")" ]]
[[ "$(sha256sum "${PROFILE}" | awk '{print $1}')" == "$(jq -er '.fixed_cross_edge_recovery.profile.sha256' "${CONTRACT}")" ]]

RESULT_DIR="${TEMPO_GO_C6_FIXED_CROSS_RESULT_DIR:-${REPO_ROOT}/results/tempo_go_c6_fixed_cross_edge_job_${SLURM_JOB_ID}}"
case "${RESULT_DIR}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ ! -e "${RESULT_DIR}" ]]
mkdir -p -- "${RESULT_DIR}"

module reset
module load pytorch/2.8.0
[[ "${NCCL_NET:-}" == "AWS Libfabric" ]] || {
  echo "expected NERSC NCCL AWS Libfabric transport, got ${NCCL_NET:-<unset>}" >&2
  exit 1
}
unset NCCL_IB_DISABLE
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
export TEMPO_GO_C6_QUALIFICATION_CONTRACT="${CONTRACT}"
export TEMPO_ELASTIC_PD_PROFILE="${PROFILE}"
export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_VLLM_MAX_NUM_SEQS=16
export TEMPO_VLLM_DECODER_PREFIX_CACHING=0
export TEMPO_LMCACHE_NIXL_BACKEND=UCX
export TEMPO_PD_REMOTE_DECODE_PLACEMENT=cross
unset TEMPO_CXI_BACKGROUND_DUTY_CYCLE TEMPO_CXI_BACKGROUND_START_FILE

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
PORT_SLOT=$((1760 + SLURM_JOB_ID % 20))
REQUEST_RATE=$(jq -er '.fixed_cross_edge_recovery.victim.offered_rate_per_s' "${CONTRACT}")
MAX_WORKERS=$(jq -er '.fixed_cross_edge_recovery.max_workers' "${CONTRACT}")
TPOT_SLO=$(jq -er '.fixed_cross_edge_recovery.slo.tpot_ms' "${CONTRACT}")
E2E_SLO=$(jq -er '.fixed_cross_edge_recovery.slo.e2e_ms' "${CONTRACT}")

cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=30s 2400s \
  /usr/bin/srun --overlap --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
  --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
  --time=00:39:00 --export=ALL \
  --output="${RESULT_DIR}/slurm-node-%N.stdout.log" \
  --error="${RESULT_DIR}/slurm-node-%N.stderr.log" \
  bash "${SCRIPT_DIR}/c6_decoder_victim_node_entry.sh" \
  "${REPO_ROOT}" "${RESULT_DIR}" "${SOURCE_WORKLOAD}" "${HOSTS_CSV}" \
  "${PORT_SLOT}" "${REQUEST_RATE}" "${MAX_WORKERS}" 128 8 3000 \
  "${TPOT_SLO}" "${E2E_SLO}"

[[ -s "${RESULT_DIR}/result.json" ]]
echo "TEMPO-GO C6 fixed cross-edge receipt: ${RESULT_DIR}/result.json"
