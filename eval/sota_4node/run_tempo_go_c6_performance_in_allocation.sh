#!/usr/bin/env bash
set -euo pipefail

[[ "${TEMPO_GO_C6_PERFORMANCE_APPROVED:-}" == YES ]] || exit 2
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 2
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_JOB_NODES:-}}" == 4 ]] || exit 2
[[ -z "${SHIFTER_RUNTIME:-}${SHIFTER_IMAGE:-}${UDI:-}${CRAY_ROOTFS:-}${SLURM_CONTAINER:-}" ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONTRACT="${TEMPO_GO_C6_PERFORMANCE_CONTRACT:-${REPO_ROOT}/eval/sota_4node/tempo_go_c6_performance_contract_v1.json}"
CONTRACT=$(realpath -e -- "${CONTRACT}")
case "${CONTRACT}" in "${REPO_ROOT}/"*) ;; *) exit 2 ;; esac
[[ -s "${CONTRACT}" ]]
[[ "$(jq -er '.schema' "${CONTRACT}")" == tempo-go-c6-performance-contract-v1 ]]
[[ "$(jq -er '.claim_boundary.controller_performance_claim_allowed' "${CONTRACT}")" == true ]]

SOURCE_REL=$(jq -er '.c6_performance.source_workload.path' "${CONTRACT}")
PROFILE_REL=$(jq -er '.c6_performance.profile.path' "${CONTRACT}")
SOURCE_WORKLOAD=$(realpath -e -- "${REPO_ROOT}/${SOURCE_REL}")
PROFILE=$(realpath -e -- "${REPO_ROOT}/${PROFILE_REL}")
[[ "$(sha256sum "${SOURCE_WORKLOAD}" | awk '{print $1}')" == "$(jq -er '.c6_performance.source_workload.sha256' "${CONTRACT}")" ]]
[[ "$(sha256sum "${PROFILE}" | awk '{print $1}')" == "$(jq -er '.c6_performance.profile.sha256' "${CONTRACT}")" ]]

RESULT_ROOT="${TEMPO_GO_C6_PERFORMANCE_RESULT_DIR:-${REPO_ROOT}/results/tempo_go_c6_performance_job_${SLURM_JOB_ID}}"
EPOCH_ONLY="${TEMPO_GO_C6_EPOCH_ONLY:-}"
case "${EPOCH_ONLY}" in
  ""|full_c6|predictor|queue_gpu|network_request_only|app_global_only) ;;
  *) exit 2 ;;
esac
case "${RESULT_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ ! -e "${RESULT_ROOT}" ]]
mkdir -p -- "${RESULT_ROOT}"

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
export TEMPO_GO_C6_PERFORMANCE_CONTRACT="${CONTRACT}"
export TEMPO_ELASTIC_PD_PROFILE="${PROFILE}"
export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_VLLM_MAX_NUM_SEQS=16
export TEMPO_VLLM_DECODER_PREFIX_CACHING=0
export TEMPO_LMCACHE_NIXL_BACKEND=UCX
unset TEMPO_CXI_BACKGROUND_DUTY_CYCLE TEMPO_CXI_BACKGROUND_START_FILE

if [[ -n "${TEMPO_GO_C6_HOSTS_CSV:-}" ]]; then
  IFS=, read -r -a HOSTS <<< "${TEMPO_GO_C6_HOSTS_CSV}"
else
  mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
fi
[[ ${#HOSTS[@]} -eq 4 ]]
[[ -n "${HOSTS[0]}" && -n "${HOSTS[1]}" && -n "${HOSTS[2]}" && -n "${HOSTS[3]}" ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
BASE_PORT_SLOT=$((1780 + SLURM_JOB_ID % 10))
REQUEST_RATE=$(jq -er '.c6_performance.victim.offered_rate_per_s' "${CONTRACT}")
MAX_WORKERS=$(jq -er '.c6_performance.max_workers' "${CONTRACT}")

run_epoch() {
  local label=$1
  local arm=$2
  local fixed_policy=$3
  local port_slot=$4
  local result_dir="${RESULT_ROOT}/${label}"
  mkdir -p -- "${result_dir}"
  export TEMPO_GO_C6_PERFORMANCE_ARM="${arm}"
  if [[ -n "${fixed_policy}" ]]; then
    export TEMPO_GO_C6_FIXED_POLICY="${fixed_policy}"
  else
    unset TEMPO_GO_C6_FIXED_POLICY
  fi
  cd -- "${REPO_ROOT}"
  timeout --foreground --signal=TERM --kill-after=30s 1500s \
    /usr/bin/srun --jobid="${SLURM_JOB_ID}" --overlap --exact \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
    --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
    --time=00:24:00 --export=ALL \
    --output="${result_dir}/slurm-node-%N.stdout.log" \
    --error="${result_dir}/slurm-node-%N.stderr.log" \
    bash "${SCRIPT_DIR}/c6_performance_node_entry.sh" \
    "${REPO_ROOT}" "${result_dir}" "${SOURCE_WORKLOAD}" "${HOSTS_CSV}" \
    "${port_slot}" "${REQUEST_RATE}" "${MAX_WORKERS}" 128 8 3000 250 16000
  [[ -s "${result_dir}/result.json" ]]
}

if [[ -n "${EPOCH_ONLY}" ]]; then
  case "${EPOCH_ONLY}" in
    full_c6) PORT_OFFSET=40 ;;
    predictor) PORT_OFFSET=50 ;;
    queue_gpu) PORT_OFFSET=60 ;;
    network_request_only) PORT_OFFSET=70 ;;
    app_global_only) PORT_OFFSET=80 ;;
  esac
  run_epoch "${EPOCH_ONLY}" "${EPOCH_ONLY}" "" \
    "$((BASE_PORT_SLOT + PORT_OFFSET))"
  echo "TEMPO-GO C6 ${EPOCH_ONLY} receipt: ${RESULT_ROOT}/${EPOCH_ONLY}/result.json"
  exit 0
fi

run_epoch fixed_p0d1 fixed fixed_p0d1 "${BASE_PORT_SLOT}"
run_epoch fixed_p1d0 fixed fixed_p1d0 "$((BASE_PORT_SLOT + 20))"
run_epoch full_c6 full_c6 "" "$((BASE_PORT_SLOT + 40))"

"${REPO_ROOT}/.vllm_venv/bin/python" \
  -m eval.sota_4node.analyze_tempo_go_c6_performance \
  --fixed-p0d1-result "${RESULT_ROOT}/fixed_p0d1/result.json" \
  --fixed-p1d0-result "${RESULT_ROOT}/fixed_p1d0/result.json" \
  --full-result "${RESULT_ROOT}/full_c6/result.json" \
  --contract "${CONTRACT}" \
  --output "${RESULT_ROOT}/analysis.json"
[[ -s "${RESULT_ROOT}/analysis.json" ]]
echo "TEMPO-GO C6 performance receipt: ${RESULT_ROOT}/analysis.json"
