#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/require_perlmutter_4node_4h_interactive.sh"
: "${TEMPO_PD_C4_SEMANTIC_APPROVED:?explicit semantic-screen approval required}"
: "${TEMPO_PD_C4_SEMANTIC_RUN_CONTRACT_SHA256:?semantic contract SHA required}"
[[ "${TEMPO_PD_C4_SEMANTIC_APPROVED}" == YES ]]
[[ "${TEMPO_PD_C4_SEMANTIC_RUN_CONTRACT_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ $# -eq 2 ]]

while IFS= read -r name; do
  case "${name}" in
    TEMPO_PD_C4_SEMANTIC_APPROVED|TEMPO_PD_C4_SEMANTIC_RUN_CONTRACT_SHA256|TEMPO_PD_C4_STEP_TIME|TEMPO_PD_C4_READINESS_S|TEMPO_PD_C4_LIFECYCLE_S|TEMPO_C4_OVERLAY_TAG) ;;
    *)
      echo "semantic screen refuses inherited experiment variable: ${name}" >&2
      exit 2
      ;;
  esac
done < <(compgen -e TEMPO_)

while IFS= read -r name; do
  case "${name}" in
    NCCL_NET_GDR_LEVEL)
      [[ "${!name}" == PHB ]] || {
        echo "semantic screen refuses inherited transport variable: ${name}" >&2
        exit 2
      }
      ;;
    FI_*|UCX_*|NIXL_*|LMCACHE_*|VLLM_*|NCCL_*|CUDA_VISIBLE_DEVICES)
      echo "semantic screen refuses inherited transport variable: ${name}" >&2
      exit 2
      ;;
  esac
done < <(compgen -e)

REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RUN_CONTRACT=$(realpath -e -- "$1")
RESULT_DIR=$(realpath -m -- "$2")
case "${RUN_CONTRACT}" in "${REPO_ROOT}"/results/*) ;; *) exit 2 ;; esac
case "${RESULT_DIR}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ -s "${RUN_CONTRACT}" && ! -e "${RESULT_DIR}" ]]
[[ "$(sha256sum "${RUN_CONTRACT}" | awk '{print $1}')" == \
  "${TEMPO_PD_C4_SEMANTIC_RUN_CONTRACT_SHA256}" ]]

resolve_contract_path() {
  local raw
  raw=$(jq -er "$1.path" "${RUN_CONTRACT}")
  case "${raw}" in
    /*) realpath -e -- "${raw}" ;;
    *) realpath -e -- "${REPO_ROOT}/${raw}" ;;
  esac
}

WORKLOAD=$(resolve_contract_path '.source_workload')
PHASE_MANIFEST=$(resolve_contract_path '.phase_manifest')
ELASTIC_PROFILE=$(resolve_contract_path '.elastic_profile')
ENDPOINT_PROFILE=$(resolve_contract_path '.endpoint_service_profile')
[[ "$(sha256sum "${WORKLOAD}" | awk '{print $1}')" == \
  "$(jq -er '.source_workload.sha256' "${RUN_CONTRACT}")" ]]
[[ "$(sha256sum "${PHASE_MANIFEST}" | awk '{print $1}')" == \
  "$(jq -er '.phase_manifest.sha256' "${RUN_CONTRACT}")" ]]
[[ "$(sha256sum "${ELASTIC_PROFILE}" | awk '{print $1}')" == \
  "$(jq -er '.elastic_profile.sha256' "${RUN_CONTRACT}")" ]]
[[ "$(sha256sum "${ENDPOINT_PROFILE}" | awk '{print $1}')" == \
  "$(jq -er '.endpoint_service_profile.sha256' "${RUN_CONTRACT}")" ]]

module reset
module load pytorch/2.8.0
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
PYTHON="${REPO_ROOT}/.vllm_venv/bin/python"
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
PORT_SLOT=$((1860 + SLURM_JOB_ID % 20))
STEP_TIME=${TEMPO_PD_C4_STEP_TIME:-01:30:00}

export TEMPO_PD_C4_APPROVED=YES
export TEMPO_PD_C4_RUN_CONTRACT="${RUN_CONTRACT}"
export TEMPO_PD_C4_RUN_CONTRACT_SHA256="${TEMPO_PD_C4_SEMANTIC_RUN_CONTRACT_SHA256}"
export TEMPO_PD_C4_PHASE_DURATION_MS=15000
export TEMPO_PD_C4_COOLDOWN_S=2
export TEMPO_PD_C4_READINESS_S=${TEMPO_PD_C4_READINESS_S:-3600}
export TEMPO_ELASTIC_PD_PROFILE="${ELASTIC_PROFILE}"
export TEMPO_ELASTIC_PD_PROFILE_SCOPE=screen_only
export TEMPO_PD_ENDPOINT_FEEDBACK_MODE=adaptive
export TEMPO_PD_ENDPOINT_ROUTING_POLICY=semantic_epoch_v1
export TEMPO_PD_ENDPOINT_SERVICE_PROFILE="${ENDPOINT_PROFILE}"
export TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256
TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256=$(sha256sum \
  "${PHASE_MANIFEST}" | awk '{print $1}')
export TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK=1
export TEMPO_PD_PRESSURE_MODE=disabled
export TEMPO_VLLM_LOAD_SNAPSHOT_MODE=disabled
export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_PD_BENCHMARK_RESET_DECODER_APC=0
export TEMPO_VLLM_DECODER_PREFIX_CACHING=0
export TEMPO_PD_FRONTEND_PAIR_POLICY=tempo-min-outstanding-decode-tokens-v1
export TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY=1
export TEMPO_PD_DECODER_REUSE_ITEMS=all
export TEMPO_PD_FORWARD_TOKEN_IDS=0
export TEMPO_PD_PROXY_KV_CONTROL_OVERLAP=0
export TEMPO_PD_REMOTE_DECODE_PLACEMENT=paired
export TEMPO_PD_PROXY_TOKENIZER_PLACEMENT=round_robin
export TEMPO_VLLM_MAX_NUM_SEQS=16
export TEMPO_VLLM_ASYNC_SCHEDULING=0
export TEMPO_VLLM_DECODER_MAX_NUM_BATCHED_TOKENS=32768
export TEMPO_VLLM_SCHEDULING_POLICY=fcfs
export TEMPO_PD_REMOTE_CATCHUP_PRIORITY=0
export TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY=0
export TEMPO_PD_LONG_REMOTE_CATCHUP_PRIORITY=0
export TEMPO_PD_LONG_REMOTE_CATCHUP_MIN_PROMPT_TOKENS=0
export TEMPO_PD_MEDIAN_GUARD_PRIORITY=0
export TEMPO_PD_MEDIUM_REMOTE_CATCHUP_PRIORITY=0
export TEMPO_PD_REMOTE_CATCHUP_MIN_OUTPUT_TOKENS=256
export TEMPO_LMCACHE_NIXL_BACKEND=UCX
export TEMPO_LMCACHE_LOCAL_CPU_GB=16
export TEMPO_LMCACHE_PD_BUFFER_BYTES=2147483648

mkdir -p -- "${RESULT_DIR}"
"${SCRIPT_DIR}/prepare_c4_python_overlay.sh" "${REPO_ROOT}" "${RESULT_DIR}"
export TEMPO_C4_PYTHON_OVERLAY_PREPARE_ARTIFACT="${RESULT_DIR}/python-overlay-prepare.json"
cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=30s 5400s \
  srun --overlap --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
  --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
  --time="${STEP_TIME}" --export=ALL \
  --output="${RESULT_DIR}/slurm-node-%N.stdout.log" \
  --error="${RESULT_DIR}/slurm-node-%N.stderr.log" \
  bash "${SCRIPT_DIR}/c4_phase_screen_pd_node_entry.sh" \
  "${REPO_ROOT}" "${RESULT_DIR}" "${WORKLOAD}" "${HOSTS_CSV}" \
  "${PORT_SLOT}" 2 128 256 8 3000 250 16000

[[ -s "${RESULT_DIR}/tempo_pd_c4_phase_screen/raw.json" ]]
[[ -s "${RESULT_DIR}/result.json" ]]
RESULT="${RESULT_DIR}/result.json"
RESULT_SHA=$(sha256sum "${RESULT}" | awk '{print $1}')
BASE_ANALYSIS="${RESULT_DIR}/phase_screen_analysis.json"
"${PYTHON}" -m eval.sota_4node.analyze_tempo_pd_c4_phase_screen \
  --result "${RESULT}" --output "${BASE_ANALYSIS}"
BASE_ANALYSIS_SHA=$(sha256sum "${BASE_ANALYSIS}" | awk '{print $1}')
SEMANTIC_ANALYSIS="${RESULT_DIR}/semantic_epoch_analysis.json"
"${PYTHON}" -m eval.sota_4node.analyze_tempo_pd_c4_semantic_epoch_screen \
  --result "${RESULT}" --expected-result-sha256 "${RESULT_SHA}" \
  --base-analysis "${BASE_ANALYSIS}" \
  --expected-base-analysis-sha256 "${BASE_ANALYSIS_SHA}" \
  --output "${SEMANTIC_ANALYSIS}"
echo "C4 semantic-epoch screen result: ${RESULT}"
echo "C4 semantic-epoch verdict: ${SEMANTIC_ANALYSIS}"
