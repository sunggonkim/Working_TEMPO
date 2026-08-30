#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/require_perlmutter_4node_4h_interactive.sh"
: "${TEMPO_PD_C4_SEMANTIC_INTEGRATION_APPROVED:?explicit semantic integration approval required}"
: "${TEMPO_PD_C4_SEMANTIC_INTEGRATION_RUN_CONTRACT_SHA256:?semantic integration run-contract SHA-256 required}"
[[ "${TEMPO_PD_C4_SEMANTIC_INTEGRATION_APPROVED}" == YES ]]
[[ "${TEMPO_PD_C4_SEMANTIC_INTEGRATION_RUN_CONTRACT_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ $# -eq 2 ]]

while IFS= read -r name; do
  case "${name}" in
    TEMPO_PD_C4_SEMANTIC_INTEGRATION_APPROVED|TEMPO_PD_C4_SEMANTIC_INTEGRATION_RUN_CONTRACT_SHA256|TEMPO_PD_C4_STEP_TIME|TEMPO_PD_C4_READINESS_S) ;;
    *)
      echo "semantic integration screen refuses inherited experiment variable: ${name}" >&2
      exit 2
      ;;
  esac
done < <(compgen -e TEMPO_)

while IFS= read -r name; do
  case "${name}" in
    NCCL_NET_GDR_LEVEL)
      [[ "${!name}" == PHB ]] || {
        echo "semantic integration screen refuses inherited transport variable: ${name}" >&2
        exit 2
      }
      ;;
    FI_*|UCX_*|NIXL_*|LMCACHE_*|VLLM_*|NCCL_*|CUDA_VISIBLE_DEVICES)
      echo "semantic integration screen refuses inherited transport variable: ${name}" >&2
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
  "${TEMPO_PD_C4_SEMANTIC_INTEGRATION_RUN_CONTRACT_SHA256}" ]]
[[ -z "${TEMPO_CXI_BACKGROUND_DUTY_CYCLE:-}" ]]
[[ -z "${TEMPO_CXI_BACKGROUND_START_FILE:-}" ]]

module reset
module load pytorch/2.8.0
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
PORT_SLOT=$((1880 + SLURM_JOB_ID % 20))
REQUEST_RATE=2
MAX_WORKERS=128
STEP_TIME=${TEMPO_PD_C4_STEP_TIME:-01:15:00}
unset TEMPO_PD_C4_STEP_TIME

export TEMPO_PD_C4_SEMANTIC_INTEGRATION_APPROVED=YES
export TEMPO_PD_C4_SEMANTIC_INTEGRATION_RUN_CONTRACT="${RUN_CONTRACT}"
export TEMPO_PD_C4_PHASE_DURATION_MS=8000
export TEMPO_PD_C4_COOLDOWN_S=2
export TEMPO_PD_C4_READINESS_S=${TEMPO_PD_C4_READINESS_S:-3600}
export TEMPO_PD_BENCHMARK_COLD_MEASURED=0
export TEMPO_PD_BENCHMARK_RESET_DECODER_APC=1
export TEMPO_VLLM_DECODER_PREFIX_CACHING=1
export TEMPO_PD_FRONTEND_PAIR_POLICY=tempo-min-outstanding-decode-tokens-v1
export TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY=1
export TEMPO_PD_DECODER_REUSE_ITEMS=all
export TEMPO_PD_FORWARD_TOKEN_IDS=0
export TEMPO_PD_PROXY_KV_CONTROL_OVERLAP=0
export TEMPO_PD_REMOTE_DECODE_PLACEMENT=paired
export TEMPO_PD_PROXY_TOKENIZER_PLACEMENT=round_robin
export TEMPO_LMCACHE_NIXL_BACKEND=UCX
export TEMPO_LMCACHE_LOCAL_CPU_GB=16
export TEMPO_LMCACHE_PD_BUFFER_BYTES=2147483648
export TEMPO_ELASTIC_PD_PROFILE_SCOPE=screen_only
export TEMPO_PD_ENDPOINT_FEEDBACK_MODE=adaptive
export TEMPO_PD_ENDPOINT_ROUTING_POLICY=semantic_epoch_v1
export TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK=1
export TEMPO_PD_PRESSURE_MODE=disabled
export TEMPO_VLLM_LOAD_SNAPSHOT_MODE=disabled
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

mkdir -p -- "${RESULT_DIR}"
cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=30s 5400s \
  srun --overlap --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
  --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
  --time="${STEP_TIME}" --export=ALL \
  --output="${RESULT_DIR}/slurm-node-%N.stdout.log" \
  --error="${RESULT_DIR}/slurm-node-%N.stderr.log" \
  bash "${SCRIPT_DIR}/c4_adaptive_screen_pd_node_entry.sh" \
  "${REPO_ROOT}" "${RESULT_DIR}" "${RUN_CONTRACT}" "${HOSTS_CSV}" \
  "${PORT_SLOT}" "${REQUEST_RATE}" "${MAX_WORKERS}" 256 8 3000 250 16000

[[ -s "${RESULT_DIR}/tempo_pd_c4_semantic_integration_screen/raw.json" ]]
[[ -s "${RESULT_DIR}/result.json" ]]
echo "Post-C4 semantic integration result: ${RESULT_DIR}/result.json"
