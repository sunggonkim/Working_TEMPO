#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/require_perlmutter_4node_4h_interactive.sh"
[[ $# -eq 2 ]]

REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKLOAD_INPUT=$(realpath -e -- "$1")
RESULT_DIR=$(realpath -m -- "$2")
: "${TEMPO_GO_C5_RUN_CONTRACT:?frozen C5 run-contract path required}"
: "${TEMPO_GO_C5_RUN_CONTRACT_SHA256:?frozen C5 run-contract SHA-256 required}"
[[ "${TEMPO_GO_C5_RUN_CONTRACT_SHA256}" =~ ^[0-9a-f]{64}$ ]]
RUN_CONTRACT=$(realpath -e -- "${TEMPO_GO_C5_RUN_CONTRACT}")
case "${WORKLOAD_INPUT}" in "${REPO_ROOT}"/*) ;; *) exit 2 ;; esac
case "${RESULT_DIR}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
case "${RUN_CONTRACT}" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
if [[ -d "${WORKLOAD_INPUT}" ]]; then
  WORKLOAD_DIR="${WORKLOAD_INPUT}"
  WORKLOAD="${WORKLOAD_DIR}/workloads/validation.jsonl"
else
  WORKLOAD="${WORKLOAD_INPUT}"
  WORKLOAD_DIR=$(cd -- "$(dirname -- "${WORKLOAD}")/.." && pwd)
fi
MANIFEST="${WORKLOAD_DIR}/tempo_go_workload_manifest.json"
[[ -s "${WORKLOAD}" && -s "${MANIFEST}" && ! -e "${RESULT_DIR}" ]]

export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
"${REPO_ROOT}/.vllm_venv/bin/python" \
  -m eval.sota_4node.tempo_go_c5_run_contract verify \
  --repo-root "${REPO_ROOT}" --contract "${RUN_CONTRACT}" \
  --sha256 "${TEMPO_GO_C5_RUN_CONTRACT_SHA256}" \
  --workload-input "${WORKLOAD_INPUT}" --arm-only tempo

module reset
module load pytorch/2.8.0
export TEMPO_LMCACHE_NIXL_BACKEND=UCX
export TEMPO_LMCACHE_LOCAL_CPU_GB=16
export TEMPO_LMCACHE_PD_BUFFER_BYTES=2147483648
export TEMPO_PD_ENDPOINT_FEEDBACK_MODE=adaptive
export TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK=1
export TEMPO_PD_ENDPOINT_ROUTING_POLICY=semantic_epoch_v1
export TEMPO_GO_GLOBAL_PROFILE="$(jq -er '.artifacts.global_profile.path' "${RUN_CONTRACT}")"
export TEMPO_GO_ELASTIC_PROFILE_PATH="$(jq -er '.artifacts.elastic_profile.path' "${RUN_CONTRACT}")"
export TEMPO_GO_ENDPOINT_PROFILE_PATH="$(jq -er '.artifacts.endpoint_profile.path' "${RUN_CONTRACT}")"
export TEMPO_PD_ENDPOINT_SERVICE_PROFILE="${TEMPO_GO_ENDPOINT_PROFILE_PATH}"
export TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256
TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256=$(sha256sum "${MANIFEST}" | awk '{print $1}')
export TEMPO_PD_PRESSURE_MODE=disabled
export TEMPO_VLLM_LOAD_SNAPSHOT_MODE=disabled
export TEMPO_VLLM_DECODER_PREFIX_CACHING=0
export TEMPO_PD_FRONTEND_PAIR_POLICY=tempo-min-outstanding-decode-tokens-v1
export TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY=1
export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_GO_C5_ARM=tempo
export TEMPO_PD_BENCHMARK_RESET_DECODER_APC=0
export TEMPO_PD_DECODER_REUSE_ITEMS=all
export TEMPO_VLLM_MAX_NUM_SEQS=16
export TEMPO_VLLM_ASYNC_SCHEDULING=0
export TEMPO_VLLM_DECODER_MAX_NUM_BATCHED_TOKENS=32768
export TEMPO_VLLM_SCHEDULING_POLICY=fcfs
export TEMPO_PD_REMOTE_DECODE_PLACEMENT=paired
export TEMPO_PD_PROXY_TOKENIZER_PLACEMENT=round_robin
mkdir -p -- "${RESULT_DIR}"
"${SCRIPT_DIR}/prepare_c4_python_overlay.sh" "${REPO_ROOT}" "${RESULT_DIR}"
export TEMPO_C4_PYTHON_OVERLAY_PREPARE_ARTIFACT="${RESULT_DIR}/python-overlay-prepare.json"
export TEMPO_PD_C5_APPROVED=YES
export TEMPO_PD_C5_RUN_CONTRACT_SHA256="${TEMPO_GO_C5_RUN_CONTRACT_SHA256}"
export TEMPO_GO_C5_RUN_CONTRACT="${RUN_CONTRACT}"
export TEMPO_GO_C5_ARM=tempo
export TEMPO_GO_C5_REQUEST_RATE="$(jq -er '.launcher.node_parameters.request_rate' "${RUN_CONTRACT}")"
export TEMPO_GO_C5_MAX_WORKERS="$(jq -er '.launcher.node_parameters.max_workers' "${RUN_CONTRACT}")"
export TEMPO_GO_C5_OUTPUT_TOKENS="$(jq -er '.launcher.node_parameters.output_tokens' "${RUN_CONTRACT}")"
cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=30s 7200s \
  /usr/bin/srun --overlap --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
  --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
  --time=02:30:00 --export=ALL \
  --output="${RESULT_DIR}/slurm-node-%N.stdout.log" \
  --error="${RESULT_DIR}/slurm-node-%N.stderr.log" \
  bash "${SCRIPT_DIR}/c5_tempo_go_node_entry.sh" \
  "${REPO_ROOT}" "${RESULT_DIR}" "${WORKLOAD}" \
  "$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | paste -sd, -)" \
  "$((1860 + SLURM_JOB_ID % 20))" "${RUN_CONTRACT}"

[[ -s "${RESULT_DIR}/result.json" ]]
echo "TEMPO-GO C5 discovery result: ${RESULT_DIR}/result.json"
