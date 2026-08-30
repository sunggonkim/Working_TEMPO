#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 6 ]]
: "${SLURM_NODEID:?node index is required}"
REPO_ROOT=$(realpath -e -- "$1")
RUN_CONTRACT=$(realpath -e -- "$6")
case "${RUN_CONTRACT}" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
: "${TEMPO_GO_SOURCE_SNAPSHOT:=${REPO_ROOT}}"
SOURCE_ROOT=$(realpath -e -- "${TEMPO_GO_SOURCE_SNAPSHOT}")
if [[ "${SOURCE_ROOT}" != "${REPO_ROOT}" ]]; then
  case "${SOURCE_ROOT}/" in
    "${REPO_ROOT}/results/"*) ;;
    *) exit 2 ;;
  esac
fi
: "${TEMPO_GO_C5_RUN_CONTRACT_SHA256:?frozen C5 run-contract SHA required}"
[[ "${TEMPO_GO_C5_RUN_CONTRACT_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "$(sha256sum "${RUN_CONTRACT}" | awk '{print $1}')" == \
  "${TEMPO_GO_C5_RUN_CONTRACT_SHA256}" ]]
export TEMPO_GO_C5_RUN_CONTRACT="${RUN_CONTRACT}"
export TEMPO_GO_SOURCE_SNAPSHOT="${SOURCE_ROOT}"
source "${SOURCE_ROOT}/eval/sota_4node/stage_c4_python_overlay.sh" "${REPO_ROOT}"
export PYTHONPATH="${TEMPO_C4_PYTHON_OVERLAY}:${SOURCE_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${SOURCE_ROOT}" != "${REPO_ROOT}" ]]; then
  cd -- "${SOURCE_ROOT}"
fi
exec "${REPO_ROOT}/.vllm_venv/bin/python" \
  -m eval.sota_4node.vllm_lmcache_tempo_go_c5_node \
  --repo-root "${REPO_ROOT}" --result-dir "$2" --scout-root "$3" \
  --node-index "${SLURM_NODEID}" --hosts "$4" --port-slot "$5" \
  --request-rate "${TEMPO_GO_C5_REQUEST_RATE:-8}" \
  --max-workers "${TEMPO_GO_C5_MAX_WORKERS:-128}" \
  --output-tokens "${TEMPO_GO_C5_OUTPUT_TOKENS:-2}" \
  --samples-per-bucket 3 --ttft-slo-ms 3000 --tpot-slo-ms 250 \
  --e2e-slo-ms 16000
