#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 12 ]]
: "${SLURM_NODEID:?nodeid}"
REPO_ROOT=$(realpath -e -- "$1")
source "${REPO_ROOT}/eval/sota_4node/stage_c4_python_overlay.sh" \
  "${REPO_ROOT}"
exec "${REPO_ROOT}/.vllm_venv/bin/python" \
  -m eval.sota_4node.vllm_lmcache_pd_c4_phase_screen_node \
  --repo-root "${REPO_ROOT}" --result-dir "$2" --scout-root "$3" \
  --node-index "${SLURM_NODEID}" --hosts "$4" --port-slot "$5" \
  --request-rate "$6" --max-workers "$7" --output-tokens "$8" \
  --samples-per-bucket "$9" --ttft-slo-ms "${10}" \
  --tpot-slo-ms "${11}" --e2e-slo-ms "${12}"
