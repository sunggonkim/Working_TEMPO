#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 12 ]]
: "${SLURM_NODEID:?SLURM_NODEID is required}"
exec "$1/.vllm_venv/bin/python" -m eval.sota_4node.vllm_lmcache_nixl_polling_snapshot_node_v3 \
  --repo-root "$1" --result-dir "$2" --stock-reference "$3" \
  --node-index "${SLURM_NODEID}" --hosts "$4" --port-slot "$5" \
  --request-rate "$6" --max-workers "$7" --output-tokens "$8" \
  --samples-per-bucket "$9" --ttft-slo-ms "${10}" --tpot-slo-ms "${11}" \
  --e2e-slo-ms "${12}"
