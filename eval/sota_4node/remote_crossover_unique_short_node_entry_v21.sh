#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 11 ]]
: "${SLURM_NODEID:?SLURM_NODEID is required inside srun}"
exec "$1/.vllm_venv/bin/python" \
  -m eval.sota_4node.vllm_lmcache_remote_crossover_unique_short_node_v21 \
  --repo-root "$1" --result-dir "$2" --node-index "${SLURM_NODEID}" \
  --hosts "$3" --port-slot "$4" --request-rate "$5" --max-workers "$6" \
  --output-tokens "$7" --samples-per-bucket "$8" --ttft-slo-ms "$9" \
  --tpot-slo-ms "${10}" --e2e-slo-ms "${11}"
