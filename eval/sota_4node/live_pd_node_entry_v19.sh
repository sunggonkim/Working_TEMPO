#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 4 ]]
: "${SLURM_NODEID:?SLURM_NODEID is required inside srun}"
exec "$1/.vllm_venv/bin/python" \
    -m eval.sota_4node.vllm_lmcache_live_pd_node_v19 \
    --repo-root "$1" --result-dir "$2" --node-index "${SLURM_NODEID}" \
    --hosts "$3" --port-slot "$4"
