#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 4 ]]
: "${SLURM_NODEID:?SLURM_NODEID is required inside srun}"
REPO_ROOT=$1
RESULT_DIR=$2
HOSTS_CSV=$3
PORT_SLOT=$4
exec "${REPO_ROOT}/.vllm_venv/bin/python" \
    -m eval.sota_4node.vllm_lmcache_live_pd_node_v3 \
    --repo-root "${REPO_ROOT}" \
    --result-dir "${RESULT_DIR}" \
    --node-index "${SLURM_NODEID}" \
    --hosts "${HOSTS_CSV}" \
    --port-slot "${PORT_SLOT}"
