#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$1
shift
cd -- "${REPO_ROOT}"
exec .vllm_venv/bin/python -m eval.sota_4node.vllm_lmcache_same_server_online_regime_salted_node_v300 "$@"
