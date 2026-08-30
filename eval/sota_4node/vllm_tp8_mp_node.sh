#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 REPO_ROOT RESULT_DIR" >&2
    exit 2
fi

readonly REPO_ROOT=$1
readonly RESULT_DIR=$2
: "${SLURM_JOB_ID:?run inside an existing allocation}"
: "${SLURM_NODEID:?one launcher task per node is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"
: "${TEMPO_VLLM_API_PORT:?TEMPO_VLLM_API_PORT is required}"

[[ "${SLURM_JOB_NUM_NODES:-}" == 2 ]]
[[ "${SLURM_NODEID}" == 0 || "${SLURM_NODEID}" == 1 ]]

readonly VLLM_BIN="${REPO_ROOT}/.vllm_venv/bin/vllm"
readonly MODEL_DIR="${REPO_ROOT}/models/TinyLlama-1.1B-Chat-v1.0"
[[ -x "${VLLM_BIN}" ]]
[[ -f "${MODEL_DIR}/config.json" ]]
[[ -f "${MODEL_DIR}/model.safetensors" ]]
[[ -d "${RESULT_DIR}" ]]

# FlashInfer reads this at import time. /tmp is node-local on Perlmutter, so
# concurrent workers do not compile into a shared Lustre cache.
readonly NODE_CACHE="/tmp/tempo-vllm-${SLURM_JOB_ID}-node${SLURM_NODEID}"
mkdir -p -- \
    "${NODE_CACHE}/flashinfer" \
    "${NODE_CACHE}/huggingface" \
    "${NODE_CACHE}/torch-extensions" \
    "${NODE_CACHE}/triton"
export FLASHINFER_WORKSPACE_BASE="${NODE_CACHE}/flashinfer"
export HF_HOME="${NODE_CACHE}/huggingface"
export TORCH_EXTENSIONS_DIR="${NODE_CACHE}/torch-extensions"
export TRITON_CACHE_DIR="${NODE_CACHE}/triton"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1

VLLM_ARGS=(
    serve "${MODEL_DIR}"
    --tensor-parallel-size 8
    --distributed-executor-backend mp
    --nnodes 2
    --node-rank "${SLURM_NODEID}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
    --dtype bfloat16
    --max-model-len 2048
    --max-num-seqs 4
    --gpu-memory-utilization 0.50
    --enforce-eager
)

if [[ "${SLURM_NODEID}" == 0 ]]; then
    VLLM_ARGS+=(
        --host 0.0.0.0
        --port "${TEMPO_VLLM_API_PORT}"
    )
else
    VLLM_ARGS+=(--headless)
fi

exec "${VLLM_BIN}" "${VLLM_ARGS[@]}"
