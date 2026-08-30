#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]] || exit 2
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
readonly VLLM_PYTHON="${REPO_ROOT}/.vllm_venv/bin/python"
readonly MODEL_DIR="${REPO_ROOT}/models/TinyLlama-1.1B-Chat-v1.0"
readonly NODE_DRIVER="${REPO_ROOT}/eval/sota_4node/vllm_tp8_mp_smoke_node_v3.py"
[[ -x "${VLLM_BIN}" && -x "${VLLM_PYTHON}" ]]
[[ -f "${MODEL_DIR}/config.json" && -f "${MODEL_DIR}/model.safetensors" ]]
[[ -f "${NODE_DRIVER}" && -d "${RESULT_DIR}" ]]

readonly NODE_CACHE="/tmp/tempo-vllm-${SLURM_JOB_ID}-node${SLURM_NODEID}"
mkdir -p -- "${NODE_CACHE}/flashinfer" "${NODE_CACHE}/huggingface" \
    "${NODE_CACHE}/torch-extensions" "${NODE_CACHE}/triton"
export FLASHINFER_WORKSPACE_BASE="${NODE_CACHE}/flashinfer"
export HF_HOME="${NODE_CACHE}/huggingface"
export TORCH_EXTENSIONS_DIR="${NODE_CACHE}/torch-extensions"
export TRITON_CACHE_DIR="${NODE_CACHE}/triton"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1

exec "${VLLM_PYTHON}" "${NODE_DRIVER}" \
    --vllm-bin "${VLLM_BIN}" --model "${MODEL_DIR}" \
    --result-dir "${RESULT_DIR}" --node-rank "${SLURM_NODEID}" \
    --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT}" \
    --api-port "${TEMPO_VLLM_API_PORT}"
