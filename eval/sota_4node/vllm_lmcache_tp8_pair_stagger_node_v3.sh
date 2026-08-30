#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]] || exit 2
readonly REPO_ROOT=$1
readonly RESULT_DIR=$2
: "${SLURM_JOB_ID:?run inside an existing allocation}"
: "${SLURM_NODEID:?one launcher task per node is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${TEMPO_VLLM_MASTER_PORT:?TEMPO_VLLM_MASTER_PORT is required}"
: "${TEMPO_SIDECAR_MASTER_PORT:?TEMPO_SIDECAR_MASTER_PORT is required}"
: "${TEMPO_VLLM_API_PORT:?TEMPO_VLLM_API_PORT is required}"
: "${TEMPO_NIXL_PORT_BASE:?TEMPO_NIXL_PORT_BASE is required}"
: "${TEMPO_VLLM_STARTUP_GRACE_S:?TEMPO_VLLM_STARTUP_GRACE_S is required}"
[[ "${SLURM_JOB_NUM_NODES:-}" == 2 ]]
[[ "${SLURM_NODEID}" == 0 || "${SLURM_NODEID}" == 1 ]]
(( TEMPO_VLLM_STARTUP_GRACE_S > 0 && TEMPO_VLLM_STARTUP_GRACE_S <= 180 ))

readonly VLLM_BIN="${REPO_ROOT}/.vllm_venv/bin/vllm"
readonly TORCHRUN_BIN="${REPO_ROOT}/.vllm_venv/bin/torchrun"
readonly MODEL_DIR="${REPO_ROOT}/models/TinyLlama-1.1B-Chat-v1.0"
readonly PLAN_PATH="${REPO_ROOT}/eval/sota_4node/real_tp8_pair_stagger_v1.json"
[[ -x "${VLLM_BIN}" && -x "${TORCHRUN_BIN}" ]]
[[ -f "${MODEL_DIR}/config.json" && -f "${MODEL_DIR}/model.safetensors" ]]
[[ -f "${PLAN_PATH}" && -d "${RESULT_DIR}" ]]

readonly NODE_CACHE="/tmp/tempo-vllm-${SLURM_JOB_ID}-node${SLURM_NODEID}"
mkdir -p -- "${NODE_CACHE}/flashinfer" "${NODE_CACHE}/huggingface" \
    "${NODE_CACHE}/torch-extensions" "${NODE_CACHE}/triton"
export FLASHINFER_WORKSPACE_BASE="${NODE_CACHE}/flashinfer"
export HF_HOME="${NODE_CACHE}/huggingface"
export TORCH_EXTENSIONS_DIR="${NODE_CACHE}/torch-extensions"
export TRITON_CACHE_DIR="${NODE_CACHE}/triton"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"

VLLM_ARGS=(
    serve "${MODEL_DIR}"
    --tensor-parallel-size 8
    --distributed-executor-backend mp
    --nnodes 2
    --node-rank "${SLURM_NODEID}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${TEMPO_VLLM_MASTER_PORT}"
    --dtype bfloat16
    --max-model-len 2048
    --max-num-seqs 4
    --gpu-memory-utilization 0.50
    --enforce-eager
    --no-enable-prefix-caching
)
if [[ "${SLURM_NODEID}" == 0 ]]; then
    VLLM_ARGS+=(--host 0.0.0.0 --port "${TEMPO_VLLM_API_PORT}")
else
    VLLM_ARGS+=(--headless)
fi

"${VLLM_BIN}" "${VLLM_ARGS[@]}" \
    > "${RESULT_DIR}/vllm-node-${SLURM_NODEID}.stdout.log" \
    2> "${RESULT_DIR}/vllm-node-${SLURM_NODEID}.stderr.log" &
VLLM_PID=$!
cleanup() {
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -TERM "${VLLM_PID}" 2>/dev/null || true
    fi
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

sleep "${TEMPO_VLLM_STARTUP_GRACE_S}"
kill -0 "${VLLM_PID}"
"${TORCHRUN_BIN}" \
    --nnodes=2 --nproc-per-node=4 \
    --node-rank="${SLURM_NODEID}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${TEMPO_SIDECAR_MASTER_PORT}" \
    -m eval.sota_4node.run_vllm_lmcache_tp8_pair_stagger_v3 \
    --output-dir "${RESULT_DIR}" \
    --plan "${PLAN_PATH}" \
    --api-host "${MASTER_ADDR}" \
    --api-port "${TEMPO_VLLM_API_PORT}" \
    --model "${MODEL_DIR}" \
    --nixl-port-base "${TEMPO_NIXL_PORT_BASE}" \
    --request-timeout-s 120 \
    > "${RESULT_DIR}/sidecar-node-${SLURM_NODEID}.stdout.log" \
    2> "${RESULT_DIR}/sidecar-node-${SLURM_NODEID}.stderr.log"
[[ -f "${RESULT_DIR}/result.json" ]]
