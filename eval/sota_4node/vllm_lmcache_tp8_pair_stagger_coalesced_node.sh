#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || exit 2
readonly REPO_ROOT=$1 RESULT_DIR=$2
: "${SLURM_JOB_ID:?}" "${SLURM_NODEID:?}" "${MASTER_ADDR:?}"
: "${TEMPO_VLLM_MASTER_PORT:?}" "${TEMPO_SIDECAR_MASTER_PORT:?}"
: "${TEMPO_VLLM_API_PORT:?}" "${TEMPO_NIXL_PORT_BASE:?}"
readonly MODEL_DIR="${REPO_ROOT}/models/TinyLlama-1.1B-Chat-v1.0"
readonly PLAN_PATH="${REPO_ROOT}/eval/sota_4node/real_tp8_pair_stagger_coalesced_v1.json"
readonly CACHE="/tmp/tempo-vllm-${SLURM_JOB_ID}-node${SLURM_NODEID}"
mkdir -p -- "${CACHE}/flashinfer" "${CACHE}/huggingface" "${CACHE}/torch-extensions" "${CACHE}/triton"
export FLASHINFER_WORKSPACE_BASE="${CACHE}/flashinfer" HF_HOME="${CACHE}/huggingface"
export TORCH_EXTENSIONS_DIR="${CACHE}/torch-extensions" TRITON_CACHE_DIR="${CACHE}/triton"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${REPO_ROOT}"
ARGS=(serve "${MODEL_DIR}" --tensor-parallel-size 8 --distributed-executor-backend mp
    --nnodes 2 --node-rank "${SLURM_NODEID}" --master-addr "${MASTER_ADDR}"
    --master-port "${TEMPO_VLLM_MASTER_PORT}" --dtype bfloat16 --max-model-len 2048
    --max-num-seqs 4 --gpu-memory-utilization 0.50 --enforce-eager --no-enable-prefix-caching)
if [[ "${SLURM_NODEID}" == 0 ]]; then ARGS+=(--host 0.0.0.0 --port "${TEMPO_VLLM_API_PORT}"); else ARGS+=(--headless); fi
"${REPO_ROOT}/.vllm_venv/bin/vllm" "${ARGS[@]}" >"${RESULT_DIR}/vllm-node-${SLURM_NODEID}.stdout.log" 2>"${RESULT_DIR}/vllm-node-${SLURM_NODEID}.stderr.log" &
PID=$!
cleanup(){ kill -TERM "${PID}" 2>/dev/null || true; wait "${PID}" 2>/dev/null || true; }
trap cleanup EXIT TERM INT
sleep 90
kill -0 "${PID}"
"${REPO_ROOT}/.vllm_venv/bin/torchrun" --nnodes=2 --nproc-per-node=4 \
    --node-rank="${SLURM_NODEID}" --master-addr="${MASTER_ADDR}" --master-port="${TEMPO_SIDECAR_MASTER_PORT}" \
    -m eval.sota_4node.run_vllm_lmcache_tp8_pair_stagger_coalesced_v1 \
    --output-dir "${RESULT_DIR}" --plan "${PLAN_PATH}" --api-host "${MASTER_ADDR}" \
    --api-port "${TEMPO_VLLM_API_PORT}" --model "${MODEL_DIR}" --nixl-port-base "${TEMPO_NIXL_PORT_BASE}" \
    --request-timeout-s 120 >"${RESULT_DIR}/sidecar-node-${SLURM_NODEID}.stdout.log" 2>"${RESULT_DIR}/sidecar-node-${SLURM_NODEID}.stderr.log"
[[ -f "${RESULT_DIR}/result.json" ]]
