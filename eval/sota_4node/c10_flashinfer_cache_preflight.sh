#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 3 ]]
REPO_ROOT=$(realpath -e -- "$1")
RESULT_ROOT=$(realpath -m -- "$2")
CACHE_MODE=$3
[[ -d "$REPO_ROOT" && "$REPO_ROOT" == */Skim-Tempo ]]
[[ "$RESULT_ROOT" == "$REPO_ROOT/results/"* ]]
[[ -n "$CACHE_MODE" && "$CACHE_MODE" != */* ]]
: "${SLURM_PROCID:?srun must provide a task index}"

NODE_INDEX=${SLURM_PROCID}
CACHE_ROOT="/tmp/tempo-live-pd-${SLURM_JOB_ID}-${CACHE_MODE}-n${NODE_INDEX}"
FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}/flashinfer"
export FLASHINFER_WORKSPACE_BASE
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1

MODEL="${REPO_ROOT}/models/Qwen2.5-7B-Instruct"
LOG_DIR="${RESULT_ROOT}/cache_preflight"
mkdir -p -- "$LOG_DIR"
PORT=$((29000 + NODE_INDEX))
SERVER_LOG="${LOG_DIR}/node-${NODE_INDEX}-server.log"
RECEIPT="${LOG_DIR}/node-${NODE_INDEX}.json"
SERVER_PID=

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 50); do
            kill -0 "${SERVER_PID}" 2>/dev/null || return 0
            sleep 0.1
        done
        kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

setsid "${REPO_ROOT}/.vllm_venv/bin/vllm" serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --served-model-name tempo-flashinfer-preflight \
    --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 1024 \
    --max-num-seqs 4 --gpu-memory-utilization 0.80 --enforce-eager \
    --no-enable-prefix-caching --disable-hybrid-kv-cache-manager \
    --no-async-scheduling --max-num-batched-tokens 512 \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 600); do
    if curl --silent --show-error --fail --max-time 1 \
        "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "preflight vLLM exited before readiness: node=${NODE_INDEX}" >&2
        exit 1
    fi
    sleep 0.5
done
curl --silent --show-error --fail --max-time 2 \
    "http://127.0.0.1:${PORT}/health" >/dev/null

SAMPLING_SO=
for candidate in "${FLASHINFER_WORKSPACE_BASE}"/.cache/flashinfer/*/*/cached_ops/sampling/sampling.so; do
    if [[ -s "$candidate" ]]; then
        SAMPLING_SO=$candidate
        break
    fi
done
[[ -n "$SAMPLING_SO" ]]
sha256=$(sha256sum "$SAMPLING_SO" | awk '{print $1}')
printf '{"schema":"tempo-go-c10-flashinfer-preflight-v1","job_id":"%s","node_index":%d,"cache_mode":"%s","sampling_so":"%s","sampling_so_sha256":"%s"}\n' \
    "$SLURM_JOB_ID" "$NODE_INDEX" "$CACHE_MODE" "$SAMPLING_SO" "$sha256" >"$RECEIPT"
