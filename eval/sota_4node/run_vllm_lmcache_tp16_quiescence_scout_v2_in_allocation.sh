#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 CAMPAIGN_INDEX [RESULT_DIR]" >&2
    exit 2
fi
readonly CAMPAIGN_INDEX=$1
case "${CAMPAIGN_INDEX}" in
    0|1|2) ;;
    *) echo "CAMPAIGN_INDEX must be 0, 1, or 2" >&2; exit 2 ;;
esac

: "${SLURM_JOB_ID:?run inside an existing allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
[[ "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 4 ]]
[[ "${TEMPO_VLLM_LMCACHE_TP16_APPROVED:-}" == YES ]] || exit 2

readonly RETRY_PORT_STRIDE=${TEMPO_TP16_QUIESCENCE_V2_PORT_STRIDE:-0}
if [[ ! "${RETRY_PORT_STRIDE}" =~ ^[0-9]+$ ]] || (( ${#RETRY_PORT_STRIDE} > 2 )); then
    echo "TEMPO_TP16_QUIESCENCE_V2_PORT_STRIDE must be an integer in 0..15" >&2
    exit 2
fi
readonly PORT_STRIDE_VALUE=$((10#${RETRY_PORT_STRIDE}))
if (( PORT_STRIDE_VALUE > 15 )); then
    echo "TEMPO_TP16_QUIESCENCE_V2_PORT_STRIDE must be an integer in 0..15" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
readonly SCRIPT_DIR REPO_ROOT
readonly NODE_DRIVER="${SCRIPT_DIR}/vllm_lmcache_tp16_quiescence_scout_node_v2.py"
readonly PYTHON_BIN="${REPO_ROOT}/.vllm_venv/bin/python"
readonly MODEL_CONFIG="${REPO_ROOT}/models/TinyLlama-1.1B-Chat-v1.0/config.json"
readonly PLAN_PATH="${SCRIPT_DIR}/real_tp16_quiescence_scout_v1.json"
[[ -f "${NODE_DRIVER}" ]]
[[ -x "${PYTHON_BIN}" ]]
[[ -f "${MODEL_CONFIG}" ]]
[[ -f "${PLAN_PATH}" ]]

RESULT_CANDIDATE=${2:-"${REPO_ROOT}/results/vllm_lmcache_tp16_quiescence_scout_v2_${CAMPAIGN_INDEX}_job_${SLURM_JOB_ID}_port_stride_${PORT_STRIDE_VALUE}"}
[[ "${RESULT_CANDIDATE}" == /* ]] || RESULT_CANDIDATE="${REPO_ROOT}/${RESULT_CANDIDATE}"
RESULT_DIR=$(realpath -m -- "${RESULT_CANDIDATE}")
readonly RESULT_DIR
case "${RESULT_DIR}/" in
    "${REPO_ROOT}/"*) ;;
    *) echo "RESULT_DIR must be below the repository root" >&2; exit 2 ;;
esac
[[ "${RESULT_DIR}" != "${REPO_ROOT}" ]]
if [[ -e "${RESULT_DIR}/result.json" || -e "${RESULT_DIR}/vllm-quiescence-trace-node-0.jsonl" ]]; then
    echo "refusing to overwrite stale result or trace" >&2
    exit 2
fi

module reset
module load pytorch/2.8.0
mapfile -t TEMPO_JOB_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#TEMPO_JOB_HOSTS[$]} -eq 4 ]]
readonly MASTER_ADDR="${TEMPO_JOB_HOSTS[0]}"
readonly SLOT=$((10#${SLURM_JOB_ID} % 500 + 2500 + CAMPAIGN_INDEX * 32 + PORT_STRIDE_VALUE))
readonly VLLM_MASTER_PORT=$((9000 + SLOT))
readonly SIDECAR_MASTER_PORT=$((13000 + SLOT))
readonly VLLM_API_PORT=$((17000 + SLOT))
readonly NIXL_PORT_BASE=$((21000 + SLOT))
(( SLOT < 3200 ))
(( VLLM_MASTER_PORT < 12200 ))
(( SIDECAR_MASTER_PORT < 16200 ))
(( VLLM_API_PORT < 20200 ))
(( NIXL_PORT_BASE + 7 < 24200 ))

readonly GATE_TAG="${SLURM_JOB_ID}-v2-c${CAMPAIGN_INDEX}-r${PORT_STRIDE_VALUE}"
readonly QUIESCENCE_SOCKET="/tmp/tempo-vllm-quiescence-${GATE_TAG}.sock"
readonly QUIESCENCE_TRACE="/tmp/tempo-step-gate-${GATE_TAG}.jsonl"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_CROSS_NIC NCCL_ALGO NCCL_PROTO UCX_TLS UCX_LOG_LEVEL
unset TEMPO_VLLM_QUIESCENCE_ENABLED TEMPO_VLLM_QUIESCENCE_NODE_RANK
unset TEMPO_VLLM_QUIESCENCE_SOCKET TEMPO_VLLM_QUIESCENCE_TRACE
unset TEMPO_VLLM_QUIESCENCE_TOKEN_INDEX TEMPO_VLLM_QUIESCENCE_TIMEOUT_S

mkdir -p -- "${RESULT_DIR}"
cd -- "${REPO_ROOT}"

timeout --foreground --signal=TERM --kill-after=20s 1800s \
    srun --exact \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
    --cpus-per-task=64 --cpu-bind=cores \
    --kill-on-bad-exit=1 --wait=10 --time=00:29:30 --export=ALL \
    --output="${RESULT_DIR}/wrapper-node-%N.stdout.log" \
    --error="${RESULT_DIR}/wrapper-node-%N.stderr.log" \
    "${PYTHON_BIN}" "${NODE_DRIVER}" \
    --repo-root "${REPO_ROOT}" \
    --result-dir "${RESULT_DIR}" \
    --plan "${PLAN_PATH}" \
    --campaign-index "${CAMPAIGN_INDEX}" \
    --master-addr "${MASTER_ADDR}" \
    --vllm-master-port "${VLLM_MASTER_PORT}" \
    --sidecar-master-port "${SIDECAR_MASTER_PORT}" \
    --api-port "${VLLM_API_PORT}" \
    --nixl-port-base "${NIXL_PORT_BASE}" \
    --quiescence-socket "${QUIESCENCE_SOCKET}" \
    --quiescence-trace "${QUIESCENCE_TRACE}" \
    --readiness-timeout-s 600 \
    --sidecar-timeout-s 1100

[[ -s "${RESULT_DIR}/result.json" ]]
[[ -s "${RESULT_DIR}/vllm-quiescence-trace-node-0.jsonl" ]]
echo "TP16 quiescence scout v2 campaign ${CAMPAIGN_INDEX}: ${RESULT_DIR}/result.json"
