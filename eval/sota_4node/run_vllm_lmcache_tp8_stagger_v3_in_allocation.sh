#!/usr/bin/env bash
set -euo pipefail

readonly LMCACHE_COMMIT="227d13f5c9fdb52ddb933641d34331f678de03a0"
if [[ $# -gt 1 ]]; then
    echo "usage: $0 [RESULT_DIR]" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?run inside an existing allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
[[ "${SLURM_JOB_NUM_NODES:-}" == 2 ]]
[[ "${TEMPO_VLLM_LMCACHE_STAGGER_APPROVED:-}" == YES ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
NODE_ENTRY="${SCRIPT_DIR}/vllm_lmcache_tp8_stagger_node_v3.sh"
RESULT_CANDIDATE=${1:-"${REPO_ROOT}/results/vllm_lmcache_stagger_v3_job_${SLURM_JOB_ID}"}
[[ "${RESULT_CANDIDATE}" == /* ]] || RESULT_CANDIDATE="${REPO_ROOT}/${RESULT_CANDIDATE}"
RESULT_DIR=$(realpath -m -- "${RESULT_CANDIDATE}")
case "${RESULT_DIR}/" in "${REPO_ROOT}/"*) ;; *) exit 2 ;; esac
[[ "${RESULT_DIR}" != "${REPO_ROOT}" ]]
[[ -x "${NODE_ENTRY}" ]]
[[ -x "${REPO_ROOT}/.vllm_venv/bin/vllm" ]]
[[ -x "${REPO_ROOT}/.vllm_venv/bin/torchrun" ]]
[[ -f "${REPO_ROOT}/models/TinyLlama-1.1B-Chat-v1.0/model.safetensors" ]]
[[ -f "${REPO_ROOT}/eval/sota_4node/real_tp8_pair_stagger_v1.json" ]]
[[ "$(git -C "${REPO_ROOT}/third_party/lmcache" rev-parse HEAD)" == "${LMCACHE_COMMIT}" ]]

module reset
module load pytorch/2.8.0
mapfile -t TEMPO_JOB_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#TEMPO_JOB_HOSTS[@]} -eq 2 ]]
export MASTER_ADDR="${TEMPO_JOB_HOSTS[0]}"
export TEMPO_VLLM_MASTER_PORT=$((23000 + SLURM_JOB_ID % 7000))
export TEMPO_SIDECAR_MASTER_PORT=$((33000 + SLURM_JOB_ID % 7000))
export TEMPO_VLLM_API_PORT=$((43000 + SLURM_JOB_ID % 7000))
export TEMPO_NIXL_PORT_BASE=$((53000 + SLURM_JOB_ID % 7000))
export TEMPO_VLLM_STARTUP_GRACE_S=90
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_CROSS_NIC NCCL_ALGO NCCL_PROTO UCX_TLS

mkdir -p -- "${RESULT_DIR}"
cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=15s 1200s \
    srun --exact \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
    --cpus-per-task=64 --cpu-bind=cores \
    --kill-on-bad-exit=1 --wait=5 --time=00:19:30 --export=ALL \
    --output="${RESULT_DIR}/wrapper-node-%N.stdout.log" \
    --error="${RESULT_DIR}/wrapper-node-%N.stderr.log" \
    "${NODE_ENTRY}" "${REPO_ROOT}" "${RESULT_DIR}"
echo "vLLM TP8 + LMCache pair-stagger v3 result: ${RESULT_DIR}/result.json"
