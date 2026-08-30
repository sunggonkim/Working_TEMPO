#!/usr/bin/env bash
set -euo pipefail

[[ $# -le 1 ]] || exit 2
: "${SLURM_JOB_ID:?run inside an existing allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
[[ "${SLURM_JOB_NUM_NODES:-}" == 2 ]]
[[ "${TEMPO_VLLM_TP8_SMOKE_APPROVED:-}" == YES ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
NODE_ENTRY="${SCRIPT_DIR}/vllm_tp8_mp_smoke_node_entry_v3.sh"
RESULT_CANDIDATE=${1:-"${REPO_ROOT}/results/vllm_tp8_mp_smoke_v3_job_${SLURM_JOB_ID}"}
[[ "${RESULT_CANDIDATE}" == /* ]] || RESULT_CANDIDATE="${REPO_ROOT}/${RESULT_CANDIDATE}"
RESULT_DIR=$(realpath -m -- "${RESULT_CANDIDATE}")
case "${RESULT_DIR}/" in "${REPO_ROOT}/"*) ;; *) exit 2 ;; esac
[[ "${RESULT_DIR}" != "${REPO_ROOT}" ]]
[[ -f "${NODE_ENTRY}" && -x "${REPO_ROOT}/.vllm_venv/bin/vllm" ]]
[[ -f "${REPO_ROOT}/models/TinyLlama-1.1B-Chat-v1.0/config.json" ]]

module reset
module load pytorch/2.8.0
mapfile -t TEMPO_JOB_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#TEMPO_JOB_HOSTS[@]} -eq 2 ]]
export MASTER_ADDR="${TEMPO_JOB_HOSTS[0]}"
export MASTER_PORT=$((32000 + SLURM_JOB_ID % 10000))
export TEMPO_VLLM_API_PORT=$((42000 + SLURM_JOB_ID % 10000))
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=hsn NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_CROSS_NIC NCCL_ALGO NCCL_PROTO

mkdir -p -- "${RESULT_DIR}"
cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=15s 600s \
    srun --exact --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
    --cpus-per-task=64 --cpu-bind=cores --kill-on-bad-exit=1 --wait=5 \
    --time=00:09:30 --export=ALL \
    --output="${RESULT_DIR}/node-%N.stdout.log" \
    --error="${RESULT_DIR}/node-%N.stderr.log" \
    bash "${NODE_ENTRY}" "${REPO_ROOT}" "${RESULT_DIR}"
echo "vLLM TP8 smoke v3 result: ${RESULT_DIR}/smoke_result.json"
