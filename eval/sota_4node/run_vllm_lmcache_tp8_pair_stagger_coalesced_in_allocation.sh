#!/usr/bin/env bash
set -euo pipefail
[[ $# -le 1 ]] || exit 2
: "${SLURM_JOB_ID:?}" "${SLURM_JOB_NODELIST:?}"
[[ "${SLURM_JOB_NUM_NODES:-}" == 2 && "${TEMPO_VLLM_LMCACHE_STAGGER_APPROVED:-}" == YES ]]
DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); ROOT=$(cd -- "${DIR}/../.." && pwd)
OUT=${1:-"${ROOT}/results/vllm_lmcache_tp8_pair_stagger_coalesced_job_${SLURM_JOB_ID}"}; [[ "${OUT}" == /* ]] || OUT="${ROOT}/${OUT}"; OUT=$(realpath -m -- "${OUT}")
case "${OUT}/" in "${ROOT}/"*) ;; *) exit 2 ;; esac
module reset; module load pytorch/2.8.0
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}"); [[ ${#HOSTS[@]} -eq 2 ]]
export MASTER_ADDR="${HOSTS[0]}" TEMPO_VLLM_MASTER_PORT=$((23000+SLURM_JOB_ID%10000))
export TEMPO_SIDECAR_MASTER_PORT=$((34000+SLURM_JOB_ID%10000)) TEMPO_VLLM_API_PORT=$((45000+SLURM_JOB_ID%10000)) TEMPO_NIXL_PORT_BASE=$((55000+SLURM_JOB_ID%10000))
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${ROOT}"
export NCCL_NET=Socket NCCL_SOCKET_IFNAME=hsn NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_CROSS_NIC NCCL_ALGO NCCL_PROTO UCX_TLS
mkdir -p -- "${OUT}"; cd -- "${ROOT}"
timeout --foreground --signal=TERM --kill-after=15s 480s srun --exact --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --distribution=block:block --gpus-per-task=4 --gpu-bind=none --cpus-per-task=64 --cpu-bind=cores \
    --kill-on-bad-exit=1 --wait=10 --time=00:07:30 --export=ALL \
    --output="${OUT}/wrapper-node-%N.stdout.log" --error="${OUT}/wrapper-node-%N.stderr.log" \
    "${DIR}/vllm_lmcache_tp8_pair_stagger_coalesced_node.sh" "${ROOT}" "${OUT}"
echo "coalesced result: ${OUT}/result.json"
