#!/usr/bin/env bash
#SBATCH --job-name=tempo_inf_2node
#SBATCH --account=m5320_g
#SBATCH --qos=debug_preempt
#SBATCH --constraint=gpu&hbm80g
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --no-requeue
#SBATCH --mail-type=NONE
#SBATCH --output=results/tempo_inf_2node_%j.out
#SBATCH --error=results/tempo_inf_2node_%j.err

set -euo pipefail

[[ "${TEMPO_INFERENCE_2NODE_APPROVED:-}" == YES ]] || exit 2
[[ "${SLURM_JOB_NUM_NODES:-}" == 2 ]] || exit 2
[[ "${SLURM_NTASKS:-}" == 8 ]] || exit 2

REPO_ROOT=$(realpath -e -- "${SLURM_SUBMIT_DIR:?submit from repository root}")
RUNNER="${REPO_ROOT}/eval/sota_4node/run_inference_interconnect_2node.py"
[[ -f "${RUNNER}" ]] || exit 2

module reset
module load pytorch/2.8.0

mapfile -t TEMPO_JOB_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ "${#TEMPO_JOB_HOSTS[@]}" == 2 ]] || exit 2
export MASTER_ADDR="${TEMPO_JOB_HOSTS[0]}"
export MASTER_PORT=$((26000 + SLURM_JOB_ID % 10000))
export WORLD_SIZE=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_IB_DISABLE NCCL_CROSS_NIC NCCL_ALGO NCCL_PROTO

RESULT_DIR="${REPO_ROOT}/results/inference_interconnect_2node_job_${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"

timeout --signal=TERM --kill-after=5s 240s \
  srun --time=00:04:00 --ntasks=8 --ntasks-per-node=4 \
    --distribution=block:block --gpus-per-node=4 --gpu-bind=none \
    --cpu-bind=map_ldom:3,2,1,0 --kill-on-bad-exit=1 --wait=3 --export=ALL \
    python "${RUNNER}" \
      --output-dir "${RESULT_DIR}" \
      --requests-per-block 2 \
      --tokens 16 \
      --layers 4 \
      --hidden-size 1024 \
      --context 128 \
      --kv-mib 128 \
      --chunk-mib 32
