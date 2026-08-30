#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "usage: $0 [RESULT_DIR]" >&2
    exit 2
fi

: "${SLURM_JOB_ID:?run inside an existing allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
[[ "${SLURM_JOB_NUM_NODES:-}" == 2 ]] || {
    echo "requires exactly two allocated nodes" >&2
    exit 2
}
[[ "${TEMPO_INFERENCE_EPOCH_APPROVED:-}" == YES ]] || {
    echo "set TEMPO_INFERENCE_EPOCH_APPROVED=YES after approving this run" >&2
    exit 2
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
if [[ $# -eq 1 ]]; then
    if [[ $1 == /* ]]; then
        RESULT_CANDIDATE=$1
    else
        RESULT_CANDIDATE="${REPO_ROOT}/$1"
    fi
else
    RESULT_CANDIDATE="${REPO_ROOT}/results/inference_epoch_job_${SLURM_JOB_ID}"
fi
RESULT_DIR=$(realpath -m -- "${RESULT_CANDIDATE}")
case "${RESULT_DIR}/" in
    "${REPO_ROOT}/"*) ;;
    *)
        echo "RESULT_DIR must stay inside ${REPO_ROOT}" >&2
        exit 2
        ;;
esac
[[ "${RESULT_DIR}" != "${REPO_ROOT}" ]] || exit 2

module reset
module load pytorch/2.8.0

mapfile -t TEMPO_JOB_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#TEMPO_JOB_HOSTS[@]} -eq 2 ]] || exit 2

export MASTER_ADDR="${TEMPO_JOB_HOSTS[0]}"
export MASTER_PORT=$((27000 + SLURM_JOB_ID % 10000))
export WORLD_SIZE=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_IB_DISABLE NCCL_CROSS_NIC NCCL_ALGO NCCL_PROTO

mkdir -p -- "${RESULT_DIR}"
PLAN_PATH="${RESULT_DIR}/epoch_plan.json"

cd -- "${REPO_ROOT}"
python -m eval.sota_4node.compile_inference_epoch_plan \
    --output "${PLAN_PATH}" \
    --total-quanta 16 \
    --deadline-tokens 10 \
    --token-slack-ms 1x4,3x6,0x6 \
    --width-penalty-ms 0:0,1:1,2:3,4:9 \
    --max-width 2 \
    --protect-prefix-tokens 4 \
    --protect-prefix-max-width 1
export TEMPO_EPOCH_PLAN="${PLAN_PATH}"

timeout --signal=TERM --kill-after=5s 240s \
    srun --exact \
    --nodes=2 \
    --ntasks=8 \
    --ntasks-per-node=4 \
    --distribution=block:block \
    --gpus-per-node=4 \
    --gpu-bind=none \
    --cpus-per-task=32 \
    --cpu-bind=map_ldom:3,2,1,0 \
    --kill-on-bad-exit=1 \
    --wait=3 \
    --output="${RESULT_DIR}/rank-%t.stdout.log" \
    --error="${RESULT_DIR}/rank-%t.stderr.log" \
    python -m eval.sota_4node.run_inference_epoch_2node \
        --output-dir "${RESULT_DIR}" \
        --requests-per-block 2 \
        --tokens 16 \
        --layers 4 \
        --hidden-size 1024 \
        --context 128 \
        --kv-mib 128 \
        --chunk-mib 32

echo "TEMPO epoch result: ${RESULT_DIR}/result.json"
