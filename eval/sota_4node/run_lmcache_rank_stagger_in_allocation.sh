#!/usr/bin/env bash
set -euo pipefail

readonly LMCACHE_COMMIT="227d13f5c9fdb52ddb933641d34331f678de03a0"
if [[ $# -gt 1 ]]; then
    echo "usage: $0 [RESULT_DIR]" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?run inside an existing allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
[[ "${SLURM_JOB_NUM_NODES:-}" == 2 ]] || exit 2
[[ "${TEMPO_LMCACHE_RANK_STAGGER_APPROVED:-}" == YES ]] || {
    echo "set TEMPO_LMCACHE_RANK_STAGGER_APPROVED=YES after approving this run" >&2
    exit 2
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RESULT_CANDIDATE=${1:-"${REPO_ROOT}/results/lmcache_rank_stagger_job_${SLURM_JOB_ID}"}
[[ "${RESULT_CANDIDATE}" == /* ]] || RESULT_CANDIDATE="${REPO_ROOT}/${RESULT_CANDIDATE}"
RESULT_DIR=$(realpath -m -- "${RESULT_CANDIDATE}")
case "${RESULT_DIR}/" in
    "${REPO_ROOT}/"*) ;;
    *) exit 2 ;;
esac
[[ "${RESULT_DIR}" != "${REPO_ROOT}" ]] || exit 2

module reset
module load pytorch/2.8.0
MODULE_PYTHON=$(command -v python)
SOTA_SITE="${REPO_ROOT}/.sota_venv/lib/python3.12/site-packages"
LMCACHE_REPO="${REPO_ROOT}/third_party/lmcache"
[[ -x "${MODULE_PYTHON}" && -d "${SOTA_SITE}" && -d "${LMCACHE_REPO}" ]]
[[ "$(git -C "${LMCACHE_REPO}" rev-parse HEAD)" == "${LMCACHE_COMMIT}" ]] || exit 2
mapfile -t TEMPO_JOB_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#TEMPO_JOB_HOSTS[@]} -eq 2 ]] || exit 2

export MASTER_ADDR="${TEMPO_JOB_HOSTS[0]}"
export MASTER_PORT=$((22000 + SLURM_JOB_ID % 10000))
export WORLD_SIZE=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
export TEMPO_LMCACHE_EXTRA_SITE="${SOTA_SITE}"
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_CROSS_NIC NCCL_ALGO NCCL_PROTO UCX_TLS

mkdir -p -- "${RESULT_DIR}"
cd -- "${REPO_ROOT}"
export TEMPO_LMCACHE_PREFLIGHT=YES
"${MODULE_PYTHON}" -m eval.sota_4node.lmcache_epoch_bootstrap \
    > "${RESULT_DIR}/runtime_preflight.json"
unset TEMPO_LMCACHE_PREFLIGHT

PLAN_PATH="${RESULT_DIR}/rank_stagger_plan.json"
"${MODULE_PYTHON}" -m eval.sota_4node.compile_inference_epoch_plan \
    --output "${PLAN_PATH}" \
    --total-quanta 4 \
    --deadline-tokens 10 \
    --token-slack-ms 3x16 \
    --width-penalty-ms 0:0,1:1 \
    --max-width 1
export TEMPO_EPOCH_PLAN="${PLAN_PATH}"

NIXL_PORT_BASE=$((42000 + SLURM_JOB_ID % 10000))
timeout --foreground --signal=TERM --kill-after=5s 240s \
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
    --time=00:04:00 \
    --export=ALL \
    --output="${RESULT_DIR}/rank-%t.stdout.log" \
    --error="${RESULT_DIR}/rank-%t.stderr.log" \
    "${MODULE_PYTHON}" -m eval.sota_4node.lmcache_rank_stagger_bootstrap \
        --output-dir "${RESULT_DIR}" \
        --requests 2 \
        --kv-mib 4 \
        --chunk-mib 1 \
        --tokens 16 \
        --layers 8 \
        --hidden-size 1024 \
        --context 128 \
        --port-base "${NIXL_PORT_BASE}"

echo "LMCache/TEMPO rank-stagger result: ${RESULT_DIR}/result.json"
