#!/usr/bin/env bash
set -euo pipefail

readonly LMCACHE_COMMIT="227d13f5c9fdb52ddb933641d34331f678de03a0"
if [[ $# -gt 1 ]]; then exit 2; fi
: "${SLURM_JOB_ID:?run inside an existing allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
[[ "${SLURM_JOB_NUM_NODES:-}" == 2 ]]
[[ "${TEMPO_LMCACHE_ACTIVE_PULSE_APPROVED:-}" == YES ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RESULT_CANDIDATE=${1:-"${REPO_ROOT}/results/lmcache_active_pulse_job_${SLURM_JOB_ID}"}
[[ "${RESULT_CANDIDATE}" == /* ]] || RESULT_CANDIDATE="${REPO_ROOT}/${RESULT_CANDIDATE}"
RESULT_DIR=$(realpath -m -- "${RESULT_CANDIDATE}")
case "${RESULT_DIR}/" in "${REPO_ROOT}/"*) ;; *) exit 2 ;; esac
[[ "${RESULT_DIR}" != "${REPO_ROOT}" ]]

module reset
module load pytorch/2.8.0
MODULE_PYTHON=$(command -v python)
SOTA_SITE="${REPO_ROOT}/.sota_venv/lib/python3.12/site-packages"
LMCACHE_REPO="${REPO_ROOT}/third_party/lmcache"
[[ -x "${MODULE_PYTHON}" && -d "${SOTA_SITE}" && -d "${LMCACHE_REPO}" ]]
[[ "$(git -C "${LMCACHE_REPO}" rev-parse HEAD)" == "${LMCACHE_COMMIT}" ]]
mapfile -t TEMPO_JOB_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#TEMPO_JOB_HOSTS[@]} -eq 2 ]]

export MASTER_ADDR="${TEMPO_JOB_HOSTS[0]}"
export MASTER_PORT=$((24000 + SLURM_JOB_ID % 10000))
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

ACTIVE_PLAN_PATH="${RESULT_DIR}/active_pulse_plan.json"
"${MODULE_PYTHON}" -m eval.sota_4node.compile_lmcache_active_pulse_plan \
    --output "${ACTIVE_PLAN_PATH}"
export TEMPO_ACTIVE_SERVICE_PLAN="${ACTIVE_PLAN_PATH}"

NIXL_PORT_BASE=$((44000 + SLURM_JOB_ID % 10000))
timeout --foreground --signal=TERM --kill-after=5s 240s \
    srun --exact \
    --nodes=2 --ntasks=8 --ntasks-per-node=4 \
    --distribution=block:block --gpus-per-node=4 --gpu-bind=none \
    --cpus-per-task=32 --cpu-bind=map_ldom:3,2,1,0 \
    --kill-on-bad-exit=1 --wait=3 --time=00:04:00 --export=ALL \
    --output="${RESULT_DIR}/rank-%t.stdout.log" \
    --error="${RESULT_DIR}/rank-%t.stderr.log" \
    "${MODULE_PYTHON}" -m eval.sota_4node.lmcache_active_pulse_bootstrap \
        --output-dir "${RESULT_DIR}" \
        --requests 2 --kv-kib 8192 --chunk-kib 512 \
        --tokens 64 --layers 8 --hidden-size 1024 --context 128 \
        --port-base "${NIXL_PORT_BASE}"

echo "LMCache/TEMPO active-pulse result: ${RESULT_DIR}/result.json"
