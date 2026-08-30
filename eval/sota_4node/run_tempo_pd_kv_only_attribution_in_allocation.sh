#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?existing allocation required}"
: "${SLURM_JOB_NODELIST:?nodelist required}"
: "${TEMPO_PD_KV_ATTR_APPROVED:?explicit approval required}"
[[ "${TEMPO_PD_KV_ATTR_APPROVED}" == YES ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 4 ]]
[[ $# -eq 2 ]]

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKLOAD=$(realpath -e -- "$1")
RESULT_DIR=$(realpath -m -- "$2")
case "${WORKLOAD}" in "${REPO_ROOT}"/results/*) ;; *) exit 2 ;; esac
case "${RESULT_DIR}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ -s "${WORKLOAD}" && ! -e "${RESULT_DIR}" ]]
[[ -z "${TEMPO_CXI_BACKGROUND_DUTY_CYCLE:-}" ]]
[[ -z "${TEMPO_CXI_BACKGROUND_START_FILE:-}" ]]

module reset
module load pytorch/2.8.0
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
PORT_SLOT=$((1720 + SLURM_JOB_ID % 20))
REQUEST_RATE=2
MAX_WORKERS=64

export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_PD_KV_ATTR_PHASE_DURATION_MS=${TEMPO_PD_KV_ATTR_PHASE_DURATION_MS:-8000}
export TEMPO_PD_KV_ATTR_COOLDOWN_S=${TEMPO_PD_KV_ATTR_COOLDOWN_S:-2}
export TEMPO_PD_KV_ATTR_RATES=${TEMPO_PD_KV_ATTR_RATES:-4,8,12,16,24,32}
export TEMPO_PD_KV_ATTR_REPETITIONS=${TEMPO_PD_KV_ATTR_REPETITIONS:-1}
export TEMPO_PD_KV_ATTR_ARM_ORDER=${TEMPO_PD_KV_ATTR_ARM_ORDER:-local_remote}
export TEMPO_PD_KV_ATTR_READINESS_S=${TEMPO_PD_KV_ATTR_READINESS_S:-1200}
export TEMPO_VLLM_MAX_NUM_SEQS=16
export TEMPO_VLLM_DECODER_PREFIX_CACHING=0
export TEMPO_LMCACHE_NIXL_BACKEND=${TEMPO_LMCACHE_NIXL_BACKEND:-UCX}
mkdir -p -- "${RESULT_DIR}"
cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=30s 2640s \
  srun --overlap --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
  --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
  --time=00:43:00 --export=ALL \
  --output="${RESULT_DIR}/slurm-node-%N.stdout.log" \
  --error="${RESULT_DIR}/slurm-node-%N.stderr.log" \
  bash "${SCRIPT_DIR}/kv_only_pd_node_entry.sh" \
  "${REPO_ROOT}" "${RESULT_DIR}" "${WORKLOAD}" "${HOSTS_CSV}" \
  "${PORT_SLOT}" "${REQUEST_RATE}" "${MAX_WORKERS}" 128 8 3000 250 16000

[[ -s "${RESULT_DIR}/tempo_pd_kv_only_attribution/raw.json" ]]
[[ -s "${RESULT_DIR}/result.json" ]]
echo "P-only attribution result: ${RESULT_DIR}/result.json"
