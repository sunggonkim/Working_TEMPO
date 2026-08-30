#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?existing allocation required}"
: "${SLURM_JOB_NODELIST:?nodelist required}"
: "${TEMPO_PD_CONTENTION_APPROVED:?explicit approval required}"
[[ "${TEMPO_PD_CONTENTION_APPROVED}" == YES ]]
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
PORT_SLOT=$((1680 + SLURM_JOB_ID % 20))
REQUEST_RATE=${TEMPO_PD_CONTENTION_FOREGROUND_RATE:-2}
MAX_WORKERS=${TEMPO_PD_CONTENTION_MAX_WORKERS:-64}
[[ "${REQUEST_RATE}" =~ ^[0-9]+([.][0-9]+)?$ ]]
[[ "${REQUEST_RATE}" == 2 ]]
[[ "${MAX_WORKERS}" == 64 ]]
[[ "${TEMPO_PD_CONTENTION_DECODER_REFERENCE_RATE:-32}" == 32 ]]
[[ "${TEMPO_PD_CONTENTION_REMOTE_REFERENCE_RATE:-6.8}" == 6.8 ]]
[[ "${TEMPO_PD_CONTENTION_LOAD_FRACTION:-0.70}" == 0.70 ]]
[[ "${TEMPO_PD_CONTENTION_PHASE_DURATION_MS:-15000}" == 15000 ]]
[[ "${TEMPO_PD_CONTENTION_COOLDOWN_S:-2}" == 2 ]]

export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_PD_CONTENTION_DECODER_REFERENCE_RATE=32
export TEMPO_PD_CONTENTION_REMOTE_REFERENCE_RATE=6.8
export TEMPO_PD_CONTENTION_LOAD_FRACTION=0.70
export TEMPO_PD_CONTENTION_PHASE_DURATION_MS=15000
export TEMPO_PD_CONTENTION_COOLDOWN_S=2
export TEMPO_PD_CONTENTION_FROZEN_MANIFEST="${REPO_ROOT}/eval/sota_4node/tempo_pd_contention_workload_v4_frozen.json"
export TEMPO_VLLM_MAX_NUM_SEQS=${TEMPO_VLLM_MAX_NUM_SEQS:-16}
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
  bash "${SCRIPT_DIR}/contention_pd_node_entry.sh" \
  "${REPO_ROOT}" "${RESULT_DIR}" "${WORKLOAD}" "${HOSTS_CSV}" \
  "${PORT_SLOT}" "${REQUEST_RATE}" "${MAX_WORKERS}" 128 8 3000 250 16000

[[ -s "${RESULT_DIR}/tempo_pd_contention_fixed/raw.json" ]]
[[ -s "${RESULT_DIR}/result.json" ]]
echo "Fixed contention result: ${RESULT_DIR}/result.json"
