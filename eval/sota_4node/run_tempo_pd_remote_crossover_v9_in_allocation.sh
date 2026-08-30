#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?run inside an existing allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
: "${TEMPO_PD_REMOTE_CROSSOVER_APPROVED:?set approval to YES}"
[[ "${TEMPO_PD_REMOTE_CROSSOVER_APPROVED}" == YES ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 4 ]]
[[ $# -eq 4 ]]
RATE=$1
WORKERS=$2
PORT_STRIDE=$3
RESULT_CANDIDATE=$4
[[ "${RATE}" =~ ^[0-9]+([.][0-9]+)?$ ]]
[[ "${WORKERS}" =~ ^[0-9]+$ && "${WORKERS}" -ge 2 && "${WORKERS}" -le 32 ]]
[[ "${PORT_STRIDE}" =~ ^[0-7]$ ]]
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
[[ "${RESULT_CANDIDATE}" == /* ]] || RESULT_CANDIDATE="${REPO_ROOT}/${RESULT_CANDIDATE}"
RESULT_DIR=$(realpath -m -- "${RESULT_CANDIDATE}")
case "${RESULT_DIR}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ ! -e "${RESULT_DIR}" ]]
module reset
module load pytorch/2.8.0
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
PORT_SLOT=$((2000 + SLURM_JOB_ID % 20 + PORT_STRIDE * 220))
mkdir -p -- "${RESULT_DIR}"
cp -- "${SCRIPT_DIR}/tempo_pd_remote_crossover_contract_v9.json" "${RESULT_DIR}/contract.json"
printf '%s\n' "rate=${RATE}" "workers=${WORKERS}" "port_stride=${PORT_STRIDE}" > "${RESULT_DIR}/launch_parameters.txt"
cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=30s 5400s \
  srun --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 --distribution=block:block \
  --gpus-per-task=4 --gpu-bind=none --cpus-per-task=64 --cpu-bind=cores \
  --kill-on-bad-exit=1 --wait=10 --time=01:29:00 --export=ALL \
  --output="${RESULT_DIR}/slurm-node-%N.stdout.log" \
  --error="${RESULT_DIR}/slurm-node-%N.stderr.log" \
  bash "${SCRIPT_DIR}/remote_crossover_node_entry_v9.sh" \
  "${REPO_ROOT}" "${RESULT_DIR}" "${HOSTS_CSV}" "${PORT_SLOT}" \
  "${RATE}" "${WORKERS}" 32 3 3000 250 12000
[[ -s "${RESULT_DIR}/result.json" ]]
echo "Remote crossover result: ${RESULT_DIR}/result.json"
