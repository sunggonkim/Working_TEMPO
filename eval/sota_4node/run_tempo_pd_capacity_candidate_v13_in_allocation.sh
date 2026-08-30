#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?run inside an existing allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
: "${TEMPO_PD_CAPACITY_APPROVED:?set approval to YES}"
[[ "${TEMPO_PD_CAPACITY_APPROVED}" == YES ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 4 && $# -eq 2 ]]
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
SCOUT_ROOT=$(realpath -e -- "$1")
RESULT_DIR=$(realpath -m -- "$2")
case "${SCOUT_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
case "${RESULT_DIR}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ -s "${SCOUT_ROOT}/crossover_local/raw.json" && ! -e "${RESULT_DIR}" ]]
[[ ! -e "${SCOUT_ROOT}/crossover_remote/raw.json" ]]
module reset
module load pytorch/2.8.0
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
PORT_SLOT=$((3000 + SLURM_JOB_ID % 20))
mkdir -p -- "${RESULT_DIR}"
cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=30s 3600s \
  srun --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 --distribution=block:block \
  --gpus-per-task=4 --gpu-bind=none --cpus-per-task=64 --cpu-bind=cores \
  --kill-on-bad-exit=1 --wait=10 --time=00:59:00 --export=ALL \
  --output="${RESULT_DIR}/slurm-node-%N.stdout.log" \
  --error="${RESULT_DIR}/slurm-node-%N.stderr.log" \
  bash "${SCRIPT_DIR}/capacity_candidate_node_entry_v13.sh" \
  "${REPO_ROOT}" "${RESULT_DIR}" "${SCOUT_ROOT}" "${HOSTS_CSV}" "${PORT_SLOT}" \
  16 16 2 3 3000 250 12000
[[ -s "${RESULT_DIR}/result.json" ]]
echo "Capacity candidate result: ${RESULT_DIR}/result.json"
