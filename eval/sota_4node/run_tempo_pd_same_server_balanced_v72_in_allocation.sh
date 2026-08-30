#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?run inside an existing allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
: "${TEMPO_PD_SAME_SERVER_APPROVED:?set approval to YES}"
[[ "${TEMPO_PD_SAME_SERVER_APPROVED}" == YES ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 4 && $# -eq 2 ]]
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
REFERENCE=$(realpath -e -- "$1")
RESULT_DIR=$(realpath -m -- "$2")
case "${REFERENCE}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
case "${RESULT_DIR}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ -s "${REFERENCE}/workloads/validation.jsonl" && ! -e "${RESULT_DIR}" ]]
module reset
module load pytorch/2.8.0
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
PORT_SLOT=$((160 + SLURM_JOB_ID % 20))
mkdir -p -- "${RESULT_DIR}"
cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=30s 5400s \
  srun --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 --distribution=block:block \
  --gpus-per-task=4 --gpu-bind=none --cpus-per-task=64 --cpu-bind=cores \
  --kill-on-bad-exit=1 --wait=10 --time=01:29:00 --export=ALL \
  --output="${RESULT_DIR}/slurm-node-%N.stdout.log" \
  --error="${RESULT_DIR}/slurm-node-%N.stderr.log" \
  bash "${SCRIPT_DIR}/same_server_balanced_node_entry_v72.sh" \
  "${REPO_ROOT}" "${RESULT_DIR}" "${REFERENCE}" "${HOSTS_CSV}" "${PORT_SLOT}" \
  32 32 32 8 3000 250 12000
[[ -s "${RESULT_DIR}/same_server_final.json" ]]
echo "Order-balanced result: ${RESULT_DIR}/same_server_final.json"
