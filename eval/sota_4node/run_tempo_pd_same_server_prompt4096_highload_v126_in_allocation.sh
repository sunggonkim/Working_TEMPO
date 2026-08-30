#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?run inside an existing allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
: "${TEMPO_PD_SAME_SERVER_APPROVED:?set approval to YES}"
[[ "${TEMPO_PD_SAME_SERVER_APPROVED}" == YES ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 4 && $# -eq 2 ]]
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKLOAD_ROOT=$(realpath -e -- "$1")
RESULT_DIR=$(realpath -m -- "$2")
case "${WORKLOAD_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
case "${RESULT_DIR}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ -s "${WORKLOAD_ROOT}/workloads/validation.jsonl" && ! -e "${RESULT_DIR}" ]]
module reset
module load pytorch/2.8.0
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
PORT_SLOT=$((1580 + SLURM_JOB_ID % 20))
mkdir -p -- "${RESULT_DIR}"
cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=30s 2640s \
  srun --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 --distribution=block:block \
  --gpus-per-task=4 --gpu-bind=none --cpus-per-task=64 --cpu-bind=cores \
  --kill-on-bad-exit=1 --wait=10 --time=00:43:00 --export=ALL \
  --output="${RESULT_DIR}/slurm-node-%N.stdout.log" \
  --error="${RESULT_DIR}/slurm-node-%N.stderr.log" \
  bash "${SCRIPT_DIR}/same_server_prompt4096_production_node_entry_v125.sh" \
  "${REPO_ROOT}" "${RESULT_DIR}" "${WORKLOAD_ROOT}" "${HOSTS_CSV}" "${PORT_SLOT}" \
  16 32 128 8 6000 250 20000
[[ -s "${RESULT_DIR}/production_final.json" ]]
"${REPO_ROOT}/.sota_venv/bin/python" -m \
  eval.sota_4node.analyze_tempo_pd_prompt4096_highload_v126 \
  --input "${RESULT_DIR}/production_final.json" \
  --output "${RESULT_DIR}/highload_final.json"
[[ -s "${RESULT_DIR}/highload_final.json" ]]
echo "High-load prompt4096: ${RESULT_DIR}/highload_final.json"
