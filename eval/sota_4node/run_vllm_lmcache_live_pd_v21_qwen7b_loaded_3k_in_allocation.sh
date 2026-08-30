#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?run inside an existing allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
[[ "${SLURM_JOB_NUM_NODES:-}" == 4 && "${TEMPO_LIVE_PD_APPROVED:-}" == YES ]]
[[ $# -le 1 ]] || exit 2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RESULT_CANDIDATE=${1:-"${REPO_ROOT}/results/live_vllm_lmcache_pd_v21_qwen7b_loaded_3k_job_${SLURM_JOB_ID}"}
[[ "${RESULT_CANDIDATE}" == /* ]] || RESULT_CANDIDATE="${REPO_ROOT}/${RESULT_CANDIDATE}"
RESULT_DIR=$(realpath -m -- "${RESULT_CANDIDATE}")
case "${RESULT_DIR}/" in "${REPO_ROOT}/"*) ;; *) exit 2 ;; esac
[[ "${RESULT_DIR}" != "${REPO_ROOT}" && ! -e "${RESULT_DIR}" ]]
module reset
module load pytorch/2.8.0
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
PORT_SLOT=$((2340 + SLURM_JOB_ID % 30))
mkdir -p -- "${RESULT_DIR}"
cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=30s 5400s \
 srun --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 --distribution=block:block \
 --gpus-per-task=4 --gpu-bind=none --cpus-per-task=64 --cpu-bind=cores \
 --kill-on-bad-exit=1 --wait=10 --time=01:29:00 --export=ALL \
 --output="${RESULT_DIR}/slurm-node-%N.stdout.log" \
 --error="${RESULT_DIR}/slurm-node-%N.stderr.log" \
 bash "${SCRIPT_DIR}/live_pd_node_entry_v20.sh" \
 "${REPO_ROOT}" "${RESULT_DIR}" "${HOSTS_CSV}" "${PORT_SLOT}"
[[ -s "${RESULT_DIR}/result.json" ]]
echo "Loaded Qwen2.5-7B 3K P/D result: ${RESULT_DIR}/result.json"
