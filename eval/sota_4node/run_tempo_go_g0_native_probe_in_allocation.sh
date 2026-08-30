#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 0 ]] || {
  echo "usage: $0" >&2
  exit 2
}

readonly REPO_ROOT="/pscratch/sd/s/sgkim/Skim-Tempo"
readonly PYTHON="${REPO_ROOT}/.vllm_venv/bin/python"
readonly PROBE="${REPO_ROOT}/eval/sota_4node/probe_tempo_go_native_rank.py"
readonly RESULT_ROOT="${REPO_ROOT}/eval/sota_4node/results/tempo_go_g0"

source "${REPO_ROOT}/eval/sota_4node/require_perlmutter_4node_4h_interactive.sh"
[[ "$(id -u)" -ne 0 ]]
[[ -x "${PYTHON}" ]]
[[ -f "${PROBE}" ]]

mapfile -t TEMPO_GO_HOSTS < <(
  /usr/bin/scontrol show hostnames "${SLURM_JOB_NODELIST}"
)
[[ "${#TEMPO_GO_HOSTS[@]}" -eq 4 ]]
[[ "$(printf '%s\n' "${TEMPO_GO_HOSTS[@]}" | sort -u | wc -l)" -eq 4 ]]
printf -v TEMPO_GO_EXPECTED_HOSTS '%s,' "${TEMPO_GO_HOSTS[@]}"
TEMPO_GO_EXPECTED_HOSTS=${TEMPO_GO_EXPECTED_HOSTS%,}
export TEMPO_GO_EXPECTED_HOSTS

readonly OUTPUT_DIR="${RESULT_ROOT}/job-${SLURM_JOB_ID}-native-v1"
mkdir -p "${RESULT_ROOT}"
mkdir "${OUTPUT_DIR}"

readonly TEMPO_GO_PROBE_SRUN_COMMAND="/usr/bin/srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --gpus-per-task=4 --gpu-bind=none --kill-on-bad-exit=1 ${PYTHON} ${PROBE} rank --repo-root ${REPO_ROOT} --output-dir ${OUTPUT_DIR}"
export TEMPO_GO_PROBE_SRUN_COMMAND

/usr/bin/srun \
  --nodes=4 \
  --ntasks=4 \
  --ntasks-per-node=1 \
  --gpus-per-task=4 \
  --gpu-bind=none \
  --kill-on-bad-exit=1 \
  "${PYTHON}" "${PROBE}" rank \
  --repo-root "${REPO_ROOT}" \
  --output-dir "${OUTPUT_DIR}"

"${PYTHON}" "${PROBE}" aggregate \
  --repo-root "${REPO_ROOT}" \
  --output-dir "${OUTPUT_DIR}"

printf 'TEMPO-GO G0 native capability passed: %s\n' \
  "${OUTPUT_DIR}/manifest.json"
