#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 && $# -le 2 ]] || {
  echo "usage: $0 SLURM_JOB_ID [RESULT_DIR]" >&2
  exit 2
}
[[ "${TEMPO_GO_C9_CAUSAL_BURST_APPROVED:-}" == YES ]] || exit 2

ALLOC_JOB_ID=$1
[[ "${ALLOC_JOB_ID}" =~ ^[0-9]+$ ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RESULT_DIR=${2:-"${REPO_ROOT}/results/tempo_go_c9_bounded_observer_job_${ALLOC_JOB_ID}"}
[[ "${RESULT_DIR}" == /* ]] || RESULT_DIR="${REPO_ROOT}/${RESULT_DIR}"
RESULT_DIR=$(realpath -m -- "${RESULT_DIR}")
case "${RESULT_DIR}/" in
  "${REPO_ROOT}/results/"*) ;;
  *) exit 2 ;;
esac
[[ ! -e "${RESULT_DIR}" ]] || exit 2

JOB_RECORD=$(scontrol show job "${ALLOC_JOB_ID}")
USER_NAME=$(id -un)
USER_ID=$(id -u)
grep -Fq "UserId=${USER_NAME}(${USER_ID})" <<<"${JOB_RECORD}"
grep -Eq '(^|[[:space:]])JobState=RUNNING([[:space:]]|$)' <<<"${JOB_RECORD}"
grep -Eq '(^|[[:space:]])NumNodes=4([[:space:]]|$)' <<<"${JOB_RECORD}"
grep -Eq '(^|[[:space:]])QOS=gpu_interactive([[:space:]]|$)' <<<"${JOB_RECORD}"
grep -Eq '(^|[[:space:]])Network=job_vni([[:space:]]|$)' <<<"${JOB_RECORD}"
grep -Eq '(^|,)gres/gpu=16(,|$)' <<<"$(sed -n 's/^ *AllocTRES=//p' <<<"${JOB_RECORD}")"

mapfile -t ACTIVE_STEPS < <(
  squeue --steps --noheader -j "${ALLOC_JOB_ID}" -o '%i' \
    | awk -v job="${ALLOC_JOB_ID}" '$1 != job ".extern" {print $1}'
)
if (( ${#ACTIVE_STEPS[@]} != 0 )); then
  printf 'active step already exists: %s\n' "${ACTIVE_STEPS[@]}" >&2
  exit 2
fi

export TEMPO_GO_C9_CAUSAL_BURST_RESULT_DIR="${RESULT_DIR}"
exec env -u SLURM_GPUS_PER_TASK -u SLURM_TRES_PER_TASK \
  /usr/bin/srun --jobid="${ALLOC_JOB_ID}" --overlap --exact \
  --nodes=4 --ntasks=4 --ntasks-per-node=1 --distribution=block:block \
  --gpus=0 --gres=none --cpus-per-task=128 --cpu-bind=none \
  --network=no_vni \
  bash "${SCRIPT_DIR}/run_tempo_go_c9_candidate_p_bounded_observer_in_allocation.sh"
