#!/usr/bin/env bash
set -euo pipefail

# Attach Candidate O to one already-approved, no-shell Perlmutter allocation.
# This wrapper deliberately reserves no GPU in the outer orchestration step.
# The child vLLM and NCCL/LMCache steps own and overlap the allocation GPUs.
# Reserving all 16 GPUs here makes those native co-job steps fail with
# "Requested node configuration is not available".
[[ $# -ge 1 && $# -le 2 ]] || {
  echo "usage: $0 SLURM_JOB_ID [RESULT_DIR]" >&2
  exit 2
}
[[ "${TEMPO_GO_C9_CAUSAL_BURST_APPROVED:-}" == YES ]] || {
  echo "TEMPO_GO_C9_CAUSAL_BURST_APPROVED=YES is required" >&2
  exit 2
}

ALLOC_JOB_ID=$1
[[ "${ALLOC_JOB_ID}" =~ ^[0-9]+$ ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RESULT_DIR=${2:-"${REPO_ROOT}/results/tempo_go_c9_route_liveness_job_${ALLOC_JOB_ID}"}
[[ "${RESULT_DIR}" == /* ]] || RESULT_DIR="${REPO_ROOT}/${RESULT_DIR}"
RESULT_DIR=$(realpath -m -- "${RESULT_DIR}")
case "${RESULT_DIR}/" in
  "${REPO_ROOT}/results/"*) ;;
  *) echo "result directory must be under ${REPO_ROOT}/results" >&2; exit 2 ;;
esac
[[ ! -e "${RESULT_DIR}" ]] || {
  echo "result directory already exists: ${RESULT_DIR}" >&2
  exit 2
}

JOB_RECORD=$(scontrol show job "${ALLOC_JOB_ID}")
USER_NAME=$(id -un)
USER_ID=$(id -u)
grep -Fq "UserId=${USER_NAME}(${USER_ID})" <<<"${JOB_RECORD}"
grep -Eq '(^|[[:space:]])JobState=RUNNING([[:space:]]|$)' <<<"${JOB_RECORD}"
grep -Eq '(^|[[:space:]])NumNodes=4([[:space:]]|$)' <<<"${JOB_RECORD}"
grep -Eq '(^|[[:space:]])QOS=gpu_interactive([[:space:]]|$)' <<<"${JOB_RECORD}"
grep -Eq '(^|[[:space:]])Network=job_vni([[:space:]]|$)' <<<"${JOB_RECORD}"
grep -Eq '(^|,)gres/gpu=16(,|$)' <<<"$(sed -n 's/^ *AllocTRES=//p' <<<"${JOB_RECORD}")"

# Refuse to overlap an unknown active step.  The allocation's extern step is
# expected; every measured child is created only after this check.
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
  bash "${SCRIPT_DIR}/run_tempo_go_c9_candidate_o_route_liveness_in_allocation.sh"
