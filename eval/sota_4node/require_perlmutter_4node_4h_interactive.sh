#!/usr/bin/env bash
TEMPO_PD_GUARD_CALLER_FLAGS=$-
if shopt -qo pipefail; then
  TEMPO_PD_GUARD_CALLER_PIPEFAIL=1
else
  TEMPO_PD_GUARD_CALLER_PIPEFAIL=0
fi
set -euo pipefail

tempo_pd_native_guard_fail() {
  echo "TEMPO native-allocation guard: $*" >&2
  exit 2
}

[[ "$(id -u)" -ne 0 ]] || \
  tempo_pd_native_guard_fail "refusing privileged execution"
[[ "$(command -v srun)" == /usr/bin/srun ]] || \
  tempo_pd_native_guard_fail "native /usr/bin/srun required"

# TEMPO-GO deliberately uses the host software stack.  Fail closed if an
# allocation inherited any container/rootfs activation.  Presence of a
# Shifter binary on Perlmutter is harmless; these environment variables are
# not, because they can route an otherwise ordinary srun through a UDI.
for tempo_pd_forbidden_var in \
  SHIFTER_RUNTIME SHIFTER_IMAGE UDI CRAY_ROOTFS SLURM_CONTAINER
do
  [[ -z "${!tempo_pd_forbidden_var:-}" ]] || \
    tempo_pd_native_guard_fail \
      "forbidden container environment: ${tempo_pd_forbidden_var}"
done
while IFS= read -r tempo_pd_env_name; do
  case "${tempo_pd_env_name}" in
    SHIFTER_*|UDI_*|SLURM_SPANK_*SHIFTER*|SLURM_SPANK_*UDI*)
      tempo_pd_native_guard_fail \
        "forbidden container environment: ${tempo_pd_env_name}"
      ;;
  esac
done < <(compgen -e)

: "${SLURM_JOB_ID:?existing Slurm allocation required}"
: "${SLURM_JOB_NODELIST:?Slurm allocation nodelist required}"
[[ "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]

# A direct shell launched by salloc normally exports one of these counts, but
# an srun job step may omit both while preserving the allocation nodelist.
# The authoritative scontrol record below still requires NumNodes=4, so an
# absent step-local count is safe; a present conflicting count remains fatal.
TEMPO_PD_STEP_NODE_COUNT="${SLURM_JOB_NUM_NODES:-${SLURM_JOB_NODES:-}}"
[[ -z "${TEMPO_PD_STEP_NODE_COUNT}" || "${TEMPO_PD_STEP_NODE_COUNT}" == 4 ]]

# Query once at experiment admission.  Never poll Slurm from a loop.  A
# parent allocation launcher may already have obtained this exact record and
# exported it to a nested GPU step.  Reusing that receipt avoids a second
# control-plane RPC from a compute node while Slurm is creating the step (the
# RPC can otherwise block indefinitely even though the allocation is healthy).
TEMPO_PD_ALLOCATION_RECORD_SOURCE="inherited"
if [[ -z "${TEMPO_PD_ALLOCATION_RECORD:-}" ]]; then
  TEMPO_PD_ALLOCATION_RECORD_SOURCE="bounded_scontrol"
  TEMPO_PD_SCONTROL_TIMEOUT_SECONDS="${TEMPO_PD_SCONTROL_TIMEOUT_SECONDS:-15}"
  [[ "${TEMPO_PD_SCONTROL_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] || \
    tempo_pd_native_guard_fail "invalid scontrol timeout"
  (( TEMPO_PD_SCONTROL_TIMEOUT_SECONDS >= 1 &&
     TEMPO_PD_SCONTROL_TIMEOUT_SECONDS <= 60 )) || \
    tempo_pd_native_guard_fail "scontrol timeout must be 1..60 seconds"
  if ! TEMPO_PD_ALLOCATION_RECORD="$(
      /usr/bin/timeout --foreground --signal=TERM --kill-after=3s \
      "${TEMPO_PD_SCONTROL_TIMEOUT_SECONDS}s" \
      scontrol show job "${SLURM_JOB_ID}" --oneliner
    )"; then
    tempo_pd_native_guard_fail \
      "bounded scontrol allocation query failed or timed out"
  fi
fi
[[ -n "${TEMPO_PD_ALLOCATION_RECORD}" ]] || \
  tempo_pd_native_guard_fail "empty allocation record"
[[ " ${TEMPO_PD_ALLOCATION_RECORD} " == *" JobState=RUNNING "* ]]
[[
  " ${TEMPO_PD_ALLOCATION_RECORD} " == *" QOS=interactive "*
  || " ${TEMPO_PD_ALLOCATION_RECORD} " == *" QOS=gpu_interactive "*
]]
[[
  " ${TEMPO_PD_ALLOCATION_RECORD} " == *" TimeLimit=04:00:00 "*
  || " ${TEMPO_PD_ALLOCATION_RECORD} " == *" TimeLimit=4:00:00 "*
]]
[[ " ${TEMPO_PD_ALLOCATION_RECORD} " == *" NumNodes=4 "* ]]
[[ "${TEMPO_PD_ALLOCATION_RECORD}" == *"gres/gpu=16"* ]]
case "${TEMPO_PD_ALLOCATION_RECORD,,}" in
  *shifter*|*udiroot*|*"--image"*)
    tempo_pd_native_guard_fail "containerized Slurm job rejected"
    ;;
esac

TEMPO_PD_NATIVE_GUARD_VERSION=1
TEMPO_PD_NATIVE_GUARD_SHA256=$(
  sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}'
)
export TEMPO_PD_NATIVE_GUARD_VERSION TEMPO_PD_NATIVE_GUARD_SHA256
export TEMPO_PD_ALLOCATION_RECORD_SOURCE
printf 'TEMPO native-allocation guard passed: job=%s sha256=%s\n' \
  "${SLURM_JOB_ID}" "${TEMPO_PD_NATIVE_GUARD_SHA256}"

unset TEMPO_PD_ALLOCATION_RECORD
unset TEMPO_PD_ALLOCATION_RECORD_SOURCE TEMPO_PD_SCONTROL_TIMEOUT_SECONDS
unset TEMPO_PD_STEP_NODE_COUNT
unset tempo_pd_forbidden_var tempo_pd_env_name
unset -f tempo_pd_native_guard_fail
if (( TEMPO_PD_GUARD_CALLER_PIPEFAIL == 0 )); then
  set +o pipefail
fi
if [[ "${TEMPO_PD_GUARD_CALLER_FLAGS}" != *u* ]]; then
  set +u
fi
if [[ "${TEMPO_PD_GUARD_CALLER_FLAGS}" != *e* ]]; then
  set +e
fi
unset TEMPO_PD_GUARD_CALLER_FLAGS TEMPO_PD_GUARD_CALLER_PIPEFAIL
