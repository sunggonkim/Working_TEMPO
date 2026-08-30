#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]]
SCRIPT_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONTRACT_REPO_ROOT=""
if [[ -n "${TEMPO_GO_C5_RUN_CONTRACT:-}" &&
      -f "${TEMPO_GO_C5_RUN_CONTRACT}" ]]; then
  CONTRACT_REPO_ROOT=$(cd -- "$(dirname -- "$(realpath -e -- "${TEMPO_GO_C5_RUN_CONTRACT}")")/../.." && pwd)
fi
REPO_ROOT="${TEMPO_GO_REPO_ROOT:-${CONTRACT_REPO_ROOT:-${SCRIPT_REPO_ROOT}}}"
WORKLOAD_INPUT=$(realpath -e -- "$1")
RESULT_ROOT=$(realpath -m -- "$2")
case "${RESULT_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ ! -e "${RESULT_ROOT}" ]]
[[ -z "${SHIFTER_RUNTIME:-}" && -z "${SHIFTER_IMAGE:-}" ]]
[[ -z "${UDI:-}" && -z "${CRAY_ROOTFS:-}" && -z "${SLURM_CONTAINER:-}" ]]
TEMPO_GO_STEP_NODE_COUNT="${SLURM_JOB_NUM_NODES:-${SLURM_JOB_NODES:-}}"
[[ -z "${TEMPO_GO_STEP_NODE_COUNT}" || "${TEMPO_GO_STEP_NODE_COUNT}" == 4 ]]
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]

: "${TEMPO_GO_C5_RUN_CONTRACT:?frozen C5 run-contract path required}"
: "${TEMPO_GO_C5_RUN_CONTRACT_SHA256:?frozen C5 run-contract SHA-256 required}"
: "${TEMPO_GO_SOURCE_SNAPSHOT:=$(jq -er '.source_snapshot.root // empty' "${TEMPO_GO_C5_RUN_CONTRACT}" 2>/dev/null || true)}"
if [[ -n "${TEMPO_GO_SOURCE_SNAPSHOT}" ]]; then
  TEMPO_GO_SOURCE_SNAPSHOT=$(realpath -e -- "${TEMPO_GO_SOURCE_SNAPSHOT}")
  case "${TEMPO_GO_SOURCE_SNAPSHOT}/" in
    "${REPO_ROOT}/results/"*) ;;
    "${REPO_ROOT}/") ;;
    *) exit 2 ;;
  esac
  export TEMPO_GO_SOURCE_SNAPSHOT
fi
CONTRACT_VERIFIER="${TEMPO_GO_SOURCE_SNAPSHOT:-${REPO_ROOT}}/eval/sota_4node/tempo_go_c5_run_contract.py"
[[ -f "${CONTRACT_VERIFIER}" ]]
PYTHONPATH="${TEMPO_GO_SOURCE_SNAPSHOT:-${REPO_ROOT}}:${REPO_ROOT}" \
  "${REPO_ROOT}/.vllm_venv/bin/python" -m eval.sota_4node.tempo_go_c5_run_contract verify \
  --repo-root "${REPO_ROOT}" \
  --contract "${TEMPO_GO_C5_RUN_CONTRACT}" \
  --sha256 "${TEMPO_GO_C5_RUN_CONTRACT_SHA256}" \
  --workload-input "${WORKLOAD_INPUT}"
command -v flock >/dev/null 2>&1
# The campaign owns an allocation-scoped result/port/source namespace.  A
# repository-global lock can be left open by an unrelated user service from a
# previous allocation and then prevents a valid four-node allocation from
# starting.  Serialize duplicate launchers for this Slurm allocation while
# preserving the old global lock file untouched for audit; never delete or
# override another holder.
CAMPAIGN_LOCK_FILE="${TEMPO_GO_NATIVE_CAMPAIGN_LOCK_FILE:-${REPO_ROOT}/results/.tempo_go_native_campaign_${SLURM_JOB_ID}.lock}"
case "${CAMPAIGN_LOCK_FILE}/" in
  "${REPO_ROOT}/results/"*) ;;
  *) exit 2 ;;
esac
exec 9>"${CAMPAIGN_LOCK_FILE}"
if ! flock -n 9; then
  echo "TEMPO native campaign lock is held for allocation ${SLURM_JOB_ID}; refusing a duplicate launcher" >&2
  exit 3
fi
# Release the descriptor on every parent exit, including a preflight failure.
# Without an explicit unlock, a launcher killed between lock acquisition and
# the later co-job EXIT trap can leave the allocation namespace held by the
# user-session parent.  Never remove the lock file or override another holder.
release_campaign_lock() {
  flock -u 9 2>/dev/null || true
  exec 9>&- 2>/dev/null || true
}
trap release_campaign_lock EXIT
mkdir -p -- "${RESULT_ROOT}"

# An interactive salloc may report a small requested CPU count while Slurm
# grants the whole GPU-node CPU envelope.  Conversely, a manually-created
# allocation can really be only cpu=4/CPUsPerTask=1.  NERSC documents that
# every GPU srun must repeat its GPU/CPU shape; fail closed before any
# interconnect step if the allocation cannot host the frozen 4-node layout.
TEMPO_GO_SCONTROL_TIMEOUT_SECONDS="${TEMPO_GO_SCONTROL_TIMEOUT_SECONDS:-15}"
[[ "${TEMPO_GO_SCONTROL_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]
(( TEMPO_GO_SCONTROL_TIMEOUT_SECONDS >= 1 &&
   TEMPO_GO_SCONTROL_TIMEOUT_SECONDS <= 60 ))
JOB_INFO="${TEMPO_PD_ALLOCATION_RECORD:-}"
if [[ -z "${JOB_INFO}" ]]; then
  if ! JOB_INFO="$(
      /usr/bin/timeout --foreground --signal=TERM --kill-after=3s \
      "${TEMPO_GO_SCONTROL_TIMEOUT_SECONDS}s" \
      scontrol show job -o "${SLURM_JOB_ID}"
    )"; then
    echo "bounded scontrol allocation query failed or timed out" >&2
    exit 124
  fi
fi
[[ -n "${JOB_INFO}" ]]
JOB_NUM_NODES="$(sed -n 's/.* NumNodes=\([0-9][0-9]*\) .*/\1/p' <<<"${JOB_INFO}")"
JOB_NUM_CPUS="$(sed -n 's/.* NumCPUs=\([0-9][0-9]*\) .*/\1/p' <<<"${JOB_INFO}")"
JOB_CPUS_PER_TASK="$(sed -n 's/.* CPUs\/Task=\([0-9][0-9]*\) .*/\1/p' <<<"${JOB_INFO}")"
JOB_ALLOC_TRES="$(sed -n 's/.* AllocTRES=\([^ ]*\) .*/\1/p' <<<"${JOB_INFO}")"
JOB_NODELIST="$(sed -n 's/.* NodeList=\([^ ]*\) .*/\1/p' <<<"${JOB_INFO}")"
JOB_NETWORK="$(sed -n 's/.* Network=\([^ ]*\) .*/\1/p' <<<"${JOB_INFO}")"
[[ "${JOB_NUM_NODES}" == 4 ]]
[[ "${JOB_NUM_CPUS}" =~ ^[0-9]+$ ]] && (( JOB_NUM_CPUS >= 512 ))
[[ "${JOB_CPUS_PER_TASK}" =~ ^[0-9]+$ ]]
[[ "${JOB_ALLOC_TRES}" == *"gres/gpu=16"* ||
   "${JOB_ALLOC_TRES}" == *"gres/gpu:a100=16"* ]]
[[ -n "${JOB_NODELIST}" ]]
[[ -n "${JOB_NETWORK}" ]]
export SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-${JOB_NODELIST}}"
# Pass the already validated allocation receipt into every nested GPU step.
# The C5 guard must not issue a second unbounded scontrol RPC from a compute
# node while the parent is creating overlapping Slurm steps.
export TEMPO_PD_ALLOCATION_RECORD="${JOB_INFO}"
printf '%s\n' "${JOB_INFO}" > "${RESULT_ROOT}/perlmutter-job-info.txt"
export TEMPO_GO_REPO_ROOT="${REPO_ROOT}"
COJOB_LAUNCHER="${TEMPO_GO_SOURCE_SNAPSHOT:-${REPO_ROOT}}/eval/sota_4node/run_lmcache_nixl_contention_2node_in_allocation.sh"
[[ -x "${COJOB_LAUNCHER}" ]]

# If entered from an ``srun --pty`` shell, the current process already owns
# a job step.  Re-targeting that parent with ``--jobid`` can make a failed
# child preflight terminate the interactive shell.  Create a child step in an
# existing step; use an explicit allocation job id only from a plain
# allocation shell or batch/extern context.
TEMPO_GO_SRUN_JOB_ARGS=()
case "${SLURM_STEP_ID:-}" in
  ""|batch|extern)
    TEMPO_GO_SRUN_JOB_ARGS=("--jobid=${SLURM_JOB_ID}")
    ;;
esac

# The co-job is an opt-in, bounded workload inside the same user allocation.
# It does not change Slingshot/NIC configuration and does not control another
# tenant.  --overlap is intentional: the experiment measures shared GPU/NCCL
# and fabric contention with the vLLM/P-D fleet.  Perlmutter nevertheless
# limits the number of concurrent applications using the Slingshot
# configuration to three per node; --overlap does not waive that network
# limit.  The valid topology is therefore: the allocation's interactive
# parent (if any) + this two-node co-job + one four-node C5 step.  An outer
# srun wrapper or an orphaned sibling step makes the fourth network user and
# is rejected below rather than retried with a different interconnect mode.
COJOB_ROOT="${TEMPO_GO_CROSS_LAYER_COJOB_ROOT:-${REPO_ROOT}/results/tempo_go_cross_layer_cojob_${SLURM_JOB_ID}}"
case "${COJOB_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ ! -e "${COJOB_ROOT}" ]]
mkdir -p -- "${COJOB_ROOT}"

# The native runner uses one 4-GPU task per node with 128 CPUs per task.
# Perlmutter may grant the full 512-CPU GPU-node envelope while retaining a
# smaller requested ``CPUs/Task`` value; Slurm then emits a misleading
# ``may never run`` warning for every nested step.  Require the documented
# shape explicitly so a future campaign stops with a local receipt instead
# of entering a partially provisioned native run.
if (( JOB_CPUS_PER_TASK < 128 )); then
  PREFLIGHT_ROOT="${COJOB_ROOT}/perlmutter-native-step-preflight"
  mkdir -p -- "${PREFLIGHT_ROOT}"
  jq -n \
    --arg schema "tempo-perlmutter-native-step-preflight-v1" \
    --arg status "failed" \
    --arg job_id "${SLURM_JOB_ID}" \
    --arg nodelist "${JOB_NODELIST}" \
    --arg network_mode "${TEMPO_GO_SRUN_NETWORK_MODE:-default}" \
    --arg allocation_network "${JOB_NETWORK}" \
    --arg allocated_cpus_per_task "${JOB_CPUS_PER_TASK}" \
    --arg error "allocation_cpu_shape_too_small" \
    '{schema:$schema,status:$status,slurm_job_id:($job_id|tonumber),nodelist:$nodelist,
      nodes:4,ntasks:4,ntasks_per_node:1,gpus_per_task:4,cpus_per_task:128,
      gpu_bind:"none",cpu_bind:"cores",network:$network_mode,
      allocation_network:$allocation_network,
      allocated_cpus_per_task:($allocated_cpus_per_task|tonumber),error:$error}' \
    >"${PREFLIGHT_ROOT}/receipt.json"
  echo "Perlmutter native step preflight refused: CPUs/Task=${JOB_CPUS_PER_TASK}; request --cpus-per-task=128" >&2
  exit 1
fi

# A previous campaign can leave a child Slurm step alive even after its
# parent shell disappeared. Starting another overlapping GPU/VNI step in
# that state produces the misleading ``Error configuring interconnect``
# symptom (and can exceed Perlmutter's three-network-user limit). Use Slurm's
# live step queue here: ``scontrol show step`` can retain
# a RUNNING accounting record after the step has no live PID, while
# ``squeue --steps`` reports the controller's launchable/running step set.
# The allocation's base extern and interactive steps are expected; any other
# live step belongs to an earlier/parallel campaign and must be cleaned up
# explicitly first.
ACTIVE_CHILD_STEPS="$(
  squeue --steps --noheader --jobs="${SLURM_JOB_ID}" \
    --states=RUNNING --format="%i" 2>/dev/null | sort -u
)"
CURRENT_LAUNCHER_STEP_ID=""
case "${SLURM_STEP_ID:-}" in
  ''|batch|extern|interactive)
    ;;
  *[!0-9]*)
    ;;
  *)
    CURRENT_LAUNCHER_STEP_ID="${SLURM_JOB_ID}.${SLURM_STEP_ID}"
    ;;
esac
EXTRA_RUNNING_STEPS=""
while IFS= read -r step_id; do
  case "${step_id}" in
    ""|"${SLURM_JOB_ID}.extern"|"${SLURM_JOB_ID}.interactive"|"${SLURM_JOB_ID}.batch"|"${CURRENT_LAUNCHER_STEP_ID}")
      ;;
    *)
      EXTRA_RUNNING_STEPS+="${EXTRA_RUNNING_STEPS:+,}${step_id}"
      ;;
  esac
done <<<"${ACTIVE_CHILD_STEPS}"
if [[ -n "${EXTRA_RUNNING_STEPS}" ]]; then
  PREFLIGHT_ROOT="${COJOB_ROOT}/perlmutter-native-step-preflight"
  mkdir -p -- "${PREFLIGHT_ROOT}"
  jq -n \
    --arg schema "tempo-perlmutter-native-step-preflight-v1" \
    --arg status "failed" \
    --arg job_id "${SLURM_JOB_ID}" \
    --arg nodelist "${JOB_NODELIST}" \
    --arg network_mode "${TEMPO_GO_SRUN_NETWORK_MODE:-default}" \
    --arg allocation_network "${JOB_NETWORK}" \
    --arg active_steps "${EXTRA_RUNNING_STEPS}" \
    --arg error "allocation_has_active_child_steps" \
    '{schema:$schema,status:$status,slurm_job_id:($job_id|tonumber),nodelist:$nodelist,
      nodes:4,ntasks:4,ntasks_per_node:1,gpus_per_task:4,cpus_per_task:128,
      gpu_bind:"none",cpu_bind:"cores",network:$network_mode,
      allocation_network:$allocation_network,active_child_steps:$active_steps,error:$error}' \
    >"${PREFLIGHT_ROOT}/receipt.json"
  echo "Perlmutter native step preflight refused: active child steps=${EXTRA_RUNNING_STEPS}" >&2
  exit 1
fi

COJOB_STOP_FILE="${COJOB_ROOT}/stop.requested"
export TEMPO_GO_CROSS_LAYER_STOP_FILE="${COJOB_STOP_FILE}"
export TEMPO_GO_CROSS_LAYER_COMPONENT_APPROVED=YES
export TEMPO_GO_CROSS_LAYER_RESULT_DIR="${COJOB_ROOT}"
export TEMPO_GO_CROSS_LAYER_EPOCH="slurm-${SLURM_JOB_ID}-c5-cross-layer"
export TEMPO_GO_NCCL_COMMUNICATOR_ID="c5-cross-layer-nixl-nccl-2node"
export TEMPO_GO_NCCL_TELEMETRY_PATH="${COJOB_ROOT}/nccl_observer.json"
# C5 startup broadcasts a ~1.6-GiB Python overlay and can legitimately pause
# the producer's filesystem-visible observer for several seconds.  Keep the
# stale guard longer than the bounded NIXL failure budget so transient telemetry
# delay is not mistaken for a dead producer, while a real dead step still
# fails the campaign in finite time.
export TEMPO_GO_NCCL_OBSERVER_MAX_AGE_MS="${TEMPO_GO_NCCL_OBSERVER_MAX_AGE_MS:-60000}"
export TEMPO_GO_NCCL_TIMEOUT_SECONDS="${TEMPO_GO_NCCL_TIMEOUT_SECONDS:-60}"
export TEMPO_GO_NCCL_OBSERVER_STARTUP_GRACE_MS="${TEMPO_GO_NCCL_OBSERVER_STARTUP_GRACE_MS:-180000}"
export TEMPO_GO_CROSS_LAYER_SRUN_OVERLAP=1
export TEMPO_GO_CROSS_LAYER_SRUN_STEP_NAME="tempo-go-cross-layer-cojob-${SLURM_JOB_ID}"
# Perlmutter's native Slingshot setup is allocation/step specific.  Do not
# force ``--network=disable_rdzv_get`` during readiness: on some otherwise
# valid allocations that option itself fails VNI configuration.  A contract
# may opt in after a bounded allocation-specific probe, but the default is
# the scheduler's native network configuration and there is no automatic
# retry with a different network mode.
TEMPO_GO_SRUN_NETWORK_MODE="${TEMPO_GO_SRUN_NETWORK_MODE:-}"
TEMPO_GO_SRUN_NETWORK_ARGS=()
if [[ -n "${TEMPO_GO_SRUN_NETWORK_MODE}" ]]; then
  [[ "${TEMPO_GO_SRUN_NETWORK_MODE}" == "disable_rdzv_get" ]] || exit 2
  TEMPO_GO_SRUN_NETWORK_ARGS=(
    "--network=${TEMPO_GO_SRUN_NETWORK_MODE}"
  )
fi
export TEMPO_GO_SRUN_NETWORK_MODE

# ``job_vni`` is an allocation property on Perlmutter.  A step-level
# ``--network=job_vni`` cannot create one after the allocation was created
# without a VNI; Slurm then fails later with the opaque ``Error configuring
# interconnect`` message.  NERSC separately documents ``no_vni`` for work
# that does not use the interconnect.  That mode is therefore not a valid
# input to this NCCL/UCX cross-layer experiment even though Slurm can launch
# ordinary bash work with it.  Require the exact allocation property before
# starting either producer or C5 and leave a machine-readable boundary
# receipt.  This is deliberately not a retry/fallback.
if [[ "${JOB_NETWORK}" != "job_vni" ]]; then
  PREFLIGHT_ROOT="${COJOB_ROOT}/perlmutter-native-step-preflight"
  mkdir -p -- "${PREFLIGHT_ROOT}"
  if [[ "${JOB_NETWORK}" == "(null)" ]]; then
    PREFLIGHT_ERROR="allocation_missing_job_vni"
  else
    PREFLIGHT_ERROR="allocation_network_not_job_vni"
  fi
  jq -n \
    --arg schema "tempo-perlmutter-native-step-preflight-v1" \
    --arg status "failed" \
    --arg job_id "${SLURM_JOB_ID}" \
    --arg nodelist "${JOB_NODELIST}" \
    --arg network_mode "${TEMPO_GO_SRUN_NETWORK_MODE:-default}" \
    --arg allocation_network "${JOB_NETWORK}" \
    --arg error "${PREFLIGHT_ERROR}" \
    '{schema:$schema,status:$status,slurm_job_id:($job_id|tonumber),nodelist:$nodelist,
      nodes:4,ntasks:4,ntasks_per_node:1,gpus_per_task:4,cpus_per_task:128,
      gpu_bind:"none",cpu_bind:"cores",network:$network_mode,
      allocation_network:$allocation_network,error:$error}' \
    >"${PREFLIGHT_ROOT}/receipt.json"
  echo "Perlmutter native step preflight refused: Network=${JOB_NETWORK}; request salloc --network=job_vni" >&2
  exit 1
fi
# Keep one producer alive across every measured arm. C5 requests the stop file
# only after its final arm, so the co-job can publish a complete observer
# receipt after the measured interval. A finite ceiling remains as a fail-safe
# if the parent is lost.
export TEMPO_GO_CROSS_LAYER_BLOCKS="${TEMPO_GO_CROSS_LAYER_BLOCKS:-30000}"
export TEMPO_GO_CROSS_LAYER_TIMEOUT_SECONDS="${TEMPO_GO_CROSS_LAYER_TIMEOUT_SECONDS:-7200}"
export TEMPO_GO_CROSS_LAYER_TIME_LIMIT="${TEMPO_GO_CROSS_LAYER_TIME_LIMIT:-02:00:00}"
export TEMPO_GO_CROSS_LAYER_READY_TIMEOUT_SECONDS="${TEMPO_GO_CROSS_LAYER_READY_TIMEOUT_SECONDS:-300}"
[[ "${TEMPO_GO_CROSS_LAYER_READY_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]
(( TEMPO_GO_CROSS_LAYER_READY_TIMEOUT_SECONDS >= 60 &&
    TEMPO_GO_CROSS_LAYER_READY_TIMEOUT_SECONDS <= 1800 ))
export TEMPO_GO_CROSS_LAYER_START_DELAY_S="${TEMPO_GO_CROSS_LAYER_START_DELAY_S:-60}"
export TEMPO_GO_CROSS_LAYER_MEM_PER_NODE="${TEMPO_GO_CROSS_LAYER_MEM_PER_NODE:-32G}"
[[ "${TEMPO_GO_CROSS_LAYER_START_DELAY_S}" =~ ^[0-9]+([.][0-9]+)?$ ]]
[[ "${TEMPO_GO_CROSS_LAYER_MEM_PER_NODE}" =~ ^[0-9]+[KMG]$ ]]
# The sustained-moderate profile preserves concurrent official LMCache/NIXL
# and NCCL activity but avoids making every measured P/D request wait behind a
# continuously saturated 256-MiB/rank burst.  The hot profile remains
# reproducible by overriding these values explicitly for a separate arm.
export TEMPO_GO_CROSS_LAYER_REQUESTS="${TEMPO_GO_CROSS_LAYER_REQUESTS:-2}"
export TEMPO_GO_CROSS_LAYER_KV_MIB="${TEMPO_GO_CROSS_LAYER_KV_MIB:-4}"
export TEMPO_GO_CROSS_LAYER_TOKEN_ITERS="${TEMPO_GO_CROSS_LAYER_TOKEN_ITERS:-8}"
export TEMPO_GO_CROSS_LAYER_FOREGROUND_MIB="${TEMPO_GO_CROSS_LAYER_FOREGROUND_MIB:-1}"
# The previous 0.10 s cadence completed the 10,000-block observer in about
# 31.5 minutes, while the seven-arm native campaign took about 38 minutes.
# Keep the same official LMCache/NIXL + NCCL workload active across every arm
# by default; the measured C5 window must not silently outlive its co-job.
export TEMPO_GO_CROSS_LAYER_BLOCK_DELAY_S="${TEMPO_GO_CROSS_LAYER_BLOCK_DELAY_S:-0.25}"

# Perlmutter can accept the allocation shape while a particular allocated
# node/pair still fails Slingshot VNI setup at job-step launch.  Do this native
# four-node smoke step before starting NCCL/LMCache so that such an allocation
# produces an explicit preflight receipt instead of a misleading co-job or
# NCCL connection failure.  The flags mirror the production vLLM step and the
# NERSC guidance: explicit GPU/CPU resources, GPU visibility, and core binding.
PREFLIGHT_ROOT="${COJOB_ROOT}/perlmutter-native-step-preflight"
mkdir -p -- "${PREFLIGHT_ROOT}"
PREFLIGHT_STDERR="${PREFLIGHT_ROOT}/srun.stderr.log"
PREFLIGHT_RECEIPT="${PREFLIGHT_ROOT}/receipt.json"
PREFLIGHT_COMMAND="${PREFLIGHT_ROOT}/command.txt"
printf '%q ' \
  /usr/bin/srun "${TEMPO_GO_SRUN_JOB_ARGS[@]}" --overlap --exact \
  --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
  --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=5 \
  "${TEMPO_GO_SRUN_NETWORK_ARGS[@]}" \
  --time=00:02:00 --export=ALL \
  --output="${PREFLIGHT_ROOT}/rank-%t.stdout.log" \
  --error="${PREFLIGHT_ROOT}/rank-%t.stderr.log" \
  bash -c 'printf "host=%s localid=%s nodeid=%s\\n" "$(hostname)" "${SLURM_LOCALID}" "${SLURM_NODEID}"; nvidia-smi -L >/dev/null' \
  >"${PREFLIGHT_COMMAND}"

PREFLIGHT_RC=0
/usr/bin/srun "${TEMPO_GO_SRUN_JOB_ARGS[@]}" --overlap --exact \
  --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
  --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=5 \
  "${TEMPO_GO_SRUN_NETWORK_ARGS[@]}" \
  --time=00:02:00 --export=ALL \
  --output="${PREFLIGHT_ROOT}/rank-%t.stdout.log" \
  --error="${PREFLIGHT_ROOT}/rank-%t.stderr.log" \
  bash -c 'printf "host=%s localid=%s nodeid=%s\\n" "$(hostname)" "${SLURM_LOCALID}" "${SLURM_NODEID}"; nvidia-smi -L >/dev/null' \
  2>"${PREFLIGHT_STDERR}" || PREFLIGHT_RC=$?

if [[ "${PREFLIGHT_RC}" -eq 0 ]]; then
  jq -n \
    --arg schema "tempo-perlmutter-native-step-preflight-v1" \
    --arg status "passed" \
    --arg job_id "${SLURM_JOB_ID}" \
    --arg nodelist "${JOB_NODELIST}" \
    --arg network_mode "${TEMPO_GO_SRUN_NETWORK_MODE:-default}" \
    --arg allocation_network "${JOB_NETWORK}" \
    --arg allocated_cpus_per_task "${JOB_CPUS_PER_TASK}" \
    --arg nccl_timeout_seconds "${TEMPO_GO_NCCL_TIMEOUT_SECONDS}" \
    --arg command_file "${PREFLIGHT_COMMAND}" \
    --arg stderr_file "${PREFLIGHT_STDERR}" \
    '{schema:$schema,status:$status,slurm_job_id:($job_id|tonumber),nodelist:$nodelist,
      nodes:4,ntasks:4,ntasks_per_node:1,gpus_per_task:4,cpus_per_task:128,
      gpu_bind:"none",cpu_bind:"cores",network:$network_mode,
      allocation_network:$allocation_network,
      allocated_cpus_per_task:($allocated_cpus_per_task|tonumber),
      nccl_collective_timeout_seconds:($nccl_timeout_seconds|tonumber),
      command_file:$command_file,stderr_file:$stderr_file}' \
    >"${PREFLIGHT_RECEIPT}"
else
  jq -n \
    --arg schema "tempo-perlmutter-native-step-preflight-v1" \
    --arg status "failed" \
    --arg job_id "${SLURM_JOB_ID}" \
    --arg nodelist "${JOB_NODELIST}" \
    --arg network_mode "${TEMPO_GO_SRUN_NETWORK_MODE:-default}" \
    --arg allocation_network "${JOB_NETWORK}" \
    --arg allocated_cpus_per_task "${JOB_CPUS_PER_TASK}" \
    --arg command_file "${PREFLIGHT_COMMAND}" \
    --arg stderr_file "${PREFLIGHT_STDERR}" \
    --argjson exit_code "${PREFLIGHT_RC}" \
    '{schema:$schema,status:$status,slurm_job_id:($job_id|tonumber),nodelist:$nodelist,
      nodes:4,ntasks:4,ntasks_per_node:1,gpus_per_task:4,cpus_per_task:128,
      gpu_bind:"none",cpu_bind:"cores",network:$network_mode,
      allocation_network:$allocation_network,
      allocated_cpus_per_task:($allocated_cpus_per_task|tonumber),
      command_file:$command_file,stderr_file:$stderr_file,exit_code:$exit_code}' \
    >"${PREFLIGHT_RECEIPT}"
  echo "Perlmutter native step preflight failed; refusing to start C5: ${PREFLIGHT_RECEIPT}" >&2
  exit "${PREFLIGHT_RC}"
fi

# P1 capability receipt: this probe only reads bounded local Cassini sysfs,
# nvidia-smi, and PyTorch/NCCL availability.  It does not create a communicator
# or exchange application traffic, so NERSC's documented --network=no_vni mode
# is the correct step boundary and does not consume a Slingshot network slot.
# Run one probe per allocated node and merge the four immutable node receipts;
# do not collapse missing optional counters into zero.
CAPABILITY_ROOT="${PREFLIGHT_ROOT}/cross-layer-capability"
CAPABILITY_SOURCE="${TEMPO_GO_SOURCE_SNAPSHOT:-${REPO_ROOT}}"
CAPABILITY_PROBE="${CAPABILITY_SOURCE}/eval/sota_4node/probe_tempo_go_cross_layer_capability.py"
CAPABILITY_PYTHON="${REPO_ROOT}/.vllm_venv/bin/python"
CAPABILITY_WAIT_SECONDS="${TEMPO_GO_CROSS_LAYER_CAPABILITY_WAIT_SECONDS:-60}"
[[ "${CAPABILITY_WAIT_SECONDS}" =~ ^[0-9]+$ ]]
(( CAPABILITY_WAIT_SECONDS >= 10 && CAPABILITY_WAIT_SECONDS <= 120 ))
[[ -f "${CAPABILITY_PROBE}" && -x "${CAPABILITY_PYTHON}" ]]
mkdir -p -- "${CAPABILITY_ROOT}"
CAPABILITY_COMMAND="${CAPABILITY_ROOT}/command.txt"
CAPABILITY_STDERR="${CAPABILITY_ROOT}/srun.stderr.log"
printf '%q ' \
  /usr/bin/srun "${TEMPO_GO_SRUN_JOB_ARGS[@]}" --overlap --exact \
  --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
  --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait="${CAPABILITY_WAIT_SECONDS}" \
  --network=no_vni --time=00:02:00 --export=ALL \
  bash -c 'set -e; root="$1"; source_root="$2"; python="$3"; cd -- "$source_root"; TEMPO_PYTHON="$python" "$python" -m eval.sota_4node.probe_tempo_go_cross_layer_capability --output "$root/capability-${SLURM_PROCID}.json"' \
  bash "${CAPABILITY_ROOT}" "${CAPABILITY_SOURCE}" "${CAPABILITY_PYTHON}" \
  >"${CAPABILITY_COMMAND}"
CAPABILITY_RC=0
/usr/bin/srun "${TEMPO_GO_SRUN_JOB_ARGS[@]}" --overlap --exact \
  --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
  --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait="${CAPABILITY_WAIT_SECONDS}" \
  --network=no_vni --time=00:02:00 --export=ALL \
  bash -c 'set -e; root="$1"; source_root="$2"; python="$3"; cd -- "$source_root"; TEMPO_PYTHON="$python" "$python" -m eval.sota_4node.probe_tempo_go_cross_layer_capability --output "$root/capability-${SLURM_PROCID}.json"' \
  bash "${CAPABILITY_ROOT}" "${CAPABILITY_SOURCE}" "${CAPABILITY_PYTHON}" \
  2>"${CAPABILITY_STDERR}" || CAPABILITY_RC=$?
if [[ "${CAPABILITY_RC}" -ne 0 ]]; then
  jq -n \
    --arg schema "tempo-cross-layer-capability-receipt-v1" \
    --arg status "failed" \
    --arg job_id "${SLURM_JOB_ID}" \
    --arg nodelist "${JOB_NODELIST}" \
    --arg command_file "${CAPABILITY_COMMAND}" \
    --arg stderr_file "${CAPABILITY_STDERR}" \
    --argjson exit_code "${CAPABILITY_RC}" \
    '{schema:$schema,status:$status,slurm_job_id:($job_id|tonumber),nodelist:$nodelist,
      node_count:4,command_file:$command_file,stderr_file:$stderr_file,exit_code:$exit_code}' \
    >"${CAPABILITY_ROOT}/receipt.json"
  echo "Cross-layer capability probe failed; refusing to start C5: ${CAPABILITY_ROOT}/receipt.json" >&2
  exit "${CAPABILITY_RC}"
fi
CAPABILITY_ROOT_ENV="${CAPABILITY_ROOT}" \
CAPABILITY_JOB_ID_ENV="${SLURM_JOB_ID}" \
CAPABILITY_NODELIST_ENV="${JOB_NODELIST}" \
CAPABILITY_COMMAND_ENV="${CAPABILITY_COMMAND}" \
CAPABILITY_STDERR_ENV="${CAPABILITY_STDERR}" \
  "${REPO_ROOT}/.vllm_venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["CAPABILITY_ROOT_ENV"])
paths = sorted(root.glob("capability-*.json"))
if len(paths) != 4:
    raise SystemExit(f"expected 4 node capability receipts, found {len(paths)}")
nodes = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
node_names = [str(item.get("node")) for item in nodes]
if len(set(node_names)) != 4 or any(not name or name == "None" for name in node_names):
    raise SystemExit("capability node identities are not unique")
payload = {
    "schema": "tempo-cross-layer-capability-receipt-v1",
    "status": "passed",
    "native_only": True,
    "slurm_job_id": int(os.environ["CAPABILITY_JOB_ID_ENV"]),
    "nodelist": os.environ["CAPABILITY_NODELIST_ENV"],
    "node_count": len(nodes),
    "command_file": os.environ["CAPABILITY_COMMAND_ENV"],
    "stderr_file": os.environ["CAPABILITY_STDERR_ENV"],
    "nodes": nodes,
}
(root / "receipt.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

COJOB_LOG="${COJOB_ROOT}/cojob.stdout.log"
COJOB_ERR="${COJOB_ROOT}/cojob.stderr.log"
"${COJOB_LAUNCHER}" \
  >"${COJOB_LOG}" 2>"${COJOB_ERR}" &
COJOB_PID=$!

write_cojob_failure_receipt() {
  local failure_kind="${1:-cojob_ended_before_c5_end}"
  [[ ! -s "${COJOB_ROOT}/campaign_failure.json" ]] || return 0
  COJOB_ROOT_ENV="${COJOB_ROOT}" \
  COJOB_PID_ENV="${COJOB_PID:-}" \
  C5_RESULT_ROOT_ENV="${RESULT_ROOT}" \
  FAILURE_KIND_ENV="${failure_kind}" \
  "${REPO_ROOT}/.vllm_venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["COJOB_ROOT_ENV"]).resolve()
observer = root / "nccl_observer.json"
state = None
sequence = None
if observer.is_file():
    try:
        value = json.loads(observer.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            state = value.get("producer_state")
            sequence = value.get("sequence")
    except (OSError, ValueError):
        pass
payload = {
    "schema": "tempo-go-cross-layer-campaign-failure-v1",
    "failure": os.environ["FAILURE_KIND_ENV"],
    "cojob_pid": os.environ.get("COJOB_PID_ENV"),
    "cojob_root": str(root),
    "observer": str(observer),
    "observer_producer_state": state,
    "observer_sequence": sequence,
    "c5_result_root": os.environ["C5_RESULT_ROOT_ENV"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "native_only": True,
}

(root / "campaign_failure.json").write_text(
    json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
}

observer_is_stale_during_c5() {
  local observer_path="${COJOB_ROOT}/nccl_observer.json"
  [[ -s "${observer_path}" ]] || return 1
  local sampled_ns now_ns max_age_ns startup_grace_ns
  sampled_ns="$(jq -er '.sampled_unix_ns // 0' "${observer_path}" 2>/dev/null)" || return 1
  [[ "${sampled_ns}" =~ ^[0-9]+$ ]] || return 1
  now_ns="$(date +%s%N)"
  max_age_ns=$((10#${TEMPO_GO_NCCL_OBSERVER_MAX_AGE_MS} * 1000000))
  startup_grace_ns=$((10#${TEMPO_GO_NCCL_OBSERVER_STARTUP_GRACE_MS} * 1000000))
  (( now_ns >= sampled_ns && now_ns - sampled_ns > max_age_ns &&
     now_ns >= C5_START_UNIX_NS + startup_grace_ns ))
}

# Do not let the C5 runner begin its measured lifecycle until the co-job has
# completed its NIXL peer-handler handshake.  The first observer snapshot is
# published only after the producer's first synchronized block, and the
# producer may intentionally hold after handler readiness while C5 starts.
# Waiting for the observer here therefore deadlocks the campaign when a
# nonzero start delay is used.  The observer freshness gate below remains the
# binding condition once C5 is running.
COJOB_READY=0
for _ in $(seq 1 "${TEMPO_GO_CROSS_LAYER_READY_TIMEOUT_SECONDS}"); do
  if [[ -s "${COJOB_ROOT}/nixl-ready" ]]; then
    COJOB_READY=1
    break
  fi
  if ! kill -0 "${COJOB_PID}" 2>/dev/null; then
    wait "${COJOB_PID}" 2>/dev/null || true
    write_cojob_failure_receipt
    exit 1
  fi
  sleep 1
done
if [[ "${COJOB_READY}" -ne 1 ]]; then
  write_cojob_failure_receipt cojob_readiness_timeout
  echo "TEMPO cross-layer campaign stopped: co-job readiness timeout" >&2
  exit 1
fi
C5_START_UNIX_NS="$(date +%s%N)"
[[ "${TEMPO_GO_NCCL_OBSERVER_STARTUP_GRACE_MS}" =~ ^[0-9]+$ ]]
(( TEMPO_GO_NCCL_OBSERVER_STARTUP_GRACE_MS >= 60000 &&
    TEMPO_GO_NCCL_OBSERVER_STARTUP_GRACE_MS <= 600000 ))

terminate_process_tree() {
  local root="$1"
  local signal="$2"
  local child
  while read -r child; do
    [[ "${child}" =~ ^[0-9]+$ ]] || continue
    terminate_process_tree "${child}" "${signal}"
  done < <(pgrep -P "${root}" 2>/dev/null || true)
  kill -"${signal}" "${root}" 2>/dev/null || true
}

stop_c5() {
  if [[ -n "${C5_PID:-}" ]] && kill -0 "${C5_PID}" 2>/dev/null; then
    terminate_process_tree "${C5_PID}" TERM
    for _ in $(seq 1 25); do
      [[ ! -e "/proc/${C5_PID}" ]] && break
      sleep 0.2
    done
    if kill -0 "${C5_PID}" 2>/dev/null; then
      terminate_process_tree "${C5_PID}" KILL
    fi
  fi
  if [[ -n "${C5_PID:-}" ]]; then
    wait "${C5_PID}" 2>/dev/null || true
  fi
}

stop_cojob() {

  : > "${COJOB_STOP_FILE}" 2>/dev/null || true
  if [[ -n "${COJOB_PID:-}" ]] && kill -0 "${COJOB_PID}" 2>/dev/null; then
    # The launcher is a shell which owns timeout/srun grandchildren.  Killing
    # only the shell leaves the nested Slurm step behind and makes the EXIT
    # trap wait forever after a C5 arm failure or Ctrl-C.
    terminate_process_tree "${COJOB_PID}" TERM
  fi
  for _ in $(seq 1 25); do
    [[ -z "${COJOB_PID:-}" ]] || ! kill -0 "${COJOB_PID}" 2>/dev/null && break
    sleep 0.2
  done
  if [[ -n "${COJOB_PID:-}" ]] && kill -0 "${COJOB_PID}" 2>/dev/null; then
    terminate_process_tree "${COJOB_PID}" KILL
  fi
  if [[ -n "${COJOB_PID:-}" ]]; then
    wait "${COJOB_PID}" 2>/dev/null || true
  fi
}
trap 'stop_c5; stop_cojob; release_campaign_lock' EXIT

# The canonical five-arm runner remains responsible for its frozen contract,
# vLLM lifecycle, request ledger, and arm failure receipts.  Keep it behind a
# foreground lifecycle loop so a dead observer cannot silently turn the later
# arms into a different workload.  This is a campaign-level global gate, not
# a background watcher: the bounded parent process owns both lifecycles.
C5_LOG="${COJOB_ROOT}/c5-runner.stdout.log"
set +e
  "${TEMPO_GO_SOURCE_SNAPSHOT:-${REPO_ROOT}}/eval/sota_4node/run_tempo_go_c5_five_arm_in_allocation.sh" \
  "${WORKLOAD_INPUT}" "${RESULT_ROOT}" >"${C5_LOG}" 2>&1 &
C5_PID=$!
set -e
C5_RC=0
while kill -0 "${C5_PID}" 2>/dev/null; do
  if ! kill -0 "${COJOB_PID}" 2>/dev/null; then
    write_cojob_failure_receipt
    kill -TERM "${C5_PID}" 2>/dev/null || true
    wait "${C5_PID}" 2>/dev/null || true
    echo "TEMPO cross-layer campaign stopped: co-job ended before C5 completion" >&2
    exit 1
  fi
  if observer_is_stale_during_c5; then
    write_cojob_failure_receipt cojob_observer_stale_during_c5
    echo "TEMPO cross-layer campaign stopped: co-job observer became stale during C5" >&2
    exit 1
  fi
  sleep 1
done
wait "${C5_PID}" || C5_RC=$?
if [[ "${C5_RC}" -ne 0 ]]; then
  : > "${COJOB_STOP_FILE}"
  exit "${C5_RC}"
fi
C5_END_UNIX_NS="$(date +%s%N)"

# Let the producer finish its current synchronized block and publish a final
# complete observer snapshot. The binding gate below requires that snapshot
# to be sampled after C5_END_UNIX_NS.
: > "${COJOB_STOP_FILE}"

COJOB_RC=0
wait "${COJOB_PID}" || COJOB_RC=$?
COJOB_PID=""
[[ "${COJOB_RC}" -eq 0 ]]
[[ -s "${COJOB_ROOT}/result.json" ]]
[[ -s "${COJOB_ROOT}/nccl_observer.json" ]]

# Preserve a non-mutating binding receipt under the C5 root.  The raw observer
# remains in its own immutable co-job root and was the path consumed live by
# the routers.
COJOB_ROOT_ENV="${COJOB_ROOT}" \
RESULT_ROOT_ENV="${RESULT_ROOT}" \
C5_START_UNIX_NS_ENV="${C5_START_UNIX_NS}" \
C5_END_UNIX_NS_ENV="${C5_END_UNIX_NS}" \
"${REPO_ROOT}/.vllm_venv/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["COJOB_ROOT_ENV"]).resolve()
result = Path(os.environ["RESULT_ROOT_ENV"]).resolve()
observer_path = root / "nccl_observer.json"
def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
observer = json.loads(observer_path.read_text(encoding="utf-8"))
c5_end_unix_ns = int(os.environ["C5_END_UNIX_NS_ENV"])
observer_sampled_unix_ns = int(observer.get("sampled_unix_ns", 0))
covered_c5_end = observer_sampled_unix_ns >= c5_end_unix_ns
payload = {
    "schema": "tempo-go-c5-cross-layer-cojob-binding-v1",
    "same_allocation": True,
    "cojob_root": str(root),
    "cojob_result": str(root / "result.json"),
    "cojob_result_sha256": digest(root / "result.json"),
    "observer": str(observer_path),
    "observer_sha256": digest(observer_path),
    "epoch": os.environ.get("TEMPO_GO_CROSS_LAYER_EPOCH"),
    "communicator_id": os.environ.get("TEMPO_GO_NCCL_COMMUNICATOR_ID"),
    "blocks": int(os.environ["TEMPO_GO_CROSS_LAYER_BLOCKS"]),
    "requests": int(os.environ["TEMPO_GO_CROSS_LAYER_REQUESTS"]),
    "kv_mib": int(os.environ["TEMPO_GO_CROSS_LAYER_KV_MIB"]),
    "token_iters": int(os.environ["TEMPO_GO_CROSS_LAYER_TOKEN_ITERS"]),
    "foreground_mib": int(os.environ["TEMPO_GO_CROSS_LAYER_FOREGROUND_MIB"]),
    "block_delay_s": float(os.environ["TEMPO_GO_CROSS_LAYER_BLOCK_DELAY_S"]),
    "timeout_seconds": int(os.environ["TEMPO_GO_CROSS_LAYER_TIMEOUT_SECONDS"]),
    "time_limit": os.environ["TEMPO_GO_CROSS_LAYER_TIME_LIMIT"],
    "cojob_ready_before_c5": True,
    "c5_start_unix_ns": int(os.environ["C5_START_UNIX_NS_ENV"]),
    "c5_end_unix_ns": c5_end_unix_ns,
    "observer_sampled_unix_ns": observer_sampled_unix_ns,
    "cojob_covered_c5_end": covered_c5_end,
    "gpu_overlap_intentional": True,
    "privileged_nic_control": False,
    "native_only": True,
}
(result / "cross_layer_cojob_binding.json").write_text(
    json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
if not covered_c5_end:
    raise SystemExit(
        "co-job completed before the C5 measured campaign ended; "
        "performance claim is invalid")
PY

echo "TEMPO cross-layer same-allocation campaign complete: ${RESULT_ROOT}"
