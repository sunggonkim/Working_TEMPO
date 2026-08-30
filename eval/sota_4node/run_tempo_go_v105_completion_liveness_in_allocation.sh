#!/usr/bin/env bash
set -euo pipefail

# Run only from the foreground shell of a user-approved 4-node
# gpu_interactive allocation. This wrapper never submits/cancels a job and
# never launches a second allocation. The frozen cross-layer runner owns the
# capability probe, moderate co-job, seven arms, teardown, and receipts.
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKLOAD="${REPO_ROOT}/results/tempo_go_c5_cross_layer_short_slice_v4/workloads/validation.jsonl"
CONTRACT="${REPO_ROOT}/results/tempo_go_c5_source_snapshot_v105_completion_liveness/native_run_contract.json"
CONTRACT_SHA="6606e3da1b21074f4cb8fc6bb9b6663cfde6963c255b7174a62f1f4bf6c8d165"

[[ "$(id -u)" -ne 0 ]]
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]
[[ -n "${SLURM_JOB_NODELIST:-}" ]]
[[ -z "${UDI:-}" && -z "${CRAY_ROOTFS:-}" && -z "${SLURM_CONTAINER:-}" ]]
[[ -z "${SHIFTER_RUNTIME:-}" && -z "${SHIFTER_IMAGE:-}" ]]
[[ -s "${WORKLOAD}" && -s "${CONTRACT}" ]]
command -v scontrol >/dev/null 2>&1

JOB_INFO=$(scontrol show job -o "${SLURM_JOB_ID}")
[[ " ${JOB_INFO} " == *" JobState=RUNNING "* ]]
[[ " ${JOB_INFO} " == *" QOS=gpu_interactive "* ||
   " ${JOB_INFO} " == *" QOS=interactive "* ]]
[[ " ${JOB_INFO} " == *" NumNodes=4 "* ]]
[[ " ${JOB_INFO} " == *" NumCPUs=512 "* ]]
[[ " ${JOB_INFO} " == *" CPUs/Task=128 "* ]]
[[ "${JOB_INFO}" =~ (^|[[:space:],])gres/gpu=16($|[[:space:],]) ||
   "${JOB_INFO}" =~ (^|[[:space:],])gres/gpu:a100=16($|[[:space:],]) ]]
[[ " ${JOB_INFO} " == *" Network=job_vni "* ]]

RUN_ATTEMPT="${TEMPO_GO_V105_RUN_ATTEMPT:-}"
[[ -z "${RUN_ATTEMPT}" || "${RUN_ATTEMPT}" =~ ^attempt[2-9][0-9]*$ ]]
RESULT_SUFFIX="${RUN_ATTEMPT:+_${RUN_ATTEMPT}}"
RESULT_ROOT="${REPO_ROOT}/results/tempo_go_cross_layer_native_v105_completion_liveness_${SLURM_JOB_ID}${RESULT_SUFFIX}"
[[ ! -e "${RESULT_ROOT}" ]]

export TEMPO_GO_REPO_ROOT="${REPO_ROOT}"
export TEMPO_GO_C5_RUN_CONTRACT="${CONTRACT}"
export TEMPO_GO_C5_RUN_CONTRACT_SHA256="${CONTRACT_SHA}"
export TEMPO_GO_CROSS_LAYER_REQUESTS=2
export TEMPO_GO_CROSS_LAYER_KV_MIB=4
export TEMPO_GO_CROSS_LAYER_TOKEN_ITERS=8
export TEMPO_GO_CROSS_LAYER_FOREGROUND_MIB=1
export TEMPO_GO_CROSS_LAYER_BLOCK_DELAY_S=0.25
export TEMPO_GO_CROSS_LAYER_START_DELAY_S=60
export TEMPO_GO_CROSS_LAYER_MEM_PER_NODE=32G
export TEMPO_GO_CROSS_LAYER_TIMEOUT_SECONDS=7200
export TEMPO_GO_CROSS_LAYER_TIME_LIMIT=02:00:00
export TEMPO_GO_CROSS_LAYER_READY_TIMEOUT_SECONDS=600
export TEMPO_GO_CROSS_LAYER_CAPABILITY_WAIT_SECONDS=60
export TEMPO_GO_CROSS_LAYER_COJOB_ROOT="${REPO_ROOT}/results/tempo_go_cross_layer_cojob_${SLURM_JOB_ID}${RESULT_SUFFIX}"
export TEMPO_GO_NCCL_OBSERVER_MAX_AGE_MS=600000
export TEMPO_GO_NCCL_OBSERVER_STARTUP_GRACE_MS=600000
export TEMPO_GO_NCCL_DIAGNOSTICS=1
export TEMPO_GO_CXI_COUNTER_REPORT=2

exec "${REPO_ROOT}/eval/sota_4node/run_tempo_go_cross_layer_with_cojob_in_allocation.sh" \
  "${WORKLOAD}" "${RESULT_ROOT}"
