#!/usr/bin/env bash
set -euo pipefail

# Run only from the foreground shell of a user-approved 4-node gpu_interactive
# allocation. This wrapper never submits/cancels a job and never launches a
# second allocation. The actual C5/co-job steps are owned by the frozen
# cross-layer runner.
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKLOAD="${REPO_ROOT}/results/tempo_go_c5_cross_layer_short_slice_v4/workloads/validation.jsonl"
CONTRACT="${REPO_ROOT}/results/tempo_go_c5_source_snapshot_v104_hierarchy_reducer_cache/native_run_contract.json"
CONTRACT_SHA="4ddd901df87dc7107fac1ee4d7c76f8734d1651eb4502b8a12563d708b64ff6d"

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
[[ " ${JOB_INFO} " == *" gres/gpu=16 "* ||
   " ${JOB_INFO} " == *" gres/gpu:a100=16 "* ]]
[[ " ${JOB_INFO} " == *" Network=job_vni "* ]]

RESULT_ROOT="${REPO_ROOT}/results/tempo_go_cross_layer_native_v104_moderate_${SLURM_JOB_ID}"
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
export TEMPO_GO_NCCL_OBSERVER_MAX_AGE_MS=600000
export TEMPO_GO_NCCL_OBSERVER_STARTUP_GRACE_MS=600000
export TEMPO_GO_NCCL_DIAGNOSTICS=1
export TEMPO_GO_CXI_COUNTER_REPORT=2

exec "${REPO_ROOT}/eval/sota_4node/run_tempo_go_cross_layer_with_cojob_in_allocation.sh" \
  "${WORKLOAD}" "${RESULT_ROOT}"
