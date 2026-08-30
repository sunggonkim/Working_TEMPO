#!/usr/bin/env bash
set -euo pipefail

# Invoke only from the foreground shell of the user-approved four-node,
# four-hour gpu_interactive allocation.  This wrapper never submits a job.
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKLOAD="${REPO_ROOT}/results/tempo_go_c5_cross_layer_short_slice_v4/workloads/validation.jsonl"
CONTRACT="${REPO_ROOT}/results/tempo_go_c5_source_snapshot_v106_cxi_completion_credit/native_run_contract.json"
CONTRACT_SHA="082ad0d53948fca0c17ad8051ab24add72c0ee0594dcc72e3e044d855809b133"

[[ "$(id -u)" -ne 0 ]]
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
[[ -s "${WORKLOAD}" && -s "${CONTRACT}" ]]
[[ -z "${UDI:-}" && -z "${CRAY_ROOTFS:-}" && -z "${SLURM_CONTAINER:-}" ]]
[[ -z "${SHIFTER_RUNTIME:-}" && -z "${SHIFTER_IMAGE:-}" ]]

RUN_ATTEMPT="${TEMPO_GO_V106_RUN_ATTEMPT:-}"
[[ -z "${RUN_ATTEMPT}" || "${RUN_ATTEMPT}" =~ ^attempt[2-9][0-9]*$ ]]
SUFFIX="${RUN_ATTEMPT:+_${RUN_ATTEMPT}}"
RESULT_ROOT="${REPO_ROOT}/results/tempo_go_cxi_native_v106_completion_credit_${SLURM_JOB_ID}${SUFFIX}"
COJOB_ROOT="${REPO_ROOT}/results/tempo_go_cxi_background_v106_${SLURM_JOB_ID}${SUFFIX}"
[[ ! -e "${RESULT_ROOT}" && ! -e "${COJOB_ROOT}" ]]

export TEMPO_GO_REPO_ROOT="${REPO_ROOT}"
export TEMPO_GO_C5_RUN_CONTRACT="${CONTRACT}"
export TEMPO_GO_C5_RUN_CONTRACT_SHA256="${CONTRACT_SHA}"
export TEMPO_GO_CXI_COJOB_ROOT="${COJOB_ROOT}"
export TEMPO_GO_SOURCE_SNAPSHOT
TEMPO_GO_SOURCE_SNAPSHOT=$(jq -er '.source_snapshot.root' "${CONTRACT}")
RUNNER=$(jq -er '.launcher.runner.path' "${CONTRACT}")
[[ -x "${RUNNER}" ]]

exec "${RUNNER}" "${WORKLOAD}" "${RESULT_ROOT}"
