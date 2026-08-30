#!/usr/bin/env bash
set -euo pipefail

# Native 4-node same-allocation v107 discovery; never submits an allocation.
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKLOAD="${REPO_ROOT}/results/tempo_go_c5_cross_layer_short_slice_v4/workloads/validation.jsonl"
CONTRACT="${REPO_ROOT}/results/tempo_go_c5_source_snapshot_v107_cxi_credit_refill/native_run_contract.json"
CONTRACT_SHA="de01e9907226c699b2a8a09d6bd6ec6d6d02fe7d2d4d3bf1c48c1e8d9ce28602"

[[ "$(id -u)" -ne 0 ]]
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
[[ -s "${WORKLOAD}" && -s "${CONTRACT}" ]]
[[ -z "${UDI:-}" && -z "${CRAY_ROOTFS:-}" && -z "${SLURM_CONTAINER:-}" ]]
[[ -z "${SHIFTER_RUNTIME:-}" && -z "${SHIFTER_IMAGE:-}" ]]
RESULT_ROOT="${REPO_ROOT}/results/tempo_go_cxi_native_v107_credit_refill_${SLURM_JOB_ID}"
COJOB_ROOT="${REPO_ROOT}/results/tempo_go_cxi_background_v107_${SLURM_JOB_ID}"
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
