#!/usr/bin/env bash
set -euo pipefail

# Candidate P never submits or cancels Slurm work. It is entered only from the
# explicit zero-GPU/no-VNI attach step of one approved allocation.
[[ "${TEMPO_GO_C9_CAUSAL_BURST_APPROVED:-}" == YES ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONTRACT="${REPO_ROOT}/results/tempo_go_c9_candidate_p_bounded_observer_v1/tempo_go_c9_candidate_p_bounded_observer_contract.json"
CONTRACT_SHA256="4f76efac34eb930da1ef61e4fe883e1d5fd557fb2d976008e285b000df3d1828"

[[ -f "${CONTRACT}" ]]
[[ "$(sha256sum "${CONTRACT}" | awk '{print $1}')" == "${CONTRACT_SHA256}" ]]
export TEMPO_GO_C9_CAUSAL_BURST_CONTRACT="${CONTRACT}"

exec bash "${SCRIPT_DIR}/run_tempo_go_c9_causal_burst_discovery_in_allocation.sh"
