#!/usr/bin/env bash
set -euo pipefail

# Candidate O is a separate source-bound discovery entry point.  It never
# submits/cancels Slurm jobs and must run inside one approved 4-node allocation.
[[ "${TEMPO_GO_C9_CAUSAL_BURST_APPROVED:-}" == YES ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONTRACT="${REPO_ROOT}/results/tempo_go_c9_candidate_o_route_liveness_v1/tempo_go_c9_candidate_o_route_liveness_population_contract.json"
CONTRACT_SHA256="9936a4a980d23250ce6604494d9fa545da189de8508b931d8ffb1952c35a8cc5"

[[ -f "${CONTRACT}" ]]
[[ "$(sha256sum "${CONTRACT}" | awk '{print $1}')" == "${CONTRACT_SHA256}" ]]
export TEMPO_GO_C9_CAUSAL_BURST_CONTRACT="${CONTRACT}"

exec bash "${SCRIPT_DIR}/run_tempo_go_c9_causal_burst_discovery_in_allocation.sh"
