#!/usr/bin/env bash
set -euo pipefail

# This entry point is deliberately v13-specific.  Do not make the generic C9
# runner's historical default silently select a different population.
[[ "${TEMPO_GO_C9_CAUSAL_BURST_APPROVED:-}" == YES ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONTRACT="${REPO_ROOT}/results/tempo_go_c9_dual_route_business_lane_v13/tempo_go_c9_dual_route_business_lane_population_contract.json"
CONTRACT_SHA256="989a09e0f005967ec5f1ff1ec17b9244b5dee0b5e39f04d0b479a8e5c1de8a69"

[[ -f "${CONTRACT}" ]]
[[ "$(sha256sum "${CONTRACT}" | awk '{print $1}')" == "${CONTRACT_SHA256}" ]]
export TEMPO_GO_C9_CAUSAL_BURST_CONTRACT="${CONTRACT}"

exec bash "${SCRIPT_DIR}/run_tempo_go_c9_causal_burst_discovery_in_allocation.sh"
