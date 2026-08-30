#!/usr/bin/env bash
set -euo pipefail

# Candidate M is a separate source-bound discovery entry point.  It must not
# silently fall back to the v13 population contract.
[[ "${TEMPO_GO_C9_CAUSAL_BURST_APPROVED:-}" == YES ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONTRACT="${REPO_ROOT}/results/tempo_go_c9_candidate_m_pressure_spill_v1/tempo_go_c9_candidate_m_pressure_spill_population_contract.json"
CONTRACT_SHA256="ef876c73dc5211e0952cd9f8335a988de34cdcb19d59e53a21dc3fe39626721c"

[[ -f "${CONTRACT}" ]]
[[ "$(sha256sum "${CONTRACT}" | awk '{print $1}')" == "${CONTRACT_SHA256}" ]]
export TEMPO_GO_C9_CAUSAL_BURST_CONTRACT="${CONTRACT}"

exec bash "${SCRIPT_DIR}/run_tempo_go_c9_causal_burst_discovery_in_allocation.sh"
