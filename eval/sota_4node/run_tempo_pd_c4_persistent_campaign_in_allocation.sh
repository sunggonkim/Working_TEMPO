#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/require_perlmutter_4node_4h_interactive.sh"
: "${TEMPO_PD_C4_PERSISTENT_APPROVED:?explicit persistent-campaign approval required}"
: "${TEMPO_PD_C4_ADAPTIVE_IMPLEMENTATION_SHA256:?adaptive implementation SHA-256 required}"
[[ "${TEMPO_PD_C4_PERSISTENT_APPROVED}" == YES ]]
[[ "${TEMPO_PD_C4_ADAPTIVE_IMPLEMENTATION_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ $# -eq 2 ]]

APPROVED=${TEMPO_PD_C4_PERSISTENT_APPROVED}
IMPLEMENTATION_SHA=${TEMPO_PD_C4_ADAPTIVE_IMPLEMENTATION_SHA256}
unset TEMPO_PD_C4_PERSISTENT_APPROVED
unset TEMPO_PD_C4_ADAPTIVE_IMPLEMENTATION_SHA256
[[ "${APPROVED}" == YES ]]

while IFS= read -r name; do
  case "${name}" in
    TEMPO_PD_C4_READINESS_S) ;;
    *)
      echo "persistent campaign refuses inherited experiment variable: ${name}" >&2
      exit 2
      ;;
  esac
done < <(compgen -e TEMPO_)

REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKLOAD=$(realpath -e -- "$1")
CAMPAIGN_ROOT=$(realpath -m -- "$2")
case "${WORKLOAD}" in "${REPO_ROOT}"/results/*) ;; *) exit 2 ;; esac
case "${CAMPAIGN_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ -s "${WORKLOAD}" && ! -e "${CAMPAIGN_ROOT}" ]]

IMPLEMENTATION_CONTRACT="${REPO_ROOT}/eval/sota_4node/tempo_pd_c4_adaptive_implementation_contract_v1.json"
[[ -s "${IMPLEMENTATION_CONTRACT}" ]]
[[ "$(sha256sum "${IMPLEMENTATION_CONTRACT}" | awk '{print $1}')" == \
  "${IMPLEMENTATION_SHA}" ]]
FIXED_IMPLEMENTATION_SHA=$(jq -er \
  '.fixed_c4_implementation_contract.sha256' "${IMPLEMENTATION_CONTRACT}")
[[ "${FIXED_IMPLEMENTATION_SHA}" =~ ^[0-9a-f]{64}$ ]]

module reset
module load pytorch/2.8.0
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
PYTHON="${REPO_ROOT}/.vllm_venv/bin/python"
mkdir -p -- "${CAMPAIGN_ROOT}/post_c4"

C4_RESULT_DIR="${CAMPAIGN_ROOT}/c4_fixed"
TEMPO_PD_C4_FIXED_APPROVED=YES \
TEMPO_PD_C4_FIXED_IMPLEMENTATION_SHA256="${FIXED_IMPLEMENTATION_SHA}" \
TEMPO_PD_C4_STEP_TIME=01:15:00 \
  bash "${SCRIPT_DIR}/run_tempo_pd_c4_fixed_phase_in_allocation.sh" \
  "${WORKLOAD}" "${C4_RESULT_DIR}"

C4_RESULT="${C4_RESULT_DIR}/result.json"
C4_RESULT_SHA=$(sha256sum "${C4_RESULT}" | awk '{print $1}')
POST_ROOT="${CAMPAIGN_ROOT}/post_c4"
ANALYSIS="${POST_ROOT}/c4_analysis.json"
"${PYTHON}" -m eval.sota_4node.analyze_tempo_pd_c4_fixed_phase \
  --result "${C4_RESULT}" \
  --expected-result-sha256 "${C4_RESULT_SHA}" \
  --output "${ANALYSIS}"
ANALYSIS_SHA=$(sha256sum "${ANALYSIS}" | awk '{print $1}')

MANIFEST="${POST_ROOT}/adaptive_manifest.json"
"${PYTHON}" -m eval.sota_4node.build_tempo_pd_c4_adaptive_screen_manifest \
  --analysis "${ANALYSIS}" \
  --expected-analysis-sha256 "${ANALYSIS_SHA}" \
  --output "${MANIFEST}"
MANIFEST_SHA=$(sha256sum "${MANIFEST}" | awk '{print $1}')

ELASTIC="${POST_ROOT}/calibrated_elastic_profile.json"
ENDPOINT="${POST_ROOT}/calibrated_endpoint_profile.json"
RECEIPT="${POST_ROOT}/calibrated_profile_receipt.json"
"${PYTHON}" -m eval.sota_4node.build_tempo_pd_c4_calibrated_profiles \
  --analysis "${ANALYSIS}" \
  --expected-analysis-sha256 "${ANALYSIS_SHA}" \
  --workload-manifest "${MANIFEST}" \
  --expected-workload-manifest-sha256 "${MANIFEST_SHA}" \
  --elastic-profile-id "tempo-c4-elastic-${SLURM_JOB_ID}" \
  --endpoint-profile-id "tempo-c4-endpoint-${SLURM_JOB_ID}" \
  --elastic-output "${ELASTIC}" \
  --endpoint-output "${ENDPOINT}" \
  --receipt-output "${RECEIPT}"
ELASTIC_SHA=$(sha256sum "${ELASTIC}" | awk '{print $1}')
ENDPOINT_SHA=$(sha256sum "${ENDPOINT}" | awk '{print $1}')
RECEIPT_SHA=$(sha256sum "${RECEIPT}" | awk '{print $1}')

REPLAY="${POST_ROOT}/offline_replay.json"
"${PYTHON}" -m eval.sota_4node.replay_tempo_pd_c4_calibrated_controller \
  --analysis "${ANALYSIS}" --analysis-sha256 "${ANALYSIS_SHA}" \
  --manifest "${MANIFEST}" --manifest-sha256 "${MANIFEST_SHA}" \
  --elastic "${ELASTIC}" --elastic-sha256 "${ELASTIC_SHA}" \
  --endpoint "${ENDPOINT}" --endpoint-sha256 "${ENDPOINT_SHA}" \
  --receipt "${RECEIPT}" --receipt-sha256 "${RECEIPT_SHA}" \
  --output "${REPLAY}"
REPLAY_SHA=$(sha256sum "${REPLAY}" | awk '{print $1}')
REPLAY_AUTHORIZED=$("${PYTHON}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["live_adaptive_screen_authorized"])' \
  "${REPLAY}")
if [[ "${REPLAY_AUTHORIZED}" != True ]]; then
  echo "Offline replay did not authorize the live adaptive screen: ${REPLAY}" >&2
  exit 3
fi

RUN_CONTRACT="${POST_ROOT}/adaptive_run_contract.json"
"${PYTHON}" -m eval.sota_4node.build_tempo_pd_c4_adaptive_run_contract \
  --repo-root "${REPO_ROOT}" \
  --analysis "${ANALYSIS}" --analysis-sha256 "${ANALYSIS_SHA}" \
  --manifest "${MANIFEST}" --manifest-sha256 "${MANIFEST_SHA}" \
  --elastic "${ELASTIC}" --elastic-sha256 "${ELASTIC_SHA}" \
  --endpoint "${ENDPOINT}" --endpoint-sha256 "${ENDPOINT_SHA}" \
  --receipt "${RECEIPT}" --receipt-sha256 "${RECEIPT_SHA}" \
  --replay "${REPLAY}" --replay-sha256 "${REPLAY_SHA}" \
  --implementation "${IMPLEMENTATION_CONTRACT}" \
  --implementation-sha256 "${IMPLEMENTATION_SHA}" \
  --output "${RUN_CONTRACT}"
RUN_CONTRACT_SHA=$(sha256sum "${RUN_CONTRACT}" | awk '{print $1}')

ADAPTIVE_RESULT_DIR="${CAMPAIGN_ROOT}/adaptive_screen"
TEMPO_PD_C4_ADAPTIVE_APPROVED=YES \
TEMPO_PD_C4_ADAPTIVE_RUN_CONTRACT_SHA256="${RUN_CONTRACT_SHA}" \
TEMPO_PD_C4_STEP_TIME=01:15:00 \
  bash "${SCRIPT_DIR}/run_tempo_pd_c4_adaptive_screen_in_allocation.sh" \
  "${RUN_CONTRACT}" "${ADAPTIVE_RESULT_DIR}"

ADAPTIVE_RESULT="${ADAPTIVE_RESULT_DIR}/result.json"
ADAPTIVE_RESULT_SHA=$(sha256sum "${ADAPTIVE_RESULT}" | awk '{print $1}')
ADAPTIVE_ANALYSIS="${CAMPAIGN_ROOT}/adaptive_screen_analysis.json"
"${PYTHON}" -m eval.sota_4node.analyze_tempo_pd_c4_adaptive_screen \
  --result "${ADAPTIVE_RESULT}" \
  --expected-result-sha256 "${ADAPTIVE_RESULT_SHA}" \
  --output "${ADAPTIVE_ANALYSIS}"

echo "Persistent 4-node C4/adaptive campaign complete: ${ADAPTIVE_ANALYSIS}"
