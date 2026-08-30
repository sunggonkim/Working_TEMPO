#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/require_perlmutter_4node_4h_interactive.sh"
: "${TEMPO_PD_C4_SEMANTIC_CAMPAIGN_APPROVED:?explicit semantic campaign approval required}"
: "${TEMPO_PD_C4_SEMANTIC_RUN_CONTRACT_SHA256:?exploratory semantic contract SHA required}"
: "${TEMPO_PD_C4_SEMANTIC_INTEGRATION_IMPLEMENTATION_SHA256:?semantic integration implementation SHA required}"
[[ "${TEMPO_PD_C4_SEMANTIC_CAMPAIGN_APPROVED}" == YES ]]
[[ "${TEMPO_PD_C4_SEMANTIC_RUN_CONTRACT_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${TEMPO_PD_C4_SEMANTIC_INTEGRATION_IMPLEMENTATION_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ $# -eq 3 ]]

CAMPAIGN_APPROVED=${TEMPO_PD_C4_SEMANTIC_CAMPAIGN_APPROVED}
EXPLORATORY_SHA=${TEMPO_PD_C4_SEMANTIC_RUN_CONTRACT_SHA256}
IMPLEMENTATION_SHA=${TEMPO_PD_C4_SEMANTIC_INTEGRATION_IMPLEMENTATION_SHA256}
unset TEMPO_PD_C4_SEMANTIC_CAMPAIGN_APPROVED
unset TEMPO_PD_C4_SEMANTIC_RUN_CONTRACT_SHA256
unset TEMPO_PD_C4_SEMANTIC_INTEGRATION_IMPLEMENTATION_SHA256
[[ "${CAMPAIGN_APPROVED}" == YES ]]

while IFS= read -r name; do
  case "${name}" in
    TEMPO_PD_C4_READINESS_S) ;;
    *)
      echo "semantic campaign refuses inherited experiment variable: ${name}" >&2
      exit 2
      ;;
  esac
done < <(compgen -e TEMPO_)

REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
EXPLORATORY_CONTRACT=$(realpath -e -- "$1")
WORKLOAD=$(realpath -e -- "$2")
CAMPAIGN_ROOT=$(realpath -m -- "$3")
case "${EXPLORATORY_CONTRACT}" in "${REPO_ROOT}"/results/*) ;; *) exit 2 ;; esac
case "${WORKLOAD}" in "${REPO_ROOT}"/results/*) ;; *) exit 2 ;; esac
case "${CAMPAIGN_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ -s "${EXPLORATORY_CONTRACT}" && -s "${WORKLOAD}" ]]
[[ ! -e "${CAMPAIGN_ROOT}" ]]
[[ "$(sha256sum "${EXPLORATORY_CONTRACT}" | awk '{print $1}')" == \
  "${EXPLORATORY_SHA}" ]]

IMPLEMENTATION_CONTRACT="${REPO_ROOT}/eval/sota_4node/tempo_pd_c4_semantic_integration_implementation_contract_v1.json"
[[ -s "${IMPLEMENTATION_CONTRACT}" ]]
[[ "$(sha256sum "${IMPLEMENTATION_CONTRACT}" | awk '{print $1}')" == \
  "${IMPLEMENTATION_SHA}" ]]
ADAPTIVE_IMPLEMENTATION=$(jq -er \
  '.adaptive_implementation_contract.path' "${IMPLEMENTATION_CONTRACT}")
case "${ADAPTIVE_IMPLEMENTATION}" in
  /*) ADAPTIVE_IMPLEMENTATION=$(realpath -e -- "${ADAPTIVE_IMPLEMENTATION}") ;;
  *) ADAPTIVE_IMPLEMENTATION=$(realpath -e -- "${REPO_ROOT}/${ADAPTIVE_IMPLEMENTATION}") ;;
esac
ADAPTIVE_IMPLEMENTATION_SHA=$(jq -er \
  '.adaptive_implementation_contract.sha256' "${IMPLEMENTATION_CONTRACT}")
[[ "$(sha256sum "${ADAPTIVE_IMPLEMENTATION}" | awk '{print $1}')" == \
  "${ADAPTIVE_IMPLEMENTATION_SHA}" ]]
FIXED_IMPLEMENTATION_SHA=$(jq -er \
  '.fixed_c4_implementation_contract.sha256' "${ADAPTIVE_IMPLEMENTATION}")
[[ "${FIXED_IMPLEMENTATION_SHA}" =~ ^[0-9a-f]{64}$ ]]

module reset
module load pytorch/2.8.0
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
PYTHON="${REPO_ROOT}/.vllm_venv/bin/python"
mkdir -p -- "${CAMPAIGN_ROOT}/post_c4"

SEMANTIC_RESULT_DIR="${CAMPAIGN_ROOT}/semantic_exploratory"
TEMPO_PD_C4_SEMANTIC_APPROVED=YES \
TEMPO_PD_C4_SEMANTIC_RUN_CONTRACT_SHA256="${EXPLORATORY_SHA}" \
TEMPO_PD_C4_STEP_TIME=01:00:00 \
  bash "${SCRIPT_DIR}/run_tempo_pd_c4_semantic_epoch_screen_in_allocation.sh" \
  "${EXPLORATORY_CONTRACT}" "${SEMANTIC_RESULT_DIR}"
SEMANTIC_ANALYSIS="${SEMANTIC_RESULT_DIR}/semantic_epoch_analysis.json"
SEMANTIC_ANALYSIS_SHA=$(sha256sum "${SEMANTIC_ANALYSIS}" | awk '{print $1}')
SEMANTIC_AUTHORIZED=$(jq -er \
  '.authorizes_candidate_for_final_c4_integration' "${SEMANTIC_ANALYSIS}")
if [[ "${SEMANTIC_AUTHORIZED}" != true ]]; then
  echo "Exploratory semantic screen did not authorize Candidate B: ${SEMANTIC_ANALYSIS}" >&2
  exit 3
fi

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
ENDPOINT_V1="${POST_ROOT}/calibrated_endpoint_profile_v1.json"
RECEIPT="${POST_ROOT}/calibrated_profile_receipt.json"
"${PYTHON}" -m eval.sota_4node.build_tempo_pd_c4_calibrated_profiles \
  --analysis "${ANALYSIS}" \
  --expected-analysis-sha256 "${ANALYSIS_SHA}" \
  --workload-manifest "${MANIFEST}" \
  --expected-workload-manifest-sha256 "${MANIFEST_SHA}" \
  --elastic-profile-id "tempo-c4-elastic-${SLURM_JOB_ID}" \
  --endpoint-profile-id "tempo-c4-endpoint-v1-${SLURM_JOB_ID}" \
  --elastic-output "${ELASTIC}" \
  --endpoint-output "${ENDPOINT_V1}" \
  --receipt-output "${RECEIPT}"
ELASTIC_SHA=$(sha256sum "${ELASTIC}" | awk '{print $1}')
ENDPOINT_V1_SHA=$(sha256sum "${ENDPOINT_V1}" | awk '{print $1}')
RECEIPT_SHA=$(sha256sum "${RECEIPT}" | awk '{print $1}')

REPLAY="${POST_ROOT}/offline_replay.json"
"${PYTHON}" -m eval.sota_4node.replay_tempo_pd_c4_calibrated_controller \
  --analysis "${ANALYSIS}" --analysis-sha256 "${ANALYSIS_SHA}" \
  --manifest "${MANIFEST}" --manifest-sha256 "${MANIFEST_SHA}" \
  --elastic "${ELASTIC}" --elastic-sha256 "${ELASTIC_SHA}" \
  --endpoint "${ENDPOINT_V1}" --endpoint-sha256 "${ENDPOINT_V1_SHA}" \
  --receipt "${RECEIPT}" --receipt-sha256 "${RECEIPT_SHA}" \
  --output "${REPLAY}"
REPLAY_SHA=$(sha256sum "${REPLAY}" | awk '{print $1}')
[[ "$(jq -er '.live_adaptive_screen_authorized' "${REPLAY}")" == true ]]

ADAPTIVE_RUN_CONTRACT="${POST_ROOT}/adaptive_run_contract.json"
"${PYTHON}" -m eval.sota_4node.build_tempo_pd_c4_adaptive_run_contract \
  --repo-root "${REPO_ROOT}" \
  --analysis "${ANALYSIS}" --analysis-sha256 "${ANALYSIS_SHA}" \
  --manifest "${MANIFEST}" --manifest-sha256 "${MANIFEST_SHA}" \
  --elastic "${ELASTIC}" --elastic-sha256 "${ELASTIC_SHA}" \
  --endpoint "${ENDPOINT_V1}" --endpoint-sha256 "${ENDPOINT_V1_SHA}" \
  --receipt "${RECEIPT}" --receipt-sha256 "${RECEIPT_SHA}" \
  --replay "${REPLAY}" --replay-sha256 "${REPLAY_SHA}" \
  --implementation "${ADAPTIVE_IMPLEMENTATION}" \
  --implementation-sha256 "${ADAPTIVE_IMPLEMENTATION_SHA}" \
  --output "${ADAPTIVE_RUN_CONTRACT}"
ADAPTIVE_RUN_CONTRACT_SHA=$(sha256sum \
  "${ADAPTIVE_RUN_CONTRACT}" | awk '{print $1}')

ENDPOINT_V2="${POST_ROOT}/calibrated_semantic_endpoint_profile_v2.json"
"${PYTHON}" -m eval.sota_4node.build_tempo_pd_semantic_epoch_endpoint_profile \
  --base-profile "${ENDPOINT_V1}" \
  --expected-base-sha256 "${ENDPOINT_V1_SHA}" \
  --profile-id "tempo-c4-semantic-endpoint-v2-${SLURM_JOB_ID}" \
  --output "${ENDPOINT_V2}"
ENDPOINT_V2_SHA=$(sha256sum "${ENDPOINT_V2}" | awk '{print $1}')

RUN_CONTRACT="${POST_ROOT}/semantic_integration_run_contract.json"
"${PYTHON}" -m eval.sota_4node.build_tempo_pd_c4_semantic_integration_run_contract \
  --repo-root "${REPO_ROOT}" \
  --adaptive-contract "${ADAPTIVE_RUN_CONTRACT}" \
  --adaptive-contract-sha256 "${ADAPTIVE_RUN_CONTRACT_SHA}" \
  --semantic-analysis "${SEMANTIC_ANALYSIS}" \
  --semantic-analysis-sha256 "${SEMANTIC_ANALYSIS_SHA}" \
  --semantic-endpoint "${ENDPOINT_V2}" \
  --semantic-endpoint-sha256 "${ENDPOINT_V2_SHA}" \
  --implementation "${IMPLEMENTATION_CONTRACT}" \
  --implementation-sha256 "${IMPLEMENTATION_SHA}" \
  --output "${RUN_CONTRACT}"
RUN_CONTRACT_SHA=$(sha256sum "${RUN_CONTRACT}" | awk '{print $1}')

INTEGRATION_RESULT_DIR="${CAMPAIGN_ROOT}/semantic_integration_screen"
TEMPO_PD_C4_SEMANTIC_INTEGRATION_APPROVED=YES \
TEMPO_PD_C4_SEMANTIC_INTEGRATION_RUN_CONTRACT_SHA256="${RUN_CONTRACT_SHA}" \
TEMPO_PD_C4_STEP_TIME=01:15:00 \
  bash "${SCRIPT_DIR}/run_tempo_pd_c4_semantic_integration_screen_in_allocation.sh" \
  "${RUN_CONTRACT}" "${INTEGRATION_RESULT_DIR}"

INTEGRATION_RESULT="${INTEGRATION_RESULT_DIR}/result.json"
INTEGRATION_RESULT_SHA=$(sha256sum \
  "${INTEGRATION_RESULT}" | awk '{print $1}')
INTEGRATION_ANALYSIS="${CAMPAIGN_ROOT}/semantic_integration_analysis.json"
"${PYTHON}" -m eval.sota_4node.analyze_tempo_pd_c4_semantic_integration_screen \
  --result "${INTEGRATION_RESULT}" \
  --expected-result-sha256 "${INTEGRATION_RESULT_SHA}" \
  --output "${INTEGRATION_ANALYSIS}"

echo "Persistent 4-node semantic integration campaign complete: ${INTEGRATION_ANALYSIS}"
