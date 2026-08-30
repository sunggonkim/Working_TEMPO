#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
: "${TEMPO_PD_INDEPENDENT_IMPLEMENTATION_SHA256:?independent implementation SHA-256 required}"
[[ "${TEMPO_PD_INDEPENDENT_IMPLEMENTATION_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ $# -eq 2 ]]

ANALYSIS=$(realpath -e -- "$1")
OUTPUT_ROOT=$(realpath -m -- "$2")
case "${ANALYSIS}" in "${REPO_ROOT}"/results/*) ;; *) exit 2 ;; esac
case "${OUTPUT_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ -s "${ANALYSIS}" && ! -e "${OUTPUT_ROOT}" ]]

IMPLEMENTATION="${SCRIPT_DIR}/tempo_pd_independent_validation_implementation_contract_v1.json"
PREREGISTRATION="${SCRIPT_DIR}/tempo_pd_independent_validation_preregistration_v1.json"
PREREGISTRATION_SHA=c1de45c97025739e94b91b2da770a38b86b7a170d288c3f47c8cd4b774af7f86
[[ -s "${IMPLEMENTATION}" && -s "${PREREGISTRATION}" ]]
[[ "$(sha256sum "${IMPLEMENTATION}" | awk '{print $1}')" == \
  "${TEMPO_PD_INDEPENDENT_IMPLEMENTATION_SHA256}" ]]
[[ "$(sha256sum "${PREREGISTRATION}" | awk '{print $1}')" == \
  "${PREREGISTRATION_SHA}" ]]

export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
PYTHON="${REPO_ROOT}/.vllm_venv/bin/python"
ANALYSIS_SHA=$(sha256sum "${ANALYSIS}" | awk '{print $1}')
mkdir -p -- "${OUTPUT_ROOT}"

MANIFEST="${OUTPUT_ROOT}/independent_manifest.json"
"${PYTHON}" -m eval.sota_4node.build_tempo_pd_independent_validation_manifest \
  --candidate-analysis "${ANALYSIS}" \
  --candidate-analysis-sha256 "${ANALYSIS_SHA}" \
  --preregistration "${PREREGISTRATION}" \
  --preregistration-sha256 "${PREREGISTRATION_SHA}" \
  --output "${MANIFEST}"
MANIFEST_SHA=$(sha256sum "${MANIFEST}" | awk '{print $1}')

readarray -t SOURCE_PATHS < <("${PYTHON}" -c '
import json, pathlib, sys
analysis = json.load(open(sys.argv[1]))
contract = json.load(open(pathlib.Path(analysis["run_contract"]["path"])))
for name in ("elastic_profile", "endpoint_service_profile", "profile_receipt"):
    print(pathlib.Path(contract[name]["path"]).resolve())
' "${ANALYSIS}")
[[ ${#SOURCE_PATHS[@]} -eq 3 ]]
SOURCE_ELASTIC=${SOURCE_PATHS[0]}
SOURCE_ENDPOINT=${SOURCE_PATHS[1]}
SOURCE_RECEIPT=${SOURCE_PATHS[2]}
SOURCE_ELASTIC_SHA=$(sha256sum "${SOURCE_ELASTIC}" | awk '{print $1}')
SOURCE_ENDPOINT_SHA=$(sha256sum "${SOURCE_ENDPOINT}" | awk '{print $1}')
SOURCE_RECEIPT_SHA=$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')

ELASTIC="${OUTPUT_ROOT}/frozen_validation_elastic_profile.json"
ENDPOINT="${OUTPUT_ROOT}/frozen_validation_endpoint_profile.json"
PROMOTION_RECEIPT="${OUTPUT_ROOT}/profile_promotion_receipt.json"
"${PYTHON}" -m eval.sota_4node.promote_tempo_pd_profiles_for_independent_validation \
  --manifest "${MANIFEST}" --manifest-sha256 "${MANIFEST_SHA}" \
  --candidate-analysis "${ANALYSIS}" \
  --candidate-analysis-sha256 "${ANALYSIS_SHA}" \
  --preregistration "${PREREGISTRATION}" \
  --preregistration-sha256 "${PREREGISTRATION_SHA}" \
  --source-elastic "${SOURCE_ELASTIC}" \
  --source-elastic-sha256 "${SOURCE_ELASTIC_SHA}" \
  --source-endpoint "${SOURCE_ENDPOINT}" \
  --source-endpoint-sha256 "${SOURCE_ENDPOINT_SHA}" \
  --source-receipt "${SOURCE_RECEIPT}" \
  --source-receipt-sha256 "${SOURCE_RECEIPT_SHA}" \
  --elastic-output "${ELASTIC}" \
  --endpoint-output "${ENDPOINT}" \
  --receipt-output "${PROMOTION_RECEIPT}"
ELASTIC_SHA=$(sha256sum "${ELASTIC}" | awk '{print $1}')
ENDPOINT_SHA=$(sha256sum "${ENDPOINT}" | awk '{print $1}')
PROMOTION_RECEIPT_SHA=$(sha256sum "${PROMOTION_RECEIPT}" | awk '{print $1}')

RUN_CONTRACT="${OUTPUT_ROOT}/independent_run_contract.json"
"${PYTHON}" -m eval.sota_4node.build_tempo_pd_independent_validation_run_contract \
  --repo-root "${REPO_ROOT}" \
  --manifest "${MANIFEST}" --manifest-sha256 "${MANIFEST_SHA}" \
  --candidate-analysis "${ANALYSIS}" \
  --candidate-analysis-sha256 "${ANALYSIS_SHA}" \
  --preregistration "${PREREGISTRATION}" \
  --preregistration-sha256 "${PREREGISTRATION_SHA}" \
  --elastic "${ELASTIC}" --elastic-sha256 "${ELASTIC_SHA}" \
  --endpoint "${ENDPOINT}" --endpoint-sha256 "${ENDPOINT_SHA}" \
  --promotion-receipt "${PROMOTION_RECEIPT}" \
  --promotion-receipt-sha256 "${PROMOTION_RECEIPT_SHA}" \
  --implementation "${IMPLEMENTATION}" \
  --implementation-sha256 "${TEMPO_PD_INDEPENDENT_IMPLEMENTATION_SHA256}" \
  --output "${RUN_CONTRACT}"

RUN_CONTRACT_SHA=$(sha256sum "${RUN_CONTRACT}" | awk '{print $1}')
echo "Independent run contract: ${RUN_CONTRACT}"
echo "Independent run contract SHA-256: ${RUN_CONTRACT_SHA}"
