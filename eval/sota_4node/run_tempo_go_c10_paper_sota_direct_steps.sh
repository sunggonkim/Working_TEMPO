#!/usr/bin/env bash
set -euo pipefail

# Direct no-shell launcher for a pre-existing 4-node interactive allocation.
# It never submits or cancels a job and deliberately avoids nested srun steps.
: "${TEMPO_GO_C10_PAPER_SOTA_APPROVED:?set to YES after explicit approval}"
: "${TEMPO_GO_C10_DIRECT_JOB_ID:?existing allocation job id required}"
[[ "${TEMPO_GO_C10_PAPER_SOTA_APPROVED}" == YES ]]
ALLOC_JOB_ID=${TEMPO_GO_C10_DIRECT_JOB_ID}
[[ "${ALLOC_JOB_ID}" =~ ^[0-9]+$ ]]

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
MANIFEST=$(realpath -e -- "${TEMPO_GO_C10_MANIFEST:-${REPO_ROOT}/results/tempo_go_c10_paper_sota_independent_manifest_v1.json}")
PARENT_CONTRACT=$(realpath -e -- "${TEMPO_GO_C10_PARENT_CONTRACT:-${REPO_ROOT}/results/tempo_go_c8_independent_validation_contract_v4_current.json}")
RESULT_ROOT=${TEMPO_GO_C10_RESULT_DIR:-${REPO_ROOT}/results/tempo_go_c10_independent_job_${ALLOC_JOB_ID}}
[[ "${RESULT_ROOT}" == "${REPO_ROOT}/results/"* ]]
[[ ! -e "${RESULT_ROOT}" ]]
[[ "$(jq -er '.schema' "${MANIFEST}")" == tempo-go-c10-paper-sota-extension-v1 ]]
[[ "$(jq -er '.claim_boundary.post_hoc_extension' "${MANIFEST}")" == false ]]
[[ "$(jq -er '.claim_boundary.independent_validation_claim_allowed' "${MANIFEST}")" == true ]]
if jq -e --arg job "${ALLOC_JOB_ID}" \
    '.claim_boundary.forbidden_allocation_job_ids | index($job) != null' \
    "${MANIFEST}" >/dev/null; then
  echo "allocation is forbidden by independent manifest: ${ALLOC_JOB_ID}" >&2
  exit 2
fi

RECEIPT=$(scontrol show job "${ALLOC_JOB_ID}" --oneliner)
[[ " ${RECEIPT} " == *" JobState=RUNNING "* ]]
[[ " ${RECEIPT} " == *" JobName=no-shell "* ]]
[[ " ${RECEIPT} " == *" QOS=gpu_interactive "* || " ${RECEIPT} " == *" QOS=interactive "* ]]
[[ " ${RECEIPT} " == *" TimeLimit=04:00:00 "* || " ${RECEIPT} " == *" TimeLimit=4:00:00 "* ]]
[[ " ${RECEIPT} " == *" NumNodes=4 "* && " ${RECEIPT} " == *" gres/gpu=16 "* ]]
NODELIST=$(sed -n 's/.*NodeList=\([^ ]*\).*/\1/p' <<<"${RECEIPT}")
[[ -n "${NODELIST}" ]]
mapfile -t HOSTS < <(scontrol show hostnames "${NODELIST}")
[[ "${#HOSTS[@]}" -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")

SOURCE_REL=$(jq -er '.joint_control.source_workload.path' "${PARENT_CONTRACT}")
PROFILE_REL=$(jq -er '.joint_control.profile.path' "${PARENT_CONTRACT}")
SOURCE_WORKLOAD=$(realpath -e -- "${REPO_ROOT}/${SOURCE_REL}")
PROFILE=$(realpath -e -- "${REPO_ROOT}/${PROFILE_REL}")
REQUEST_RATE=$(jq -er '.joint_control.victim.offered_rate_per_s' "${PARENT_CONTRACT}")
MAX_WORKERS=$(jq -er '.joint_control.max_workers' "${PARENT_CONTRACT}")
PARENT_ANALYSIS=$(jq -er '.parent_independent_validation.analysis' "${MANIFEST}")
TEMPO_RESULT=$(jq -er '.parent_independent_validation.tempo_result' "${MANIFEST}")
PARENT_ANALYSIS="${REPO_ROOT}/${PARENT_ANALYSIS}"
TEMPO_RESULT="${REPO_ROOT}/${TEMPO_RESULT}"
[[ -s "${SOURCE_WORKLOAD}" && -s "${PROFILE}" && -s "${PARENT_ANALYSIS}" && -s "${TEMPO_RESULT}" ]]

mkdir -p -- "${RESULT_ROOT}/cache_preflight"
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${REPO_ROOT}"
export NCCL_NET="${NCCL_NET:-AWS Libfabric}"
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}"
export TEMPO_GO_C8_DUAL_REGIME_CONTRACT="${PARENT_CONTRACT}"
export TEMPO_GO_C10_PAPER_SOTA_MANIFEST="${MANIFEST}"
export TEMPO_ELASTIC_PD_PROFILE="${PROFILE}"
export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY=1
export TEMPO_VLLM_MAX_NUM_SEQS=16
export TEMPO_VLLM_DECODER_PREFIX_CACHING=0
export TEMPO_LMCACHE_NIXL_BACKEND=UCX
export TEMPO_SHARED_CACHE_MODE="${TEMPO_SHARED_CACHE_MODE:-tempo_go_c10_independent_${ALLOC_JOB_ID}}"
unset TEMPO_CXI_BACKGROUND_DUTY_CYCLE TEMPO_CXI_BACKGROUND_START_FILE

run_step() {
  local output_dir=$1
  shift
  local gpus_per_task=$1
  local cpus_per_task=$2
  shift 2
  mkdir -p -- "${output_dir}"
  srun --jobid="${ALLOC_JOB_ID}" --overlap --exact \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 --distribution=block:block \
    --gpus-per-task="${gpus_per_task}" --gpu-bind=none --cpus-per-task="${cpus_per_task}" --cpu-bind=cores \
    --kill-on-bad-exit=1 --wait=600 --time=00:29:00 \
    --export=ALL \
    --output="${output_dir}/slurm-node-%N.stdout.log" \
    --error="${output_dir}/slurm-node-%N.stderr.log" \
    bash -lc "export PYTHONPATH='${REPO_ROOT}' PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1; export TEMPO_GO_C8_DUAL_REGIME_CONTRACT='${PARENT_CONTRACT}' TEMPO_GO_C10_PAPER_SOTA_MANIFEST='${MANIFEST}' TEMPO_ELASTIC_PD_PROFILE='${PROFILE}' TEMPO_SHARED_CACHE_MODE='${TEMPO_SHARED_CACHE_MODE}' NCCL_NET='${NCCL_NET}' FI_CXI_RX_MATCH_MODE='${FI_CXI_RX_MATCH_MODE}'; module reset >/dev/null 2>&1; module load pytorch/2.8.0 >/dev/null 2>&1; exec $*"
}

PREFLIGHT_ROOT="${RESULT_ROOT}/cache_preflight"
run_step "${PREFLIGHT_ROOT}" 1 32 bash "${SCRIPT_DIR}/c10_flashinfer_cache_preflight.sh" "${REPO_ROOT}" "${RESULT_ROOT}" "${TEMPO_SHARED_CACHE_MODE}"
for node_index in 0 1 2 3; do
  receipt="${PREFLIGHT_ROOT}/node-${node_index}.json"
  [[ -s "${receipt}" ]]
  jq -e --arg job "${ALLOC_JOB_ID}" --arg mode "${TEMPO_SHARED_CACHE_MODE}" \
    '.schema == "tempo-go-c10-flashinfer-preflight-v1" and .job_id == $job and .cache_mode == $mode and (.sampling_so_sha256 | length == 64)' \
    "${receipt}" >/dev/null
done

mapfile -t POLICIES < <(jq -er '.execution_order[]' "${MANIFEST}")
[[ "${#POLICIES[@]}" -eq 2 ]]
for index in 0 1; do
  policy=${POLICIES[index]}
  result_dir="${RESULT_ROOT}/${policy}"
  export TEMPO_PAPER_BASELINE_POLICY="${policy}"
  export TEMPO_GO_C8_DUAL_REGIME_ARM=app_global_only
  run_step "${result_dir}" 4 128 bash "${SCRIPT_DIR}/c10_paper_sota_node_entry.sh" \
    "${REPO_ROOT}" "${result_dir}" "${SOURCE_WORKLOAD}" "${HOSTS_CSV}" \
    "$((2420 + index * 40))" "${REQUEST_RATE}" "${MAX_WORKERS}" 128 8 3000 150 8000
  [[ -s "${result_dir}/result.json" ]]
done

PYTHONPATH="${REPO_ROOT}" "${REPO_ROOT}/.vllm_venv/bin/python" -m eval.sota_4node.analyze_tempo_go_c10_paper_sota \
  --manifest "${MANIFEST}" \
  --tempo-result "${TEMPO_RESULT}" \
  --parent-analysis "${PARENT_ANALYSIS}" \
  --baseline "${POLICIES[0]}=${RESULT_ROOT}/${POLICIES[0]}/result.json" \
  --baseline "${POLICIES[1]}=${RESULT_ROOT}/${POLICIES[1]}/result.json" \
  --output "${RESULT_ROOT}/analysis.json"
jq '{actual_sota_extension_positive,claim_boundary,gates}' "${RESULT_ROOT}/analysis.json"
