#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

[[ "${TEMPO_GO_C8_INDEPENDENT_VALIDATION_APPROVED:-}" == YES ]] || exit 2
source "${SCRIPT_DIR}/require_perlmutter_4node_4h_interactive.sh"

REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONTRACT="${TEMPO_GO_C8_INDEPENDENT_VALIDATION_CONTRACT:-${REPO_ROOT}/eval/sota_4node/tempo_go_c8_independent_validation_contract_v3.json}"
CONTRACT=$(realpath -e -- "${CONTRACT}")
case "${CONTRACT}" in "${REPO_ROOT}/"*) ;; *) exit 2 ;; esac
[[ "$(jq -er '.schema' "${CONTRACT}")" == tempo-go-c8-dual-regime-contract-v1 ]]
[[ "$(jq -er '.independent_validation.schema' "${CONTRACT}")" == tempo-go-c8-independent-validation-v1 ]]
[[ "$(jq -er '.independent_validation.one_shot_no_retry' "${CONTRACT}")" == true ]]
[[ "$(jq -er '.claim_boundary.performance_claim_allowed' "${CONTRACT}")" == false ]]
[[ "$(jq -er '.claim_boundary.independent_validation_claim_allowed' "${CONTRACT}")" == false ]]

if jq -e --arg job "${SLURM_JOB_ID}" \
  '.independent_validation.forbidden_discovery_job_ids | index($job) != null' \
  "${CONTRACT}" >/dev/null; then
  echo "independent validation requires a fresh allocation" >&2
  exit 2
fi
JOB_RECEIPT=$(scontrol show job "${SLURM_JOB_ID}" -o)
[[ "${JOB_RECEIPT}" == *"JobName=no-shell"* ]]
[[ "${JOB_RECEIPT}" == *"Command=(null)"* ]]

SOURCE_REL=$(jq -er '.joint_control.source_workload.path' "${CONTRACT}")
PROFILE_REL=$(jq -er '.joint_control.profile.path' "${CONTRACT}")
GLOBAL_PROFILE_REL=$(jq -er '.joint_control.global_profile.path' "${CONTRACT}")
SOURCE_WORKLOAD=$(realpath -e -- "${REPO_ROOT}/${SOURCE_REL}")
PROFILE=$(realpath -e -- "${REPO_ROOT}/${PROFILE_REL}")
GLOBAL_PROFILE=$(realpath -e -- "${REPO_ROOT}/${GLOBAL_PROFILE_REL}")
[[ "$(sha256sum "${SOURCE_WORKLOAD}" | awk '{print $1}')" == "$(jq -er '.joint_control.source_workload.sha256' "${CONTRACT}")" ]]
[[ "$(sha256sum "${PROFILE}" | awk '{print $1}')" == "$(jq -er '.joint_control.profile.sha256' "${CONTRACT}")" ]]
[[ "$(sha256sum "${GLOBAL_PROFILE}" | awk '{print $1}')" == "$(jq -er '.joint_control.global_profile.sha256' "${CONTRACT}")" ]]

RESULT_ROOT="${TEMPO_GO_C8_INDEPENDENT_VALIDATION_RESULT_DIR:-${REPO_ROOT}/results/tempo_go_c8_independent_validation_job_${SLURM_JOB_ID}}"
case "${RESULT_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ ! -e "${RESULT_ROOT}" ]]
mkdir -p -- "${RESULT_ROOT}"

CONTRACT_SHA=$(sha256sum "${CONTRACT}" | awk '{print $1}')
jq -n \
  --arg schema tempo-go-c8-independent-attempt-v1 \
  --arg status running \
  --arg job_id "${SLURM_JOB_ID}" \
  --arg contract "${CONTRACT}" \
  --arg contract_sha256 "${CONTRACT_SHA}" \
  '{schema:$schema,status:$status,slurm_job_id:$job_id,contract:$contract,contract_sha256:$contract_sha256,one_shot_no_retry:true}' \
  >"${RESULT_ROOT}/attempt.json"

module reset
module load pytorch/2.8.0
[[ "${NCCL_NET:-}" == "AWS Libfabric" ]] || {
  echo "expected NERSC NCCL AWS Libfabric transport, got ${NCCL_NET:-<unset>}" >&2
  exit 1
}
unset NCCL_IB_DISABLE
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
export TEMPO_GO_C8_DUAL_REGIME_CONTRACT="${CONTRACT}"
export TEMPO_ELASTIC_PD_PROFILE="${PROFILE}"
export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY=1
export TEMPO_VLLM_MAX_NUM_SEQS=16
export TEMPO_VLLM_DECODER_PREFIX_CACHING=0
export TEMPO_LMCACHE_NIXL_BACKEND=UCX
unset TEMPO_CXI_BACKGROUND_DUTY_CYCLE TEMPO_CXI_BACKGROUND_START_FILE

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
mapfile -t CONTRACT_ARMS < <(jq -er '.joint_control.arms[].name' "${CONTRACT}")
BASE_PORT_SLOT=$(jq -er '.independent_validation.runtime_port_schedule.port_slot_base' "${CONTRACT}")
PORT_SLOT_STRIDE=$(jq -er '.independent_validation.runtime_port_schedule.port_slot_stride_per_arm' "${CONTRACT}")
MAXIMUM_PORT_SLOT=$(jq -er '.independent_validation.runtime_port_schedule.maximum_port_slot' "${CONTRACT}")
[[ $((BASE_PORT_SLOT + PORT_SLOT_STRIDE * (${#CONTRACT_ARMS[@]} - 1))) -eq ${MAXIMUM_PORT_SLOT} ]]
[[ $((30000 + MAXIMUM_PORT_SLOT)) -lt 32768 ]]
REQUEST_RATE=$(jq -er '.joint_control.victim.offered_rate_per_s' "${CONTRACT}")
MAX_WORKERS=$(jq -er '.joint_control.max_workers' "${CONTRACT}")

run_arm() {
  local arm=$1
  local index=$2
  local result_dir="${RESULT_ROOT}/${arm}"
  mkdir -p -- "${result_dir}"
  export TEMPO_GO_C8_DUAL_REGIME_ARM="${arm}"
  timeout --foreground --signal=TERM --kill-after=30s 1800s \
    /usr/bin/srun --overlap --exact \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
    --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
    --time=00:29:00 --export=ALL \
    --output="${result_dir}/slurm-node-%N.stdout.log" \
    --error="${result_dir}/slurm-node-%N.stderr.log" \
    bash "${SCRIPT_DIR}/c8_independent_validation_node_entry.sh" \
    "${REPO_ROOT}" "${result_dir}" "${SOURCE_WORKLOAD}" "${HOSTS_CSV}" \
    "$((BASE_PORT_SLOT + index * PORT_SLOT_STRIDE))" \
    "${REQUEST_RATE}" "${MAX_WORKERS}" 128 8 3000 150 8000
  [[ -s "${result_dir}/result.json" ]]
  jq -ce '.analysis | {
    arm,
    miss_hot_slo_good:.miss_hot.slo_good_victims,
    miss_hot_p99_ms:.miss_hot.victim.e2e_ms.p99,
    remote_slo_good:.remote_favorable.slo_good_victims,
    remote_p99_ms:.remote_favorable.victim.e2e_ms.p99,
    remote_routes:.remote_favorable.route_counts,
    route_counts,
    edge_counts
  }' "${result_dir}/result.json"
}

for index in "${!CONTRACT_ARMS[@]}"; do
  arm=${CONTRACT_ARMS[index]}
  if ! run_arm "${arm}" "${index}"; then
    jq -n \
      --arg schema tempo-go-c8-independent-failure-v1 \
      --arg job_id "${SLURM_JOB_ID}" \
      --arg arm "${arm}" \
      --arg contract_sha256 "${CONTRACT_SHA}" \
      '{schema:$schema,slurm_job_id:$job_id,failed_arm:$arm,contract_sha256:$contract_sha256,terminal:false,retry_allowed:false}' \
      >"${RESULT_ROOT}/execution_failure_receipt.json"
    exit 1
  fi
done

ANALYZE=(
  "${REPO_ROOT}/.vllm_venv/bin/python"
  -m eval.sota_4node.analyze_tempo_go_c8_independent_validation
  --contract "${CONTRACT}"
)
for arm in "${CONTRACT_ARMS[@]}"; do
  ANALYZE+=(--result "${arm}=${RESULT_ROOT}/${arm}/result.json")
done
ANALYZE+=(--output "${RESULT_ROOT}/analysis.json")
"${ANALYZE[@]}"
[[ -s "${RESULT_ROOT}/analysis.json" ]]

jq -n \
  --arg schema tempo-go-c8-independent-attempt-v1 \
  --arg status complete \
  --arg job_id "${SLURM_JOB_ID}" \
  --arg contract "${CONTRACT}" \
  --arg contract_sha256 "${CONTRACT_SHA}" \
  --arg analysis "${RESULT_ROOT}/analysis.json" \
  --arg analysis_sha256 "$(sha256sum "${RESULT_ROOT}/analysis.json" | awk '{print $1}')" \
  '{schema:$schema,status:$status,slurm_job_id:$job_id,contract:$contract,contract_sha256:$contract_sha256,analysis:$analysis,analysis_sha256:$analysis_sha256,one_shot_no_retry:true}' \
  >"${RESULT_ROOT}/completed_attempt.json"

jq '{
  slurm_job_ids,
  fresh_allocation_gate,
  one_shot_execution_receipt_gate,
  base_performance_gate,
  background:.background | {
    c7_completion_fraction,
    c7_minimum_block_tenant_completion_fraction,
    c7_tenant_jain_fairness,
    c7_service_lane_failure_fraction,
    c8_completion_fraction,
    background_utility_and_fairness_gate
  },
  telemetry:.telemetry | {
    complete_batch_fraction,
    collection_ms,
    admission_wait_ms,
    source_virtual_service_binding_fraction,
    required_supported_signal_gate,
    required_explicit_status_gate,
    telemetry_and_overhead_gate
  },
  independent_validation_positive,
  performance_claim_allowed,
  independent_validation_claim_allowed
}' "${RESULT_ROOT}/analysis.json"
echo "TEMPO-GO C8 independent receipt: ${RESULT_ROOT}/analysis.json"
