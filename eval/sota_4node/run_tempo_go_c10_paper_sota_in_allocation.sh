#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ "${TEMPO_GO_C10_PAPER_SOTA_APPROVED:-}" == YES ]] || exit 2
source "${SCRIPT_DIR}/require_perlmutter_4node_4h_interactive.sh"

REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PARENT_CONTRACT=$(realpath -e -- "${TEMPO_GO_C10_PARENT_CONTRACT:-${REPO_ROOT}/eval/sota_4node/tempo_go_c8_independent_validation_contract_v3.json}")
MANIFEST=$(realpath -e -- "${TEMPO_GO_C10_MANIFEST:-${REPO_ROOT}/eval/sota_4node/tempo_go_c10_paper_sota_extension_v1.json}")
case "${PARENT_CONTRACT}" in "${REPO_ROOT}/"*) ;; *) exit 2 ;; esac
case "${MANIFEST}" in "${REPO_ROOT}/"*) ;; *) exit 2 ;; esac
[[ "$(jq -er '.schema' "${MANIFEST}")" == tempo-go-c10-paper-sota-extension-v1 ]]
[[ "$(jq -er '.claim_boundary.post_hoc_extension' "${MANIFEST}")" == true ]]
[[ "$(jq -er '.claim_boundary.independent_validation_claim_allowed' "${MANIFEST}")" == false ]]
[[ "$(sha256sum "${PARENT_CONTRACT}" | awk '{print $1}')" == "$(jq -er '.parent_independent_validation.sha256' "${MANIFEST}")" ]]

JOB_RECEIPT=$(scontrol show job "${SLURM_JOB_ID}" -o)
[[ "${JOB_RECEIPT}" == *"JobName=no-shell"* ]]
[[ "${JOB_RECEIPT}" == *"Command=(null)"* ]]

SOURCE_REL=$(jq -er '.joint_control.source_workload.path' "${PARENT_CONTRACT}")
PROFILE_REL=$(jq -er '.joint_control.profile.path' "${PARENT_CONTRACT}")
GLOBAL_PROFILE_REL=$(jq -er '.joint_control.global_profile.path' "${PARENT_CONTRACT}")
SOURCE_WORKLOAD=$(realpath -e -- "${REPO_ROOT}/${SOURCE_REL}")
PROFILE=$(realpath -e -- "${REPO_ROOT}/${PROFILE_REL}")
GLOBAL_PROFILE=$(realpath -e -- "${REPO_ROOT}/${GLOBAL_PROFILE_REL}")
[[ "$(sha256sum "${SOURCE_WORKLOAD}" | awk '{print $1}')" == "$(jq -er '.joint_control.source_workload.sha256' "${PARENT_CONTRACT}")" ]]
[[ "$(sha256sum "${PROFILE}" | awk '{print $1}')" == "$(jq -er '.joint_control.profile.sha256' "${PARENT_CONTRACT}")" ]]
[[ "$(sha256sum "${GLOBAL_PROFILE}" | awk '{print $1}')" == "$(jq -er '.joint_control.global_profile.sha256' "${PARENT_CONTRACT}")" ]]

while IFS=$'\t' read -r relative expected; do
  source_path=$(realpath -e -- "${REPO_ROOT}/${relative}")
  case "${source_path}" in "${REPO_ROOT}/"*) ;; *) exit 2 ;; esac
  [[ "$(sha256sum "${source_path}" | awk '{print $1}')" == "${expected}" ]]
done < <(jq -r '.source_inventory | to_entries[] | [.key,.value] | @tsv' "${MANIFEST}")

RESULT_ROOT="${TEMPO_GO_C10_RESULT_DIR:-${REPO_ROOT}/results/tempo_go_c10_paper_sota_job_${SLURM_JOB_ID}_v1}"
case "${RESULT_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ ! -e "${RESULT_ROOT}" ]]
mkdir -p -- "${RESULT_ROOT}"

module reset
module load pytorch/2.8.0
[[ "${NCCL_NET:-}" == "AWS Libfabric" ]]
unset NCCL_IB_DISABLE
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
export TEMPO_GO_C8_DUAL_REGIME_CONTRACT="${PARENT_CONTRACT}"
export TEMPO_GO_C10_PAPER_SOTA_MANIFEST="${MANIFEST}"
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
REQUEST_RATE=$(jq -er '.joint_control.victim.offered_rate_per_s' "${PARENT_CONTRACT}")
MAX_WORKERS=$(jq -er '.joint_control.max_workers' "${PARENT_CONTRACT}")
mapfile -t POLICIES < <(jq -er '.execution_order[]' "${MANIFEST}")
[[ ${#POLICIES[@]} -eq 2 ]]
POLICY_FILTER=${TEMPO_GO_C10_POLICY_FILTER:-}
if [[ -n "${POLICY_FILTER}" ]]; then
  [[ "${POLICY_FILTER}" == netkv || "${POLICY_FILTER}" == kairos_x512 ]]
  jq -e --arg policy "${POLICY_FILTER}" \
    '.execution_order | index($policy) != null' "${MANIFEST}" >/dev/null
  POLICIES=("${POLICY_FILTER}")
fi

run_policy() {
  local policy=$1
  local index=$2
  local result_dir="${RESULT_ROOT}/${policy}"
  mkdir -p -- "${result_dir}"
  export TEMPO_PAPER_BASELINE_POLICY="${policy}"
  export TEMPO_GO_C8_DUAL_REGIME_ARM=app_global_only
  timeout --foreground --signal=TERM --kill-after=30s 1800s \
    /usr/bin/srun --overlap --exact \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
    --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
    --time=00:29:00 --export=ALL \
    --output="${result_dir}/slurm-node-%N.stdout.log" \
    --error="${result_dir}/slurm-node-%N.stderr.log" \
    bash "${SCRIPT_DIR}/c10_paper_sota_node_entry.sh" \
    "${REPO_ROOT}" "${result_dir}" "${SOURCE_WORKLOAD}" "${HOSTS_CSV}" \
    "$((2420 + index * 40))" \
    "${REQUEST_RATE}" "${MAX_WORKERS}" 128 8 3000 150 8000
  [[ -s "${result_dir}/result.json" ]]
  jq -ce --arg policy "${policy}" '.analysis | {
    policy:$policy,
    miss_hot_slo_good:.miss_hot.slo_good_victims,
    miss_hot_p99_ms:.miss_hot.victim.e2e_ms.p99,
    remote_slo_good:.remote_favorable.slo_good_victims,
    remote_p99_ms:.remote_favorable.victim.e2e_ms.p99,
    route_counts,
    edge_counts
  }' "${result_dir}/result.json"
}

for index in "${!POLICIES[@]}"; do
  run_policy "${POLICIES[index]}" "${index}"
done

if [[ -n "${POLICY_FILTER}" ]]; then
  sha256sum "${RESULT_ROOT}/${POLICY_FILTER}/result.json"
  exit 0
fi

PARENT_ROOT="${REPO_ROOT}/results/tempo_go_c8_independent_validation_job_57586612_v3"
PARENT_ANALYSIS="${PARENT_ROOT}/analysis.json"
TEMPO_RESULT="${PARENT_ROOT}/full_c7_managed_background/result.json"
[[ -s "${PARENT_ANALYSIS}" && -s "${TEMPO_RESULT}" ]]

ANALYZE=(
  "${REPO_ROOT}/.vllm_venv/bin/python"
  -m eval.sota_4node.analyze_tempo_go_c10_paper_sota
  --manifest "${MANIFEST}"
  --tempo-result "${TEMPO_RESULT}"
  --parent-analysis "${PARENT_ANALYSIS}"
)
for policy in "${POLICIES[@]}"; do
  ANALYZE+=(--baseline "${policy}=${RESULT_ROOT}/${policy}/result.json")
done
ANALYZE+=(--output "${RESULT_ROOT}/analysis.json")
"${ANALYZE[@]}"
[[ -s "${RESULT_ROOT}/analysis.json" ]]
sha256sum "${RESULT_ROOT}/analysis.json"
