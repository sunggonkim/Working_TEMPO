#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

[[ "${TEMPO_GO_C8_DUAL_REGIME_APPROVED:-}" == YES ]] || exit 2
source "${SCRIPT_DIR}/require_perlmutter_4node_4h_interactive.sh"

REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONTRACT="${TEMPO_GO_C8_DUAL_REGIME_CONTRACT:-${REPO_ROOT}/eval/sota_4node/tempo_go_c8_dual_regime_contract_v39.json}"
CONTRACT=$(realpath -e -- "${CONTRACT}")
case "${CONTRACT}" in "${REPO_ROOT}/"*) ;; *) exit 2 ;; esac
[[ "$(jq -er '.schema' "${CONTRACT}")" == tempo-go-c8-dual-regime-contract-v1 ]]
[[ "$(jq -er '.claim_boundary.independent_validation_claim_allowed' "${CONTRACT}")" == false ]]

SOURCE_REL=$(jq -er '.joint_control.source_workload.path' "${CONTRACT}")
PROFILE_REL=$(jq -er '.joint_control.profile.path' "${CONTRACT}")
SOURCE_WORKLOAD=$(realpath -e -- "${REPO_ROOT}/${SOURCE_REL}")
PROFILE=$(realpath -e -- "${REPO_ROOT}/${PROFILE_REL}")
[[ "$(sha256sum "${SOURCE_WORKLOAD}" | awk '{print $1}')" == "$(jq -er '.joint_control.source_workload.sha256' "${CONTRACT}")" ]]
[[ "$(sha256sum "${PROFILE}" | awk '{print $1}')" == "$(jq -er '.joint_control.profile.sha256' "${CONTRACT}")" ]]

RESULT_ROOT="${TEMPO_GO_C8_DUAL_REGIME_RESULT_DIR:-${REPO_ROOT}/results/tempo_go_c8_dual_regime_job_${SLURM_JOB_ID}}"
case "${RESULT_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ ! -e "${RESULT_ROOT}" ]]
mkdir -p -- "${RESULT_ROOT}"

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

if [[ -n "${TEMPO_GO_C8_HOSTS_CSV:-}" ]]; then
  IFS=, read -r -a HOSTS <<< "${TEMPO_GO_C8_HOSTS_CSV}"
else
  mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
fi
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
BASE_PORT_SLOT=$((2140 + SLURM_JOB_ID % 20))
REQUEST_RATE=$(jq -er '.joint_control.victim.offered_rate_per_s' "${CONTRACT}")
MAX_WORKERS=$(jq -er '.joint_control.max_workers' "${CONTRACT}")
mapfile -t CONTRACT_ARMS < <(jq -er '.joint_control.arms[].name' "${CONTRACT}")
ARM_ONLY="${TEMPO_GO_C8_DUAL_REGIME_ARM_ONLY:-}"

run_arm() {
  local arm=$1
  local index=$2
  local result_dir="${RESULT_ROOT}/${arm}"
  mkdir -p -- "${result_dir}"
  export TEMPO_GO_C8_DUAL_REGIME_ARM="${arm}"
  if ! timeout --foreground --signal=TERM --kill-after=30s 1800s \
    /usr/bin/srun --overlap --exact \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
    --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
    --time=00:29:00 --export=ALL \
    --output="${result_dir}/slurm-node-%N.stdout.log" \
    --error="${result_dir}/slurm-node-%N.stderr.log" \
    bash "${SCRIPT_DIR}/c8_dual_regime_node_entry.sh" \
    "${REPO_ROOT}" "${result_dir}" "${SOURCE_WORKLOAD}" "${HOSTS_CSV}" \
    "$((BASE_PORT_SLOT + index * 40))" \
    "${REQUEST_RATE}" "${MAX_WORKERS}" 128 8 3000 150 8000; then
    echo "TEMPO-GO C8 arm failed: ${arm}" >&2
    return 1
  fi
  [[ -s "${result_dir}/result.json" ]] || {
    echo "TEMPO-GO C8 arm produced no result: ${arm}" >&2
    return 1
  }
  jq -ce '.analysis | {
    arm,
    miss_hot_slo_good:.miss_hot.slo_good_victims,
    miss_hot_p99_ms:.miss_hot.victim.e2e_ms.p99,
    remote_slo_good:.remote_favorable.slo_good_victims,
    remote_p99_ms:.remote_favorable.victim.e2e_ms.p99,
    remote_routes:.remote_favorable.route_counts,
    route_counts,
    edge_counts
  }' "${result_dir}/result.json" || return 1
}

if [[ -n "${ARM_ONLY}" ]]; then
  selected=-1
  for index in "${!CONTRACT_ARMS[@]}"; do
    if [[ "${CONTRACT_ARMS[index]}" == "${ARM_ONLY}" ]]; then
      selected=${index}
      break
    fi
  done
  (( selected >= 0 )) || exit 2
  run_arm "${ARM_ONLY}" "${selected}"
  echo "TEMPO-GO C8 arm receipt: ${RESULT_ROOT}/${ARM_ONLY}/result.json"
  exit 0
fi

for index in "${!CONTRACT_ARMS[@]}"; do
  run_arm "${CONTRACT_ARMS[index]}" "${index}"
done

ANALYZE=(
  "${REPO_ROOT}/.vllm_venv/bin/python"
  -m eval.sota_4node.analyze_tempo_go_c8_dual_regime
  --contract "${CONTRACT}"
)
for arm in "${CONTRACT_ARMS[@]}"; do
  ANALYZE+=(--result "${arm}=${RESULT_ROOT}/${arm}/result.json")
done
ANALYZE+=(--output "${RESULT_ROOT}/analysis.json")
"${ANALYZE[@]}"
[[ -s "${RESULT_ROOT}/analysis.json" ]]
jq '{
  strongest_fixed_miss_hot_arm,
  strongest_fixed_remote_favorable_arm,
  effects,
  correctness_gate,
  miss_hot_vs_strongest_fixed_robustness_gate,
  miss_hot_vs_predictor_robustness_gate,
  remote_favorable_vs_predictor_robustness_gate,
  remote_favorable_best_fixed_noninferiority_gate,
  remote_favorable_remote_fraction,
  remote_favorable_activation_gate,
  remote_favorable_exact_lmcache_full_hit_gate,
  full_uses_both_local_and_remote,
  c8_dual_regime_discovery_positive
}' "${RESULT_ROOT}/analysis.json"
echo "TEMPO-GO C8 dual-regime receipt: ${RESULT_ROOT}/analysis.json"
