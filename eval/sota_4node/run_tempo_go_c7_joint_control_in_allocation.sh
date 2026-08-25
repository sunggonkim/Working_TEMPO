#!/usr/bin/env bash
set -euo pipefail

[[ "${TEMPO_GO_C7_JOINT_CONTROL_APPROVED:-}" == YES ]] || exit 2
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 2
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_JOB_NODES:-}}" == 4 ]] || exit 2
[[ -z "${SHIFTER_RUNTIME:-}${SHIFTER_IMAGE:-}${UDI:-}${CRAY_ROOTFS:-}${SLURM_CONTAINER:-}" ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONTRACT="${TEMPO_GO_C7_JOINT_CONTROL_CONTRACT:-${REPO_ROOT}/eval/sota_4node/tempo_go_c7_joint_control_contract_v1.json}"
CONTRACT=$(realpath -e -- "${CONTRACT}")
case "${CONTRACT}" in "${REPO_ROOT}/"*) ;; *) exit 2 ;; esac
[[ "$(jq -er '.schema' "${CONTRACT}")" == tempo-go-c7-joint-control-contract-v1 ]]
[[ "$(jq -er '.claim_boundary.independent_validation_claim_allowed' "${CONTRACT}")" == false ]]

SOURCE_REL=$(jq -er '.joint_control.source_workload.path' "${CONTRACT}")
PROFILE_REL=$(jq -er '.joint_control.profile.path' "${CONTRACT}")
SOURCE_WORKLOAD=$(realpath -e -- "${REPO_ROOT}/${SOURCE_REL}")
PROFILE=$(realpath -e -- "${REPO_ROOT}/${PROFILE_REL}")
[[ "$(sha256sum "${SOURCE_WORKLOAD}" | awk '{print $1}')" == "$(jq -er '.joint_control.source_workload.sha256' "${CONTRACT}")" ]]
[[ "$(sha256sum "${PROFILE}" | awk '{print $1}')" == "$(jq -er '.joint_control.profile.sha256' "${CONTRACT}")" ]]

RESULT_ROOT="${TEMPO_GO_C7_JOINT_CONTROL_RESULT_DIR:-${REPO_ROOT}/results/tempo_go_c7_joint_control_job_${SLURM_JOB_ID}}"
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
export TEMPO_GO_C7_JOINT_CONTROL_CONTRACT="${CONTRACT}"
export TEMPO_ELASTIC_PD_PROFILE="${PROFILE}"
export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_VLLM_MAX_NUM_SEQS=16
export TEMPO_VLLM_DECODER_PREFIX_CACHING=0
export TEMPO_LMCACHE_NIXL_BACKEND=UCX
unset TEMPO_CXI_BACKGROUND_DUTY_CYCLE TEMPO_CXI_BACKGROUND_START_FILE

if [[ -n "${TEMPO_GO_C7_HOSTS_CSV:-}" ]]; then
  IFS=, read -r -a HOSTS <<< "${TEMPO_GO_C7_HOSTS_CSV}"
else
  mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
fi
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
BASE_PORT_SLOT=$((1940 + SLURM_JOB_ID % 20))
REQUEST_RATE=$(jq -er '.joint_control.victim.offered_rate_per_s' "${CONTRACT}")
MAX_WORKERS=$(jq -er '.joint_control.max_workers' "${CONTRACT}")
mapfile -t CONTRACT_ARMS < <(jq -er '.joint_control.arms[].name' "${CONTRACT}")
ARM_ONLY="${TEMPO_GO_C7_JOINT_CONTROL_ARM_ONLY:-}"

run_arm() {
  local arm=$1
  local index=$2
  local result_dir="${RESULT_ROOT}/${arm}"
  mkdir -p -- "${result_dir}"
  export TEMPO_GO_C7_JOINT_CONTROL_ARM="${arm}"
  timeout --foreground --signal=TERM --kill-after=30s 1500s \
    /usr/bin/srun --overlap --exact \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
    --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
    --time=00:24:00 --export=ALL \
    --output="${result_dir}/slurm-node-%N.stdout.log" \
    --error="${result_dir}/slurm-node-%N.stderr.log" \
    bash "${SCRIPT_DIR}/c7_joint_control_node_entry.sh" \
    "${REPO_ROOT}" "${result_dir}" "${SOURCE_WORKLOAD}" "${HOSTS_CSV}" \
    "$((BASE_PORT_SLOT + index * 40))" \
    "${REQUEST_RATE}" "${MAX_WORKERS}" 128 8 3000 150 8000
  [[ -s "${result_dir}/result.json" ]]
  jq -c '.analysis | {
    arm, hot_slo_good:.hot.slo_good_victims,
    hot_p99_ms:.hot.victim.e2e_ms.p99,
    normal_p50_ms:.normal.victim.e2e_ms.p50,
    route_counts, edge_counts
  }' "${result_dir}/result.json"
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
  echo "TEMPO-GO C7 joint arm receipt: ${RESULT_ROOT}/${ARM_ONLY}/result.json"
  exit 0
fi

for index in "${!CONTRACT_ARMS[@]}"; do
  run_arm "${CONTRACT_ARMS[index]}" "${index}"
done

ANALYZE=(
  "${REPO_ROOT}/.vllm_venv/bin/python"
  -m eval.sota_4node.analyze_tempo_go_c7_joint_control
  --contract "${CONTRACT}"
)
for arm in "${CONTRACT_ARMS[@]}"; do
  ANALYZE+=(--result "${arm}=${RESULT_ROOT}/${arm}/result.json")
done
ANALYZE+=(--output "${RESULT_ROOT}/analysis.json")
"${ANALYZE[@]}"
[[ -s "${RESULT_ROOT}/analysis.json" ]]
jq '{
  strongest_fixed_arm,
  full_effects,
  full_vs_strongest_fixed_robustness_gate,
  full_vs_predictor_robustness_gate,
  full_vs_queue_gpu_robustness_gate,
  cross_layer_incremental_gate,
  full_uses_both_local_and_remote,
  full_switches_away_from_hot_receiver,
  c7_joint_control_discovery_positive
}' "${RESULT_ROOT}/analysis.json"
echo "TEMPO-GO C7 joint-control receipt: ${RESULT_ROOT}/analysis.json"
