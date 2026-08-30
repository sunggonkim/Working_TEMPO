#!/usr/bin/env bash
set -euo pipefail

[[ "${TEMPO_GO_C7_ACTUAL_VLLM_FANIN_APPROVED:-}" == YES ]] || exit 2
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 2
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_JOB_NODES:-}}" == 4 ]] || exit 2
[[ -z "${SHIFTER_RUNTIME:-}${SHIFTER_IMAGE:-}${UDI:-}${CRAY_ROOTFS:-}${SLURM_CONTAINER:-}" ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONTRACT="${TEMPO_GO_C7_ACTUAL_VLLM_FANIN_CONTRACT:-${REPO_ROOT}/eval/sota_4node/tempo_go_c7_actual_vllm_fanin_contract_v1.json}"
CONTRACT=$(realpath -e -- "${CONTRACT}")
case "${CONTRACT}" in "${REPO_ROOT}/"*) ;; *) exit 2 ;; esac
[[ "$(jq -er '.schema' "${CONTRACT}")" == tempo-go-c7-actual-vllm-fanin-contract-v1 ]]
[[ "$(jq -er '.claim_boundary.performance_claim_allowed' "${CONTRACT}")" == false ]]

SOURCE_REL=$(jq -er '.actual_vllm_fanin.source_workload.path' "${CONTRACT}")
PROFILE_REL=$(jq -er '.actual_vllm_fanin.profile.path' "${CONTRACT}")
SOURCE_WORKLOAD=$(realpath -e -- "${REPO_ROOT}/${SOURCE_REL}")
PROFILE=$(realpath -e -- "${REPO_ROOT}/${PROFILE_REL}")
[[ "$(sha256sum "${SOURCE_WORKLOAD}" | awk '{print $1}')" == "$(jq -er '.actual_vllm_fanin.source_workload.sha256' "${CONTRACT}")" ]]
[[ "$(sha256sum "${PROFILE}" | awk '{print $1}')" == "$(jq -er '.actual_vllm_fanin.profile.sha256' "${CONTRACT}")" ]]

RESULT_ROOT="${TEMPO_GO_C7_ACTUAL_VLLM_FANIN_RESULT_DIR:-${REPO_ROOT}/results/tempo_go_c7_actual_vllm_fanin_job_${SLURM_JOB_ID}}"
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
export TEMPO_GO_C7_ACTUAL_VLLM_FANIN_CONTRACT="${CONTRACT}"
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
PORT_SLOT=$((1900 + SLURM_JOB_ID % 20))
REQUEST_RATE=$(jq -er '.actual_vllm_fanin.victim.offered_rate_per_s' "${CONTRACT}")
MAX_WORKERS=$(jq -er '.actual_vllm_fanin.max_workers' "${CONTRACT}")

timeout --foreground --signal=TERM --kill-after=30s 1500s \
  /usr/bin/srun --overlap --exact \
  --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
  --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
  --time=00:24:00 --export=ALL \
  --output="${RESULT_ROOT}/slurm-node-%N.stdout.log" \
  --error="${RESULT_ROOT}/slurm-node-%N.stderr.log" \
  bash "${SCRIPT_DIR}/c7_actual_vllm_fanin_node_entry.sh" \
  "${REPO_ROOT}" "${RESULT_ROOT}" "${SOURCE_WORKLOAD}" "${HOSTS_CSV}" \
  "${PORT_SLOT}" "${REQUEST_RATE}" "${MAX_WORKERS}" 128 8 3000 150 8000

[[ -s "${RESULT_ROOT}/result.json" ]]
jq '.analysis | {
  c7_actual_vllm_fanin_qualification_pass,
  first_material_knee_rate_per_s,
  actual_two_prefill_to_one_decoder_fanin,
  material_independent_victim_degradation,
  joint_control_discovery_run_allowed
}' "${RESULT_ROOT}/result.json"
echo "TEMPO-GO C7 actual-vLLM fan-in receipt: ${RESULT_ROOT}/result.json"
