#!/usr/bin/env bash
set -u -o pipefail

# Run only inside an already allocated Perlmutter interactive job.  This
# launcher never calls sbatch/salloc/scancel and uses one explicit srun step
# per immutable arm.
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
: "${SLURM_JOB_ID:?existing interactive allocation is required}"
: "${SLURM_JOB_NODELIST:?allocation nodelist is required}"

if [[ "${SLURM_JOB_NUM_NODES:-0}" -ne 4 ]]; then
  echo "requires the existing four-node allocation" >&2
  exit 2
fi

WORKLOAD=${TEMPO_REAL_TRACE_WORKLOAD:?set TEMPO_REAL_TRACE_WORKLOAD}
POPULATION=${TEMPO_REAL_TRACE_POPULATION_MANIFEST:?set TEMPO_REAL_TRACE_POPULATION_MANIFEST}
BUSINESS=${TEMPO_REAL_TRACE_BUSINESS_PROFILE:?set TEMPO_REAL_TRACE_BUSINESS_PROFILE}
PROFILE=${TEMPO_REAL_TRACE_ELASTIC_PROFILE:?set TEMPO_REAL_TRACE_ELASTIC_PROFILE}
GLOBAL_PROFILE=${TEMPO_REAL_TRACE_GLOBAL_PROFILE:?set TEMPO_REAL_TRACE_GLOBAL_PROFILE}
ENDPOINT_PROFILE=${TEMPO_REAL_TRACE_ENDPOINT_PROFILE:?set TEMPO_REAL_TRACE_ENDPOINT_PROFILE}
RESULT_ROOT=${TEMPO_REAL_TRACE_RESULT_ROOT:?set TEMPO_REAL_TRACE_RESULT_ROOT}

case "${RESULT_ROOT}/" in
  "${REPO_ROOT}/results/"*) ;;
  *) echo "result root must be below repository results/" >&2; exit 2 ;;
esac
[[ -f "${WORKLOAD}" && -f "${POPULATION}" && -f "${BUSINESS}" && -f "${PROFILE}" ]]
[[ -f "${GLOBAL_PROFILE}" && -f "${ENDPOINT_PROFILE}" ]]
[[ ! -e "${RESULT_ROOT}" ]]
mkdir -p -- "${RESULT_ROOT}"

module reset
module load pytorch/2.8.0
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
export NCCL_NET="${NCCL_NET:-AWS Libfabric}"
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}"
export TEMPO_ELASTIC_PD_PROFILE_SCOPE=screen_only
export TEMPO_ELASTIC_PD_PROFILE="${PROFILE}"
export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_LMCACHE_NIXL_BACKEND=UCX
export TEMPO_VLLM_MAX_NUM_SEQS=16
export TEMPO_VLLM_DECODER_PREFIX_CACHING=0
export TEMPO_VLLM_SCHEDULING_POLICY=priority
export TEMPO_PD_REMOTE_CATCHUP_PRIORITY=0
export TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY=-2
export TEMPO_PD_LONG_REMOTE_CATCHUP_PRIORITY=0
export TEMPO_PD_MEDIUM_REMOTE_CATCHUP_PRIORITY=0
export TEMPO_PD_MEDIAN_GUARD_PRIORITY=0

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ "${#HOSTS[@]}" -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
REQUEST_RATE=$("${REPO_ROOT}/.vllm_venv/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["arrival"]["effective_offered_rate_per_s"])' "${POPULATION}")
PROFILE_FINGERPRINT=$(jq -er '.fingerprint_sha256' "${GLOBAL_PROFILE}")
ENDPOINT_MANIFEST_SHA=$(jq -er '.workload_manifest_sha256' "${ENDPOINT_PROFILE}")
export TEMPO_GO_PROFILE="${GLOBAL_PROFILE}"
export TEMPO_GO_PROFILE_SHA256="${PROFILE_FINGERPRINT}"
export TEMPO_GO_ELASTIC_PROFILE="${PROFILE}"
export TEMPO_GO_ENDPOINT_PROFILE="${ENDPOINT_PROFILE}"
export TEMPO_PD_ENDPOINT_SERVICE_PROFILE="${ENDPOINT_PROFILE}"
export TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256="${ENDPOINT_MANIFEST_SHA}"
export TEMPO_PD_REMOTE_DECODE_PLACEMENT=global_mesh
export TEMPO_PD_ENDPOINT_FEEDBACK_MODE=adaptive
export TEMPO_PD_ENDPOINT_ROUTING_POLICY=semantic_epoch_v1
export TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK=1

run_arm() {
  local arm=$1
  local result_dir="${RESULT_ROOT}/${arm}"
  local slot=$2
  export TEMPO_GO_TOKENIZER_URL="http://${HOSTS[1]}:$((14000 + slot))"
  if [[ "${arm}" == "local" || "${arm}" == "remote" ]]; then
    export TEMPO_PD_REMOTE_DECODE_PLACEMENT=paired
    # Fixed baselines do not consume the global priority service-lane
    # commitment; retain the same vLLM scheduler but disable TEMPO actions.
    export TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY=0
  else
    export TEMPO_PD_REMOTE_DECODE_PLACEMENT=global_mesh
    export TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY=-2
  fi
  if [[ "${arm}" == "remote" || "${arm}" == "predictor" ]]; then
    # The official trace contains repeated prefixes; remote LMCache is
    # expected to observe those natural hits rather than reject them as a
    # synthetic cold-only experiment.  Dynamic arms may select remote too.
    export TEMPO_PD_BENCHMARK_COLD_MEASURED=0
  else
    export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
  fi
  [[ ! -e "${result_dir}" ]]
  mkdir -p -- "${result_dir}"
  echo "START arm=${arm} result=${result_dir}"
  /usr/bin/srun --jobid="${SLURM_JOB_ID}" --overlap --exact \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 --distribution=block:block \
    --gpus-per-task=4 --gpu-bind=none --cpus-per-task=128 --cpu-bind=cores \
    --kill-on-bad-exit=1 --wait=10 --time=00:29:00 --export=ALL \
    --output="${result_dir}/slurm-node-%N.stdout.log" \
    --error="${result_dir}/slurm-node-%N.stderr.log" \
    bash -c 'exec "$1/.vllm_venv/bin/python" -m \
      eval.sota_4node.vllm_lmcache_tempo_go_real_trace_node \
      --repo-root "$1" --result-dir "$2" --scout-root "$3" \
      --node-index "${SLURM_NODEID}" --hosts "$4" --port-slot "$5" \
      --request-rate "$6" --max-workers "$7" --output-tokens 128 \
      --population-manifest "$8" --business-profile "$9" \
      --wire-arm "${10}" --profile "${11}"' bash \
    "${REPO_ROOT}" "${result_dir}" "${WORKLOAD}" "${HOSTS_CSV}" \
    "${slot}" "${REQUEST_RATE}" 64 "${POPULATION}" "${BUSINESS}" \
    "${arm}" "${PROFILE}"
  [[ -s "${result_dir}/result.json" ]]
  jq '{schema,wire_arm,request_count,performance_claim_allowed,raw}' \
    "${result_dir}/result.json"
}

# Preserve the first measured real-trace comparison as one fixed order.  The
# workload/profile/arrival contract is unchanged across arms.
if [[ -n "${TEMPO_REAL_TRACE_ARMS:-}" ]]; then
  read -r -a ARMS <<< "${TEMPO_REAL_TRACE_ARMS}"
else
  ARMS=(tempo local remote predictor)
fi
[[ "${#ARMS[@]}" -gt 0 ]]
for index in "${!ARMS[@]}"; do
  if (( index > 0 )); then
    sleep 10
    /usr/bin/srun --jobid="${SLURM_JOB_ID}" --overlap --exact \
      --nodes=4 --ntasks=4 --ntasks-per-node=1 --gpus-per-task=4 \
      --gpu-bind=none --cpus-per-task=1 --cpu-bind=none --time=00:02:00 \
      --output=/dev/null --error=/dev/null bash -c \
      'test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"'
  fi
  run_arm "${ARMS[index]}" "$((2300 + index * 40))"
done

echo "REAL_TRACE_CAMPAIGN_COMPLETE=${RESULT_ROOT}"
