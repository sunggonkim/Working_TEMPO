#!/usr/bin/env bash
set -euo pipefail

[[ "${TEMPO_GO_C9_CAUSAL_BURST_APPROVED:-}" == YES ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/require_perlmutter_4node_4h_interactive.sh"

# The outer attach step may span all four allocated nodes so nested native
# steps can use the complete allocation, but the campaign runner itself is a
# single-owner control process.  Without this guard every outer task races on
# the same result root and launches duplicate cleanup/finalization logic.
if [[ "${SLURM_PROCID:-0}" != 0 ]]; then
  exit 0
fi

REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONTRACT=$(realpath -e -- "${TEMPO_GO_C9_CAUSAL_BURST_CONTRACT:-${REPO_ROOT}/results/tempo_go_c9_causal_burst_current_source.json}")
BASE_CONTRACT=$(realpath -e -- "${REPO_ROOT}/$(jq -er '.system_under_test.base_contract' "${CONTRACT}")")
NODE_ENTRY_REL=$(jq -er '.system_under_test.node_entry' "${CONTRACT}")
case "${NODE_ENTRY_REL}" in
  eval/sota_4node/c8_independent_validation_node_entry.sh|eval/sota_4node/c8_dual_regime_node_entry.sh|eval/sota_4node/c9_gate_node_entry.sh) ;;
  *) exit 2 ;;
esac
NODE_ENTRY=$(realpath -e -- "${REPO_ROOT}/${NODE_ENTRY_REL}")
[[ "$(jq -er '.schema' "${CONTRACT}")" == tempo-go-c9-causal-burst-discovery-v1 ]]
[[ "$(jq -er '.claim_boundary.discovery_only' "${CONTRACT}")" == true ]]
[[ "$(jq -er '.execution.one_campaign_no_retry' "${CONTRACT}")" == true ]]
[[ "$(sha256sum "${BASE_CONTRACT}" | awk '{print $1}')" == "$(jq -er '.system_under_test.base_contract_sha256' "${CONTRACT}")" ]]

# Every population arm owns all four nodes and arms run sequentially. Refuse a
# short outer step before creating a result root or launching any GPU child.
# Older frozen contracts use the same explicit 150-minute default.
duration_to_seconds() {
  local value="$1"
  if [[ "${value}" =~ ^([0-9]+)-([0-9]{2}):([0-9]{2}):([0-9]{2})$ ]]; then
    echo $((10#${BASH_REMATCH[1]} * 86400 + 10#${BASH_REMATCH[2]} * 3600 + 10#${BASH_REMATCH[3]} * 60 + 10#${BASH_REMATCH[4]}))
  elif [[ "${value}" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
    echo $((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
  elif [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo $((10#${value} * 60))
  else
    return 1
  fi
}

MINIMUM_OUTER_TIME_S=$(jq -er '.execution.minimum_outer_time_s // 9000' "${CONTRACT}")
[[ "${MINIMUM_OUTER_TIME_S}" =~ ^[0-9]+$ ]]
(( MINIMUM_OUTER_TIME_S > 0 ))
if [[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]]; then
  PARENT_STEP_ID="${SLURM_JOB_ID}.${SLURM_STEP_ID}"
  PARENT_STEP_INFO=$(scontrol show step "${PARENT_STEP_ID}" --oneliner 2>/dev/null)
  [[ -n "${PARENT_STEP_INFO}" ]] || {
    echo "cannot inspect outer step ${PARENT_STEP_ID} time budget" >&2
    exit 1
  }
  PARENT_TIME_LIMIT=$(sed -n 's/.* TimeLimit=\([^ ]*\).*/\1/p' <<<" ${PARENT_STEP_INFO}")
  PARENT_ELAPSED=$(sed -n 's/.* Elapsed=\([^ ]*\).*/\1/p' <<<" ${PARENT_STEP_INFO}")
  PARENT_START_TIME=$(sed -n 's/.* StartTime=\([^ ]*\).*/\1/p' <<<" ${PARENT_STEP_INFO}")
  if [[ "${PARENT_TIME_LIMIT}" != "UNLIMITED" ]]; then
    PARENT_TIME_LIMIT_S=$(duration_to_seconds "${PARENT_TIME_LIMIT}") || exit 1
    if [[ -n "${PARENT_ELAPSED}" ]]; then
      PARENT_ELAPSED_S=$(duration_to_seconds "${PARENT_ELAPSED}") || exit 1
    else
      # Active Perlmutter step records may omit Elapsed, and squeue -j
      # job.step reports allocation age rather than step age. Compute the
      # exact age from scontrol's authoritative StartTime instead.
      [[ -n "${PARENT_START_TIME}" ]]
      PARENT_START_EPOCH=$(date --date="${PARENT_START_TIME}" +%s) || exit 1
      PARENT_NOW_EPOCH=$(date +%s)
      PARENT_ELAPSED_S=$((PARENT_NOW_EPOCH - PARENT_START_EPOCH))
      (( PARENT_ELAPSED_S >= 0 ))
    fi
    PARENT_REMAINING_S=$((PARENT_TIME_LIMIT_S - PARENT_ELAPSED_S))
    if (( PARENT_REMAINING_S < MINIMUM_OUTER_TIME_S )); then
      echo "C9 campaign requires >=${MINIMUM_OUTER_TIME_S}s outer time; remaining=${PARENT_REMAINING_S}s step=${PARENT_STEP_ID}" >&2
      exit 2
    fi
  fi
fi

# Freeze the C9 parent/child launcher boundary before any GPU step starts.
# A current-source contract carries hashes for the nested network launcher and
# node entry; refuse a stale contract instead of discovering drift after
# vLLM has already loaded four GPUs.
while IFS=$'\t' read -r relative expected; do
  source_path=$(realpath -e -- "${REPO_ROOT}/${relative}")
  [[ "${source_path}" == "${REPO_ROOT}/"* ]]
  [[ "$(sha256sum "${source_path}" | awk '{print $1}')" == "${expected}" ]]
done < <(jq -er '.provenance.source_inventory // {} | to_entries[] | [.key, .value] | @tsv' "${CONTRACT}")
bash -n "${SCRIPT_DIR}/run_lmcache_nixl_contention_2node_in_allocation.sh" "${NODE_ENTRY}"

# The outer attach step is only a control shell.  If this runner is entered
# from an explicit srun parent, require that parent to use no_vni so it does
# not consume a third Slingshot network slot.  The actual vLLM and native
# NCCL/LMCache child steps below retain job_vni.
case "${SLURM_STEP_ID:-}" in
  ""|batch|extern) ;;
  *)
    PARENT_STEP_ID="${SLURM_JOB_ID}.${SLURM_STEP_ID}"
    mapfile -t PARENT_STEP_NETWORKS < <(
      scontrol show step "${PARENT_STEP_ID}" --oneliner \
        | awk -v step_id="${PARENT_STEP_ID}" '
            $1 == ("StepId=" step_id) {
              for (field = 1; field <= NF; field++) {
                if (substr($field, 1, 8) == "Network=") {
                  print substr($field, 9)
                }
              }
            }'
    )
    [[ "${#PARENT_STEP_NETWORKS[@]}" -eq 1 ]] || {
      echo "could not identify exactly one orchestration parent step ${PARENT_STEP_ID}" >&2
      exit 1
    }
    PARENT_STEP_NETWORK="${PARENT_STEP_NETWORKS[0]}"
    [[ "${PARENT_STEP_NETWORK}" == no_vni ]] || {
      echo "C9 orchestration parent must use --network=no_vni; got ${PARENT_STEP_NETWORK:-<unknown>}" >&2
      exit 1
    }
    ;;
esac

RESULT_ROOT="${TEMPO_GO_C9_CAUSAL_BURST_RESULT_DIR:-${REPO_ROOT}/results/tempo_go_c9_causal_burst_job_${SLURM_JOB_ID}}"
case "${RESULT_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ ! -e "${RESULT_ROOT}" ]]
mkdir -p -- "${RESULT_ROOT}"

SOURCE_REL=$(jq -er '.joint_control.source_workload.path' "${BASE_CONTRACT}")
PROFILE_REL=$(jq -er '.joint_control.profile.path' "${BASE_CONTRACT}")
GLOBAL_PROFILE_REL=$(jq -er '.joint_control.global_profile.path' "${BASE_CONTRACT}")
SOURCE_WORKLOAD=$(realpath -e -- "${REPO_ROOT}/${SOURCE_REL}")
PROFILE=$(realpath -e -- "${REPO_ROOT}/${PROFILE_REL}")
GLOBAL_PROFILE=$(realpath -e -- "${REPO_ROOT}/${GLOBAL_PROFILE_REL}")
[[ "$(sha256sum "${SOURCE_WORKLOAD}" | awk '{print $1}')" == "$(jq -er '.joint_control.source_workload.sha256' "${BASE_CONTRACT}")" ]]
[[ "$(sha256sum "${PROFILE}" | awk '{print $1}')" == "$(jq -er '.joint_control.profile.sha256' "${BASE_CONTRACT}")" ]]
[[ "$(sha256sum "${GLOBAL_PROFILE}" | awk '{print $1}')" == "$(jq -er '.joint_control.global_profile.sha256' "${BASE_CONTRACT}")" ]]

module reset
module load pytorch/2.8.0
[[ "${NCCL_NET:-}" == "AWS Libfabric" ]] || {
  echo "expected NERSC NCCL AWS Libfabric transport, got ${NCCL_NET:-<unset>}" >&2
  exit 1
}
unset NCCL_IB_DISABLE NCCL_RAS_ADDR
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
export TEMPO_GO_C8_DUAL_REGIME_CONTRACT="${BASE_CONTRACT}"
export TEMPO_ELASTIC_PD_PROFILE="${PROFILE}"
export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY=1
export TEMPO_VLLM_MAX_NUM_SEQS=16
export TEMPO_VLLM_DECODER_PREFIX_CACHING=0
export TEMPO_LMCACHE_NIXL_BACKEND=UCX
unset TEMPO_CXI_BACKGROUND_DUTY_CYCLE TEMPO_CXI_BACKGROUND_START_FILE

if [[ -n "${TEMPO_GO_C9_HOSTS_CSV:-}" ]]; then
  IFS=, read -r -a HOSTS <<< "${TEMPO_GO_C9_HOSTS_CSV}"
else
  ALLOCATION_NODELIST=$(
    scontrol show job "${SLURM_JOB_ID}" --oneliner \
      | sed -n 's/.* NodeList=\([^ ]*\).*/\1/p'
  )
  [[ -n "${ALLOCATION_NODELIST}" ]]
  mapfile -t HOSTS < <(scontrol show hostnames "${ALLOCATION_NODELIST}")
fi
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
export TEMPO_GO_C9_HOSTS_CSV="${HOSTS_CSV}"
REQUEST_RATE=$(jq -er '.joint_control.victim.offered_rate_per_s' "${BASE_CONTRACT}")
MAX_WORKERS=$(jq -er '.joint_control.max_workers' "${BASE_CONTRACT}")
CONTRACT_SHA=$(sha256sum "${CONTRACT}" | awk '{print $1}')
BASE_CONTRACT_SHA=$(sha256sum "${BASE_CONTRACT}" | awk '{print $1}')
# The same allocation may be used for a new, explicitly separated discovery
# result after a setup failure.  Reusing only ``job_id + arm_index`` can leave
# a ZMQ listener in TIME_WAIT or a slow-draining prior process on the same
# NIXL port (the v15 rank-4 ``tcp://*:37030`` collision).  Derive the native
# rendezvous ports from the unique result root instead.  The receipt records
# the resulting ports, and the result-root existence guard makes reuse of the
# same seed impossible within this runner.  The endpoint probe helper maps a
# slot to ``30000 + slot`` and accepts ports below 32768; the seven-arm contract
# reaches slot 2660, so keep the per-root offset below 108 rather than allowing
# a hash offset to make the final arm invalid.
RESULT_ROOT_PORT_HASH=$(printf '%s' "${RESULT_ROOT}" | sha256sum | cut -c1-6)
RESULT_ROOT_PORT_OFFSET=$((16#${RESULT_ROOT_PORT_HASH} % 100))

jq -n \
  --arg schema tempo-go-c9-causal-burst-attempt-v1 \
  --arg status running \
  --arg job_id "${SLURM_JOB_ID}" \
  --arg contract "${CONTRACT}" \
  --arg contract_sha256 "${CONTRACT_SHA}" \
  --arg base_contract "${BASE_CONTRACT}" \
  --arg base_contract_sha256 "${BASE_CONTRACT_SHA}" \
  --arg hosts "${HOSTS_CSV}" \
  '{schema:$schema,status:$status,slurm_job_id:($job_id|tonumber),contract:$contract,contract_sha256:$contract_sha256,base_contract:$base_contract,base_contract_sha256:$base_contract_sha256,hosts:($hosts|split(",")),one_campaign_no_retry:true,discovery_only:true}' \
  >"${RESULT_ROOT}/attempt.json"

CURRENT_COJOB_PIDS=()
CURRENT_STOP_FILES=()
CURRENT_INFERENCE_PID=""
CURRENT_STEP_NAMES=()
cancel_owned_steps() {
  local step_name step_id
  local -a owned_steps=()
  for step_name in "${CURRENT_STEP_NAMES[@]}"; do
    mapfile -t owned_steps < <(
      scontrol show step "${SLURM_JOB_ID}" --oneliner 2>/dev/null \
        | awk -v wanted="${step_name}" '
            $1 ~ /^StepId=/ {
              for (field = 1; field <= NF; field++) {
                if ($field == ("Name=" wanted)) {
                  split($1, parts, "=")
                  print parts[2]
                }
              }
            }'
    )
    for step_id in "${owned_steps[@]}"; do
      [[ "${step_id}" =~ ^${SLURM_JOB_ID//./\\.}\\.[0-9]+$ ]] || continue
      scancel "${step_id}" 2>/dev/null || true
    done
  done
}
cleanup_cojob() {
  local stop_file cojob_pid
  for stop_file in "${CURRENT_STOP_FILES[@]}"; do
    : >"${stop_file}" 2>/dev/null || true
  done
  for cojob_pid in "${CURRENT_COJOB_PIDS[@]}"; do
    if [[ -n "${cojob_pid}" ]] && kill -0 "${cojob_pid}" 2>/dev/null; then
      kill -TERM "${cojob_pid}" 2>/dev/null || true
    fi
  done
  for cojob_pid in "${CURRENT_COJOB_PIDS[@]}"; do
    if [[ -n "${cojob_pid}" ]]; then
      wait "${cojob_pid}" 2>/dev/null || true
    fi
  done
  if [[ -n "${CURRENT_INFERENCE_PID}" ]] && kill -0 "${CURRENT_INFERENCE_PID}" 2>/dev/null; then
    kill -TERM "${CURRENT_INFERENCE_PID}" 2>/dev/null || true
    wait "${CURRENT_INFERENCE_PID}" 2>/dev/null || true
  fi
  cancel_owned_steps
  CURRENT_COJOB_PIDS=()
  CURRENT_STOP_FILES=()
  CURRENT_INFERENCE_PID=""
  CURRENT_STEP_NAMES=()
}

write_preperformance_failure() {
  local failure_stage="$1"
  local failure_detail="$2"
  local receipt="${block_dir}/block_failure_receipt.json"
  jq -n \
    --arg schema tempo-go-c9-causal-burst-block-failure-v1 \
    --arg block "${block_name}" \
    --arg arm "${arm}" \
    --arg failure_stage "${failure_stage}" \
    --arg failure_detail "${failure_detail}" \
    --arg failure_receipt "${receipt}" \
    '{schema:$schema,block:$block,arm:$arm,failure_stage:$failure_stage,failure_detail:$failure_detail,inference_result_exists:false,measured_arm_retried:false,performance_result:false,failure_receipt:$failure_receipt}' \
    >"${receipt}"
  jq -n \
    --arg schema tempo-go-c9-causal-burst-failed-attempt-v1 \
    --arg status failed_before_performance_result \
    --arg block "${block_name}" \
    --arg contract "${CONTRACT}" \
    --arg contract_sha256 "${CONTRACT_SHA}" \
    --arg base_contract "${BASE_CONTRACT}" \
    --arg base_contract_sha256 "${BASE_CONTRACT_SHA}" \
    --arg failure_receipt "${receipt}" \
    '{schema:$schema,status:$status,failed_block:$block,contract:$contract,contract_sha256:$contract_sha256,base_contract:$base_contract,base_contract_sha256:$base_contract_sha256,failure_receipt:$failure_receipt,one_campaign_no_retry:true,measured_arm_retried:false,performance_result:false}' \
    >"${RESULT_ROOT}/failed_attempt.json"
}
write_interrupted_attempt() {
  local signal_name="${1:-unknown}"
  local root="${RESULT_ROOT:-}"
  [[ -n "${root}" && -d "${root}" ]] || return 0
  [[ ! -e "${root}/completed_attempt.json" ]] || return 0
  [[ ! -e "${root}/failed_attempt.json" ]] || return 0
  [[ ! -e "${root}/interrupted_attempt.json" ]] || return 0
  jq -n \
    --arg schema tempo-go-c9-causal-burst-interrupted-attempt-v1 \
    --arg status interrupted \
    --arg signal "${signal_name}" \
    --arg block "${block_name:-not_started}" \
    --arg arm "${arm:-not_started}" \
    --arg contract "${CONTRACT:-}" \
    --arg contract_sha256 "${CONTRACT_SHA:-}" \
    --arg reason "outer step interrupted before seven-arm finalization" \
    '{schema:$schema,status:$status,signal:$signal,current_block:$block,current_arm:$arm,contract:$contract,contract_sha256:$contract_sha256,reason:$reason,performance_claim_allowed:false,one_campaign_no_retry:true}' \
    >"${root}/interrupted_attempt.json"
}
handle_campaign_signal() {
  local signal_name="${1:-unknown}"
  write_interrupted_attempt "${signal_name}"
  cleanup_cojob
  exit 143
}
trap cleanup_cojob EXIT
trap 'handle_campaign_signal TERM' TERM
trap 'handle_campaign_signal INT' INT

mapfile -t BLOCK_NAMES < <(jq -er '.execution.order[].name' "${CONTRACT}")
mapfile -t ARMS < <(jq -er '.execution.order[].arm' "${CONTRACT}")
mapfile -t PORT_SLOTS < <(jq -er '.execution.order[].port_slot' "${CONTRACT}")
[[ ${#BLOCK_NAMES[@]} -gt 0 ]]
[[ ${#BLOCK_NAMES[@]} -eq ${#ARMS[@]} && ${#ARMS[@]} -eq ${#PORT_SLOTS[@]} ]]

for index in "${!BLOCK_NAMES[@]}"; do
  if (( index > 0 )); then
    sleep "$(jq -er '.execution.cooldown_s' "${CONTRACT}")"
    # A prior multi-process vLLM lifecycle can outlive the srun host by a few
    # seconds.  Fail closed before starting another measured arm; never kill
    # an unknown GPU process and never retry an arm.
    /usr/bin/srun --jobid="${SLURM_JOB_ID}" --overlap \
      --nodes=4 --ntasks=4 --ntasks-per-node=1 \
      --gpus-per-task=4 --gpu-bind=none --cpus-per-task=1 \
      --cpu-bind=none \
      --network=no_vni \
      --time=00:02:00 --output=/dev/null --error=/dev/null \
      bash -c '
        attempt=0
        while (( attempt < 120 )); do
          if test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"; then
            exit 0
          fi
          attempt=$((attempt + 1))
          sleep 1
        done
        exit 1
      '
  fi
  block_name="${BLOCK_NAMES[index]}"
  arm="${ARMS[index]}"
  port_slot=$((10#${PORT_SLOTS[index]} + RESULT_ROOT_PORT_OFFSET))
  block_dir="${RESULT_ROOT}/${block_name}"
  inference_dir="${block_dir}/inference"
  cojob_dir="${block_dir}/cojob"
  cojob_pair1_dir="${block_dir}/cojob-pair-1"
  mkdir -p -- "${inference_dir}" "${cojob_dir}" "${cojob_pair1_dir}"
  stop_file="${cojob_dir}/stop"
  ready_file="${cojob_dir}/nixl-ready"
  stop_pair1_file="${cojob_pair1_dir}/stop"
  ready_pair1_file="${cojob_pair1_dir}/nixl-ready"
  cojob_dirs=("${cojob_dir}" "${cojob_pair1_dir}")
  stop_files=("${stop_file}" "${stop_pair1_file}")
  ready_files=("${ready_file}" "${ready_pair1_file}")
  cojob_hosts=(
    "${HOSTS[0]},${HOSTS[1]}"
    "${HOSTS[2]},${HOSTS[3]}"
  )
  start_file="${inference_dir}/measurement-start"
  CURRENT_VLLM_STEP_NAME="c9-vllm-${index}-${SLURM_JOB_ID}"
  CURRENT_COJOB_STEP_NAMES=(
    "c9-causal-cojob-p0-${index}-${SLURM_JOB_ID}"
    "c9-causal-cojob-p1-${index}-${SLURM_JOB_ID}"
  )
  CURRENT_STEP_NAMES=(
    "${CURRENT_VLLM_STEP_NAME}"
    "${CURRENT_COJOB_STEP_NAMES[@]}"
  )
  [[ ! -e "${start_file}" ]]
  epoch="slurm-${SLURM_JOB_ID}-c9-causal-${index}"
  communicator="c9-causal-burst-${index}"
  export TEMPO_GO_CROSS_LAYER_EPOCH="${epoch}"
  export TEMPO_GO_NCCL_COMMUNICATOR_ID="${communicator}"
  export TEMPO_GO_NCCL_TELEMETRY_PATH="${cojob_dir}/nccl_observer.json"
  # Run one real NCCL/LMCache co-job on each physical P/D pair.  The two
  # observers are exposed through the same pair-indexed template consumed by
  # vLLM; no pair-1 evidence is synthesized when its child is unavailable.
  ln -s -- "nccl_observer.json" "${cojob_dir}/nccl_observer_pair-0.json"
  ln -s -- "../cojob-pair-1/nccl_observer.json" "${cojob_dir}/nccl_observer_pair-1.json"
  export TEMPO_GO_NCCL_OBSERVER_PATH_TEMPLATE="${cojob_dir}/nccl_observer_pair-{pair}.json"
  export TEMPO_GO_NCCL_OBSERVER_MAX_AGE_MS
  TEMPO_GO_NCCL_OBSERVER_MAX_AGE_MS=$(jq -er '.burst.observer_max_age_ms' "${CONTRACT}")
  export TEMPO_GO_C8_DUAL_REGIME_ARM="${arm}"

  # Start the actual vLLM P/D lifecycle first.  Each node publishes a marker
  # only after its health endpoint is live and then waits on start_file, so
  # model loading cannot consume the short causal burst before the victim is
  # ready.  The co-job intentionally does not wait on this marker: it must
  # execute a bootstrap block and publish a fresh observer snapshot before
  # the victim release gate below.
  inference_timeout=$(jq -er '.execution.inference_timeout_s' "${CONTRACT}")
  export TEMPO_GO_C9_INFERENCE_START_FILE="${start_file}"
  export TEMPO_GO_C9_INFERENCE_START_TIMEOUT_S="${inference_timeout}"
  # This is the actual vLLM P/D child step and must retain native job_vni.
  # The outer orchestration attach is checked above for no_vni; only this
  # child and the native NCCL/LMCache co-job consume Slingshot network slots.
  set +e
  timeout --foreground --signal=TERM --kill-after=30s "${inference_timeout}s" \
    /usr/bin/srun --jobid="${SLURM_JOB_ID}" --overlap \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    --distribution=block:block --gpus-per-node=4 --gpu-bind=none \
    --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 \
    --job-name="${CURRENT_VLLM_STEP_NAME}" \
    --network=job_vni --time=00:29:00 --export=ALL \
    --output="${inference_dir}/slurm-node-%N.stdout.log" \
    --error="${inference_dir}/slurm-node-%N.stderr.log" \
    bash "${NODE_ENTRY}" \
    "${REPO_ROOT}" "${inference_dir}" "${SOURCE_WORKLOAD}" "${HOSTS_CSV}" \
    "${port_slot}" "${REQUEST_RATE}" "${MAX_WORKERS}" 128 8 3000 150 8000 \
    &
  CURRENT_INFERENCE_PID=$!
  set -e

  inference_ready=0
  for (( second=0; second<inference_timeout; second++ )); do
    inference_ready=1
    for node_index in 0 1 2 3; do
      if [[ ! -s "${inference_dir}/node-${node_index}-vllm-ready" ]]; then
        inference_ready=0
        break
      fi
    done
    [[ "${inference_ready}" -eq 1 ]] && break
    if ! kill -0 "${CURRENT_INFERENCE_PID}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if [[ "${inference_ready}" -ne 1 ]]; then
    cleanup_cojob
    write_preperformance_failure inference_readiness "vLLM health readiness failed"
    echo "C9 vLLM health readiness failed: ${block_name}" >&2
    exit 1
  fi

  CURRENT_COJOB_PIDS=()
  CURRENT_STOP_FILES=("${stop_files[@]}")
  for pair_index in 0 1; do
    pair_dir="${cojob_dirs[pair_index]}"
    pair_stop_file="${stop_files[pair_index]}"
    pair_hosts="${cojob_hosts[pair_index]}"
    pair_step_name="${CURRENT_COJOB_STEP_NAMES[pair_index]}"
    pair_epoch="${epoch}"
    pair_communicator="${communicator}"
    if [[ "${pair_index}" -eq 1 ]]; then
      pair_epoch="${epoch}-pair-1"
      pair_communicator="${communicator}-pair-1"
    fi
    TEMPO_GO_CROSS_LAYER_COMPONENT_APPROVED=YES \
    TEMPO_GO_C9_COJOB_HOSTS_CSV="${pair_hosts}" \
    TEMPO_GO_CROSS_LAYER_RESULT_DIR="${pair_dir}" \
    TEMPO_GO_CROSS_LAYER_SRUN_STEP_NAME="${pair_step_name}" \
    TEMPO_GO_CROSS_LAYER_SRUN_OVERLAP=1 \
    TEMPO_GO_CROSS_LAYER_STOP_FILE="${pair_stop_file}" \
    TEMPO_GO_CROSS_LAYER_EPOCH="${pair_epoch}" \
    TEMPO_GO_NCCL_COMMUNICATOR_ID="${pair_communicator}" \
    TEMPO_GO_CROSS_LAYER_MASTER_PORT="$((36000 + index * 512 + pair_index * 64 + RESULT_ROOT_PORT_OFFSET))" \
    TEMPO_GO_CROSS_LAYER_NIXL_PORT_BASE="$((37000 + index * 16 + pair_index * 8 + RESULT_ROOT_PORT_OFFSET))" \
    TEMPO_GO_NCCL_RAS_ADDR="127.0.0.1:$((29500 + index * 2 + pair_index + SLURM_JOB_ID % 100))" \
    TEMPO_GO_CROSS_LAYER_TRAFFIC_PATTERN="$(jq -er '.burst.traffic_pattern' "${CONTRACT}")" \
    TEMPO_GO_CROSS_LAYER_BLOCKS="$(jq -er '.burst.minimum_blocks' "${CONTRACT}")" \
    TEMPO_GO_CROSS_LAYER_MAXIMUM_BLOCKS="$(jq -er '.burst.maximum_blocks' "${CONTRACT}")" \
    TEMPO_GO_CROSS_LAYER_MINIMUM_ACTIVE_DURATION_S="$(jq -er '.burst.minimum_active_duration_s' "${CONTRACT}")" \
    TEMPO_GO_CROSS_LAYER_START_DELAY_S=0 \
    TEMPO_GO_CROSS_LAYER_BLOCK_DELAY_S="$(jq -er '.burst.block_delay_s' "${CONTRACT}")" \
    TEMPO_GO_CROSS_LAYER_REQUESTS="$(jq -er '.burst.requests_per_source' "${CONTRACT}")" \
    TEMPO_GO_CROSS_LAYER_KV_MIB="$(jq -er '.burst.kv_mib_per_request' "${CONTRACT}")" \
    TEMPO_GO_CROSS_LAYER_TOKEN_ITERS="$(jq -er '.burst.token_iters' "${CONTRACT}")" \
    TEMPO_GO_CROSS_LAYER_FOREGROUND_MIB="$(jq -er '.burst.foreground_mib' "${CONTRACT}")" \
    TEMPO_GO_NCCL_TIMEOUT_SECONDS="$(jq -er '.burst.process_group_timeout_s' "${CONTRACT}")" \
    TEMPO_GO_NIXL_TIMEOUT_SECONDS="$(jq -er '.burst.nixl_transfer_timeout_s' "${CONTRACT}")" \
    TEMPO_GO_CROSS_LAYER_TIMEOUT_SECONDS="$(jq -er '.burst.cojob_timeout_s' "${CONTRACT}")" \
    TEMPO_GO_CROSS_LAYER_TIME_LIMIT="$(jq -er '.burst.cojob_time_limit' "${CONTRACT}")" \
      bash "${SCRIPT_DIR}/run_lmcache_nixl_contention_2node_in_allocation.sh" \
      >"${pair_dir}/launcher.stdout.log" 2>"${pair_dir}/launcher.stderr.log" &
    CURRENT_COJOB_PIDS+=("$!")
  done

  ready=0
  readiness_timeout=$(jq -er '.execution.cojob_readiness_timeout_s' "${CONTRACT}")
  for (( second=0; second<readiness_timeout; second++ )); do
    ready=1
    for pair_index in 0 1; do
      [[ -s "${ready_files[pair_index]}" ]] || ready=0
      if [[ -n "${CURRENT_COJOB_PIDS[pair_index]}" ]] \
        && ! kill -0 "${CURRENT_COJOB_PIDS[pair_index]}" 2>/dev/null; then
        ready=0
      fi
    done
    [[ "${ready}" -eq 1 ]] && break
    sleep 1
  done
  if [[ "${ready}" -ne 1 ]]; then
    cleanup_cojob
    write_preperformance_failure cojob_readiness "co-job ready file was not published"
    echo "C9 causal burst co-job readiness failed: ${block_name}" >&2
    exit 1
  fi

  # Peer initialization is not observer readiness.  The producer publishes
  # its first atomic, correctness-checked active window only after the first
  # collective/NIXL block completes.  Do not release the victim workload
  # before that receipt exists: otherwise the co-job can hit its intentional
  # overload timeout before any victim request has a fresh cross-layer
  # sample, making the entire measured arm appear observer-blind.
  observer_ready=0
  for (( second=0; second<readiness_timeout; second++ )); do
    observer_ready=1
    for pair_index in 0 1; do
      observer_path="${cojob_dirs[pair_index]}/nccl_observer.json"
      if [[ ! -s "${observer_path}" ]] \
        || [[ "$(jq -er '.producer_state' "${observer_path}" 2>/dev/null)" != active ]] \
        || [[ "$(jq -er '.correctness_met' "${observer_path}" 2>/dev/null)" != true ]] \
        || (( $(jq -er '.sequence' "${observer_path}" 2>/dev/null) < 1 )); then
        observer_ready=0
      fi
      if [[ -n "${CURRENT_COJOB_PIDS[pair_index]}" ]] \
        && ! kill -0 "${CURRENT_COJOB_PIDS[pair_index]}" 2>/dev/null; then
        observer_ready=0
      fi
    done
    [[ "${observer_ready}" -eq 1 ]] && break
    sleep 1
  done
  if [[ "${observer_ready}" -ne 1 ]]; then
    cleanup_cojob
    write_preperformance_failure observer_readiness "correctness-checked observer snapshot was not published"
    echo "C9 causal burst observer readiness failed: ${block_name}" >&2
    exit 1
  fi

  # The co-job has already published a fresh, correctness-checked bootstrap
  # snapshot; release the victim nodes while the co-job continues producing
  # blocks.  This avoids a start-file cycle while preserving a concurrent
  # measurement window after observer readiness.
  : >"${start_file}"
  set +e
  wait "${CURRENT_INFERENCE_PID}"
  inference_rc=$?
  set -e
  CURRENT_INFERENCE_PID=""

  for pair_stop_file in "${stop_files[@]}"; do
    : >"${pair_stop_file}"
  done
  cojob_rcs=()
  if [[ "${inference_rc}" -ne 0 || ! -s "${inference_dir}/result.json" ]]; then
    # No measured inference result exists.  Stop this launcher's exact child
    # co-job immediately instead of waiting through its pre-burst delay.
    for cojob_pid in "${CURRENT_COJOB_PIDS[@]}"; do
      if [[ -n "${cojob_pid}" ]] && kill -0 "${cojob_pid}" 2>/dev/null; then
        kill -TERM "${cojob_pid}" 2>/dev/null || true
      fi
    done
  fi
  for pair_index in 0 1; do
    cojob_rc=0
    wait "${CURRENT_COJOB_PIDS[pair_index]}" || cojob_rc=$?
    cojob_rcs+=("${cojob_rc}")
  done
  CURRENT_COJOB_PIDS=()
  CURRENT_STOP_FILES=()
  cojob_rc="${cojob_rcs[0]}"
  if [[ "${inference_rc}" -ne 0 || ! -s "${inference_dir}/result.json" ]]; then
    jq -n \
      --arg schema tempo-go-c9-causal-burst-block-failure-v1 \
      --arg block "${block_name}" \
      --arg arm "${arm}" \
      --argjson inference_exit_code "${inference_rc}" \
      --argjson cojob_exit_code "${cojob_rc}" \
      '{schema:$schema,block:$block,arm:$arm,failure_stage:"inference_lifecycle",inference_exit_code:$inference_exit_code,cojob_exit_code:$cojob_exit_code,inference_result_exists:false,measured_arm_retried:false,performance_result:false}' \
      >"${block_dir}/block_failure_receipt.json"
    jq -n \
      --arg schema tempo-go-c9-causal-burst-failed-attempt-v1 \
      --arg status failed_before_performance_result \
      --arg block "${block_name}" \
      --arg contract "${CONTRACT}" \
      --arg contract_sha256 "${CONTRACT_SHA}" \
      --arg base_contract "${BASE_CONTRACT}" \
      --arg base_contract_sha256 "${BASE_CONTRACT_SHA}" \
      --arg failure_receipt "${block_dir}/block_failure_receipt.json" \
      '{schema:$schema,status:$status,failed_block:$block,contract:$contract,contract_sha256:$contract_sha256,base_contract:$base_contract,base_contract_sha256:$base_contract_sha256,failure_receipt:$failure_receipt,one_campaign_no_retry:true,measured_arm_retried:false,performance_result:false}' \
      >"${RESULT_ROOT}/failed_attempt.json"
    exit 1
  fi
  [[ -s "${cojob_dir}/nccl_observer.json" ]]

  cojob_outcome=complete
  if [[ "${cojob_rcs[0]}" -ne 0 ]]; then
    cojob_outcome=overload_timeout
  fi
  cojob_pair_outcomes=()
  for pair_index in 0 1; do
    pair_dir="${cojob_dirs[pair_index]}"
    if [[ "${cojob_rcs[pair_index]}" -eq 0 ]]; then
      [[ -s "${pair_dir}/result.json" ]]
      [[ "$(jq -er '.producer_state' "${pair_dir}/nccl_observer.json")" == complete ]]
      [[ "$(jq -er '.overall_correctness_met' "${pair_dir}/result.json")" == true ]]
      cojob_pair_outcomes+=(complete)
    else
      # Once readiness and at least one correct active snapshot have been
      # published, an official NIXL transfer timeout is the overload outcome
      # being measured.  Validate that outcome independently for both real
      # physical pairs; setup, rendezvous, CUDA, NCCL, or unrelated failures
      # remain campaign-terminal.
      [[ -s "${pair_dir}/cojob_failure.json" ]]
      [[ "$(jq -er '.failure' "${pair_dir}/cojob_failure.json")" == cojob_step_failed ]]
      [[ "$(jq -er '.producer_state' "${pair_dir}/nccl_observer.json")" == active ]]
      [[ "$(jq -er '.correctness_met' "${pair_dir}/nccl_observer.json")" == true ]]
      (( $(jq -er '.sequence' "${pair_dir}/nccl_observer.json") >= 1 ))
      grep -Eq 'official LMCache/NIXL batched_write exceeded [0-9.]+s' \
        "${pair_dir}"/cojob-rank-*.stderr.log
      cojob_pair_outcomes+=(overload_timeout)
    fi
    [[ -s "${pair_dir}/native_transport_receipt.json" ]]
  done
  cojob_pair_outcomes_json=$(printf '%s\n' "${cojob_pair_outcomes[@]}" | jq -Rsc 'split("\n") | map(select(length > 0))')
  cojob_dirs_json=$(printf '%s\n' "${cojob_dirs[@]}" | jq -Rsc 'split("\n") | map(select(length > 0))')
  transport_receipts_json=$(printf '%s\n' "${cojob_dirs[@]}" | sed 's#$#/native_transport_receipt.json#' | jq -Rsc 'split("\n") | map(select(length > 0))')

  jq -n \
    --arg schema tempo-go-c9-causal-burst-block-execution-v1 \
    --arg block "${block_name}" \
    --arg arm "${arm}" \
    --arg cojob_outcome "${cojob_outcome}" \
    --argjson cojob_exit_code "${cojob_rc}" \
    --argjson cojob_exit_codes "$(printf '%s\n' "${cojob_rcs[@]}" | jq -Rsc 'split("\n") | map(select(length > 0) | tonumber)')" \
    --arg inference_result "${inference_dir}/result.json" \
    --arg inference_result_sha256 "$(sha256sum "${inference_dir}/result.json" | awk '{print $1}')" \
    --arg observer "${cojob_dir}/nccl_observer.json" \
    --arg observer_sha256 "$(sha256sum "${cojob_dir}/nccl_observer.json" | awk '{print $1}')" \
    --arg observer_pair1 "${cojob_pair1_dir}/nccl_observer.json" \
    --arg observer_pair1_sha256 "$(sha256sum "${cojob_pair1_dir}/nccl_observer.json" | awk '{print $1}')" \
    --argjson cojob_pair_outcomes "${cojob_pair_outcomes_json}" \
    --argjson cojob_dirs "${cojob_dirs_json}" \
    --argjson transport_receipts "${transport_receipts_json}" \
    '{schema:$schema,block:$block,arm:$arm,inference_status:"complete",cojob_outcome:$cojob_outcome,cojob_exit_code:$cojob_exit_code,cojob_exit_codes:$cojob_exit_codes,cojob_pair_outcomes:$cojob_pair_outcomes,cojob_dirs:$cojob_dirs,transport_receipts:$transport_receipts,inference_result:$inference_result,inference_result_sha256:$inference_result_sha256,observer:$observer,observer_sha256:$observer_sha256,observers:[$observer,$observer_pair1],observer_sha256s:[$observer_sha256,$observer_pair1_sha256],cojob_pair_count:2,measured_arm_retried:false}' \
    >"${block_dir}/block_execution_receipt.json"
  jq -ce '.analysis | {arm,normal_slo:.normal.slo_good_victims,normal_p99:.normal.victim.e2e_ms.p99,miss_slo:.miss_hot.slo_good_victims,miss_p99:.miss_hot.victim.e2e_ms.p99,remote_slo:.remote_favorable.slo_good_victims,remote_p99:.remote_favorable.victim.e2e_ms.p99,route_counts,edge_counts}' "${inference_dir}/result.json"
done

"${REPO_ROOT}/.vllm_venv/bin/python" \
  -m eval.sota_4node.analyze_tempo_go_c9_causal_burst_discovery \
  --contract "${CONTRACT}" --root "${RESULT_ROOT}" \
  --output "${RESULT_ROOT}/analysis.json"
[[ -s "${RESULT_ROOT}/analysis.json" ]]
jq -n \
  --arg schema tempo-go-c9-causal-burst-attempt-v1 \
  --arg status complete \
  --arg analysis "${RESULT_ROOT}/analysis.json" \
  --arg analysis_sha256 "$(sha256sum "${RESULT_ROOT}/analysis.json" | awk '{print $1}')" \
  '{schema:$schema,status:$status,analysis:$analysis,analysis_sha256:$analysis_sha256,one_campaign_no_retry:true,discovery_only:true}' \
  >"${RESULT_ROOT}/completed_attempt.json"
jq '{gates,effects,telemetry,causal_discovery_positive,claim_boundary}' "${RESULT_ROOT}/analysis.json"
echo "TEMPO C9 causal-burst discovery receipt: ${RESULT_ROOT}/analysis.json"
