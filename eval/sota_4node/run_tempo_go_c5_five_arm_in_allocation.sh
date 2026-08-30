#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_ROOT_HINT="${TEMPO_GO_SOURCE_SNAPSHOT:-${SCRIPT_DIR}/../..}"
source "$(realpath -e -- "${SOURCE_ROOT_HINT}/eval/sota_4node/require_perlmutter_4node_4h_interactive.sh")"
[[ $# -eq 2 ]]

SCRIPT_REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
REPO_ROOT="${TEMPO_GO_REPO_ROOT:-${SCRIPT_REPO_ROOT}}"
WORKLOAD_INPUT=$(realpath -e -- "$1")
RESULT_ROOT=$(realpath -m -- "$2")
: "${TEMPO_GO_C5_RUN_CONTRACT:?frozen C5 run-contract path required}"
: "${TEMPO_GO_C5_RUN_CONTRACT_SHA256:?frozen C5 run-contract SHA-256 required}"
[[ "${TEMPO_GO_C5_RUN_CONTRACT_SHA256}" =~ ^[0-9a-f]{64}$ ]]
RUN_CONTRACT=$(realpath -e -- "${TEMPO_GO_C5_RUN_CONTRACT}")
case "${WORKLOAD_INPUT}" in "${REPO_ROOT}"/*) ;; *) exit 2 ;; esac
case "${RESULT_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
case "${RUN_CONTRACT}" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
: "${TEMPO_GO_SOURCE_SNAPSHOT:=$(jq -er '.source_snapshot.root // empty' "${RUN_CONTRACT}" 2>/dev/null || true)}"
if [[ -n "${TEMPO_GO_SOURCE_SNAPSHOT}" ]]; then
  TEMPO_GO_SOURCE_SNAPSHOT=$(realpath -e -- "${TEMPO_GO_SOURCE_SNAPSHOT}")
  case "${TEMPO_GO_SOURCE_SNAPSHOT}/" in
    "${REPO_ROOT}/results/"*) ;;
    "${REPO_ROOT}/") ;;
    *) exit 2 ;;
  esac
  export TEMPO_GO_SOURCE_SNAPSHOT
fi
if [[ -d "${WORKLOAD_INPUT}" ]]; then
  WORKLOAD="${WORKLOAD_INPUT}/workloads/validation.jsonl"
else
  WORKLOAD="${WORKLOAD_INPUT}"
fi
[[ -s "${WORKLOAD}" && ! -e "${RESULT_ROOT}" ]]

export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
if [[ -n "${TEMPO_GO_SOURCE_SNAPSHOT:-}" ]]; then
  export PYTHONPATH="${TEMPO_GO_SOURCE_SNAPSHOT}:${REPO_ROOT}"
else
  export PYTHONPATH="${REPO_ROOT}"
fi
if [[ -n "${TEMPO_GO_SOURCE_SNAPSHOT:-}" ]]; then
  cd -- "${TEMPO_GO_SOURCE_SNAPSHOT}"
fi
ARM_ONLY=${TEMPO_GO_C5_ARM_ONLY:-}
CONTRACT_VERIFY_ARGS=()
if [[ -n "${ARM_ONLY}" ]]; then
  CONTRACT_VERIFY_ARGS=(--arm-only "${ARM_ONLY}")
fi
"${REPO_ROOT}/.vllm_venv/bin/python" -m eval.sota_4node.tempo_go_c5_run_contract verify \
  --repo-root "${REPO_ROOT}" --contract "${RUN_CONTRACT}" \
  --sha256 "${TEMPO_GO_C5_RUN_CONTRACT_SHA256}" \
  --workload-input "${WORKLOAD_INPUT}" \
  "${CONTRACT_VERIFY_ARGS[@]}"

NODE_ENTRY_PATH=$(jq -er '.launcher.node_entry.path' "${RUN_CONTRACT}")
ANALYZER_PATH=$(jq -er '.launcher.analyzer.path' "${RUN_CONTRACT}")
[[ -f "${NODE_ENTRY_PATH}" && -f "${ANALYZER_PATH}" ]]
if [[ -n "${TEMPO_GO_SOURCE_SNAPSHOT:-}" ]]; then
  [[ "$(jq -er '.source_snapshot.root' "${RUN_CONTRACT}")" == "${TEMPO_GO_SOURCE_SNAPSHOT}" ]]
fi

CONTRACT_WORKLOAD=$(jq -er '.artifacts.workload.path' "${RUN_CONTRACT}")
WORKLOAD_MANIFEST=$(jq -er '.artifacts.manifest.path' "${RUN_CONTRACT}")
[[ "${CONTRACT_WORKLOAD}" == "$(realpath -e -- "${WORKLOAD}")" ]]
[[ -s "${WORKLOAD_MANIFEST}" ]]
for inherited in TEMPO_GO_GLOBAL_PROFILE TEMPO_GO_ELASTIC_PROFILE_PATH \
  TEMPO_GO_ENDPOINT_PROFILE_PATH TEMPO_PD_ENDPOINT_SERVICE_PROFILE \
  TEMPO_GO_C5_STEP_TIME TEMPO_GO_C5_REQUEST_RATE \
  TEMPO_GO_C5_MAX_WORKERS TEMPO_GO_C5_OUTPUT_TOKENS; do
  if [[ -n "${!inherited+x}" ]]; then
    echo "frozen C5 runner refuses inherited override: ${inherited}" >&2
    exit 2
  fi
done

mkdir -p -- "${RESULT_ROOT}"

module reset
module load pytorch/2.8.0
[[ "${NCCL_NET:-}" == "AWS Libfabric" ]] || {
  echo "expected NERSC NCCL AWS Libfabric transport, got ${NCCL_NET:-<unset>}" >&2
  exit 1
}
unset NCCL_IB_DISABLE
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
if [[ -n "${TEMPO_GO_SOURCE_SNAPSHOT:-}" ]]; then
  export PYTHONPATH="${TEMPO_GO_SOURCE_SNAPSHOT}:${REPO_ROOT}"
else
  export PYTHONPATH="${REPO_ROOT}"
fi
export TEMPO_LMCACHE_NIXL_BACKEND=UCX
export TEMPO_LMCACHE_LOCAL_CPU_GB=16
export TEMPO_LMCACHE_PD_BUFFER_BYTES=2147483648
export TEMPO_PD_PRESSURE_MODE=disabled
export TEMPO_VLLM_DECODER_PREFIX_CACHING=0
export TEMPO_PD_FRONTEND_PAIR_POLICY=tempo-min-outstanding-decode-tokens-v1
export TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY=1
export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_PD_BENCHMARK_RESET_DECODER_APC=0
export TEMPO_PD_DECODER_REUSE_ITEMS=all
export TEMPO_VLLM_MAX_NUM_SEQS=16
export TEMPO_VLLM_ASYNC_SCHEDULING=0
export TEMPO_VLLM_DECODER_MAX_NUM_BATCHED_TOKENS=32768
export TEMPO_VLLM_SCHEDULING_POLICY=fcfs
export TEMPO_PD_REMOTE_DECODE_PLACEMENT=paired
export TEMPO_PD_PROXY_TOKENIZER_PLACEMENT=round_robin
prepare_overlay="${TEMPO_GO_SOURCE_SNAPSHOT:-${REPO_ROOT}}/eval/sota_4node/prepare_c4_python_overlay.sh"
[[ -x "${prepare_overlay}" ]]
"${prepare_overlay}" "${REPO_ROOT}" "${RESULT_ROOT}"
export TEMPO_C4_PYTHON_OVERLAY_PREPARE_ARTIFACT="${RESULT_ROOT}/python-overlay-prepare.json"
export TEMPO_PD_C5_APPROVED=YES
export TEMPO_PD_C5_RUN_CONTRACT_SHA256="${TEMPO_GO_C5_RUN_CONTRACT_SHA256}"
export TEMPO_GO_C5_RUN_CONTRACT="${RUN_CONTRACT}"
set -euo pipefail
export TEMPO_GO_GLOBAL_PROFILE="$(jq -er '.artifacts.global_profile.path' "${RUN_CONTRACT}")"
export TEMPO_GO_ELASTIC_PROFILE_PATH="$(jq -er '.artifacts.elastic_profile.path' "${RUN_CONTRACT}")"
export TEMPO_GO_ENDPOINT_PROFILE_PATH="$(jq -er '.artifacts.endpoint_profile.path' "${RUN_CONTRACT}")"
export TEMPO_PD_ENDPOINT_SERVICE_PROFILE="${TEMPO_GO_ENDPOINT_PROFILE_PATH}"
export TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256
TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256=$(sha256sum "${WORKLOAD_MANIFEST}" | awk '{print $1}')
export TEMPO_GO_C5_REQUEST_RATE="$(jq -er '.launcher.node_parameters.request_rate' "${RUN_CONTRACT}")"
export TEMPO_GO_C5_MAX_WORKERS="$(jq -er '.launcher.node_parameters.max_workers' "${RUN_CONTRACT}")"
export TEMPO_GO_C5_OUTPUT_TOKENS="$(jq -er '.launcher.node_parameters.output_tokens' "${RUN_CONTRACT}")"
printf '%s\n' "${SLURM_JOB_ID}" > "${RESULT_ROOT}/slurm_job_id.txt"

# Slurm can terminate the launcher while an inner native step is being
# torn down.  In that case the normal `if srun; then ... else ... fi` branch
# is never reached.  Preserve an explicit current-arm failure receipt so the
# absence of a raw file cannot be confused with an unobserved experiment.
write_native_signal_receipt() {
  local signal_name="$1"
  local arm="${TEMPO_GO_C5_ARM:-}"
  [[ -n "${arm}" ]] || return 0
  local arm_result="${RESULT_ROOT}/${arm}"
  [[ -d "${arm_result}" ]] || return 0
  [[ ! -e "${arm_result}/result.json" && \
    ! -e "${arm_result}/failure.json" ]] || return 0
  failure_path="${arm_result}/failure.json" \
  failure_arm="${arm}" \
  failure_signal="${signal_name}" \
  failure_workload="${WORKLOAD}" \
  failure_manifest="${WORKLOAD_MANIFEST}" \
  failure_contract="${RUN_CONTRACT}" \
  failure_contract_sha256="${TEMPO_GO_C5_RUN_CONTRACT_SHA256}" \
  failure_contract_fingerprint="$(jq -er '.fingerprint_sha256' "${RUN_CONTRACT}")" \
  "${REPO_ROOT}/.vllm_venv/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

workload = Path(os.environ["failure_workload"]).resolve()
manifest = Path(os.environ["failure_manifest"]).resolve()
value = {
    "schema": "tempo-go-c5-native-arm-signal-failure-v1",
    "arm": os.environ["failure_arm"],
    "failure": "native_arm_step_signal",
    "signal": os.environ["failure_signal"],
    "native_only": True,
    "node_count": 4,
    "gpu_count": 16,
    "transport": "LMCacheConnectorV1:UCX",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "result_dir": str(Path(os.environ["failure_path"]).parent.resolve()),
    "native_logs": {
        "stdout_glob": "slurm-node-*.stdout.log",
        "stderr_glob": "slurm-node-*.stderr.log",
    },
    "workload": str(workload),
    "workload_sha256": digest(workload),
    "workload_manifest": str(manifest),
    "workload_manifest_sha256": digest(manifest),
    "run_contract": os.environ["failure_contract"],
    "run_contract_sha256": os.environ["failure_contract_sha256"],
    "run_contract_fingerprint_sha256": os.environ["failure_contract_fingerprint"],
}
Path(os.environ["failure_path"]).write_text(
    json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
}

trap 'write_native_signal_receipt TERM; exit 143' TERM
trap 'write_native_signal_receipt INT; exit 130' INT
trap 'write_native_signal_receipt HUP; exit 129' HUP

if [[ -n "${ARM_ONLY}" ]]; then
  case "${ARM_ONLY}" in
    local|remote|predictor|queue_gpu|network_request_only|app_global_only|tempo) ;;
    *) echo "invalid TEMPO_GO_C5_ARM_ONLY=${ARM_ONLY}" >&2; exit 2 ;;
  esac
  # A single-arm continuation is used only to close a missing receipt inside
  # an already approved interactive allocation.  It never re-runs a valid arm.
  ARM_ORDER=("${ARM_ONLY}")
else
  mapfile -t ARM_ORDER < <(jq -er '.arm_order[]' "${RUN_CONTRACT}")
fi
if [[ -n "${ARM_ONLY}" ]]; then
  ARM_ORDER=("${ARM_ONLY}")
fi
printf '%s\n' "${ARM_ORDER[@]}" > "${RESULT_ROOT}/arm_order.txt"
HOSTS=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | paste -sd, -)
[[ -n "${HOSTS}" ]]

# Five sequential arms must fit inside one four-hour allocation.  The step and
# timeout are frozen in the run contract, leaving teardown/artifact time in the
# allocation.
STEP_TIME=$(jq -er '.launcher.step_time' "${RUN_CONTRACT}")
TIMEOUT_SECONDS=$(jq -er '.launcher.timeout_seconds' "${RUN_CONTRACT}")
SRUN_JOB_ARGS=()
# `--jobid=$SLURM_JOB_ID` is valid from a plain allocation/batch shell, but
# it is unsafe from an existing interactive step: a failed child setup can
# terminate the parent step and relinquish the whole allocation.  Let srun
# create a nested child step when this launcher is already inside one.
case "${SLURM_STEP_ID:-}" in
  ""|batch|extern)
    if [[ -n "${TEMPO_GO_C5_SRUN_JOBID:-}" ]]; then
      [[ "${TEMPO_GO_C5_SRUN_JOBID}" =~ ^[0-9]+$ ]]
      SRUN_JOB_ARGS=("--jobid=${TEMPO_GO_C5_SRUN_JOBID}")
    else
      SRUN_JOB_ARGS=("--jobid=${SLURM_JOB_ID}")
    fi
    ;;
  *)
    if [[ -n "${TEMPO_GO_C5_SRUN_JOBID:-}" ]]; then
      echo "ignoring TEMPO_GO_C5_SRUN_JOBID inside nested Slurm step ${SLURM_STEP_ID}" >&2
    fi
    ;;
esac
# Match the allocation's native Slingshot setup by default.  An explicit
# disable_rdzv_get opt-in remains available only when a bounded allocation
# probe has established that it is valid; never force it on every measured
# arm because that can fail VNI setup on otherwise healthy allocations.
C5_SRUN_NETWORK_ARGS=()
if [[ -n "${TEMPO_GO_SRUN_NETWORK_MODE:-}" ]]; then
  [[ "${TEMPO_GO_SRUN_NETWORK_MODE}" == "disable_rdzv_get" ]]
  C5_SRUN_NETWORK_ARGS=("--network=${TEMPO_GO_SRUN_NETWORK_MODE}")
fi

for index in "${!ARM_ORDER[@]}"; do
  arm=${ARM_ORDER[$index]}
  arm_result="${RESULT_ROOT}/${arm}"
  [[ ! -e "${arm_result}" ]]
  mkdir -p -- "${arm_result}"
  export TEMPO_GO_C5_ARM="${arm}"
  if [[ "${arm}" == tempo || "${arm}" == app_global_only ]]; then
    export TEMPO_PD_ENDPOINT_FEEDBACK_MODE=adaptive
    export TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK=1
    export TEMPO_PD_ENDPOINT_ROUTING_POLICY=semantic_epoch_v1
    export TEMPO_VLLM_LOAD_SNAPSHOT_MODE=disabled
    if [[ "${arm}" == app_global_only ]]; then
      export TEMPO_GO_ABLATION=app_global_only
    else
      export TEMPO_GO_ABLATION=disabled
    fi
  else
    export TEMPO_PD_ENDPOINT_FEEDBACK_MODE=disabled
    export TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK=0
    export TEMPO_PD_ENDPOINT_ROUTING_POLICY=instant_score_v1
    if [[ "${arm}" == queue_gpu ]]; then
      export TEMPO_VLLM_LOAD_SNAPSHOT_MODE=observe_only
    else
      export TEMPO_VLLM_LOAD_SNAPSHOT_MODE=disabled
    fi
    export TEMPO_GO_ABLATION=disabled
  fi
  port_slot=$((1860 + SLURM_JOB_ID % 20 + index * 40))
  if timeout --foreground --signal=TERM --kill-after=30s "${TIMEOUT_SECONDS}s" \
    /usr/bin/srun "${SRUN_JOB_ARGS[@]}" \
      --overlap --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
    --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 \
    "${C5_SRUN_NETWORK_ARGS[@]}" \
    --time="${STEP_TIME}" --export=ALL \
    --output="${arm_result}/slurm-node-%N.stdout.log" \
    --error="${arm_result}/slurm-node-%N.stderr.log" \
    bash "${NODE_ENTRY_PATH}" \
    "${REPO_ROOT}" "${arm_result}" "${WORKLOAD}" "${HOSTS}" \
    "${port_slot}" "${RUN_CONTRACT}"; then
    [[ -s "${arm_result}/result.json" ]]
  else
    rc=$?
    # Any native arm failure is an evidence-bearing outcome.  A fixed baseline
    # failure must not prevent the remaining fixed/global arms from producing
    # their own receipts; it is excluded from latency comparison and retained
    # as robustness evidence.  TEMPO failure still stops the campaign because
    # a partial global-controller run cannot be mistaken for a complete arm.
    failure_path="${arm_result}/failure.json" \
    failure_arm="${arm}" \
    failure_rc="${rc}" \
    failure_workload="${WORKLOAD}" \
    failure_manifest="${WORKLOAD_MANIFEST}" \
    failure_contract="${RUN_CONTRACT}" \
    failure_contract_sha256="${TEMPO_GO_C5_RUN_CONTRACT_SHA256}" \
    failure_contract_fingerprint="$(jq -er '.fingerprint_sha256' "${RUN_CONTRACT}")" \
    "${REPO_ROOT}/.vllm_venv/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

workload = Path(os.environ["failure_workload"]).resolve()
manifest = Path(os.environ["failure_manifest"]).resolve()
value = {
    "schema": "tempo-go-c5-native-arm-failure-v1",
    "arm": os.environ["failure_arm"],
    "failure": "native_arm_process_failed",
    "exit_code": int(os.environ["failure_rc"]),
    "native_only": True,
    "node_count": 4,
    "gpu_count": 16,
    "transport": "LMCacheConnectorV1:UCX",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "result_dir": str(Path(os.environ["failure_path"]).parent.resolve()),
    "native_logs": {
        "stdout_glob": "slurm-node-*.stdout.log",
        "stderr_glob": "slurm-node-*.stderr.log",
    },
    "workload": str(workload),
    "workload_sha256": digest(workload),
    "workload_manifest": str(manifest),
    "workload_manifest_sha256": digest(manifest),
    "run_contract": os.environ["failure_contract"],
    "run_contract_sha256": os.environ["failure_contract_sha256"],
    "run_contract_fingerprint_sha256": os.environ["failure_contract_fingerprint"],
}
Path(os.environ["failure_path"]).write_text(
    json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
    [[ -s "${arm_result}/failure.json" ]]
    if [[ "${arm}" == tempo ]]; then
      echo "TEMPO-GO native arm failure recorded: arm=${arm} rc=${rc}" >&2
      exit "${rc}"
    fi
    echo "TEMPO-GO native baseline failure recorded: arm=${arm} rc=${rc}" >&2
  fi
done

if [[ -z "${ARM_ONLY}" ]]; then
  "${REPO_ROOT}/.vllm_venv/bin/python" "${ANALYZER_PATH}" \
    --result-root "${RESULT_ROOT}" \
    --output "${RESULT_ROOT}/native_five_arm_analysis.json"
  echo "TEMPO-GO native five-arm discovery completed: ${RESULT_ROOT}"
else
  echo "TEMPO-GO native single-arm receipt completed: arm=${ARM_ONLY} root=${RESULT_ROOT}"
fi
