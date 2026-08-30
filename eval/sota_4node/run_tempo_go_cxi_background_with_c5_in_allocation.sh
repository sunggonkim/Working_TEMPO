#!/usr/bin/env bash
set -euo pipefail

# One bounded foreground campaign inside an already-approved native four-node
# gpu_interactive allocation.  The CPU MPI producer occupies four Cassini
# rails per node but no GPU; the canonical C5 runner owns all 16 GPUs.
[[ $# -eq 2 ]]
SCRIPT_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REPO_ROOT="${TEMPO_GO_REPO_ROOT:-${SCRIPT_REPO_ROOT}}"
WORKLOAD_INPUT=$(realpath -e -- "$1")
RESULT_ROOT=$(realpath -m -- "$2")
case "${RESULT_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ ! -e "${RESULT_ROOT}" ]]
[[ "$(id -u)" -ne 0 ]]
[[ -z "${UDI:-}" && -z "${CRAY_ROOTFS:-}" && -z "${SLURM_CONTAINER:-}" ]]
[[ -z "${SHIFTER_RUNTIME:-}" && -z "${SHIFTER_IMAGE:-}" ]]
: "${SLURM_JOB_ID:?existing allocation required}"
: "${SLURM_JOB_NODELIST:?allocation nodelist required}"
: "${TEMPO_GO_C5_RUN_CONTRACT:?frozen C5 run contract required}"
: "${TEMPO_GO_C5_RUN_CONTRACT_SHA256:?frozen contract SHA required}"

RUN_CONTRACT=$(realpath -e -- "${TEMPO_GO_C5_RUN_CONTRACT}")
: "${TEMPO_GO_SOURCE_SNAPSHOT:=$(jq -er '.source_snapshot.root' "${RUN_CONTRACT}")}"
TEMPO_GO_SOURCE_SNAPSHOT=$(realpath -e -- "${TEMPO_GO_SOURCE_SNAPSHOT}")
case "${TEMPO_GO_SOURCE_SNAPSHOT}/" in
  "${REPO_ROOT}/results/"*) ;;
  *) exit 2 ;;
esac
export TEMPO_GO_SOURCE_SNAPSHOT TEMPO_GO_REPO_ROOT="${REPO_ROOT}"
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1
export PYTHONPATH="${TEMPO_GO_SOURCE_SNAPSHOT}:${REPO_ROOT}"
PYTHON="${REPO_ROOT}/.vllm_venv/bin/python"
[[ -x "${PYTHON}" ]]
"${PYTHON}" -m eval.sota_4node.tempo_go_c5_run_contract verify \
  --repo-root "${REPO_ROOT}" --contract "${RUN_CONTRACT}" \
  --sha256 "${TEMPO_GO_C5_RUN_CONTRACT_SHA256}" \
  --workload-input "${WORKLOAD_INPUT}"
[[ "$(jq -er '.launcher.cxi_background_cojob' "${RUN_CONTRACT}")" == true ]]

command -v flock >/dev/null 2>&1
LOCK_FILE="${REPO_ROOT}/results/.tempo_go_cxi_campaign_${SLURM_JOB_ID}.lock"
exec 9>"${LOCK_FILE}"
flock -n 9 || {
  echo "CXI/C5 campaign already active in allocation ${SLURM_JOB_ID}" >&2
  exit 3
}
release_lock() {
  flock -u 9 2>/dev/null || true
  exec 9>&- 2>/dev/null || true
}

JOB_INFO="${TEMPO_PD_ALLOCATION_RECORD:-}"
if [[ -z "${JOB_INFO}" ]]; then
  JOB_INFO=$(/usr/bin/timeout --foreground --signal=TERM --kill-after=3s 15s \
    scontrol show job -o "${SLURM_JOB_ID}")
fi
[[ " ${JOB_INFO} " == *" JobState=RUNNING "* ]]
[[ " ${JOB_INFO} " == *" NumNodes=4 "* ]]
[[ " ${JOB_INFO} " == *" NumCPUs=512 "* ]]
[[ " ${JOB_INFO} " == *" CPUs/Task=128 "* ]]
[[ " ${JOB_INFO} " == *" Network=job_vni "* ]]
[[ "${JOB_INFO}" == *"gres/gpu=16"* || \
   "${JOB_INFO}" == *"gres/gpu:a100=16"* ]]
export TEMPO_PD_ALLOCATION_RECORD="${JOB_INFO}"

COJOB_ROOT="${TEMPO_GO_CXI_COJOB_ROOT:-${REPO_ROOT}/results/tempo_go_cxi_background_${SLURM_JOB_ID}}"
case "${COJOB_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ ! -e "${COJOB_ROOT}" ]]
mkdir -p -- "${COJOB_ROOT}"
printf '%s\n' "${JOB_INFO}" >"${COJOB_ROOT}/perlmutter-job-info.txt"

ACTIVE_CHILD_STEPS=$(squeue --steps --noheader --jobs="${SLURM_JOB_ID}" \
  --states=RUNNING --format="%i" 2>/dev/null | sort -u)
CURRENT_STEP=""
case "${SLURM_STEP_ID:-}" in
  ''|batch|extern|interactive|*[!0-9]*) ;;
  *) CURRENT_STEP="${SLURM_JOB_ID}.${SLURM_STEP_ID}" ;;
esac
EXTRA_STEPS=""
while IFS= read -r step_id; do
  case "${step_id}" in
    ""|"${SLURM_JOB_ID}.extern"|"${SLURM_JOB_ID}.interactive"|\
      "${SLURM_JOB_ID}.batch"|"${CURRENT_STEP}") ;;
    *) EXTRA_STEPS+="${EXTRA_STEPS:+,}${step_id}" ;;
  esac
done <<<"${ACTIVE_CHILD_STEPS}"
if [[ -n "${EXTRA_STEPS}" ]]; then
  jq -n --arg schema tempo-go-cxi-c5-preflight-v1 \
    --arg status failed --arg active_steps "${EXTRA_STEPS}" \
    '{schema:$schema,status:$status,error:"allocation_has_active_child_steps",active_steps:$active_steps}' \
    >"${COJOB_ROOT}/preflight.json"
  echo "active child steps prevent a bounded campaign: ${EXTRA_STEPS}" >&2
  release_lock
  exit 1
fi

module reset
module load python/3.12-26.1.0
TRAFFIC_SOURCE="${TEMPO_GO_SOURCE_SNAPSHOT}/eval/sota_4node/cxi_background_traffic.c"
TRAFFIC_BINARY="${COJOB_ROOT}/cxi_background_traffic"
cc -O2 -std=c11 -Wall -Wextra -Werror "${TRAFFIC_SOURCE}" \
  -lm -o "${TRAFFIC_BINARY}"

TRAFFIC_READY="${COJOB_ROOT}/traffic.ready"
TRAFFIC_START="${COJOB_ROOT}/traffic.start"
TRAFFIC_STOP="${COJOB_ROOT}/traffic.stop"
TRAFFIC_LOG="${COJOB_ROOT}/traffic.log"
C5_LOG="${COJOB_ROOT}/c5-runner.log"
TRAFFIC_PID=""
C5_PID=""

terminate_tree() {
  local root="$1" signal="$2" child
  while read -r child; do
    [[ "${child}" =~ ^[0-9]+$ ]] || continue
    terminate_tree "${child}" "${signal}"
  done < <(pgrep -P "${root}" 2>/dev/null || true)
  kill -"${signal}" "${root}" 2>/dev/null || true
}

cleanup() {
  : >"${TRAFFIC_STOP}" 2>/dev/null || true
  for process in "${C5_PID}" "${TRAFFIC_PID}"; do
    if [[ -n "${process}" ]] && kill -0 "${process}" 2>/dev/null; then
      terminate_tree "${process}" TERM
    fi
  done
  for process in "${C5_PID}" "${TRAFFIC_PID}"; do
    [[ -z "${process}" ]] || wait "${process}" 2>/dev/null || true
  done
  release_lock
}
trap cleanup EXIT INT TERM HUP

SRUN_JOB_ARGS=()
case "${SLURM_STEP_ID:-}" in
  ''|batch|extern) SRUN_JOB_ARGS=("--jobid=${SLURM_JOB_ID}") ;;
esac

/usr/bin/timeout --foreground --signal=TERM --kill-after=15s 9300s \
  /usr/bin/srun "${SRUN_JOB_ARGS[@]}" --overlap --exact \
  --nodes=4 --ntasks=16 --ntasks-per-node=4 \
  --distribution=block:block --gpus=0 --cpus-per-task=1 \
  --cpu-bind=cores --kill-on-bad-exit=1 --wait=15 \
  --time=02:35:00 --export=ALL \
  --output="${TRAFFIC_LOG}" --error="${TRAFFIC_LOG}" \
  env MPICH_GPU_SUPPORT_ENABLED=0 FI_PROVIDER=cxi \
  MPICH_OFI_NIC_POLICY=ROUND-ROBIN MPICH_OFI_CXI_COUNTER_REPORT=2 \
  "${TRAFFIC_BINARY}" --duration-s 9000 --message-bytes 16777216 \
  --inflight 8 --duty-cycle 1.0 --pattern pd-3p1d-incast \
  --ready-file "${TRAFFIC_READY}" --start-file "${TRAFFIC_START}" \
  --stop-file "${TRAFFIC_STOP}" &
TRAFFIC_PID=$!

READY=0
for _ in $(seq 1 600); do
  if [[ -s "${TRAFFIC_READY}" ]]; then READY=1; break; fi
  kill -0 "${TRAFFIC_PID}" 2>/dev/null || {
    wait "${TRAFFIC_PID}" || true
    echo "CXI producer ended before readiness" >&2
    exit 1
  }
  sleep 0.1
done
[[ "${READY}" -eq 1 ]]
TRAFFIC_START_UNIX_NS=$(date +%s%N)
: >"${TRAFFIC_START}"
C5_START_UNIX_NS=$(date +%s%N)

C5_RUNNER="${TEMPO_GO_SOURCE_SNAPSHOT}/eval/sota_4node/run_tempo_go_c5_five_arm_in_allocation.sh"
[[ -x "${C5_RUNNER}" ]]
"${C5_RUNNER}" "${WORKLOAD_INPUT}" "${RESULT_ROOT}" \
  >"${C5_LOG}" 2>&1 &
C5_PID=$!
C5_RC=0
while kill -0 "${C5_PID}" 2>/dev/null; do
  if ! kill -0 "${TRAFFIC_PID}" 2>/dev/null; then
    wait "${TRAFFIC_PID}" 2>/dev/null || true
    echo "CXI producer ended before C5 completion" >&2
    terminate_tree "${C5_PID}" TERM
    wait "${C5_PID}" 2>/dev/null || true
    exit 1
  fi
  sleep 1
done
wait "${C5_PID}" || C5_RC=$?
C5_PID=""
C5_END_UNIX_NS=$(date +%s%N)
: >"${TRAFFIC_STOP}"
TRAFFIC_RC=0
wait "${TRAFFIC_PID}" || TRAFFIC_RC=$?
TRAFFIC_PID=""
[[ "${C5_RC}" -eq 0 && "${TRAFFIC_RC}" -eq 0 ]]

CALIBRATION="${REPO_ROOT}/results/tempo_cxi_fabric_ladder_57490824_v7/fabric_ladder_summary_v2.json"
[[ -s "${CALIBRATION}" ]]
COJOB_ROOT_ENV="${COJOB_ROOT}" RESULT_ROOT_ENV="${RESULT_ROOT}" \
TRAFFIC_LOG_ENV="${TRAFFIC_LOG}" CALIBRATION_ENV="${CALIBRATION}" \
TRAFFIC_START_NS_ENV="${TRAFFIC_START_UNIX_NS}" \
C5_START_NS_ENV="${C5_START_UNIX_NS}" C5_END_NS_ENV="${C5_END_UNIX_NS}" \
RUN_CONTRACT_ENV="${RUN_CONTRACT}" \
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

log = Path(os.environ["TRAFFIC_LOG_ENV"]).resolve()
records = []
for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
    if not line.startswith("{"):
        continue
    try:
        value = json.loads(line)
    except ValueError:
        continue
    if value.get("schema") == "tempo-cxi-background-traffic-3":
        records.append(value)
if len(records) != 1:
    raise SystemExit(f"expected one CXI traffic record, found {len(records)}")
traffic = records[0]
required = (
    traffic.get("correctness") is True
    and traffic.get("pattern") == "pd-3p1d-incast"
    and traffic.get("message_bytes") == 16 * 1024 * 1024
    and traffic.get("inflight") == 8
    and traffic.get("duty_cycle") == 1.0
    and traffic.get("start_gated") is True
    and traffic.get("start_observed") is True
    and traffic.get("stop_requested") is True
)
if not required:
    raise SystemExit("CXI traffic identity/correctness gate failed")
decoder_ingress = traffic.get("node_received_gbps", [0, 0, 0, 0])[3]
if not isinstance(decoder_ingress, (int, float)) or decoder_ingress < 250.0:
    raise SystemExit(
        f"p07 decoder ingress below contention gate: {decoder_ingress}")

root = Path(os.environ["RESULT_ROOT_ENV"]).resolve()
calibration = Path(os.environ["CALIBRATION_ENV"]).resolve()
contract = Path(os.environ["RUN_CONTRACT_ENV"]).resolve()
payload = {
    "schema": "tempo-go-c5-cxi-background-binding-v1",
    "same_allocation": True,
    "native_only": True,
    "privileged_nic_control": False,
    "full_campaign_coverage": True,
    "slurm_job_id": int(os.environ["SLURM_JOB_ID"]),
    "cojob_root": os.environ["COJOB_ROOT_ENV"],
    "traffic_log": str(log),
    "traffic_log_sha256": digest(log),
    "traffic": traffic,
    "selected_calibration": str(calibration),
    "selected_calibration_sha256": digest(calibration),
    "selected_profile": "p07_3p1d_16m_i8_d100",
    "minimum_decoder_ingress_gbps_gate": 250.0,
    "traffic_start_unix_ns": int(os.environ["TRAFFIC_START_NS_ENV"]),
    "c5_start_unix_ns": int(os.environ["C5_START_NS_ENV"]),
    "c5_end_unix_ns": int(os.environ["C5_END_NS_ENV"]),
    "run_contract": str(contract),
    "run_contract_sha256": digest(contract),
}
(root / "cxi_background_binding.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

trap - EXIT INT TERM HUP
release_lock
echo "TEMPO-GO p07 CXI same-allocation campaign complete: ${RESULT_ROOT}"
