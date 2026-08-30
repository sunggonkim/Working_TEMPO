#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?existing allocation required}"
: "${SLURM_JOB_NODELIST:?nodelist required}"
: "${TEMPO_ELASTIC_PD_APPROVED:?explicit approval required}"
[[ "${TEMPO_ELASTIC_PD_APPROVED}" == YES && "${SLURM_JOB_NUM_NODES:-}" == 4 ]]
[[ $# -eq 2 ]]
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKLOAD=$(realpath -e -- "$1")
RESULT_DIR=$(realpath -m -- "$2")
case "${WORKLOAD}" in "${REPO_ROOT}"/results/*) ;; *) exit 2 ;; esac
case "${RESULT_DIR}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ -s "${WORKLOAD}" && ! -e "${RESULT_DIR}" ]]
module reset
module load pytorch/2.8.0
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
PORT_SLOT=$((1640 + SLURM_JOB_ID % 20))
REQUEST_RATE=${TEMPO_ELASTIC_PD_REQUEST_RATE:-48}
[[ "${REQUEST_RATE}" =~ ^[0-9]+([.][0-9]+)?$ ]]
[[ "${REQUEST_RATE}" != 0 && "${REQUEST_RATE}" != 0.0 ]]
mkdir -p -- "${RESULT_DIR}"
[[ -z "${TEMPO_CXI_BACKGROUND_START_FILE:-}" ]] || {
  echo "TEMPO_CXI_BACKGROUND_START_FILE is launcher-reserved" >&2
  exit 2
}
BACKGROUND_DUTY=${TEMPO_CXI_BACKGROUND_DUTY_CYCLE:-}
BACKGROUND_PATTERN=${TEMPO_CXI_BACKGROUND_PATTERN:-pairwise-bidir}
BACKGROUND_MESSAGE_BYTES=${TEMPO_CXI_BACKGROUND_MESSAGE_BYTES:-16777216}
BACKGROUND_INFLIGHT=${TEMPO_CXI_BACKGROUND_INFLIGHT:-4}
MAIN_CPUS=128
SRUN_OVERLAP=(--overlap)
BACKGROUND_PID=
BACKGROUND_START=
BACKGROUND_STOP=
if [[ -n "${BACKGROUND_DUTY}" ]]; then
  case "${BACKGROUND_DUTY}" in
    0.25|0.5|0.50|0.75|1|1.0|1.00) ;;
    *)
      echo "TEMPO_CXI_BACKGROUND_DUTY_CYCLE must be 0.25, 0.5, 0.75, or 1.0" >&2
      exit 2
      ;;
  esac
  case "${BACKGROUND_PATTERN}" in
    pairwise-bidir|pd-2p2d-incast|pd-3p1d-incast) ;;
    *)
      echo "TEMPO_CXI_BACKGROUND_PATTERN is invalid" >&2
      exit 2
      ;;
  esac
  case "${BACKGROUND_MESSAGE_BYTES}" in
    8388608|16777216|33554432) ;;
    *)
      echo "TEMPO_CXI_BACKGROUND_MESSAGE_BYTES must be 8, 16, or 32 MiB" >&2
      exit 2
      ;;
  esac
  case "${BACKGROUND_INFLIGHT}" in
    1|2|4|8) ;;
    *)
      echo "TEMPO_CXI_BACKGROUND_INFLIGHT must be 1, 2, 4, or 8" >&2
      exit 2
      ;;
  esac
  TRAFFIC_SOURCE="${SCRIPT_DIR}/cxi_background_traffic.c"
  TRAFFIC_BINARY="${RESULT_DIR}/cxi_background_traffic"
  BACKGROUND_READY="${RESULT_DIR}/cxi-background.ready"
  BACKGROUND_START="${RESULT_DIR}/cxi-background.start"
  BACKGROUND_STOP="${RESULT_DIR}/cxi-background.stop"
  BACKGROUND_LOG="${RESULT_DIR}/cxi-background.log"
  [[ -f "${TRAFFIC_SOURCE}" && ! -e "${BACKGROUND_READY}" \
    && ! -e "${BACKGROUND_START}" && ! -e "${BACKGROUND_STOP}" ]]
  export TEMPO_CXI_BACKGROUND_START_FILE="${BACKGROUND_START}"
  cc -O2 -std=c11 -Wall -Wextra -Werror "${TRAFFIC_SOURCE}" \
    -L/opt/cray/libfabric/1.22.0/lib64 \
    -Wl,-rpath,/opt/cray/libfabric/1.22.0/lib64 -lfabric -lm \
    -o "${TRAFFIC_BINARY}"
  srun --overlap --exact --nodes=4 --ntasks=16 --ntasks-per-node=4 \
    --distribution=block:block \
    --gpus=0 --cpus-per-task=1 \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 --wait=10 --export=ALL \
    --output="${BACKGROUND_LOG}" --error="${BACKGROUND_LOG}" \
    env MPICH_GPU_SUPPORT_ENABLED=0 FI_PROVIDER=cxi \
    MPICH_OFI_NIC_POLICY=ROUND-ROBIN MPICH_OFI_CXI_COUNTER_REPORT=2 \
    "LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/lib64:/opt/cray/libfabric/1.22.0/lib64:${LD_LIBRARY_PATH:-}" \
    "${TRAFFIC_BINARY}" --duration-s 2400 \
    --message-bytes "${BACKGROUND_MESSAGE_BYTES}" \
    --inflight "${BACKGROUND_INFLIGHT}" --duty-cycle "${BACKGROUND_DUTY}" \
    --pattern "${BACKGROUND_PATTERN}" \
    --ready-file "${BACKGROUND_READY}" --start-file "${BACKGROUND_START}" \
    --stop-file "${BACKGROUND_STOP}" &
  BACKGROUND_PID=$!
  cleanup_background() {
    if [[ -n "${BACKGROUND_PID}" ]]; then
      touch -- "${BACKGROUND_STOP}"
      wait "${BACKGROUND_PID}" || true
    fi
  }
  trap cleanup_background EXIT
  for ((attempt = 0; attempt < 300; ++attempt)); do
    [[ -s "${BACKGROUND_READY}" ]] && break
    kill -0 "${BACKGROUND_PID}" 2>/dev/null || {
      wait "${BACKGROUND_PID}"
      exit 1
    }
    sleep 0.1
  done
  [[ -s "${BACKGROUND_READY}" ]]
  MAIN_CPUS=120
fi
cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=30s 2640s \
  srun "${SRUN_OVERLAP[@]}" --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
  --cpus-per-task="${MAIN_CPUS}" --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
  --time=00:43:00 --export=ALL \
  --output="${RESULT_DIR}/slurm-node-%N.stdout.log" \
  --error="${RESULT_DIR}/slurm-node-%N.stderr.log" \
  bash "${SCRIPT_DIR}/elastic_pd_node_entry.sh" \
  "${REPO_ROOT}" "${RESULT_DIR}" "${WORKLOAD}" "${HOSTS_CSV}" \
  "${PORT_SLOT}" "${REQUEST_RATE}" 32 128 8 3000 250 16000
[[ -s "${RESULT_DIR}/elastic_pd_final.json" ]]
if [[ -n "${BACKGROUND_PID}" ]]; then
  [[ -s "${BACKGROUND_START}" ]]
  touch -- "${BACKGROUND_STOP}"
  wait "${BACKGROUND_PID}"
  BACKGROUND_PID=
  trap - EXIT
  grep -q '"schema":"tempo-cxi-background-traffic-3"' "${BACKGROUND_LOG}"
  grep -q '"pattern":"'"${BACKGROUND_PATTERN}"'"' "${BACKGROUND_LOG}"
  grep -q '"start_gated":true' "${BACKGROUND_LOG}"
  grep -q '"start_observed":true' "${BACKGROUND_LOG}"
  grep -q '"correctness":true' "${BACKGROUND_LOG}"
fi
echo "Canonical Elastic-PD result: ${RESULT_DIR}/elastic_pd_final.json"
