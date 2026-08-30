#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?existing allocation required}"
: "${SLURM_JOB_NODELIST:?nodelist required}"
: "${TEMPO_FABRIC_LADDER_APPROVED:?explicit approval required}"
[[ "${TEMPO_FABRIC_LADDER_APPROVED}" == YES ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 4 ]]
[[ $# -eq 1 ]]

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RESULT_DIR=$(realpath -m -- "$1")
case "${RESULT_DIR}/" in
  "${REPO_ROOT}/results/"*) ;;
  *)
    echo "result directory must be under ${REPO_ROOT}/results" >&2
    exit 2
    ;;
esac
[[ ! -e "${RESULT_DIR}" ]]

module reset
module load python/3.12-26.1.0
cd -- "${REPO_ROOT}"
mkdir -p -- "${RESULT_DIR}"

TRAFFIC_SOURCE="${SCRIPT_DIR}/cxi_background_traffic.c"
TRAFFIC_BINARY="${RESULT_DIR}/cxi_background_traffic"
SAMPLER_PYTHON="${REPO_ROOT}/.vllm_venv/bin/python"
[[ -x "${SAMPLER_PYTHON}" ]]
cc -O2 -std=c11 -Wall -Wextra -Werror "${TRAFFIC_SOURCE}" \
  -lm -o "${TRAFFIC_BINARY}"

PROFILES=(
  "p00_pairwise_8m_i1_d25 pairwise-bidir 8388608 1 0.25"
  "p01_pairwise_16m_i4_d100 pairwise-bidir 16777216 4 1.0"
  "p02_pairwise_16m_i8_d100 pairwise-bidir 16777216 8 1.0"
  "p03_2p2d_16m_i4_d100 pd-2p2d-incast 16777216 4 1.0"
  "p04_2p2d_16m_i8_d100 pd-2p2d-incast 16777216 8 1.0"
  "p05_2p2d_32m_i4_d100 pd-2p2d-incast 33554432 4 1.0"
  "p06_3p1d_16m_i4_d100 pd-3p1d-incast 16777216 4 1.0"
  "p07_3p1d_16m_i8_d100 pd-3p1d-incast 16777216 8 1.0"
  "p08_3p1d_32m_i4_d100 pd-3p1d-incast 33554432 4 1.0"
)

TRAFFIC_PID=
SAMPLER_PID=
START_FILE=
cleanup_profile() {
  if [[ -n "${START_FILE}" && ! -e "${START_FILE}" ]]; then
    touch -- "${START_FILE}"
  fi
  for process in "${SAMPLER_PID}" "${TRAFFIC_PID}"; do
    if [[ -n "${process}" ]] && kill -0 "${process}" 2>/dev/null; then
      kill -TERM "${process}" 2>/dev/null || true
    fi
  done
  for process in "${SAMPLER_PID}" "${TRAFFIC_PID}"; do
    if [[ -n "${process}" ]]; then
      wait "${process}" 2>/dev/null || true
    fi
  done
}
trap cleanup_profile EXIT

for profile in "${PROFILES[@]}"; do
  read -r PROFILE_NAME PATTERN MESSAGE_BYTES INFLIGHT DUTY <<<"${profile}"
  PROFILE_DIR="${RESULT_DIR}/${PROFILE_NAME}"
  mkdir -- "${PROFILE_DIR}"
  TRAFFIC_READY="${PROFILE_DIR}/traffic.ready"
  START_FILE="${PROFILE_DIR}/window.start"
  READY_PREFIX="${PROFILE_DIR}/cassini.ready"
  TRAFFIC_PID=
  SAMPLER_PID=

  timeout --foreground --signal=TERM --kill-after=5s 30s \
    srun --overlap --exact --nodes=4 --ntasks=16 --ntasks-per-node=4 \
    --distribution=block:block --gpus=0 --cpus-per-task=1 \
    --cpu-bind=cores --kill-on-bad-exit=1 --wait=5 \
    --export=ALL --output="${PROFILE_DIR}/traffic.log" \
    --error="${PROFILE_DIR}/traffic.log" \
    env MPICH_GPU_SUPPORT_ENABLED=0 FI_PROVIDER=cxi \
    MPICH_OFI_NIC_POLICY=ROUND-ROBIN MPICH_OFI_CXI_COUNTER_REPORT=2 \
    "${TRAFFIC_BINARY}" --duration-s 7 --message-bytes "${MESSAGE_BYTES}" \
    --inflight "${INFLIGHT}" --duty-cycle "${DUTY}" \
    --pattern "${PATTERN}" --ready-file "${TRAFFIC_READY}" \
    --start-file "${START_FILE}" &
  TRAFFIC_PID=$!

  traffic_ready=0
  for ((attempt = 0; attempt < 300; ++attempt)); do
    if [[ -s "${TRAFFIC_READY}" ]]; then
      traffic_ready=1
      break
    fi
    kill -0 "${TRAFFIC_PID}" 2>/dev/null || {
      wait "${TRAFFIC_PID}"
      exit 1
    }
    sleep 0.1
  done
  [[ "${traffic_ready}" == 1 ]]

  timeout --foreground --signal=TERM --kill-after=5s 25s \
    srun --overlap --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    --distribution=block:block --gpus=0 --cpus-per-task=1 \
    --cpu-bind=cores --kill-on-bad-exit=1 --wait=5 --export=ALL \
    --output="${PROFILE_DIR}/cassini-%N.jsonl" \
    --error="${PROFILE_DIR}/cassini-%N.jsonl" \
    env "PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}" \
    "${SAMPLER_PYTHON}" "${SCRIPT_DIR}/sample_cassini_fabric_window.py" \
    --sample-seconds 6 --start-file "${START_FILE}" \
    --ready-prefix "${READY_PREFIX}" --pattern "${PATTERN}" &
  SAMPLER_PID=$!

  sampler_ready=0
  for ((attempt = 0; attempt < 300; ++attempt)); do
    ready_count=0
    for node_index in 0 1 2 3; do
      [[ -s "${READY_PREFIX}.node${node_index}" ]] \
        && ready_count=$((ready_count + 1))
    done
    if [[ "${ready_count}" == 4 ]]; then
      sampler_ready=1
      break
    fi
    kill -0 "${SAMPLER_PID}" 2>/dev/null || {
      wait "${SAMPLER_PID}"
      exit 1
    }
    sleep 0.1
  done
  [[ "${sampler_ready}" == 1 ]]

  touch -- "${START_FILE}"
  wait "${SAMPLER_PID}"
  SAMPLER_PID=
  wait "${TRAFFIC_PID}"
  TRAFFIC_PID=
  grep -q '"schema":"tempo-cxi-background-traffic-3"' \
    "${PROFILE_DIR}/traffic.log"
  grep -q '"correctness":true' "${PROFILE_DIR}/traffic.log"
  START_FILE=
done

trap - EXIT
python "${SCRIPT_DIR}/summarize_cxi_fabric_ladder.py" "${RESULT_DIR}"
echo "CXI fabric ladder: ${RESULT_DIR}/fabric_ladder_summary.json"
