#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/require_perlmutter_4node_4h_interactive.sh"
: "${TEMPO_PD_C4_APPROVED:?explicit C4 screen approval required}"
[[ "${TEMPO_PD_C4_APPROVED}" == YES ]]
[[ $# -eq 2 ]]

REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKLOAD=$(realpath -e -- "$1")
RESULT_DIR=$(realpath -m -- "$2")
case "${WORKLOAD}" in "${REPO_ROOT}"/results/*) ;; *) exit 2 ;; esac
case "${RESULT_DIR}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ -s "${WORKLOAD}" && ! -e "${RESULT_DIR}" ]]
[[ -z "${TEMPO_CXI_BACKGROUND_DUTY_CYCLE:-}" ]]
[[ -z "${TEMPO_CXI_BACKGROUND_START_FILE:-}" ]]

RUN_CONTRACT="${REPO_ROOT}/eval/sota_4node/tempo_pd_c4_phase_screen_run_contract_v1.json"
PHASE_MANIFEST="${REPO_ROOT}/eval/sota_4node/tempo_pd_c4_phase_screen_manifest_v1.json"
ELASTIC_PROFILE="${REPO_ROOT}/results/tempo_elastic_pd_canonical_discovery_57133688/profiles/real_tempo_pd_elastic_profile_run17_v452.json"
ENDPOINT_PROFILE="${REPO_ROOT}/eval/sota_4node/real_tempo_pd_endpoint_service_profile_c4_screen_v1.json"
[[ -s "${RUN_CONTRACT}" && -s "${PHASE_MANIFEST}" ]]
[[ -s "${ELASTIC_PROFILE}" && -s "${ENDPOINT_PROFILE}" ]]
[[ "$(sha256sum "${RUN_CONTRACT}" | awk '{print $1}')" == \
  89a9ebe8962a8be24db7201740be9b203b9d1d35b5c920e7d53ca48504618d87 ]]
[[ "$(sha256sum "${PHASE_MANIFEST}" | awk '{print $1}')" == \
  c029e24a7cfea6cd567ffd89dc2b49b9822d92514341f73f04582f0f70aa7544 ]]
[[ "$(sha256sum "${ELASTIC_PROFILE}" | awk '{print $1}')" == \
  db65320aa1bbb7c1e095c0d7a4312749327dafbffd904009d83355aa80231a2d ]]
[[ "$(sha256sum "${ENDPOINT_PROFILE}" | awk '{print $1}')" == \
  aa0eb9019ca6fd802901d6b8d0107dd165145c2230cfd3db03ca065c04205bcf ]]

module reset
module load pytorch/2.8.0
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
PORT_SLOT=$((1780 + SLURM_JOB_ID % 20))
REQUEST_RATE=2
MAX_WORKERS=128
STEP_TIME=${TEMPO_PD_C4_STEP_TIME:-01:58:00}

export TEMPO_PD_C4_APPROVED=YES
export TEMPO_PD_C4_RUN_CONTRACT="${RUN_CONTRACT}"
export TEMPO_PD_C4_RUN_CONTRACT_SHA256=89a9ebe8962a8be24db7201740be9b203b9d1d35b5c920e7d53ca48504618d87
export TEMPO_PD_C4_PHASE_DURATION_MS=15000
export TEMPO_PD_C4_COOLDOWN_S=2
export TEMPO_PD_C4_READINESS_S=${TEMPO_PD_C4_READINESS_S:-3600}
export TEMPO_ELASTIC_PD_PROFILE="${ELASTIC_PROFILE}"
export TEMPO_ELASTIC_PD_PROFILE_SCOPE=screen_only
export TEMPO_PD_ENDPOINT_FEEDBACK_MODE=adaptive
export TEMPO_PD_ENDPOINT_SERVICE_PROFILE="${ENDPOINT_PROFILE}"
export TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256=c029e24a7cfea6cd567ffd89dc2b49b9822d92514341f73f04582f0f70aa7544
export TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK=0
export TEMPO_PD_PRESSURE_MODE=disabled
export TEMPO_VLLM_LOAD_SNAPSHOT_MODE=disabled
export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_PD_BENCHMARK_RESET_DECODER_APC=0
export TEMPO_VLLM_DECODER_PREFIX_CACHING=0
export TEMPO_VLLM_MAX_NUM_SEQS=16
export TEMPO_VLLM_SCHEDULING_POLICY=fcfs
export TEMPO_PD_REMOTE_CATCHUP_PRIORITY=0
export TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY=0
export TEMPO_PD_LONG_REMOTE_CATCHUP_PRIORITY=0
export TEMPO_PD_MEDIAN_GUARD_PRIORITY=0
export TEMPO_PD_MEDIUM_REMOTE_CATCHUP_PRIORITY=0
export TEMPO_LMCACHE_NIXL_BACKEND=UCX
export TEMPO_LMCACHE_LOCAL_CPU_GB=16
export TEMPO_LMCACHE_PD_BUFFER_BYTES=2147483648

mkdir -p -- "${RESULT_DIR}"
"${SCRIPT_DIR}/prepare_c4_python_overlay.sh" \
  "${REPO_ROOT}" "${RESULT_DIR}"
export TEMPO_C4_PYTHON_OVERLAY_PREPARE_ARTIFACT="${RESULT_DIR}/python-overlay-prepare.json"
cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=30s 6900s \
  srun --overlap --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
  --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
  --time="${STEP_TIME}" --export=ALL \
  --output="${RESULT_DIR}/slurm-node-%N.stdout.log" \
  --error="${RESULT_DIR}/slurm-node-%N.stderr.log" \
  bash "${SCRIPT_DIR}/c4_phase_screen_pd_node_entry.sh" \
  "${REPO_ROOT}" "${RESULT_DIR}" "${WORKLOAD}" "${HOSTS_CSV}" \
  "${PORT_SLOT}" "${REQUEST_RATE}" "${MAX_WORKERS}" 256 8 3000 250 16000

[[ -s "${RESULT_DIR}/tempo_pd_c4_phase_screen/raw.json" ]]
[[ -s "${RESULT_DIR}/result.json" ]]
echo "C4 phase screen result: ${RESULT_DIR}/result.json"
