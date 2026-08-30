#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/require_perlmutter_4node_4h_interactive.sh"
: "${TEMPO_PD_C3_APPROVED:?explicit C3 approval required}"
[[ "${TEMPO_PD_C3_APPROVED}" == YES ]]
export TEMPO_PD_C3_APPROVED
[[ $# -eq 2 ]]

REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKLOAD=$(realpath -e -- "$1")
RESULT_DIR=$(realpath -m -- "$2")
case "${WORKLOAD}" in "${REPO_ROOT}"/results/*) ;; *) exit 2 ;; esac
case "${RESULT_DIR}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ -s "${WORKLOAD}" && ! -e "${RESULT_DIR}" ]]
[[ -z "${TEMPO_CXI_BACKGROUND_DUTY_CYCLE:-}" ]]
[[ -z "${TEMPO_CXI_BACKGROUND_START_FILE:-}" ]]

module reset
module load pytorch/2.8.0
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]]
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
PORT_SLOT=$((1760 + SLURM_JOB_ID % 20))
REQUEST_RATE=2
MAX_WORKERS=128

export TEMPO_PD_KV_ATTR_APPROVED=YES
export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_PD_KV_ATTR_PHASE_DURATION_MS=8000
export TEMPO_PD_KV_ATTR_COOLDOWN_S=2
export TEMPO_PD_KV_ATTR_RATES=0,4,8,12
export TEMPO_PD_KV_ATTR_REPETITIONS=2
export TEMPO_PD_KV_ATTR_ARM_ORDER=paired_abba
export TEMPO_PD_KV_ATTR_DECODER_HOT_RATE=22.4
export TEMPO_PD_KV_ATTR_READINESS_S=${TEMPO_PD_KV_ATTR_READINESS_S:-3600}
export TEMPO_PD_C3_COUPLED_MANIFEST="${REPO_ROOT}/eval/sota_4node/tempo_pd_c3_coupled_abba_manifest_v2.json"
export TEMPO_VLLM_MAX_NUM_SEQS=16
export TEMPO_VLLM_DECODER_PREFIX_CACHING=0
export TEMPO_LMCACHE_NIXL_BACKEND=UCX
mkdir -p -- "${RESULT_DIR}"
cd -- "${REPO_ROOT}"
timeout --foreground --signal=TERM --kill-after=30s 7200s \
  srun --overlap --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus-per-task=4 --gpu-bind=none \
  --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
  --time=01:58:00 --export=ALL \
  --output="${RESULT_DIR}/slurm-node-%N.stdout.log" \
  --error="${RESULT_DIR}/slurm-node-%N.stderr.log" \
  bash "${SCRIPT_DIR}/kv_only_pd_node_entry.sh" \
  "${REPO_ROOT}" "${RESULT_DIR}" "${WORKLOAD}" "${HOSTS_CSV}" \
  "${PORT_SLOT}" "${REQUEST_RATE}" "${MAX_WORKERS}" 128 8 3000 250 16000

[[ -s "${RESULT_DIR}/tempo_pd_kv_only_attribution/raw.json" ]]
[[ -s "${RESULT_DIR}/result.json" ]]
CHARACTERIZATION="${RESULT_DIR}/kv_only_characterization_v3.json"
GATE="${RESULT_DIR}/c3_abba_gate_v1.json"

env -u SLURM_GPUS_PER_TASK -u SLURM_TRES_PER_TASK \
  srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=none \
  --cpu-bind=cores --time=00:14:00 --export=ALL \
  "${REPO_ROOT}/.vllm_venv/bin/python" -m \
  eval.sota_4node.analyze_tempo_pd_kv_only_attribution \
  --input "${RESULT_DIR}/tempo_pd_kv_only_attribution/raw.json" \
  --output "${CHARACTERIZATION}"

env -u SLURM_GPUS_PER_TASK -u SLURM_TRES_PER_TASK \
  srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=none \
  --cpu-bind=cores --time=00:14:00 --export=ALL \
  "${REPO_ROOT}/.vllm_venv/bin/python" -m \
  eval.sota_4node.analyze_tempo_pd_c3_coupled_abba \
  --result "${RESULT_DIR}/result.json" \
  --characterization "${CHARACTERIZATION}" \
  --manifest "${TEMPO_PD_C3_COUPLED_MANIFEST}" \
  --output "${GATE}"

[[ -s "${CHARACTERIZATION}" && -s "${GATE}" ]]
echo "Coupled C3 ABBA result: ${RESULT_DIR}/result.json"
echo "Coupled C3 ABBA gate: ${GATE}"
