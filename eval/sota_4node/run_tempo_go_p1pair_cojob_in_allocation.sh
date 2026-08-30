#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/require_perlmutter_4node_4h_interactive.sh"
[[ $# -eq 5 ]]

REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKLOAD=$(realpath -e -- "$1")
WORKLOAD_MANIFEST=$(realpath -e -- "$2")
GLOBAL_PROFILE=$(realpath -e -- "$3")
ELASTIC_PROFILE=$(realpath -e -- "$4")
ENDPOINT_PROFILE=$(realpath -e -- "$5")
RESULT_DIR=$(realpath -m -- "${TEMPO_GO_P1PAIR_RESULT_DIR:-${REPO_ROOT}/results/tempo_go_p1pair_cojob_${SLURM_JOB_ID}}")

for path in "${WORKLOAD}" "${WORKLOAD_MANIFEST}" "${GLOBAL_PROFILE}" \
    "${ELASTIC_PROFILE}" "${ENDPOINT_PROFILE}"; do
  case "${path}"/ in "${REPO_ROOT}"/*) ;; *) exit 2 ;; esac
  [[ -s "${path}" ]]
done
case "${RESULT_DIR}"/ in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ ! -e "${RESULT_DIR}" ]]

module reset
module load pytorch/2.8.0
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}"
export TEMPO_LMCACHE_NIXL_BACKEND=UCX
export TEMPO_LMCACHE_LOCAL_CPU_GB=16
export TEMPO_LMCACHE_PD_BUFFER_BYTES=2147483648
export TEMPO_PD_ENDPOINT_FEEDBACK_MODE=adaptive
export TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK=1
export TEMPO_PD_ENDPOINT_ROUTING_POLICY=semantic_epoch_v1
export TEMPO_PD_PRESSURE_MODE=disabled
export TEMPO_VLLM_LOAD_SNAPSHOT_MODE=disabled
export TEMPO_VLLM_DECODER_PREFIX_CACHING=0
export TEMPO_VLLM_MAX_NUM_SEQS=16
export TEMPO_VLLM_ASYNC_SCHEDULING=0
export TEMPO_VLLM_DECODER_MAX_NUM_BATCHED_TOKENS=32768
export TEMPO_VLLM_SCHEDULING_POLICY=fcfs
export TEMPO_PD_REMOTE_DECODE_PLACEMENT=paired
export TEMPO_PD_FRONTEND_PAIR_POLICY=tempo-min-outstanding-decode-tokens-v1
export TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY=0
export TEMPO_PD_BENCHMARK_COLD_MEASURED=1
export TEMPO_GO_C5_ARM=tempo
export TEMPO_ELASTIC_PD_PROFILE_SCOPE=screen_only
export TEMPO_GO_PROFILE="${GLOBAL_PROFILE}"
export TEMPO_GO_ELASTIC_PROFILE="${ELASTIC_PROFILE}"
export TEMPO_GO_ENDPOINT_PROFILE="${ENDPOINT_PROFILE}"
export TEMPO_PD_ENDPOINT_SERVICE_PROFILE="${ENDPOINT_PROFILE}"
export TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256
TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256=$(sha256sum "${WORKLOAD_MANIFEST}" | awk '{print $1}')
export TEMPO_GO_CROSS_LAYER_EPOCH="slurm-${SLURM_JOB_ID}-p1pair-cojob"
export TEMPO_GO_NCCL_COMMUNICATOR_ID="p1pair-cojob-nixl-nccl-world"
export WORLD_SIZE=8
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '3p')"
export MASTER_PORT=$((30100 + SLURM_JOB_ID % 500))
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export UCX_TLS=cuda_ipc,cuda_copy,tcp
export UCX_NET_DEVICES=all
export UCX_LOG_LEVEL=warn
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_CROSS_NIC NCCL_ALGO NCCL_PROTO NIXL_PLUGIN_DIR

mkdir -p -- "${RESULT_DIR}/cojob-history"
mapfile -t TEMPO_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#TEMPO_HOSTS[@]} -eq 4 ]]
COJOB_READY="${RESULT_DIR}/cojob-ready"
COJOB_OBSERVER="${RESULT_DIR}/nccl-observer.json"
export TEMPO_GO_NCCL_TELEMETRY_PATH="${COJOB_OBSERVER}"

timeout --foreground --signal=TERM --kill-after=30s 5400s \
  /usr/bin/srun --exact --nodes=2 --ntasks=8 --ntasks-per-node=4 \
  --nodelist="${TEMPO_HOSTS[2]},${TEMPO_HOSTS[3]}" \
  --distribution=block:block --gpus-per-node=4 --gpu-bind=none \
  --cpus-per-task=8 --kill-on-bad-exit=1 --wait=5 --time=03:00:00 \
  --export=ALL \
  --output="${RESULT_DIR}/cojob-rank-%t.stdout.log" \
  --error="${RESULT_DIR}/cojob-rank-%t.stderr.log" \
  "${REPO_ROOT}/.vllm_venv/bin/python" \
  -m eval.sota_4node.run_lmcache_nixl_contention_2node \
  --output "${RESULT_DIR}/cojob-result.json" \
  --observer-output "${COJOB_OBSERVER}" \
  --observer-history-dir "${RESULT_DIR}/cojob-history" \
  --ready-file "${COJOB_READY}" \
  --requests 1 --kv-mib 8 --token-iters 8 --blocks 1000 \
  --foreground-mib 4 --start-delay-s 120 --port-base "$((30480 + SLURM_JOB_ID % 200))" &
COJOB_PID=$!
trap 'kill "${COJOB_PID}" 2>/dev/null || true; wait "${COJOB_PID}" 2>/dev/null || true' EXIT

timeout --foreground --signal=TERM --kill-after=30s 5400s \
  /usr/bin/srun --exact --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --nodelist="${TEMPO_HOSTS[0]},${TEMPO_HOSTS[1]}" \
  --distribution=block:block --gpus-per-node=4 --gpu-bind=none \
  --cpus-per-task=128 --cpu-bind=cores --kill-on-bad-exit=1 --wait=10 \
  --time=03:00:00 --export=ALL \
  --output="${RESULT_DIR}/inference-node-%N.stdout.log" \
  --error="${RESULT_DIR}/inference-node-%N.stderr.log" \
  "${REPO_ROOT}/.vllm_venv/bin/python" \
  -m eval.sota_4node.vllm_lmcache_tempo_go_p1pair_node \
  --repo-root "${REPO_ROOT}" --result-dir "${RESULT_DIR}" \
  --hosts "${TEMPO_HOSTS[0]},${TEMPO_HOSTS[1]}" \
  --port-slot "$((2500 + SLURM_JOB_ID % 400))" \
  --workload "${WORKLOAD}" --workload-manifest "${WORKLOAD_MANIFEST}" \
  --global-profile "${GLOBAL_PROFILE}" --elastic-profile "${ELASTIC_PROFILE}" \
  --endpoint-profile "${ENDPOINT_PROFILE}" --cojob-ready-file "${COJOB_READY}"

wait "${COJOB_PID}"
trap - EXIT
[[ -s "${RESULT_DIR}/result.json" ]]
[[ -s "${RESULT_DIR}/cojob-result.json" ]]
echo "TEMPO-GO P1PAIR+COJOB result: ${RESULT_DIR}"
