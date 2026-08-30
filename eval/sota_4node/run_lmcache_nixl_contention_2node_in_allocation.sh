#!/usr/bin/env bash
set -euo pipefail

[[ "${TEMPO_GO_CROSS_LAYER_COMPONENT_APPROVED:-}" == YES ]] || exit 2
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 2
if [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
  [[ "${SLURM_JOB_NUM_NODES:-}" == 4 ]]
elif [[ -n "${SLURM_JOB_NODES:-}" ]]; then
  [[ "${SLURM_JOB_NODES:-}" == 4 ]]
fi
[[ -z "${SHIFTER_RUNTIME:-}" && -z "${SHIFTER_IMAGE:-}" ]] || exit 2
[[ -z "${UDI:-}" && -z "${CRAY_ROOTFS:-}" && -z "${SLURM_CONTAINER:-}" ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
REPO_ROOT="${TEMPO_GO_REPO_ROOT:-${SCRIPT_REPO_ROOT}}"
SOURCE_ROOT="${TEMPO_GO_SOURCE_SNAPSHOT:-${REPO_ROOT}}"
SOURCE_ROOT=$(realpath -e -- "${SOURCE_ROOT}")
if [[ "${SOURCE_ROOT}" != "${REPO_ROOT}" ]]; then
  case "${SOURCE_ROOT}/" in
    "${REPO_ROOT}/results/"*) ;;
    *) exit 2 ;;
  esac
fi
RESULT_DIR="${TEMPO_GO_CROSS_LAYER_RESULT_DIR:-${REPO_ROOT}/results/tempo_go_cross_layer_component_job_${SLURM_JOB_ID}}"
case "${RESULT_DIR}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
# The same-allocation wrapper creates this directory first so it can bind
# stdout/stderr receipts.  Refuse stale data files rather than requiring a
# nonexistent parent directory, which previously caused an immediate silent
# co-job exit before any NCCL/LMCache process started.
[[ ! -e "${RESULT_DIR}/result.json" ]] || exit 2
[[ ! -e "${RESULT_DIR}/nccl_observer.json" ]] || exit 2
mkdir -p -- "${RESULT_DIR}"

module reset
module load pytorch/2.8.0
# Perlmutter's pytorch module loads the CUDA-matched NERSC NCCL package and
# its AWS OFI/libfabric plugin.  Do not replace that production Slingshot
# path with NCCL's TCP Socket transport: the co-job is meant to exercise the
# same fabric that the global orchestrator observes.
[[ "${NCCL_NET:-}" == "AWS Libfabric" ]] || {
  echo "expected NERSC NCCL AWS Libfabric transport, got ${NCCL_NET:-<unset>}" >&2
  exit 1
}
unset NCCL_IB_DISABLE
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${SOURCE_ROOT}:${REPO_ROOT}"
export TEMPO_GO_SOURCE_SNAPSHOT="${SOURCE_ROOT}"
cd -- "${SOURCE_ROOT}"
export WORLD_SIZE=8
# A co-job is intentionally launched as a nested child step from the
# one-node inference parent.  SLURM_JOB_NODELIST in that child therefore
# describes only the parent step, not the four-node interactive allocation.
# Use an explicit caller-provided host list when available; otherwise resolve
# the authoritative allocation record once.  Never silently reduce a
# cross-layer co-job to one node.
if [[ -n "${TEMPO_GO_C9_COJOB_HOSTS_CSV:-}" ]]; then
  IFS=, read -r -a COJOB_HOSTS <<< "${TEMPO_GO_C9_COJOB_HOSTS_CSV}"
  [[ ${#COJOB_HOSTS[@]} -eq 2 ]]
elif [[ -n "${TEMPO_GO_C9_HOSTS_CSV:-}" ]]; then
  IFS=, read -r -a COJOB_HOSTS <<< "${TEMPO_GO_C9_HOSTS_CSV}"
else
  ALLOCATION_NODELIST=$(
    scontrol show job "${SLURM_JOB_ID}" --oneliner \
      | sed -n 's/.* NodeList=\([^ ]*\).*/\1/p'
  )
  [[ -n "${ALLOCATION_NODELIST}" ]]
  mapfile -t COJOB_HOSTS < <(scontrol show hostnames "${ALLOCATION_NODELIST}")
fi
[[ ${#COJOB_HOSTS[@]} -eq 2 || ${#COJOB_HOSTS[@]} -eq 4 ]]
COJOB_NODELIST="${COJOB_HOSTS[0]},${COJOB_HOSTS[1]}"
export MASTER_ADDR="${COJOB_HOSTS[0]}"
MASTER_PORT_VALUE="${TEMPO_GO_CROSS_LAYER_MASTER_PORT:-$((30100 + SLURM_JOB_ID % 500))}"
NIXL_PORT_BASE_VALUE="${TEMPO_GO_CROSS_LAYER_NIXL_PORT_BASE:-$((30200 + SLURM_JOB_ID % 300))}"
[[ "${MASTER_PORT_VALUE}" =~ ^[0-9]+$ ]]
[[ "${NIXL_PORT_BASE_VALUE}" =~ ^[0-9]+$ ]]
MASTER_PORT_DEC=$((10#${MASTER_PORT_VALUE}))
NIXL_PORT_BASE_DEC=$((10#${NIXL_PORT_BASE_VALUE}))
(( MASTER_PORT_DEC >= 1024 && MASTER_PORT_DEC <= 65535 ))
(( NIXL_PORT_BASE_DEC >= 1024 && NIXL_PORT_BASE_DEC <= 65531 ))
export MASTER_PORT="${MASTER_PORT_DEC}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export TEMPO_GO_CROSS_LAYER_EPOCH="${TEMPO_GO_CROSS_LAYER_EPOCH:-slurm-${SLURM_JOB_ID}-component}"
export TEMPO_GO_NCCL_COMMUNICATOR_ID="${TEMPO_GO_NCCL_COMMUNICATOR_ID:-nixl-nccl-2node-world}"
# NERSC documents hybrid Slingshot message matching as the safe mode when a
# job can fill the NIC hardware receive queues.  Preserve an allocation's
# explicit choice, but make the native contention harness fail less often due
# to pure-hardware queue exhaustion.
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}"
COJOB_STEP_NAME="${TEMPO_GO_CROSS_LAYER_SRUN_STEP_NAME:-tempo-go-cross-layer-cojob-${SLURM_JOB_ID}}"
[[ "${COJOB_STEP_NAME}" =~ ^[A-Za-z0-9_.-]{1,64}$ ]]
# The launcher must be entered from a four-node parent step.  Within that
# parent, the measured inference and co-job are nested ``--overlap`` child
# steps so Slurm can split the parent's 16-GPU shape between the four-node
# inference and two-node co-job.  If entered from a plain allocation shell or
# extern context, bind directly to the allocation job id.  A one-node parent
# step is invalid for this harness because it cannot create the required
# two-node child; the caller's four-node preflight enforces that boundary.
# On Perlmutter, --overlap only permits sharing assigned CPU/GPU/memory
# resources; it does not permit more than three simultaneous Slingshot
# network users per node.  The caller keeps one parent step, this co-job, and
# one C5 step as the complete network-step budget.
TEMPO_GO_SRUN_JOB_ARGS=()
case "${SLURM_STEP_ID:-}" in
  ""|batch|extern)
    TEMPO_GO_SRUN_JOB_ARGS=("--jobid=${SLURM_JOB_ID}")
    ;;
esac
# Match the parent readiness contract.  The allocation is explicitly acquired
# with job_vni for NCCL/UCX experiments, so every nested co-job step must carry
# the same network mode.  Omitting it creates Network=default on Perlmutter
# and can fail with "Error configuring interconnect" even though the parent
# allocation has a valid VNI.
TEMPO_GO_SRUN_NETWORK_MODE="${TEMPO_GO_SRUN_NETWORK_MODE:-job_vni}"
TEMPO_GO_SRUN_NETWORK_ARGS=()
if [[ -n "${TEMPO_GO_SRUN_NETWORK_MODE}" ]]; then
  [[ "${TEMPO_GO_SRUN_NETWORK_MODE}" == "job_vni" \
    || "${TEMPO_GO_SRUN_NETWORK_MODE}" == "disable_rdzv_get" ]] || exit 2
  TEMPO_GO_SRUN_NETWORK_ARGS=(
    "--network=${TEMPO_GO_SRUN_NETWORK_MODE}"
  )
fi
COJOB_BLOCKS="${TEMPO_GO_CROSS_LAYER_BLOCKS:-3}"
[[ "${COJOB_BLOCKS}" =~ ^[0-9]+$ ]]
COJOB_BLOCKS_DEC=$((10#${COJOB_BLOCKS}))
(( COJOB_BLOCKS_DEC >= 1 && COJOB_BLOCKS_DEC <= 100000 ))
COJOB_MINIMUM_ACTIVE_DURATION_S="${TEMPO_GO_CROSS_LAYER_MINIMUM_ACTIVE_DURATION_S:-0}"
COJOB_MAXIMUM_BLOCKS="${TEMPO_GO_CROSS_LAYER_MAXIMUM_BLOCKS:-${COJOB_BLOCKS}}"
COJOB_NO_BACKGROUND_TRANSFER="${TEMPO_GO_CROSS_LAYER_NO_BACKGROUND_TRANSFER:-0}"
[[ "${COJOB_MINIMUM_ACTIVE_DURATION_S}" =~ ^[0-9]+([.][0-9]+)?$ ]]
[[ "${COJOB_MAXIMUM_BLOCKS}" =~ ^[0-9]+$ ]]
[[ "${COJOB_NO_BACKGROUND_TRANSFER}" == 0 || "${COJOB_NO_BACKGROUND_TRANSFER}" == 1 ]]
COJOB_MAXIMUM_BLOCKS_DEC=$((10#${COJOB_MAXIMUM_BLOCKS}))
(( COJOB_MAXIMUM_BLOCKS_DEC >= COJOB_BLOCKS_DEC && COJOB_MAXIMUM_BLOCKS_DEC <= 100000 ))
COJOB_TIMEOUT_SECONDS="${TEMPO_GO_CROSS_LAYER_TIMEOUT_SECONDS:-900}"
[[ "${COJOB_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]
COJOB_TIMEOUT_SECONDS_DEC=$((10#${COJOB_TIMEOUT_SECONDS}))
(( COJOB_TIMEOUT_SECONDS_DEC >= 60 && COJOB_TIMEOUT_SECONDS_DEC <= 14400 ))
COJOB_TIME_LIMIT="${TEMPO_GO_CROSS_LAYER_TIME_LIMIT:-00:20:00}"
[[ "${COJOB_TIME_LIMIT}" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]]
COJOB_REQUESTS="${TEMPO_GO_CROSS_LAYER_REQUESTS:-2}"
COJOB_KV_MIB="${TEMPO_GO_CROSS_LAYER_KV_MIB:-32}"
COJOB_TOKEN_ITERS="${TEMPO_GO_CROSS_LAYER_TOKEN_ITERS:-16}"
COJOB_TRAFFIC_PATTERN="${TEMPO_GO_CROSS_LAYER_TRAFFIC_PATTERN:-paired_1to1}"
COJOB_FOREGROUND_MIB="${TEMPO_GO_CROSS_LAYER_FOREGROUND_MIB:-4}"
COJOB_BLOCK_DELAY_S="${TEMPO_GO_CROSS_LAYER_BLOCK_DELAY_S:-0}"
COJOB_START_DELAY_S="${TEMPO_GO_CROSS_LAYER_START_DELAY_S:-0}"
COJOB_START_FILE="${TEMPO_GO_CROSS_LAYER_START_FILE:-}"
COJOB_START_FILE_TIMEOUT_S="${TEMPO_GO_CROSS_LAYER_START_FILE_TIMEOUT_S:-1800}"
COJOB_MEM_PER_NODE="${TEMPO_GO_CROSS_LAYER_MEM_PER_NODE:-32G}"
COJOB_NCCL_TIMEOUT_SECONDS="${TEMPO_GO_NCCL_TIMEOUT_SECONDS:-60}"
COJOB_NIXL_TIMEOUT_SECONDS="${TEMPO_GO_NIXL_TIMEOUT_SECONDS:-30}"
COJOB_STOP_FILE="${TEMPO_GO_CROSS_LAYER_STOP_FILE:-}"
if [[ -n "${COJOB_STOP_FILE}" ]]; then
  case "${COJOB_STOP_FILE}/" in
    "${RESULT_DIR}/"*) ;;
    *) exit 2 ;;
  esac
fi
if [[ -n "${COJOB_START_FILE}" ]]; then
  [[ "${COJOB_START_FILE}" == /* ]]
  case "${COJOB_START_FILE}/" in
    "${REPO_ROOT}/results/"*) ;;
    *) exit 2 ;;
  esac
fi
[[ "${COJOB_REQUESTS}" =~ ^[0-9]+$ ]]
[[ "${COJOB_KV_MIB}" =~ ^[0-9]+$ ]]
[[ "${COJOB_TOKEN_ITERS}" =~ ^[0-9]+$ ]]
[[ "${COJOB_TRAFFIC_PATTERN}" == paired_1to1 || "${COJOB_TRAFFIC_PATTERN}" == incast_4to1 ]]
[[ "${COJOB_FOREGROUND_MIB}" =~ ^[0-9]+$ ]]
[[ "${COJOB_BLOCK_DELAY_S}" =~ ^[0-9]+([.][0-9]+)?$ ]]
[[ "${COJOB_START_DELAY_S}" =~ ^[0-9]+([.][0-9]+)?$ ]]
[[ "${COJOB_START_FILE_TIMEOUT_S}" =~ ^[0-9]+([.][0-9]+)?$ ]]
[[ "${COJOB_MEM_PER_NODE}" =~ ^[0-9]+[KMG]$ ]]
[[ "${COJOB_NCCL_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]
[[ "${COJOB_NIXL_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]
COJOB_REQUESTS_DEC=$((10#${COJOB_REQUESTS}))
COJOB_KV_MIB_DEC=$((10#${COJOB_KV_MIB}))
COJOB_TOKEN_ITERS_DEC=$((10#${COJOB_TOKEN_ITERS}))
COJOB_FOREGROUND_MIB_DEC=$((10#${COJOB_FOREGROUND_MIB}))
COJOB_START_DELAY_S_DEC="${COJOB_START_DELAY_S}"
COJOB_START_FILE_TIMEOUT_S_DEC="${COJOB_START_FILE_TIMEOUT_S}"
COJOB_NCCL_TIMEOUT_SECONDS_DEC=$((10#${COJOB_NCCL_TIMEOUT_SECONDS}))
COJOB_NIXL_TIMEOUT_SECONDS_DEC=$((10#${COJOB_NIXL_TIMEOUT_SECONDS}))
(( COJOB_REQUESTS_DEC >= 1 && COJOB_REQUESTS_DEC <= 16 ))
(( COJOB_KV_MIB_DEC >= 1 && COJOB_KV_MIB_DEC <= 256 ))
(( COJOB_TOKEN_ITERS_DEC >= 1 && COJOB_TOKEN_ITERS_DEC <= 8192 ))
(( COJOB_FOREGROUND_MIB_DEC >= 1 && COJOB_FOREGROUND_MIB_DEC <= 64 ))
(( COJOB_NCCL_TIMEOUT_SECONDS_DEC >= 5 && COJOB_NCCL_TIMEOUT_SECONDS_DEC <= 3600 ))
(( COJOB_NIXL_TIMEOUT_SECONDS_DEC >= 1 && COJOB_NIXL_TIMEOUT_SECONDS_DEC <= 3600 ))
export NCCL_DEBUG="${TEMPO_GO_NCCL_DEBUG:-WARN}"
# NCCL RAS listens on a localhost socket.  The co-job and the measured C5
# step are independent NCCL jobs that intentionally overlap on two nodes; a
# job-local port prevents their RAS listeners from colliding while leaving the
# production NCCL/Slingshot data path unchanged.  NCCL documents this
# separation for multiple independent jobs sharing a node.
COJOB_RAS_ADDR="${TEMPO_GO_NCCL_RAS_ADDR:-127.0.0.1:$((28028 + SLURM_JOB_ID % 1000))}"
[[ "${COJOB_RAS_ADDR}" =~ ^(localhost|127\.0\.0\.1):[0-9]+$ ]]
export NCCL_RAS_ADDR="${COJOB_RAS_ADDR}"
if [[ "${TEMPO_GO_NCCL_DIAGNOSTICS:-0}" == 1 ]]; then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=ENV,INIT,BOOTSTRAP,NET,COLL,PROXY,RAS,REG,GRAPH,TUNING
  export NCCL_DEBUG_TIMESTAMP_LEVELS=WARN,INFO
  export NCCL_DEBUG_TIMESTAMP_FORMAT='[%F %T.%3f] '
  export NCCL_DEBUG_FILE="${RESULT_DIR}/nccl.%h.%p.log"
else
  unset NCCL_DEBUG_SUBSYS NCCL_DEBUG_TIMESTAMP_LEVELS NCCL_DEBUG_TIMESTAMP_FORMAT NCCL_DEBUG_FILE
fi
# NERSC's MPICH_OFI_CXI_COUNTER_REPORT is an MPI network report.  This
# process is a native PyTorch/NCCL + LMCache/NIXL co-job, not an MPI job; do
# not publish an MPI-only counter as NCCL evidence.  The legacy wrapper may
# still export TEMPO_GO_CXI_COUNTER_REPORT, so retain it only as an explicit
# audit warning and keep the MPI variable unset.
if [[ -n "${TEMPO_GO_CXI_COUNTER_REPORT:-}" ]]; then
  [[ "${TEMPO_GO_CXI_COUNTER_REPORT}" =~ ^[1-5]$ ]]
  echo "ignoring MPI-only TEMPO_GO_CXI_COUNTER_REPORT for native NCCL/NIXL co-job" >&2
fi
unset MPICH_OFI_CXI_COUNTER_REPORT
# libfabric's CXI provider can emit a domain-lifetime telemetry delta when
# explicitly requested.  These are per-CXI-interface counters, not
# per-process counters; the default remains empty and the endpoint sampler
# remains the causal in-band source for vLLM decisions.
FI_CXI_TELEMETRY_VALUE="${TEMPO_GO_FI_CXI_TELEMETRY:-}"
if [[ -n "${FI_CXI_TELEMETRY_VALUE}" ]]; then
  [[ "${FI_CXI_TELEMETRY_VALUE}" =~ ^[a-z0-9_,]+$ ]]
  export FI_CXI_TELEMETRY="${FI_CXI_TELEMETRY_VALUE}"
else
  unset FI_CXI_TELEMETRY
fi
export UCX_TLS=cuda_ipc,cuda_copy,tcp
export UCX_NET_DEVICES=all
export UCX_LOG_LEVEL=warn
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_CROSS_NIC NCCL_ALGO NCCL_PROTO NIXL_PLUGIN_DIR
NCCL_RUNTIME_VERSION="$(${REPO_ROOT}/.vllm_venv/bin/python - <<'PY'
try:
    import torch
    value = torch.cuda.nccl.version()
    if isinstance(value, tuple):
        print(".".join(str(item) for item in value))
    else:
        print(str(value))
except Exception:
    print("")
PY
)"
RESULT_DIR_ENV="${RESULT_DIR}" \
  SLURM_JOB_ID_ENV="${SLURM_JOB_ID}" \
  NCCL_NET_ENV="${NCCL_NET}" \
  MASTER_PORT_ENV="${MASTER_PORT}" \
  NIXL_PORT_BASE_ENV="${NIXL_PORT_BASE_DEC}" \
  TRAFFIC_PATTERN_ENV="${COJOB_TRAFFIC_PATTERN}" \
  COJOB_NODELIST_ENV="${COJOB_NODELIST}" \
  NCCL_RAS_ADDR_ENV="${NCCL_RAS_ADDR}" \
  SLURM_NETWORK_MODE_ENV="${TEMPO_GO_SRUN_NETWORK_MODE:-default}" \
  NCCL_VERSION_ENV="${NCCL_VERSION:-}" \
  NCCL_RUNTIME_VERSION_ENV="${NCCL_RUNTIME_VERSION}" \
  NCCL_DIR_ENV="${NCCL_DIR:-}" \
  NCCL_SOCKET_IFNAME_ENV="${NCCL_SOCKET_IFNAME:-}" \
  NCCL_NET_GDR_LEVEL_ENV="${NCCL_NET_GDR_LEVEL:-}" \
  FI_CXI_DISABLE_HOST_REGISTER_ENV="${FI_CXI_DISABLE_HOST_REGISTER:-}" \
  FI_CXI_TELEMETRY_ENV="${FI_CXI_TELEMETRY:-}" \
  UCX_TLS_ENV="${UCX_TLS}" \
  UCX_NET_DEVICES_ENV="${UCX_NET_DEVICES}" \
  START_FILE_ENV="${COJOB_START_FILE}" \
  START_FILE_TIMEOUT_S_ENV="${COJOB_START_FILE_TIMEOUT_S_DEC}" \
  "${REPO_ROOT}/.vllm_venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "schema": "tempo-go-native-transport-receipt-v1",
    "scope": "same-allocation-nccl-lmcache-cojob",
    "slurm_job_id": int(os.environ["SLURM_JOB_ID_ENV"]),
  "transport": {
        "nccl_net": os.environ["NCCL_NET_ENV"],
        "master_port": int(os.environ["MASTER_PORT_ENV"]),
        "nixl_port_base": int(os.environ["NIXL_PORT_BASE_ENV"]),
        "lmcache_traffic_pattern": os.environ["TRAFFIC_PATTERN_ENV"],
        "fixed_nodelist": os.environ["COJOB_NODELIST_ENV"].split(","),
        "nccl_ras_addr": os.environ["NCCL_RAS_ADDR_ENV"],
        "nccl_version": (
            os.environ["NCCL_RUNTIME_VERSION_ENV"]
            or os.environ["NCCL_VERSION_ENV"]
        ),
        "nccl_environment_version": os.environ["NCCL_VERSION_ENV"],
        "nccl_runtime_version": os.environ["NCCL_RUNTIME_VERSION_ENV"],
        "nccl_dir": os.environ["NCCL_DIR_ENV"],
        "nccl_socket_ifname": os.environ["NCCL_SOCKET_IFNAME_ENV"],
        "nccl_net_gdr_level": os.environ["NCCL_NET_GDR_LEVEL_ENV"],
        "fi_cxi_disable_host_register": os.environ[
            "FI_CXI_DISABLE_HOST_REGISTER_ENV"
        ],
        "fi_cxi_rx_match_mode": os.environ.get("FI_CXI_RX_MATCH_MODE", ""),
        "mpich_ofi_cxi_counter_report": "",
        "fi_cxi_telemetry": os.environ.get("FI_CXI_TELEMETRY_ENV", ""),
        "cxi_telemetry_scope": (
            "per_cxi_interface_domain_delta"
            if os.environ.get("FI_CXI_TELEMETRY_ENV", "")
            else "endpoint_sysfs_delta_only"
        ),
        "ucx_tls": os.environ["UCX_TLS_ENV"],
        "ucx_net_devices": os.environ["UCX_NET_DEVICES_ENV"],
        "phase_start_file": os.environ["START_FILE_ENV"],
        "phase_start_file_timeout_s": float(
            os.environ["START_FILE_TIMEOUT_S_ENV"]
        ),
    },
    "slurm_step": {
        "network": os.environ["SLURM_NETWORK_MODE_ENV"],
        "exact": True,
        "nodes": 2,
        "ntasks": 8,
        "ntasks_per_node": 4,
        "gpus_per_node": 4,
        "cpus_per_task": 32,
        "gpu_bind": "none",
        "cpu_bind": "cores",
    },
    "slingshot_path": "nersc-nccl-ofi-libfabric",
    "production_transport_verified": os.environ["NCCL_NET_ENV"]
    == "AWS Libfabric",
}
path = Path(os.environ["RESULT_DIR_ENV"]) / "native_transport_receipt.json"
path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
PY
mkdir -p -- "${RESULT_DIR}/observer-history"
READY_FILE="${RESULT_DIR}/nixl-ready"

cleanup_component_step() {
  if [[ -n "${COMPONENT_STEP_PID:-}" ]] && kill -0 "${COMPONENT_STEP_PID}" 2>/dev/null; then
    kill -TERM "${COMPONENT_STEP_PID}" 2>/dev/null || true
  fi
  # Never call ``scancel ... $SLURM_JOB_ID`` here.  The allocation job id is
  # the parent interactive allocation, not the nested child step; Slurm can
  # interpret a name-filtered request against that id as a cancellation of
  # the whole four-node allocation.  The foreground timeout/srun process is
  # owned by this shell and is terminated above; --kill-on-bad-exit handles
  # its ranks without touching the parent allocation.
}
trap 'cleanup_component_step; exit 143' TERM
trap 'cleanup_component_step; exit 130' INT
trap cleanup_component_step EXIT

COJOB_STOP_ARGS=()
if [[ -n "${COJOB_STOP_FILE}" ]]; then
  COJOB_STOP_ARGS+=(--stop-file "${COJOB_STOP_FILE}")
fi
COJOB_START_ARGS=()
if [[ -n "${COJOB_START_FILE}" ]]; then
  COJOB_START_ARGS+=(
    --start-file "${COJOB_START_FILE}"
    --start-file-timeout-s "${COJOB_START_FILE_TIMEOUT_S_DEC}"
  )
fi
COJOB_BACKGROUND_ARGS=()
if [[ "${COJOB_NO_BACKGROUND_TRANSFER}" == 1 ]]; then
  COJOB_BACKGROUND_ARGS+=(--no-background-transfer)
fi

set +e
timeout --foreground --signal=TERM --kill-after=10s "${COJOB_TIMEOUT_SECONDS_DEC}s" \
  /usr/bin/srun "${TEMPO_GO_SRUN_JOB_ARGS[@]}" \
    ${TEMPO_GO_CROSS_LAYER_SRUN_OVERLAP:+--overlap} \
  --job-name="${COJOB_STEP_NAME}" \
  --exact --nodes=2 --ntasks=8 --ntasks-per-node=4 \
  --nodelist="${COJOB_NODELIST}" \
  --distribution=block:block --gpus-per-node=4 --gpu-bind=none \
  --cpus-per-task=32 --mem="${COJOB_MEM_PER_NODE}" \
  --cpu-bind=cores --kill-on-bad-exit=1 --wait=5 \
  "${TEMPO_GO_SRUN_NETWORK_ARGS[@]}" \
  --time="${COJOB_TIME_LIMIT}" --export=ALL \
  --output="${RESULT_DIR}/cojob-rank-%t.stdout.log" \
  --error="${RESULT_DIR}/cojob-rank-%t.stderr.log" \
  "${REPO_ROOT}/.vllm_venv/bin/python" \
  -m eval.sota_4node.run_lmcache_nixl_contention_2node \
  --output "${RESULT_DIR}/result.json" \
  --observer-output "${RESULT_DIR}/nccl_observer.json" \
  --observer-history-dir "${RESULT_DIR}/observer-history" \
  --ready-file "${READY_FILE}" \
  --requests "${COJOB_REQUESTS_DEC}" --kv-mib "${COJOB_KV_MIB_DEC}" \
  --token-iters "${COJOB_TOKEN_ITERS_DEC}" --blocks "${COJOB_BLOCKS_DEC}" \
  --traffic-pattern "${COJOB_TRAFFIC_PATTERN}" \
  --minimum-active-duration-s "${COJOB_MINIMUM_ACTIVE_DURATION_S}" \
  --maximum-blocks "${COJOB_MAXIMUM_BLOCKS_DEC}" \
  --foreground-mib "${COJOB_FOREGROUND_MIB_DEC}" \
  --block-delay-s "${COJOB_BLOCK_DELAY_S}" \
  --start-delay-s "${COJOB_START_DELAY_S_DEC}" \
  "${COJOB_START_ARGS[@]}" \
  --process-group-timeout-s "${COJOB_NCCL_TIMEOUT_SECONDS_DEC}" \
  --nixl-transfer-timeout-s "${COJOB_NIXL_TIMEOUT_SECONDS_DEC}" \
  "${COJOB_BACKGROUND_ARGS[@]}" \
  "${COJOB_STOP_ARGS[@]}" \
  --port-base "${NIXL_PORT_BASE_DEC}"
COMPONENT_RC=$?
COMPONENT_STEP_PID=""
trap - TERM INT EXIT
if [[ "${COMPONENT_RC}" -ne 0 ]]; then
  RESULT_DIR_ENV="${RESULT_DIR}" \
  COMPONENT_RC_ENV="${COMPONENT_RC}" \
  COJOB_STEP_NAME_ENV="${COJOB_STEP_NAME}" \
  "${REPO_ROOT}/.vllm_venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

result = Path(os.environ["RESULT_DIR_ENV"]).resolve()
observer = result / "nccl_observer.json"
observer_state = None
observer_sequence = None
if observer.is_file():
    try:
        value = json.loads(observer.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            observer_state = value.get("producer_state")
            observer_sequence = value.get("sequence")
    except (OSError, ValueError):
        pass
payload = {
    "schema": "tempo-go-cross-layer-cojob-failure-v1",
    "failure": "cojob_step_failed",
    "exit_code": int(os.environ["COMPONENT_RC_ENV"]),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "cojob_step_name": os.environ["COJOB_STEP_NAME_ENV"],
    "result_dir": str(result),
    "observer": str(observer),
    "observer_producer_state": observer_state,
    "observer_sequence": observer_sequence,
    "native_only": True,
}
(result / "cojob_failure.json").write_text(
    json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
  exit "${COMPONENT_RC}"
fi

[[ -s "${RESULT_DIR}/result.json" && -s "${RESULT_DIR}/nccl_observer.json" ]]
echo "TEMPO cross-layer component receipt: ${RESULT_DIR} blocks=${COJOB_BLOCKS_DEC} maximum_blocks=${COJOB_MAXIMUM_BLOCKS_DEC} minimum_active_duration_s=${COJOB_MINIMUM_ACTIVE_DURATION_S} background_transfer=$((1 - COJOB_NO_BACKGROUND_TRANSFER)) requests=${COJOB_REQUESTS_DEC} kv_mib=${COJOB_KV_MIB_DEC} token_iters=${COJOB_TOKEN_ITERS_DEC} foreground_mib=${COJOB_FOREGROUND_MIB_DEC} block_delay_s=${COJOB_BLOCK_DELAY_S} start_file=${COJOB_START_FILE:-none} timeout=${COJOB_TIMEOUT_SECONDS_DEC}s time_limit=${COJOB_TIME_LIMIT}"
