#!/usr/bin/env bash
# =============================================================================
# TEMPO Phase 1 — Background I/O Stress Injector
# =============================================================================
# Purpose:
#   Simulate an aggressive checkpoint flush: Local NVMe (/tmp) → Lustre ($PSCRATCH)
#   This stresses the Slingshot 11 NIC and the PCIe Root Complex on Perlmutter's
#   AMD EPYC CPU, causing measurable NCCL All-Reduce bandwidth degradation.
#
# Design Rationale:
#   - SPDK requires root-level NVMe driver unbind → NOT available on Perlmutter.
#   - We use fio with the io_uring engine (Linux 5.1+) for kernel-space async I/O
#     that can saturate PCIe 4.0 bandwidth without root privileges.
#   - The read + write path traverses the SAME PCIe Root Complex as NCCL traffic.
#
# PCIe Contention Mechanism:
#   NVMe read  → PCIe Root Complex (I/O Die) → DRAM
#   DRAM write → PCIe Root Complex (I/O Die) → Slingshot NIC → Lustre
#   GPU NCCL   → PCIe Root Complex (I/O Die) → Slingshot NIC → Network
#                               ▲ BOTTLENECK
#
# Usage:
#   IO_SIZE=32g NUM_JOBS=8 NUMA_NODE=0 bash background_io.sh
#   Or: bash background_io.sh --io-size 32g --num-jobs 8 --numa-node 0
#
# Environment variables (all have defaults):
#   LOCAL_NVME_DIR  : Source directory on local NVMe  (default: /tmp)
#   PSCRATCH        : Destination Lustre directory     (default: $PSCRATCH)
#   IO_SIZE         : Total I/O size per fio job       (default: 16g)
#   BLOCK_SIZE      : I/O block size                   (default: 4m)
#   NUM_JOBS        : Parallel fio jobs                (default: 8)
#   RUNTIME         : Maximum fio runtime in seconds   (default: 3600)
#   NUMA_NODE       : NUMA node for CPU pinning        (default: 0)
#   LOG_DIR         : Directory for log files          (default: /tmp)
# =============================================================================
set -euo pipefail
trap 'echo "[TEMPO BG_IO] ERROR: script failed at line $LINENO"; cleanup; exit 1' ERR
trap 'echo "[TEMPO BG_IO] Interrupted — cleaning up..."; cleanup; exit 0' INT TERM

# --------------------------------------------------------------------------- #
#  Parse arguments / environment defaults
# --------------------------------------------------------------------------- #
LOCAL_NVME_BASE="${LOCAL_NVME_DIR:-/tmp}"
LUSTRE_BASE="${PSCRATCH:-/tmp/lustre_mock}"   # fallback for testing off-cluster
IO_SIZE="${IO_SIZE:-16g}"
BLOCK_SIZE="${BLOCK_SIZE:-4m}"
NUM_JOBS="${NUM_JOBS:-8}"
RUNTIME="${RUNTIME:-3600}"
NUMA_NODE="${NUMA_NODE:-0}"
LOG_DIR="${LOG_DIR:-/tmp}"

# Unique subdirectory per SLURM process to avoid collisions on multi-node runs
PROC_ID="${SLURM_PROCID:-$$}"
LOCAL_WORK_DIR="${LOCAL_NVME_BASE}/tempo_io_rank${PROC_ID}"
LUSTRE_WORK_DIR="${LUSTRE_BASE}/tempo_flush_rank${PROC_ID}"
LOG_FILE="${LOG_DIR}/bg_io_rank${PROC_ID}.log"
FIO_JSON_LOG="${LOG_DIR}/bg_io_rank${PROC_ID}_fio.json"

DUMMY_FILE="${LOCAL_WORK_DIR}/ckpt_source.bin"
DUMMY_SIZE_MB="${DUMMY_SIZE_MB:-8192}"  # Default 8 GB (Llama-3-8B shard); override via env

# --------------------------------------------------------------------------- #
#  Cleanup on exit
# --------------------------------------------------------------------------- #
cleanup() {
    echo "[TEMPO BG_IO] Cleaning up rank ${PROC_ID}..." | tee -a "$LOG_FILE" 2>/dev/null || true
    rm -rf "${LOCAL_WORK_DIR}" "${LUSTRE_WORK_DIR}" 2>/dev/null || true
}

# --------------------------------------------------------------------------- #
#  Logging helper
# --------------------------------------------------------------------------- #
log() {
    echo "[TEMPO BG_IO $(date +%T)] (rank ${PROC_ID}) $*" | tee -a "$LOG_FILE"
}

# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
mkdir -p "${LOCAL_WORK_DIR}" "${LUSTRE_WORK_DIR}" "${LOG_DIR}"

log "===== TEMPO Background I/O Stress ====="
log "Host          : $(hostname)"
log "Local NVMe    : ${LOCAL_WORK_DIR}"
log "Lustre target : ${LUSTRE_WORK_DIR}"
log "IO_SIZE       : ${IO_SIZE}  BLOCK_SIZE: ${BLOCK_SIZE}"
log "NUM_JOBS      : ${NUM_JOBS}  NUMA_NODE: ${NUMA_NODE}"
log "RUNTIME       : ${RUNTIME}s"

# --------------------------------------------------------------------------- #
#  Step 1: Pre-populate local NVMe source file
#          (simulates a checkpoint already staged to local storage)
# --------------------------------------------------------------------------- #
log "Pre-populating ${DUMMY_SIZE_MB}MB source file on local NVMe..."
if [[ ! -f "${DUMMY_FILE}" ]]; then
    dd if=/dev/urandom of="${DUMMY_FILE}" \
        bs=64M count=$(( DUMMY_SIZE_MB / 64 )) \
        status=progress 2>&1 | tee -a "$LOG_FILE"
fi
ACTUAL_SIZE=$(stat -c%s "${DUMMY_FILE}" 2>/dev/null || echo "0")
log "Source file ready: ${DUMMY_FILE} ($(( ACTUAL_SIZE / 1024 / 1024 )) MB)"

# --------------------------------------------------------------------------- #
#  Step 2: Write fio job configuration
# --------------------------------------------------------------------------- #
FIO_CFG="${LOCAL_WORK_DIR}/tempo_stress.fio"
cat > "${FIO_CFG}" << EOF
# fio config — TEMPO background I/O stress
# Simulates checkpoint flush: local NVMe read + Lustre write
[global]
ioengine=io_uring
sqthread_poll=1
iodepth=64
numjobs=${NUM_JOBS}
group_reporting=1
direct=1
bs=${BLOCK_SIZE}
time_based=1
runtime=${RUNTIME}
# NUMA pinning: bind to node ${NUMA_NODE} so I/O and GPU NICs
# share the same PCIe Root Complex — maximises contention
numa_cpu_nodes=${NUMA_NODE}
numa_mem_nodes=${NUMA_NODE}

# ----- Job 1: Sequential read from local NVMe -----
# Simulates reading the checkpoint buffer off fast local storage
[nvme_read]
rw=read
filename=${DUMMY_FILE}
size=${IO_SIZE}
new_group

# ----- Job 2: Sequential write to Lustre ($PSCRATCH) -----
# Simulates flushing checkpoint through Slingshot NIC to Lustre
[lustre_write]
rw=write
filename=${LUSTRE_WORK_DIR}/flush_\${SLURM_PROCID:-0}.bin
size=${IO_SIZE}
new_group
EOF

# --------------------------------------------------------------------------- #
#  Step 3: Launch I/O stress
#          Primary: fio with io_uring (maximises async I/O depth)
#          Fallback: parallel dd processes (if fio is unavailable)
# --------------------------------------------------------------------------- #
if command -v fio &>/dev/null; then
    log "Launching fio (io_uring) stress..."
    numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
        fio "${FIO_CFG}" \
            --output-format=json+ \
            --output="${FIO_JSON_LOG}" \
        2>&1 | tee -a "$LOG_FILE"
    log "fio completed. JSON stats → ${FIO_JSON_LOG}"
else
    log "WARNING: fio not found; falling back to parallel dd processes"
    log "Tip: 'module load fio' or install via 'pip install fio' may not work."
    log "     Recommend building fio from source: https://github.com/axboe/fio"

    # Fallback: parallel dd (less precise but still stresses PCIe BW)
    DD_PIDS=()
    for i in $(seq 0 $(( NUM_JOBS - 1 ))); do
        numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
            bash -c "
                while true; do
                    dd if=${DUMMY_FILE} of=${LUSTRE_WORK_DIR}/flush_${i}.bin \
                        bs=${BLOCK_SIZE} conv=notrunc status=none 2>/dev/null || true
                done
            " &
        DD_PIDS+=($!)
    done

    log "DD pids: ${DD_PIDS[*]}"
    # Wait for RUNTIME seconds then kill
    sleep "${RUNTIME}"
    for pid in "${DD_PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
fi

cleanup
log "===== Background I/O stress finished ====="
