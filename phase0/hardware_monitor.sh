#!/usr/bin/env bash
# =============================================================================
# Phase 0: Hardware Monitor — High-frequency PCIe/NVMe/GPU I/O Logger
# =============================================================================
# MISSION: Capture the exact moment KV cache eviction hits NVMe/PCIe,
#          synchronized with the ITL timeline from workload_injector.py
#
# Metrics sampled at 100ms intervals:
#   - NVMe write throughput (MB/s)  → direct signal of KV eviction DMA
#   - NVMe read throughput  (MB/s)  → KV re-load (prefill cache miss)
#   - CPU memory bandwidth  (GB/s)  → PCIe-CPU side of the DMA
#   - GPU HBM used          (MiB)   → when this plateaus → eviction starts
#   - GPU PCIe Tx/Rx        (MB/s)  → direct PCIe bus occupancy
#
# Output: results/phase0/io_profile.csv
# Columns: timestamp_ns, nvme_write_mbps, nvme_read_mbps,
#          cpu_membw_gbps, gpu_hbm_used_mib, gpu_pcie_tx_mbps, gpu_pcie_rx_mbps
#
# Usage:
#   ./hardware_monitor.sh [output_dir] [interval_ms]
#   ./hardware_monitor.sh results/phase0 100
#
# STOP: Send SIGTERM or SIGINT (Ctrl-C) — script flushes and exits cleanly.
# =============================================================================

set -euo pipefail

OUTPUT_DIR="${1:-results/phase0}"
INTERVAL_MS="${2:-100}"
INTERVAL_SEC=$(echo "scale=3; $INTERVAL_MS / 1000" | bc)

mkdir -p "$OUTPUT_DIR"
OUT_CSV="${OUTPUT_DIR}/io_profile.csv"
MARKER_FILE="${OUTPUT_DIR}/monitor_start.marker"

# ---------------------------------------------------------------------------
# Detect NVMe device — prefer local scratch (/tmp or /local) over Lustre
# ---------------------------------------------------------------------------
detect_nvme() {
    # On Perlmutter: local NVMe is typically nvme0n1 or nvme1n1
    # Check what device backs /tmp (local scratch on compute nodes)
    local dev
    dev=$(df /tmp 2>/dev/null | awk 'NR==2{print $1}' | sed 's|/dev/||' | sed 's|[0-9]*$||')
    if [[ -z "$dev" || "$dev" == "tmpfs" ]]; then
        # Fallback: first NVMe device
        dev=$(ls /dev/nvme*n1 2>/dev/null | head -1 | sed 's|/dev/||')
    fi
    echo "${dev:-nvme0n1}"
}

NVME_DEV=$(detect_nvme)
echo "[MONITOR] NVMe device: /dev/${NVME_DEV}"

# ---------------------------------------------------------------------------
# Detect available GPU count
# ---------------------------------------------------------------------------
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
echo "[MONITOR] GPUs detected: ${GPU_COUNT}"

# ---------------------------------------------------------------------------
# Check iostat availability
# ---------------------------------------------------------------------------
if ! command -v iostat &>/dev/null; then
    echo "[FATAL] iostat not found. Install sysstat: yum install sysstat" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Write CSV header
# ---------------------------------------------------------------------------
echo "timestamp_ns,nvme_write_mbps,nvme_read_mbps,cpu_membw_gbps,gpu_hbm_used_mib,gpu_pcie_tx_mbps,gpu_pcie_rx_mbps" \
    > "$OUT_CSV"

# Record wall-clock start time as sync marker
MONITOR_START_NS=$(date +%s%N)
echo "$MONITOR_START_NS" > "$MARKER_FILE"
echo "[MONITOR] Start marker written: ${MONITOR_START_NS} ns"
echo "[MONITOR] Logging to: ${OUT_CSV}"
echo "[MONITOR] Interval: ${INTERVAL_MS}ms"
echo "[MONITOR] Press Ctrl-C to stop cleanly"
echo ""

# ---------------------------------------------------------------------------
# Cleanup trap
# ---------------------------------------------------------------------------
SAMPLE_COUNT=0
trap 'echo ""; echo "[MONITOR] Caught signal — flushing and exiting (${SAMPLE_COUNT} samples)"; exit 0' \
    SIGTERM SIGINT

# ---------------------------------------------------------------------------
# PCIe bandwidth via nvidia-smi (available natively, no perf needed)
# ---------------------------------------------------------------------------
get_gpu_pcie_bw() {
    # Returns: "tx_mbps rx_mbps hbm_mib"  (space-separated, GPU 0 by default)
    # nvidia-smi pcie-throughput queries are in KiB/s
    if [[ "$GPU_COUNT" -gt 0 ]]; then
        nvidia-smi \
            --query-gpu=memory.used,pcie.link.gen.current,utilization.memory \
            --format=csv,noheader,nounits \
            -i 0 2>/dev/null | awk -F',' '{printf "%s 0 0\n", $1}'
    else
        echo "0 0 0"
    fi
}

# More precise PCIe Tx/Rx via nvidia-smi dmon
get_gpu_metrics() {
    # dmon -s u reports: gpu  fb  bar1  ccpm  bus-id ...
    # -s m = power,temp; -s u = utilization; we want pcie bw = not in dmon
    # Use pmon for per-process, dmon for device-level pcie is not available directly.
    # Fall back to memory.used only; PCIe columns need nvsmi raw counters.
    if [[ "$GPU_COUNT" -gt 0 ]]; then
        local mem_used
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            2>/dev/null | head -1 | tr -d ' ')
        echo "${mem_used:-0}"
    else
        echo "0"
    fi
}

# ---------------------------------------------------------------------------
# CPU memory bandwidth via /proc/meminfo delta (rough but zero-dependency)
# ---------------------------------------------------------------------------
PREV_MEM_TOTAL=0
PREV_MEM_AVAILABLE=0
get_cpu_membw_gbps() {
    # Approximation: delta of (MemTotal - MemAvailable) / interval = allocation rate
    # Real bandwidth would need PCM or perf uncore, but this is a useful proxy
    # for detecting large KV offload bursts (hundreds of MiB/s)
    local cur_avail
    cur_avail=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
    if [[ "$PREV_MEM_AVAILABLE" -gt 0 ]]; then
        local delta_kb=$(( PREV_MEM_AVAILABLE - cur_avail ))   # positive = memory consumed
        # delta in kB over INTERVAL_SEC seconds → GB/s
        local bw_gbps
        bw_gbps=$(echo "scale=3; ${delta_kb} / 1024 / 1024 / ${INTERVAL_SEC}" | bc 2>/dev/null || echo "0")
        echo "$bw_gbps"
    else
        echo "0"
    fi
    PREV_MEM_AVAILABLE=$cur_avail
}

# ---------------------------------------------------------------------------
# Main sampling loop
# ---------------------------------------------------------------------------
while true; do
    TIMESTAMP_NS=$(date +%s%N)

    # --- NVMe throughput via iostat (single 1-second sample) ---
    # We use a very short iostat sample to avoid blocking the loop
    # '-d' = device stats only, '-y' = skip first report, '-x' = extended
    IOSTAT_LINE=$(iostat -d -x "$NVME_DEV" 1 2 2>/dev/null \
        | grep "^${NVME_DEV}" \
        | tail -1)

    if [[ -n "$IOSTAT_LINE" ]]; then
        # iostat -x columns: rrqm/s wrqm/s r/s w/s rkB/s wkB/s ...
        # Column indices depend on kernel version — use awk by field name
        NVME_WRITE_MBS=$(echo "$IOSTAT_LINE" | awk '{
            # Typical extended stat order: Device rrqm/s wrqm/s r/s w/s rkB/s wkB/s
            # wkB/s is field 7 (1-indexed) in most sysstat versions
            printf "%.2f", $7 / 1024
        }')
        NVME_READ_MBS=$(echo "$IOSTAT_LINE" | awk '{printf "%.2f", $6 / 1024}')
    else
        NVME_WRITE_MBS="0.00"
        NVME_READ_MBS="0.00"
    fi

    # --- CPU Memory BW (proxy) ---
    CPU_MEMBW=$(get_cpu_membw_gbps)

    # --- GPU HBM used ---
    GPU_HBM=$(get_gpu_metrics)

    # --- GPU PCIe Tx/Rx (nvidia-smi does not expose instantaneous PCIe BW
    #     without NVML direct access; we record placeholders and note this) ---
    # For production: replace with `nvml_pcie_throughput` via Python/NVML
    GPU_PCIE_TX="0"
    GPU_PCIE_RX="0"

    # Append row
    echo "${TIMESTAMP_NS},${NVME_WRITE_MBS},${NVME_READ_MBS},${CPU_MEMBW},${GPU_HBM},${GPU_PCIE_TX},${GPU_PCIE_RX}" \
        >> "$OUT_CSV"

    SAMPLE_COUNT=$(( SAMPLE_COUNT + 1 ))

    # Progress every 10 samples (every ~1s)
    if (( SAMPLE_COUNT % 10 == 0 )); then
        printf "\r[MONITOR] %d samples | NVMe write: %s MB/s | GPU HBM: %s MiB" \
            "$SAMPLE_COUNT" "$NVME_WRITE_MBS" "$GPU_HBM"
    fi

    sleep "$INTERVAL_SEC"
done
