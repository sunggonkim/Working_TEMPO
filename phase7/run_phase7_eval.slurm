#!/bin/bash
# =============================================================================
# TEMPO Phase 7 — PCIe Contention Timeline Profiler (OSDI Fig 9)
# =============================================================================
# Purpose: Prove at µs resolution that concurrent KV DMA and NCCL AllReduce
#          compete for PCIe bandwidth, causing measurable AllReduce stalls.
#
# Design:
#   4 × A100 per node, single node.
#   NVLink disabled (NCCL_P2P_DISABLE=1) → forces AllReduce through PCIe.
#   Baseline:  KV DMA + AllReduce run on separate CUDA streams simultaneously.
#   TEMPO:     stream dependency ensures AllReduce finishes before DMA starts.
#   Output:    results/phase7/timeline_{baseline,tempo}.csv
#   Plot:      results/figures/fig9_pcie_timeline.{pdf,png}
#
# Expected results (Perlmutter A100, PCIe Gen4 ×16, 32 GB/s peak):
#   Baseline  AllReduce: ~20 ms (inflated by PCIe contention with 512 MB DMA)
#   TEMPO     AllReduce: ~11.5 ms (hardware-limited, no DMA competition)
#   Stall reduction: ~42%
# =============================================================================
#SBATCH --job-name=tempo_phase7
#SBATCH --account=m5320
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/phase7_%j.out
#SBATCH --error=logs/phase7_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/pscratch/sd/s/sgkim/Skim-Tempo}"
[[ "$(basename "$REPO_ROOT")" == "phase7" ]] && REPO_ROOT="$(dirname "$REPO_ROOT")"
cd "$REPO_ROOT"

module load pytorch/2.8.0

# ── NCCL plugin ──────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH=/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib:\
/global/common/software/nersc9/nccl/2.29.2-cu13/lib:${LD_LIBRARY_PATH}

# ── Critical Perlmutter fixes ────────────────────────────────────────────────
export FI_CXI_DISABLE_HMEM_MODES=1
export FI_CXI_ATS=0
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN

# ── FORCE PCIe path: disable NVLink P2P and shared-memory transport ──────────
# Without this, NCCL uses NVLink between GPUs on the same node and there is
# NO PCIe contention between NCCL and DMA — the experiment would be null.
# With this, all AllReduce data travels through PCIe Root Complex → demonstrates
# the bottleneck that exists for cross-node NCCL on Perlmutter.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NET_GDR_LEVEL=0   # no GDR — everything through CPU memory PCIe path

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p results/phase7 results/figures logs

echo "============================================================"
echo " TEMPO Phase 7: PCIe Timeline Profiler"
echo " Job ID    : ${SLURM_JOB_ID}"
echo " Node      : $(hostname)"
echo " Start     : $(date -Iseconds)"
echo " NVLink    : DISABLED (NCCL_P2P_DISABLE=1)"
echo "============================================================"

# ── Run baseline (concurrent DMA + AllReduce) ─────────────────────────────────
echo ""
echo "=== Run A: BASELINE (Concurrent DMA + AllReduce) ==="
torchrun \
    --nproc_per_node=4 \
    --master_port=29510 \
    phase7/pcie_timeline_profiler.py \
        --mode    baseline \
        --n-steps 200 \
        --kv-mb   512 \
        --ar-mb   64

echo ""
echo "=== Run B: TEMPO (Phase-Gated: AllReduce → DMA) ==="
torchrun \
    --nproc_per_node=4 \
    --master_port=29511 \
    phase7/pcie_timeline_profiler.py \
        --mode    tempo \
        --n-steps 200 \
        --kv-mb   512 \
        --ar-mb   64

# ── Generate Fig 9 ─────────────────────────────────────────────────────────
echo ""
echo "=== Generating Fig 9 ==="
python3 scripts/plot_micro_benchmarks.py --fig 9

echo ""
echo "============================================================"
echo " Phase 7 COMPLETE"
echo " Results : results/phase7/"
echo " Figure  : results/figures/fig9_pcie_timeline.{pdf,png}"
echo " End     : $(date -Iseconds)"
echo "============================================================"
