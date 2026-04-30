#!/bin/bash
# =============================================================================
# TEMPO Phase 4 — Lustre I/O vs NCCL Bandwidth Sweep (OSDI Fig 10)
# =============================================================================
# Purpose: Quantitatively prove that background Lustre I/O degrades NCCL
#          AllReduce bandwidth on Perlmutter Slingshot-11 (shared fabric).
#
# Design:
#   2 nodes × 4 × A100 (8 ranks total).
#   Rank 0 writes to Lustre at swept rates: 0, 1, 2, 4, 8, 16, 32 GB/s.
#   All ranks run continuous AllReduce (256 MB tensor, bfloat16).
#   Measurement: 30 s per I/O rate point (+ 5 s warmup), ~245 AllReduce/point.
#
# Expected results (Perlmutter, 200 Gbps Slingshot-11):
#   0  GB/s I/O → NCCL BW ≈ 18–22 GB/s (P50)
#   16 GB/s I/O → NCCL BW ≈  8–11 GB/s (~50% degradation)
#   32 GB/s I/O → NCCL BW ≈  4–7  GB/s (~70% degradation)
#   Clear "knee" around 4 GB/s confirms shared Dragonfly allocation policy.
#
# Total runtime: ~7 rates × (5+30) s × overhead ≈ 8–12 min → fits in debug QOS.
# =============================================================================
#SBATCH --job-name=tempo_io_sweep
#SBATCH --account=m5320
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:29:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/phase4_sweep_%j.out
#SBATCH --error=logs/phase4_sweep_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/pscratch/sd/s/sgkim/Skim-Tempo}"
[[ "$(basename "$REPO_ROOT")" == "phase4" ]] && REPO_ROOT="$(dirname "$REPO_ROOT")"
cd "$REPO_ROOT"

module load pytorch/2.8.0

# ── NCCL plugin ──────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH=/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib:\
/global/common/software/nersc9/nccl/2.29.2-cu13/lib:${LD_LIBRARY_PATH}

# ── Critical Perlmutter fixes ────────────────────────────────────────────────
export FI_CXI_DISABLE_HMEM_MODES=1
export FI_CXI_ATS=0
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_TIMEOUT=60
export NCCL_DEBUG=WARN

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29520
export WORLD_SIZE=$(( SLURM_NNODES * 4 ))

# Lustre destination for I/O flood writes
export FLOOD_DIR="${PSCRATCH}/tempo_io_sweep_${SLURM_JOB_ID}"
mkdir -p "$FLOOD_DIR" results/phase4/io_nccl_sweep results/figures logs

SRUN_FLAGS=(
    --ntasks-per-node=4
    --gpus-per-node=4
    --cpu-bind=map_cpu:0,8,16,24
    --export=ALL
)

echo "============================================================"
echo " TEMPO I/O-NCCL Sweep Experiment"
echo " Job ID    : ${SLURM_JOB_ID}"
echo " Nodes     : ${SLURM_JOB_NODELIST}"
echo " World     : ${WORLD_SIZE} ranks"
echo " Flood dir : ${FLOOD_DIR}"
echo " Start     : $(date -Iseconds)"
echo "============================================================"

srun "${SRUN_FLAGS[@]}" \
    python3 -u phase4/io_nccl_sweep.py \
        --measure-secs 30 \
        --warmup-secs  5 \
        --ar-mb        256 \
        --io-chunk-mb  128 \
        --output-dir   results/phase4/io_nccl_sweep \
        --lustre-dir   "${FLOOD_DIR}"

# ── Cleanup Lustre flood files ────────────────────────────────────────────────
rm -rf "${FLOOD_DIR}" || true

# ── Generate Fig 10 ──────────────────────────────────────────────────────────
echo ""
echo "=== Generating Fig 10 ==="
python3 scripts/plot_micro_benchmarks.py --fig 10

echo ""
echo "============================================================"
echo " I/O-NCCL Sweep COMPLETE"
echo " Results : results/phase4/io_nccl_sweep/io_nccl_sweep.csv"
echo " Figure  : results/figures/fig10_io_nccl_sweep.{pdf,png}"
echo " End     : $(date -Iseconds)"
echo "============================================================"
