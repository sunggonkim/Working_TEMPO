#!/bin/bash
#SBATCH --job-name=tempo_phase6
#SBATCH --account=m4431
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:29:00
#SBATCH --qos=debug
#SBATCH --output=logs/phase6_%j.out
#SBATCH --error=logs/phase6_%j.err

# ============================================================================
# TEMPO Phase 6: v4 Ablation Study — 2N×4×A100, Perlmutter
# ============================================================================
# Measures NCCL BW, P99 ITL, SLO violation rate, and KV transfer bandwidth
# reduction across 6 TEMPO configurations:
#   baseline / v1 / v2 / v3 / v4-no-sparse / v4-full
#
# Expected runtime: ~5 min (simulation, no actual training needed)
# ============================================================================

set -euo pipefail

# ---------- Environment -----------------------------------------------------
module load pytorch/2.8.0

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29600

# Slingshot-11 NCCL plugin
NCCL_PLUGIN=/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib
export LD_LIBRARY_PATH="${NCCL_PLUGIN}:${LD_LIBRARY_PATH:-}"

# NCCL tuning (same as phases 1–5)
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_TIMEOUT=60
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn

# Project root
REPO_DIR=/pscratch/sd/s/sgkim/Skim-Tempo
RESULTS_DIR="${PSCRATCH}/results/phase6"
mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "  TEMPO Phase 6 — v4 Ablation"
echo "  SLURM_JOB_ID    : ${SLURM_JOB_ID}"
echo "  SLURM_NODELIST  : ${SLURM_JOB_NODELIST}"
echo "  RESULTS_DIR     : ${RESULTS_DIR}"
echo "  START TIME      : $(date -Iseconds)"
echo "============================================================"

cd "${REPO_DIR}"

# ---------- Run ablation (rank-0 only; simulation is single-process) --------
srun --ntasks=1 --ntasks-per-node=1 \
    python phase6/tempo_v4_eval.py \
        --results-dir "${RESULTS_DIR}" \
        --n-steps 300 \
        --n-layers 32 \
        --ckpt-interval 50 \
        --modes baseline v1 v2 v3 v4-no-sparse v4-full

echo "============================================================"
echo "  Ablation complete.  Results in: ${RESULTS_DIR}"
echo "  END TIME: $(date -Iseconds)"
echo "============================================================"

# ---------- Generate figures ------------------------------------------------
echo "Generating fig11 (TEMPO v4 ablation plots)..."
python scripts/make_figures.py \
    --phase6 \
    --results-dir "${RESULTS_DIR}" \
    --out-dir     "${RESULTS_DIR}/figures"

echo "Figure generation complete."
echo "============================================================"
echo "  Phase 6 DONE.  Job: ${SLURM_JOB_ID}"
echo "============================================================"
