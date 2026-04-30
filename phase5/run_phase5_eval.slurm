#!/bin/bash
#SBATCH -J tempo_p5
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:29:00
#SBATCH -A m5320
#SBATCH --gpus-per-node=4
#SBATCH -e logs/phase5_%j.err
#SBATCH -o logs/phase5_%j.out

set -euo pipefail

module load pytorch/2.8.0

export LD_LIBRARY_PATH="/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib:${LD_LIBRARY_PATH:-}"
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29600
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

ROOT="$SCRATCH/Skim-Tempo"
cd "$ROOT"

mkdir -p logs results/phase5/placement results/phase5/qos

echo "============================================================"
echo "  TEMPO v3  Phase 5 — Topology & QoS evaluation"
echo "  Nodes: $SLURM_NNODES  GPUs/node: $SLURM_NTASKS_PER_NODE"
echo "  World size: $WORLD_SIZE"
echo "  Job: $SLURM_JOB_ID"
echo "============================================================"

# ── Helper to launch one eval mode ─────────────────────────────────────────
run_mode () {
  local EXP=$1
  local MODE=$2
  echo ""
  echo ">>> Exp=${EXP}  Mode=${MODE}"
  srun --ntasks="$WORLD_SIZE" \
       --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
       --gpus-per-task=1 \
    python3 phase5/topology_qos_eval.py \
      --exp  "$EXP" \
      --mode "$MODE" \
      --out  "$ROOT/results/phase5"
}

# ── Exp A: Placement strategies ─────────────────────────────────────────────
echo ""
echo "=== Exp A: Topology-Aware Placement ==="
run_mode placement naive
run_mode placement tempo-v2
run_mode placement tempo-v3

# ── Exp B: QoS Traffic Classes ──────────────────────────────────────────────
echo ""
echo "=== Exp B: Hardware QoS Traffic Classes ==="
run_mode qos no-qos
run_mode qos qos-soft
run_mode qos qos-hard

# ── Generate figures ────────────────────────────────────────────────────────
echo ""
echo "=== Generating figures ==="
python3 scripts/make_figures.py --phase5

echo ""
echo "============================================================"
echo "  Phase 5 complete.  Figures in results/figures/"
echo "============================================================"
