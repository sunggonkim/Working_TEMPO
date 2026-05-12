#!/bin/bash
#SBATCH --job-name=tempo_nexus
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/phase8_nexus_%j.err
#SBATCH --error=logs/phase8_nexus_%j.err

# ─── TEMPO-Nexus: Distributed Staggered Checkpoint Protocol Evaluation ───────
# 8 nodes × 4 A100 = 32 GPUs total
# Compares: baseline (flood) vs tempo-nexus (DSCP staggered)
# Runtime: ~15 min per mode, 30 min total
# ─────────────────────────────────────────────────────────────────────────────

module load pytorch/2.8.0

export FI_CXI_DISABLE_HMEM_MODES=1
export NCCL_NET_PLUGIN=slingshot11
export LD_LIBRARY_PATH=/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PSCRATCH/Skim-Tempo:$PYTHONPATH

# NCCL tuning for Slingshot-11 Dragonfly+
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -1)
export MASTER_PORT=29500

cd $PSCRATCH/Skim-Tempo

N_STEPS=300
CKPT_INTERVAL=50

echo "========================================================"
echo " TEMPO-Nexus Phase 8 Evaluation"
echo " Nodes: $SLURM_NNODES  Tasks: $SLURM_NTASKS"
echo " Mode: baseline → results/phase8/nexus/baseline/"
echo "========================================================"

# ── Run 1: Baseline (simultaneous flush — collective flood) ──────────────────
echo "[$(date)] Starting baseline run..."
srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=4 \
     bash -c "
     export RANK=\$SLURM_PROCID
     export LOCAL_RANK=\$SLURM_LOCALID
     export WORLD_SIZE=\$SLURM_NTASKS
     python3 phase8/nexus_eval.py \
         --mode baseline \
         --n-steps $N_STEPS \
         --ckpt-interval $CKPT_INTERVAL \
         --out-dir results/phase8/nexus
     "

echo "[$(date)] Baseline done."
sleep 10

# ── Run 2: TEMPO-Nexus (DSCP staggered flush) ────────────────────────────────
echo "[$(date)] Starting tempo-nexus run..."
srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=4 \
     bash -c "
     export RANK=\$SLURM_PROCID
     export LOCAL_RANK=\$SLURM_LOCALID
     export WORLD_SIZE=\$SLURM_NTASKS
     python3 phase8/nexus_eval.py \
         --mode tempo-nexus \
         --n-steps $N_STEPS \
         --ckpt-interval $CKPT_INTERVAL \
         --base-window-ms 200.0 \
         --out-dir results/phase8/nexus
     "

echo "[$(date)] TEMPO-Nexus done."

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "Results written to: results/phase8/nexus/"
echo "To generate figures:"
echo "  python3 scripts/plot_micro_benchmarks.py --fig 12"
echo ""
ls -lh results/phase8/nexus/ 2>/dev/null || echo "(results not yet staged)"
