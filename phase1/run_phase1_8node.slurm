#!/bin/bash
# =============================================================================
# TEMPO Phase 1 — Scale-out: 8-node PCIe Contention Verification
# =============================================================================
# 8-node / 32-GPU All-Reduce — Slingshot rail-optimised topology.
# At this scale the 3-level tree (intra-node NVLink → ring → Slingshot)
# means that NVMe→Lustre I/O on every node simultaneously stresses EVERY
# Slingshot NIC ring segment.  The contention effect should be maximal here.
#
# Output:
#   results/8node/baseline/nccl_bw_rank0.csv
#   results/8node/contention/nccl_bw_rank0.csv
# =============================================================================
#SBATCH --job-name=tempo_p1_8n
#SBATCH -A m5320
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -t 00:30:00
#SBATCH -o logs/phase1_8n_%j.out
#SBATCH -e logs/phase1_8n_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-/pscratch/sd/s/sgkim/Skim-Tempo}"
[[ "$(basename "$REPO_ROOT")" == "phase1" ]] && REPO_ROOT="$(dirname "$REPO_ROOT")"
cd "$REPO_ROOT"

module load pytorch/2.8.0

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$(( SLURM_NNODES * 4 ))

# Slingshot 11 — minimal stable settings (same as 2-node run)
# Fix NCCL plugin API mismatch: use 2.29.2-cu13 plugin instead of module's 2.24.3
export LD_LIBRARY_PATH=/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib:/global/common/software/nersc9/nccl/2.29.2-cu13/lib:${LD_LIBRARY_PATH}
export FI_CXI_ATS=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=WARN

export NUMA_NODE_FOR_IO=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p logs results/8node/baseline results/8node/contention results/8node/figures

echo "==========================================="
echo "  TEMPO Phase 1 [8-node]: PCIe Contention"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Nodes     : $SLURM_JOB_NODELIST"
echo "  World size: $WORLD_SIZE (${SLURM_NNODES} nodes × 4 GPUs)"
echo "==========================================="

# ── Run 1: BASELINE ─────────────────────────────────────────────────────────
echo ""
echo "--- Run 1: BASELINE (no I/O stress) ---"

srun \
    --ntasks-per-node=4 \
    --cpu-bind=cores \
    python phase1/train_llm_profiling.py \
        --mode         nccl_bench \
        --tensor-gb    1.0 \
        --num-steps    100 \
        --warmup-steps 10 \
        --output-dir   results/8node \
        --run-tag      baseline

echo "Baseline complete → results/8node/baseline/"

# ── Run 2: CONTENTION ───────────────────────────────────────────────────────
echo ""
echo "--- Run 2: CONTENTION (all 8 nodes under I/O stress) ---"

srun \
    --overlap \
    --ntasks-per-node=1 \
    --cpu-bind=map_cpu:0 \
    --export=ALL,NUMA_NODE=${NUMA_NODE_FOR_IO},IO_SIZE=8g,BLOCK_SIZE=4m,NUM_JOBS=8,RUNTIME=300,DUMMY_SIZE_MB=2048,LOG_DIR=logs \
    bash phase1/background_io.sh &

BG_IO_PID=$!
echo "I/O stressors launched on all ${SLURM_NNODES} nodes (PID: $BG_IO_PID)"
echo "Waiting 5s for I/O to reach steady state..."
sleep 5

srun \
    --overlap \
    --ntasks-per-node=4 \
    --cpu-bind=cores \
    python phase1/train_llm_profiling.py \
        --mode         nccl_bench \
        --tensor-gb    1.0 \
        --num-steps    100 \
        --warmup-steps 10 \
        --output-dir   results/8node \
        --run-tag      contention

kill $BG_IO_PID 2>/dev/null || true
wait $BG_IO_PID 2>/dev/null || true
echo "Contention run complete → results/8node/contention/"

# ── Plot ─────────────────────────────────────────────────────────────────────
echo ""
echo "--- Plotting Killer Graph (8-node) ---"
python phase3/plot_killer_graph.py \
    --baseline-csv   results/8node/baseline/nccl_bw_rank0.csv \
    --contention-csv results/8node/contention/nccl_bw_rank0.csv \
    --output-dir     results/8node/figures \
    --title          "Phase 1 [8-node / 32 GPU]: PCIe Contention (Perlmutter)"

echo ""
echo "==========================================="
echo "  Phase 1 [8-node] DONE → results/8node/figures/"
echo "==========================================="
