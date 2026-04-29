#!/bin/bash
# =============================================================================
# TEMPO Phase 1 — PCIe Contention Verification on NERSC Perlmutter
# =============================================================================
# Goal: Produce the "Killer Graph" — show that background checkpoint flush
#       (NVMe → Lustre) degrades NCCL All-Reduce bandwidth on the SAME node
#       even though storage and GPU-communication NICs are physically separate.
#
# Experiment design:
#   Run 1 (BASELINE):   Only NCCL benchmark, no I/O stress → flat BW line
#   Run 2 (CONTENTION): NCCL benchmark + concurrent background_io.sh  → BW drop
#
# Output:
#   results/baseline/nccl_bw_rank0.csv
#   results/contention/nccl_bw_rank0.csv
#   → feed these to phase3/plot_killer_graph.py
#
# Perlmutter node topology:
#   1x AMD EPYC 7763 (64 cores, 2 NUMA domains in NPS=2 mode)
#   4x NVIDIA A100 40GB  (PCIe Gen4 x16)
#   4x HPE Slingshot 11 NIC (200 Gbps, PCIe Gen4 x16)
#   1.5 TB local NVMe at /tmp (PCIe Gen4)
#
# PCIe contention is maximised by pinning BOTH the I/O threads and
# GPU rank 0's NCCL traffic to NUMA node 0 (same PCIe Root Complex).
# =============================================================================
#SBATCH --job-name=tempo_phase1
#SBATCH -A m5320
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -t 00:15:00
#SBATCH -o logs/phase1_%j.out
#SBATCH -e logs/phase1_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-/pscratch/sd/s/sgkim/Skim-Tempo}"
[[ "$(basename "$REPO_ROOT")" == "phase1" ]] && REPO_ROOT="$(dirname "$REPO_ROOT")"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
#  Environment setup
# --------------------------------------------------------------------------- #
module load pytorch/2.8.0           # Loads PyTorch + CUDA 12.9 + NCCL

# Add workspace to Python path (for `tempo` package)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Distributed rendezvous
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$(( SLURM_NNODES * 4 ))   # 4 GPUs per node

# Perlmutter Slingshot 11 NIC — required NCCL/libfabric settings
export FI_CXI_ATS=0                          # Disable Address Translation Services
export NCCL_NET_GDR_LEVEL=3                  # Enable GPU Direct RDMA
export NCCL_CROSS_NIC=1                      # Allow NIC crossover for large messages
export NCCL_DEBUG=WARN

# NUMA pinning for MAXIMUM PCIe contention:
# On AMD EPYC Milan (NPS=2), NUMA 0 = cores 0-31.
# GPU0 + NIC0 share the PCIe switch closest to NUMA 0.
# Pinning both the I/O threads and NCCL rank 0 to NUMA 0 forces all
# traffic through the same PCIe Root Complex.
export NUMA_NODE_FOR_IO=0
# Expose all 4 GPUs per task — required for NCCL P2P intra-node transport.
# --gpus-per-task=1 restricts visibility and breaks peer device access.
export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p logs results/baseline results/contention

echo "==========================================="
echo "  TEMPO Phase 1: PCIe Contention Study"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Nodes     : $SLURM_JOB_NODELIST"
echo "  World size: $WORLD_SIZE"
echo "  Repo root : $REPO_ROOT"
echo "==========================================="

# --------------------------------------------------------------------------- #
#  Run 1: BASELINE — NCCL benchmark with zero I/O stress
# --------------------------------------------------------------------------- #
echo ""
echo "--- Run 1: BASELINE (no I/O stress) ---"

srun \
    --ntasks-per-node=4 \
    --cpu-bind=cores \
    python phase1/train_llm_profiling.py \
        --mode       nccl_bench \
        --tensor-gb  1.0 \
        --num-steps  100 \
        --warmup-steps 10 \
        --output-dir results \
        --run-tag    baseline

echo "Baseline complete. CSV → results/baseline/"

# --------------------------------------------------------------------------- #
#  Run 2: CONTENTION — NCCL benchmark + concurrent background I/O stress
# --------------------------------------------------------------------------- #
echo ""
echo "--- Run 2: CONTENTION (with background I/O stress) ---"

srun \
    --ntasks-per-node=1 \
    --cpu-bind=map_cpu:0 \
    --export=ALL,NUMA_NODE=${NUMA_NODE_FOR_IO},IO_SIZE=8g,BLOCK_SIZE=4m,NUM_JOBS=8,RUNTIME=300,DUMMY_SIZE_MB=2048,LOG_DIR=logs \
    bash phase1/background_io.sh &

BG_IO_PID=$!
echo "Background I/O stress PID: $BG_IO_PID"

echo "Waiting 5s for I/O to reach PCIe saturation..."
sleep 5

srun \
    --ntasks-per-node=4 \
    --cpu-bind=cores \
    python phase1/train_llm_profiling.py \
        --mode       nccl_bench \
        --tensor-gb  1.0 \
        --num-steps  100 \
        --warmup-steps 10 \
        --output-dir results \
        --run-tag    contention

# Stop background I/O
kill $BG_IO_PID 2>/dev/null || true
wait $BG_IO_PID 2>/dev/null || true
echo "Contention run complete. CSV → results/contention/"

# --------------------------------------------------------------------------- #
#  Plot the Killer Graph
# --------------------------------------------------------------------------- #
echo ""
echo "--- Plotting Killer Graph ---"
python phase3/plot_killer_graph.py \
    --baseline-csv  results/baseline/nccl_bw_rank0.csv \
    --contention-csv results/contention/nccl_bw_rank0.csv \
    --output-dir    results/figures \
    --title         "Phase 1: PCIe Contention Verification (Perlmutter)"

echo ""
echo "==========================================="
echo "  Phase 1 DONE.  Figures → results/figures/"
echo "==========================================="
