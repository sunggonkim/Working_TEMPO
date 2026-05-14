#!/bin/bash
# =============================================================================
# TEMPO Node Scaling Experiment — AllReduce Regression Curve
# =============================================================================
# Purpose: Measure how AllReduce latency degrades as concurrent checkpoint I/O
#          is injected, across 2 / 4 / 8 / 16 / 32 nodes.
#
# Hypothesis: Baseline AllReduce latency grows super-linearly with node count
#             because Slingshot quota contention probability P_conflict → 1.
#             TEMPO holds latency near the hardware floor at every scale.
#
# Each job covers ONE node count (N).  Submit all five to get the full curve:
#   for N in 2 4 8 16 32; do
#     sbatch --nodes=$N eval/node_scaling/run_node_scaling.slurm
#   done
#
# Output:
#   results/node_scaling/scaling_N{N}_baseline.csv
#   results/node_scaling/scaling_N{N}_tempo.csv
#   (combined by plot_node_scaling.py into one figure)
#
# SLURM parameters below are for N=2 (minimum).  The submit wrapper above
# overrides --nodes automatically.
# =============================================================================
#SBATCH --job-name=tempo_scaling
#SBATCH --account=m5320
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=00:40:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/scaling_%N_%j.out
#SBATCH --error=logs/scaling_%N_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/pscratch/sd/s/sgkim/Skim-Tempo}"
cd "$REPO_ROOT"

module load pytorch/2.8.0
export LD_LIBRARY_PATH=/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib:\
/global/common/software/nersc9/nccl/2.29.2-cu13/lib:${LD_LIBRARY_PATH}

export FI_CXI_DISABLE_HMEM_MODES=1
export FI_CXI_ATS=0
export NCCL_SOCKET_IFNAME=hsn
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=0
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29600
export WORLD_SIZE=$(( SLURM_NNODES * 4 ))
N=${SLURM_NNODES}

export LOCAL_NVME=/tmp/tempo_scaling_${SLURM_JOB_ID}
export CKPT_LUSTRE=${PSCRATCH}/tempo_scaling_${SLURM_JOB_ID}
mkdir -p logs results/node_scaling results/figures \
         "${CKPT_LUSTRE}/baseline" "${CKPT_LUSTRE}/tempo"

SRUN_FLAGS=(
    --ntasks-per-node=4
    --cpu-bind=map_cpu:0,8,16,24
    --export=ALL
)

TRAIN_FLAGS=(
    --model-size    1b
    --num-steps     80
    --warmup-steps  10
    --ckpt-every    20
    --batch-size    1
    --seq-len       512
    --local-nvme    "${LOCAL_NVME}"
    --flush-chunk-mb 32
    --adaptive-chunk
    --scale-exp               # enables allreduce-only CSV output format
    --n-nodes       "${N}"
)

echo "==========================================="
echo "  TEMPO Node Scaling — N=${N} nodes"
echo "  Job ID     : ${SLURM_JOB_ID}"
echo "  World size : ${WORLD_SIZE}"
echo "  Nodes      : ${SLURM_JOB_NODELIST}"
echo "==========================================="

# --------------------------------------------------------------------------- #
#  Run A: BASELINE
# --------------------------------------------------------------------------- #
echo ""
echo "====== BASELINE (N=${N}) ======"
export MASTER_PORT=29600

srun "${SRUN_FLAGS[@]}" \
    python eval/node_scaling/train_scaling.py \
        "${TRAIN_FLAGS[@]}" \
        --mode       baseline \
        --lustre-dir "${CKPT_LUSTRE}/baseline" \
        --output-csv "results/node_scaling/scaling_N${N}_baseline.csv"

sleep 5

# --------------------------------------------------------------------------- #
#  Run B: TEMPO
# --------------------------------------------------------------------------- #
echo ""
echo "====== TEMPO (N=${N}) ======"
export MASTER_PORT=29601

srun "${SRUN_FLAGS[@]}" \
    python eval/node_scaling/train_scaling.py \
        "${TRAIN_FLAGS[@]}" \
        --mode       tempo \
        --lustre-dir "${CKPT_LUSTRE}/tempo" \
        --output-csv "results/node_scaling/scaling_N${N}_tempo.csv"

# --------------------------------------------------------------------------- #
#  Incremental plot (works with partial data; full figure after all N run)
# --------------------------------------------------------------------------- #
echo ""
echo "====== Incremental plot ======"
python eval/node_scaling/plot_node_scaling.py \
    --results-dir results/node_scaling \
    --output-dir  results/figures

echo ""
echo "==========================================="
echo "  Scaling N=${N} DONE  (job ${SLURM_JOB_ID})"
echo "  CSV → results/node_scaling/scaling_N${N}_{baseline,tempo}.csv"
echo "==========================================="

rm -rf "${LOCAL_NVME}" 2>/dev/null || true
