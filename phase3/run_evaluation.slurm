#!/bin/bash
# =============================================================================
# TEMPO Phase 3 — Side-by-Side Evaluation: Baseline vs. TEMPO on Perlmutter
# =============================================================================
#SBATCH --job-name=tempo_eval
#SBATCH -A m5320
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -t 00:29:00
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-/pscratch/sd/s/sgkim/Skim-Tempo}"
[[ "$(basename "$REPO_ROOT")" == "phase3" ]] && REPO_ROOT="$(dirname "$REPO_ROOT")"
cd "$REPO_ROOT"

module load pytorch/2.8.0
export FI_CXI_DISABLE_HMEM_MODES=1
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=hsn

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29501
export WORLD_SIZE=$(( SLURM_NNODES * 4 ))

# NCCL fix: use 2.29.2 plugin compatible with NCCL 2.28.9 (same as Phase 1)
export LD_LIBRARY_PATH=/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib:/global/common/software/nersc9/nccl/2.29.2-cu13/lib:${LD_LIBRARY_PATH}
export FI_CXI_ATS=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=WARN

export LOCAL_NVME=/tmp/tempo_eval_${SLURM_JOB_ID}
export CKPT_LUSTRE=${PSCRATCH}/tempo_eval_${SLURM_JOB_ID}

mkdir -p logs results/baseline results/tempo results/figures \
         "${CKPT_LUSTRE}/baseline" "${CKPT_LUSTRE}/tempo"

# No --gpus-per-task: expose all 4 GPUs per task for NCCL P2P intra-node
export CUDA_VISIBLE_DEVICES=0,1,2,3
SRUN_FLAGS=(
    --ntasks-per-node=4
    --cpu-bind=map_cpu:0,8,16,24
    --export=ALL
)

TRAIN_FLAGS=(
    --model-size   1b
    --num-steps    60
    --warmup-steps 10
    --ckpt-every   20
    --batch-size   1
    --seq-len      512
    --output-dir   results
    --local-nvme   "${LOCAL_NVME}"
    --flush-chunk-mb 32
    --adaptive-chunk
)

echo "==========================================="
echo "  TEMPO Evaluation — Job ${SLURM_JOB_ID}"
echo "  Nodes      : ${SLURM_JOB_NODELIST}"
echo "  World size : ${WORLD_SIZE}"
echo "  Local NVMe : ${LOCAL_NVME}"
echo "  Lustre     : ${CKPT_LUSTRE}"
echo "==========================================="

# --------------------------------------------------------------------------- #
#  Run A: BASELINE — greedy flush (reproduces PCIe contention)
# --------------------------------------------------------------------------- #
echo ""
echo "====== Run A: BASELINE (Greedy Flush) ======"
export MASTER_PORT=29501

srun "${SRUN_FLAGS[@]}" \
    python phase3/train_with_tempo.py \
        "${TRAIN_FLAGS[@]}" \
        --mode       baseline \
        --lustre-dir "${CKPT_LUSTRE}/baseline"

echo "Baseline done → results/baseline/nccl_bw_rank0.csv"

sleep 10

# --------------------------------------------------------------------------- #
#  Run B: TEMPO — paced flush (paper contribution)
# --------------------------------------------------------------------------- #
echo ""
echo "====== Run B: TEMPO (Paced Flush) ======"
export MASTER_PORT=29502

srun "${SRUN_FLAGS[@]}" \
    python phase3/train_with_tempo.py \
        "${TRAIN_FLAGS[@]}" \
        --mode       tempo \
        --lustre-dir "${CKPT_LUSTRE}/tempo"

echo "TEMPO done → results/tempo/nccl_bw_rank0.csv"

# --------------------------------------------------------------------------- #
#  Plot
# --------------------------------------------------------------------------- #
echo ""
echo "====== Plotting Killer Graph ======"

python phase3/plot_killer_graph.py \
    --baseline-csv   results/baseline/nccl_bw_rank0.csv \
    --contention-csv results/baseline/nccl_bw_rank0.csv \
    --tempo-csv      results/tempo/nccl_bw_rank0.csv \
    --output-dir     results/figures \
    --title          "TEMPO: Pacing Scheduler for LLM Checkpoints (Perlmutter, 2 nodes)" \
    --ckpt-every     20 \
    --bw-metric      algbw_GBs

echo ""
echo "==========================================="
echo "  Evaluation DONE."
echo "  Figures → results/figures/"
echo "  Baseline CSV → results/baseline/"
echo "  TEMPO CSV    → results/tempo/"
echo "==========================================="

rm -rf "${LOCAL_NVME}" 2>/dev/null || true
