#!/bin/bash
# =============================================================================
# TEMPO Phase 3 — Chunk Size Sensitivity Sweep
#
# Motivation (from paper analysis):
#   Fixed chunk sizes create a trade-off:
#     - Too large → misses NCCL phase boundaries → contention leaks through
#     - Too small → too many gate checks → overhead on flush throughput
#     - Adaptive → self-tunes to 50% of observed NCCL window duration
#
# This job runs 6 conditions on 2 nodes (2N × 4×A100):
#   1. baseline      (greedy, no gating)
#   2. tempo-16mb    (very fine gating)
#   3. tempo-64mb    (fine gating)
#   4. tempo-128mb   (default)
#   5. tempo-256mb   (coarse gating)
#   6. tempo-adaptive (self-tuning, new contribution)
#
# Results go to results/chunk_sweep/{mode}/nccl_bw_rank0.csv
# =============================================================================
#SBATCH --job-name=tempo_chunk_sweep
#SBATCH -A m5320
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -t 00:29:00
#SBATCH -o logs/chunk_sweep_%j.out
#SBATCH -e logs/chunk_sweep_%j.err

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
export WORLD_SIZE=$(( SLURM_NNODES * 4 ))

# NCCL tuning (same as evaluation job)
export LD_LIBRARY_PATH=/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib:/global/common/software/nersc9/nccl/2.29.2-cu13/lib:${LD_LIBRARY_PATH}
export FI_CXI_ATS=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=WARN

export LOCAL_NVME=/tmp/tempo_sweep_${SLURM_JOB_ID}
export CKPT_LUSTRE=${PSCRATCH}/tempo_sweep_${SLURM_JOB_ID}
export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p logs results/chunk_sweep \
         "${CKPT_LUSTRE}"

SRUN_FLAGS=(
    --ntasks-per-node=4
    --cpu-bind=map_cpu:0,8,16,24
    --export=ALL
)

COMMON=(
    --model-size   1b
    --num-steps    60
    --warmup-steps 10
    --ckpt-every   20
    --batch-size   1
    --seq-len      512
    --local-nvme   "${LOCAL_NVME}"
)

# ─────────────────────────────────────────────────────────────────────────────
run_one() {
    local label="$1"; shift
    local outdir="results/chunk_sweep/${label}"
    mkdir -p "${outdir}" "${CKPT_LUSTRE}/${label}"
    echo ""
    echo "====== ${label} ======"
    export MASTER_PORT=$(( 29600 + RANDOM % 100 ))
    srun "${SRUN_FLAGS[@]}" \
        python phase3/train_with_tempo.py \
            "${COMMON[@]}" \
            --lustre-dir "${CKPT_LUSTRE}/${label}" \
            --output-dir "results/chunk_sweep/${label}" \
            "$@"
    echo "  → ${outdir}/nccl_bw_rank0.csv"
}
# ─────────────────────────────────────────────────────────────────────────────

echo "==========================================="
echo "  TEMPO Chunk Sweep — Job ${SLURM_JOB_ID}"
echo "  Nodes: ${SLURM_JOB_NODELIST}   WS: ${WORLD_SIZE}"
echo "==========================================="

run_one "baseline"      --mode baseline

run_one "tempo-16mb"    --mode tempo  --flush-chunk-mb  16

run_one "tempo-64mb"    --mode tempo  --flush-chunk-mb  64

run_one "tempo-128mb"   --mode tempo  --flush-chunk-mb 128

run_one "tempo-256mb"   --mode tempo  --flush-chunk-mb 256

run_one "tempo-adaptive" --mode tempo  --flush-chunk-mb 128 --adaptive-chunk

echo ""
echo "==========================================="
echo "  All runs complete.  Generating figure..."
python3 scripts/make_figures.py --chunk-sweep
echo "  Figure: results/figures/fig6_chunk_sweep.png"
echo "==========================================="
