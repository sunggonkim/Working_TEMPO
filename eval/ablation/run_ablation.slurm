#!/bin/bash
# =============================================================================
# TEMPO Ablation Study — Sequential 5-Mode Evaluation
# =============================================================================
# Purpose: Quantify the contribution of each TEMPO component.
#
# Modes (weakest → strongest):
#   1. baseline        — greedy flush, no gating
#   2. core            — phase-gate only (V1)
#   3. core_p1         — core + GPU-Driven NIC Doorbell (P1, dry-run)
#   4. core_p1_p2      — core + P1 + NVLink PCIe Multipath (P2)
#   5. core_p1_p2_p3   — core + P1 + P2 + CXI TC Control (P3, dry-run)
#
# Output CSVs: results/ablation/{mode}/nccl_bw_rank0.csv
# Figure:      results/figures/fig_ablation.{pdf,png}
#
# All 5 modes run within ONE SLURM job for controlled environment.
# =============================================================================
#SBATCH --job-name=tempo_ablation
#SBATCH --account=m5320
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err

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

LOCAL_NVME=/tmp/tempo_ablation_${SLURM_JOB_ID}
CKPT_LUSTRE=${PSCRATCH}/tempo_ablation_${SLURM_JOB_ID}

mkdir -p logs results/ablation results/figures "${LOCAL_NVME}" "${CKPT_LUSTRE}"

SRUN_FLAGS=(
    --ntasks-per-node=4
    --cpu-bind=map_cpu:0,8,16,24
    --export=ALL
)

# P1/P3 use dry-run by default (safe on Perlmutter; set --no-cxi-dry-run for
# real fi_setopt calls when libfabric CXI is confirmed available)
TRAIN_FLAGS=(
    --model-size    1b
    --num-steps     80
    --warmup-steps  10
    --ckpt-every    20
    --batch-size    1
    --seq-len       512
    --local-nvme    "${LOCAL_NVME}"
    --lustre-dir    "${CKPT_LUSTRE}"
    --flush-chunk-mb 32
    --adaptive-chunk
    --output-dir    results/ablation
    --cxi-dry-run
)

# Mapping from mode label → master port (avoids port conflicts between runs)
declare -A PORTS=(
    [baseline]=29700
    [core]=29701
    [core_p1]=29702
    [core_p1_p2]=29703
    [core_p1_p2_p3]=29704
)

echo "==========================================="
echo "  TEMPO Ablation Study"
echo "  Job ID     : ${SLURM_JOB_ID}"
echo "  Nodes      : ${SLURM_JOB_NODELIST}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "==========================================="

for ABLATION_MODE in baseline core core_p1 core_p1_p2 core_p1_p2_p3; do
    PORT=${PORTS[$ABLATION_MODE]}
    export MASTER_PORT="${PORT}"

    echo ""
    echo "====== Ablation: ${ABLATION_MODE}  (port=${PORT}) ======"

    srun "${SRUN_FLAGS[@]}" \
        python eval/ablation/train_ablation.py \
            "${TRAIN_FLAGS[@]}" \
            --ablation-mode "${ABLATION_MODE}"

    echo "  Done: ${ABLATION_MODE}"
    sleep 5
done

# ── Post-processing ──────────────────────────────────────────────────────────
echo ""
echo "====== Plotting ablation results ======"
python eval/ablation/plot_ablation.py \
    --results-dir results/ablation \
    --output-dir  results/figures

echo ""
echo "==========================================="
echo "  Ablation DONE  (job ${SLURM_JOB_ID})"
echo "  CSVs   → results/ablation/{mode}/nccl_bw_rank0.csv"
echo "  Figure → results/figures/fig_ablation.{pdf,png}"
echo "==========================================="

rm -rf "${LOCAL_NVME}" 2>/dev/null || true
