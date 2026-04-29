#!/bin/bash
# =============================================================================
# TEMPO Smoke Test — 1분 이내 완료, 단일 노드 (1×4 A100)
# 목적: 분산 NCCL 초기화 → nccl_bench 20스텝 정상 실행 → CSV 출력 확인
# =============================================================================
#SBATCH --job-name=tempo_smoke
#SBATCH -A m5320
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -t 00:05:00
#SBATCH -o logs/smoke_%j.out
#SBATCH -e logs/smoke_%j.err

set -euo pipefail
# SLURM_SUBMIT_DIR = directory where sbatch was called
REPO_ROOT="${SLURM_SUBMIT_DIR:-/pscratch/sd/s/sgkim/Skim-Tempo}"
# If submitted from inside tests/, step up one level
[[ "$(basename "$REPO_ROOT")" == "tests" ]] && REPO_ROOT="$(dirname "$REPO_ROOT")"
cd "$REPO_ROOT"

module load pytorch/2.8.0

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Slingshot 11 settings
export FI_CXI_ATS=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=WARN

mkdir -p logs results/smoke

echo "========================================"
echo "  TEMPO SMOKE TEST — Job ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo "  pytorch: $(python -c 'import torch; print(torch.__version__)')"
echo "========================================"

# ── Test 1: NCCL benchmark 20 steps ───────────────────────────────────────────
echo ""
echo "--- Test 1: NCCL benchmark (20 steps, 256MB tensor) ---"
srun \
    --ntasks-per-node=4 \
    --cpu-bind=cores \
    python phase1/train_llm_profiling.py \
        --mode       nccl_bench \
        --tensor-gb  0.25 \
        --num-steps  20 \
        --warmup-steps 5 \
        --output-dir results/smoke \
        --run-tag    smoke_nccl

echo "Test 1 complete."
if [[ -f results/smoke/smoke_nccl/nccl_bw_rank0.csv ]]; then
    echo "CSV output:"
    head -5 results/smoke/smoke_nccl/nccl_bw_rank0.csv
else
    echo "ERROR: CSV not found!" && exit 1
fi

# ── Test 2: TEMPO package import sanity ────────────────────────────────────────────
echo ""
echo "--- Test 2: TEMPO package import ---"
srun --ntasks=1 --gpus=1 \
    python -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from tempo import TEMPOScheduler, PhaseMonitor, CheckpointManager
s = TEMPOScheduler(rank=0, world_size=1, mode='tempo',
                   lustre_dir='/tmp/smoke_lustre_test')
s.shutdown(wait=False)
print('TEMPO package: OK')
"
echo "Test 2 complete."

# ── Test 3: Killer Graph demo plot ────────────────────────────────────────────────
echo ""
echo "--- Test 3: plot_killer_graph.py --demo ---"
python phase3/plot_killer_graph.py \
    --demo \
    --output-dir results/smoke/figures
ls -lh results/smoke/figures/killer_graph.{pdf,png} 2>/dev/null && echo "Plot: OK"

echo ""
echo "========================================"
echo "  SMOKE TEST PASSED — all 3 tests OK"
echo "========================================"
