#!/bin/bash
#SBATCH -J tempo_phase4
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 00:29:00
#SBATCH -A m5320
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH --output=logs/phase4_%j.out
#SBATCH --error=logs/phase4_%j.err
#SBATCH --mail-type=FAIL,END

# =============================================================================
# Phase 4: OSDI-level evaluation — BurstGPT + Network Interference
#
# Experiments:
#   1. Network interference probe: baseline vs tempo-v2 (fig7)
#      Proves: rank-0 I/O flood degrades AllReduce BW on ALL ranks
#      TEMPO v2 NetworkMonitor gates the flood → BW recovers
#
#   2. BurstGPT evaluation: 6 modes (fig8)
#      Modes: baseline, tempo-v1, tempo-v2, ablations (no-net/no-gain/no-il)
#      Metrics: NCCL BW, P99 ITL, SLO violation rate, Service Gain
# =============================================================================

set -euo pipefail
cd /pscratch/sd/s/sgkim/Skim-Tempo

# Environment
module load pytorch/2.8.0
export PYTHONPATH="$PWD:$PYTHONPATH"
export LD_LIBRARY_PATH="/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib:${LD_LIBRARY_PATH:-}"

# ── Perlmutter / Cassini NIC memory registration fix ─────────────────────────
# Without this, libfabric's cxi provider attempts to map GPU buffers directly
# into the NIC (HMEM mode), which fails with "cxil_map: write error" /
# "Unable to register memory RC:-14" unless hugepages are pre-allocated.
# FI_CXI_DISABLE_HMEM_MODES=1 forces host-copy path (safe, no perf penalty
# for our checkpoint-I/O workload which already goes CPU→NIC).
export FI_CXI_DISABLE_HMEM_MODES=1
# Also suppress the device_id guessing warning in PyTorch 2.x NCCL
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_TIMEOUT=60
export NCCL_SOCKET_IFNAME=hsn

# Perlmutter-specific: local NVMe on each node
export LOCAL_NVME="/tmp/tempo_phase4_${SLURM_JOB_ID}"
mkdir -p "${LOCAL_NVME}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR

echo "=========================================="
echo " TEMPO Phase 4 — Job ${SLURM_JOB_ID}"
echo " Nodes: ${SLURM_JOB_NODELIST}"
echo " Master: ${MASTER_ADDR}"
echo " Date:   $(date)"
echo "=========================================="

# ---------------------------------------------------------------------------
# Experiment 1: Network Interference Probe
# ---------------------------------------------------------------------------

echo ""
echo "=== Exp 1a: Interference Probe — BASELINE ==="
srun --ntasks-per-node=4 --gpus-per-node=4 \
     -l --label \
     python phase4/network_interference_probe.py \
       --mode baseline \
       --num-allreduce 500 \
       --tensor-gb 0.5 \
       --flood-start 100 \
       --flood-duration 200 \
       --flood-gb 4.0 \
       --output-dir results/phase4/network_interference/baseline \
       --lustre-dir "${PSCRATCH}/tempo_flood_baseline_${SLURM_JOB_ID}" \
  || echo "[WARN] Exp 1a returned non-zero — continuing"

echo ""
echo "=== Exp 1b: Interference Probe — TEMPO v2 ==="
srun --ntasks-per-node=4 --gpus-per-node=4 \
     -l --label \
     python phase4/network_interference_probe.py \
       --mode tempo-v2 \
       --num-allreduce 500 \
       --tensor-gb 0.5 \
       --flood-start 100 \
       --flood-duration 200 \
       --flood-gb 4.0 \
       --output-dir results/phase4/network_interference/tempo-v2 \
       --lustre-dir "${PSCRATCH}/tempo_flood_v2_${SLURM_JOB_ID}" \
  || echo "[WARN] Exp 1b returned non-zero — continuing"

# ---------------------------------------------------------------------------
# Experiment 2: BurstGPT Evaluation (all 6 modes, model=1b for time budget)
# ---------------------------------------------------------------------------

MODES="baseline tempo-v1 tempo-v2 tempo-v2-no-net tempo-v2-no-gain tempo-v2-no-il"

for mode in $MODES; do
    echo ""
    echo "=== Exp 2: BurstGPT — mode=${mode} ==="
    srun --ntasks-per-node=4 --gpus-per-node=4 \
         -l --label \
         python phase4/burst_traffic_workload.py \
           --mode "${mode}" \
           --model-size 1b \
           --num-steps 200 \
           --warmup-steps 20 \
           --ckpt-every 30 \
           --burst-rate 3.0 \
           --slo-itl-ms 200 \
           --output-dir "results/phase4/burst_${mode}" \
           --local-nvme "${LOCAL_NVME}" \
           --lustre-dir "${PSCRATCH}/tempo_burst_${mode}_${SLURM_JOB_ID}" \
           --milestone-interval 100 \
      || echo "[WARN] mode=${mode} returned non-zero — continuing"
done

# ---------------------------------------------------------------------------
# Figure Generation
# ---------------------------------------------------------------------------

echo ""
echo "=== Generating Phase 4 figures ==="
python scripts/make_figures.py --phase4 || echo "[WARN] figure generation failed"

echo ""
echo "=== Phase 4 complete: $(date) ==="
echo "Results:"
find results/phase4/ -name "*.csv" | head -20
