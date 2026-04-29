#!/bin/bash
#SBATCH --job-name=tempo_phase0
#SBATCH --account=m5320
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/phase0_%j.out
#SBATCH --error=logs/phase0_%j.err

# =============================================================================
# TEMPO Phase 0: Verify ITL Spike ↔ KV Eviction Correlation
# =============================================================================
# Purpose: Prove that KV cache eviction I/O causes ITL spikes in vLLM.
#          This is the Motivation experiment — OSDI Figure 1.
#
# Strategy:
#   1. Pin vLLM + I/O monitor to NUMA node 0 (same PCIe root complex → max contention)
#   2. Start hardware_monitor.sh in background
#   3. Run workload_injector.py with gpu_util=0.6 (forced eviction)
#   4. Stop monitor, generate killer graph
#
# NUMA topology on Perlmutter (AMD EPYC 7763, NPS=2):
#   NUMA 0: CPU cores 0-31,  GPU 0 & 1, NVMe 0  ← we use this
#   NUMA 1: CPU cores 32-63, GPU 2 & 3, NVMe 1
#   (PCIe root complex contention is maximized when CPU, GPU, NVMe share same NUMA)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load pytorch/2.8.0

REPO_DIR="${SLURM_SUBMIT_DIR:-/pscratch/sd/s/sgkim/Skim-Tempo}"
[[ "$(basename "$REPO_DIR")" == "phase0" ]] && REPO_DIR="$(dirname "$REPO_DIR")"
cd "$REPO_DIR"

export PHASE0_OUTPUT="${REPO_DIR}/results/phase0"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
mkdir -p "$PHASE0_OUTPUT" results/figures logs

echo "============================================================"
echo " TEMPO Phase 0: KV Eviction Interference Verification"
echo " Job ID    : ${SLURM_JOB_ID}"
echo " Node      : $(hostname)"
echo " Start     : $(date -Iseconds)"
echo " Repo      : ${REPO_DIR}"
echo " Output    : ${PHASE0_OUTPUT}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Verify vLLM is installed (install if not)
# ---------------------------------------------------------------------------
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "[PHASE0] vllm not found — installing..."
    pip3 install vllm --quiet
fi

# ---------------------------------------------------------------------------
# Check for ShareGPT dataset (optional — workload_injector has synthetic fallback)
# ---------------------------------------------------------------------------
SHAREGPT_PATH="${PSCRATCH}/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
if [[ ! -f "$SHAREGPT_PATH" ]]; then
    SHAREGPT_PATH="ShareGPT_V3_unfiltered_cleaned_split.json"
fi
export SHAREGPT_PATH

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
# Use a smaller model for quick verification; swap to 13B/70B for paper-quality results
MODEL="${TEMPO_MODEL:-facebook/opt-6.7b}"
echo "[PHASE0] Model: ${MODEL}"

# Tensor parallel = number of GPUs (all 4 A100s)
TP=4

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
GPU_UTIL=0.60         # INTENTIONALLY LOW — forces KV eviction (paper uses 0.6 vs 0.9 comparison)
CONCURRENCY=64        # simultaneous requests — saturates HBM fast
NUM_REQUESTS=300      # total requests in the run
MAX_TOKENS=256        # decode length per request
MAX_MODEL_LEN=4096    # long context = large KV footprint

# ---------------------------------------------------------------------------
# STEP 1: Start hardware monitor (background, NUMA node 0)
# ---------------------------------------------------------------------------
echo ""
echo "[PHASE0] === STEP 1: Starting hardware_monitor.sh (background) ==="
numactl --cpunodebind=0 --membind=0 \
    bash phase0/hardware_monitor.sh "$PHASE0_OUTPUT" 100 &
MONITOR_PID=$!
echo "[PHASE0] hardware_monitor PID: ${MONITOR_PID}"

# Give monitor time to write its start marker
sleep 3

# ---------------------------------------------------------------------------
# STEP 2: Run workload injector (NUMA node 0 — same PCIe root as NVMe)
# ---------------------------------------------------------------------------
echo ""
echo "[PHASE0] === STEP 2: Running workload_injector.py (foreground) ==="
echo "[PHASE0] GPU util forced to ${GPU_UTIL} to trigger KV eviction"
echo ""

# CUDA_VISIBLE_DEVICES=all so all 4 GPUs are available for tensor parallel
export CUDA_VISIBLE_DEVICES=0,1,2,3

numactl --cpunodebind=0 --membind=0 \
    python3 phase0/workload_injector.py \
        --model "$MODEL" \
        --tp "$TP" \
        --gpu-util "$GPU_UTIL" \
        --swap-space-gb 32 \
        --concurrency "$CONCURRENCY" \
        --num-requests "$NUM_REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --sharegpt "$SHAREGPT_PATH"

INJECTOR_EXIT=$?
echo ""
echo "[PHASE0] workload_injector.py exited with code: ${INJECTOR_EXIT}"

# ---------------------------------------------------------------------------
# STEP 3: Stop hardware monitor
# ---------------------------------------------------------------------------
echo ""
echo "[PHASE0] === STEP 3: Stopping hardware_monitor.sh ==="
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
echo "[PHASE0] Monitor stopped"

# ---------------------------------------------------------------------------
# STEP 4: Generate killer graph
# ---------------------------------------------------------------------------
echo ""
echo "[PHASE0] === STEP 4: Generating killer graph ==="

ITL_CSV="${PHASE0_OUTPUT}/itl_profile.csv"
IO_CSV="${PHASE0_OUTPUT}/io_profile.csv"

if [[ -f "$ITL_CSV" && -f "$IO_CSV" ]]; then
    python3 phase0/plot_inference_interference.py \
        --itl "$ITL_CSV" \
        --io  "$IO_CSV"  \
        --out-pdf "results/figures/fig1_itl_vs_kv_eviction.pdf" \
        --out-png "results/figures/fig1_itl_vs_kv_eviction.png"
    echo "[PHASE0] Killer graph saved to results/figures/"
else
    echo "[PHASE0] WARNING: CSV files missing, generating demo graph instead"
    python3 phase0/plot_inference_interference.py --demo \
        --out-pdf "results/figures/fig1_itl_vs_kv_eviction_DEMO.pdf" \
        --out-png "results/figures/fig1_itl_vs_kv_eviction_DEMO.png"
fi

# ---------------------------------------------------------------------------
# STEP 5: Print summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " TEMPO Phase 0: COMPLETE"
echo " End: $(date -Iseconds)"
echo ""
echo " Output files:"
ls -lh "${PHASE0_OUTPUT}/"*.csv 2>/dev/null || echo "  (no CSVs)"
ls -lh results/figures/fig1_*.{pdf,png} 2>/dev/null || echo "  (no figures)"
echo ""
echo " NEXT STEPS:"
echo "   If the killer graph shows correlated spikes → hypothesis CONFIRMED"
echo "   → TEMPO Pacing Scheduler is justified"
echo "   → This plot is OSDI Figure 1 (Motivation)"
echo ""
echo "   If no correlation:"
echo "   1. Try --gpu-util 0.5 or lower"
echo "   2. Increase --concurrency to 128"
echo "   3. Check NVMe swap is actually being used:"
echo "      grep swap_space vllm logs, or watch nvme iostat during run"
echo "============================================================"
