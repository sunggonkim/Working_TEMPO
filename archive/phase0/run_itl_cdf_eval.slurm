#!/bin/bash
# =============================================================================
# TEMPO Phase 0 — BurstGPT ITL Tail Latency CDF Profiler (OSDI Fig 11)
# =============================================================================
# Purpose: Prove that I/O bursts cause tail latency spikes (P99/P99.9 ITL)
#          in real LLM serving under BurstGPT workload, and that TEMPO
#          eliminates them via phase-gated KV cache writes.
#
# Design:
#   Single node, 4 × A100 40GB, vLLM with tensor_parallel=4.
#   gpu_memory_utilization=0.65 (below normal 0.9) → forced KV eviction.
#   500 requests per mode, BurstGPT inter-arrival (Pareto alpha=1.2).
#   Modes run sequentially: baseline first, then tempo.
#
# Expected results:
#   P50  ITL: baseline ≈ 25 ms,  TEMPO ≈ 25 ms  (similar — normal compute)
#   P99  ITL: baseline ≈ 200 ms, TEMPO ≈ 15 ms  (14× improvement)
#   P999 ITL: baseline ≈ 650 ms, TEMPO ≈ 18 ms  (36× improvement)
#   CDFs diverge sharply at the 97th percentile — proves I/O spike origin.
#
# BurstGPT traces:
#   Automatic download attempted; if network is down, fallback to
#   BurstGPT-calibrated synthetic inter-arrivals (same statistics, clearly
#   labelled in CSV as mode=baseline/tempo, trace_type=synthetic).
#   For AE review: place real traces at data/burstgpt/GPT3_5.csv
# =============================================================================
#SBATCH --job-name=tempo_itl_cdf
#SBATCH --account=m5320
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/itl_cdf_%j.out
#SBATCH --error=logs/itl_cdf_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/pscratch/sd/s/sgkim/Skim-Tempo}"
[[ "$(basename "$REPO_ROOT")" == "phase0" ]] && REPO_ROOT="$(dirname "$REPO_ROOT")"
cd "$REPO_ROOT"

module load pytorch/2.8.0

export LD_LIBRARY_PATH=/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib:\
/global/common/software/nersc9/nccl/2.29.2-cu13/lib:${LD_LIBRARY_PATH}

export FI_CXI_DISABLE_HMEM_MODES=1
export FI_CXI_ATS=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=WARN
export VLLM_WORKER_MULTIPROC_METHOD=fork

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PHASE0_OUTPUT="${REPO_ROOT}/results/phase0"

# HuggingFace cache on local NVMe (avoids Lustre fcntl lock issue)
export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}" results/phase0 results/figures logs

# ── Optional: BurstGPT trace path ───────────────────────────────────────────
# If you have the BurstGPT CSV, set BURSTGPT_CSV before submitting:
#   BURSTGPT_CSV=/path/to/GPT3_5.csv sbatch phase0/run_itl_cdf_eval.slurm
BURSTGPT_CSV="${BURSTGPT_CSV:-}"
TRACE_ARG=""
if [[ -n "$BURSTGPT_CSV" && -f "$BURSTGPT_CSV" ]]; then
    TRACE_ARG="--trace-path ${BURSTGPT_CSV}"
    echo "[INFO] Using real BurstGPT trace: ${BURSTGPT_CSV}"
else
    echo "[WARN] BURSTGPT_CSV not set or file not found."
    echo "[WARN] Falling back to BurstGPT-calibrated synthetic inter-arrivals."
    echo "[WARN] For real data: set BURSTGPT_CSV=<path> before sbatch."
fi

echo "============================================================"
echo " TEMPO ITL Tail Latency CDF Profiler"
echo " Job ID    : ${SLURM_JOB_ID}"
echo " Node      : $(hostname)"
echo " Start     : $(date -Iseconds)"
echo " GPU util  : 0.65 (forced KV eviction)"
echo " Requests  : 500 per mode"
echo "============================================================"

# ── Check vLLM ───────────────────────────────────────────────────────────────
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "[PHASE0] Installing vllm ..."
    pip3 install vllm --quiet
fi

COMMON_ARGS=(
    --num-requests 500
    --max-tokens   256
    --gpu-util     0.65
    --tp-size      4
    --seed         42
    $TRACE_ARG
)

# ── Mode A: BASELINE (greedy KV flush — no phase gating) ─────────────────────
echo ""
echo "=== Mode A: BASELINE ==="
python3 phase0/burstgpt_itl_cdf.py \
    --mode baseline \
    "${COMMON_ARGS[@]}"

# Allow GPU to cool down between runs
sleep 10

# ── Mode B: TEMPO (phase-gated KV flush) ─────────────────────────────────────
echo ""
echo "=== Mode B: TEMPO ==="
python3 phase0/burstgpt_itl_cdf.py \
    --mode tempo \
    "${COMMON_ARGS[@]}"

# ── Generate Fig 11 ──────────────────────────────────────────────────────────
echo ""
echo "=== Generating Fig 11 ==="
python3 scripts/plot_micro_benchmarks.py --fig 11

echo ""
echo "============================================================"
echo " ITL CDF Profiler COMPLETE"
echo " Baseline : results/phase0/itl_baseline.csv"
echo " TEMPO    : results/phase0/itl_tempo.csv"
echo " Figure   : results/figures/fig11_itl_cdf.{pdf,png}"
echo " End      : $(date -Iseconds)"
echo "============================================================"
