#!/usr/bin/env python3
"""
Phase 0: Workload Injector — ITL Spike Profiler
================================================
MISSION: Prove that KV cache eviction causes ITL spikes.

Strategy:
  - Set gpu_memory_utilization=0.6 to force aggressive KV swapping
  - Flood with long-context requests (high concurrency) to saturate HBM
  - Record EXACT per-token timestamps at microsecond resolution
  - Output: itl_profile.csv  [timestamp_ns, request_id, token_idx, itl_ms]

On Perlmutter (4x A100 40GB):
  - Normal utilization=0.9 → ~144GB usable KV pool across 4 GPUs
  - Forced utilization=0.6  →  ~96GB usable → triggers eviction ~33% sooner
  - Combined with seq_len=4096 requests at concurrency=64 → guaranteed eviction
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, List, Optional

# ---------------------------------------------------------------------------
# Dependency check — fail early with actionable message
# ---------------------------------------------------------------------------
try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    from vllm.outputs import RequestOutput
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL       = "meta-llama/Llama-2-7b-hf"   # change to 70B for full pressure
DEFAULT_GPU_UTIL    = 0.6    # INTENTIONALLY LOW — forces KV eviction
DEFAULT_CONCURRENCY = 64     # simultaneous decode streams
DEFAULT_NUM_REQUESTS= 200    # total requests in the experiment
DEFAULT_MAX_TOKENS  = 256    # tokens to generate per request
DEFAULT_PROMPT_LEN  = 512    # input prompt tokens (approx)
SHAREGPT_PATH       = os.getenv("SHAREGPT_PATH", "ShareGPT_V3_unfiltered_cleaned_split.json")
OUTPUT_DIR          = Path(os.getenv("PHASE0_OUTPUT", "results/phase0"))

SYNC_MARKER_FILE    = OUTPUT_DIR / "experiment_start.marker"  # wall-clock sync with monitor.sh

# ---------------------------------------------------------------------------
# Synthetic prompt generator (fallback when ShareGPT is not available)
# ---------------------------------------------------------------------------
LOREM = (
    "The quick brown fox jumps over the lazy dog. " * 20
    + "In the context of large language models, attention mechanism scales "
    + "quadratically with sequence length, which means that longer contexts "
    + "consume disproportionately more HBM. When the KV cache for active "
    + "requests no longer fits in GPU memory, the serving engine must evict "
    + "cold pages to either host DRAM or local NVMe storage, causing a DMA "
    + "transfer that occupies PCIe bandwidth. This contention hypothesis is "
    + "the core motivation for the TEMPO pacing scheduler. " * 3
)


def load_sharegpt_prompts(path: str, n: int, min_len: int = 200) -> List[str]:
    """Load prompts from ShareGPT dataset, filtering by minimum length."""
    try:
        with open(path) as f:
            data = json.load(f)
        prompts = []
        for item in data:
            for turn in item.get("conversations", []):
                if turn.get("from") == "human" and len(turn["value"]) >= min_len:
                    prompts.append(turn["value"])
        if not prompts:
            raise ValueError("no valid prompts found")
        random.shuffle(prompts)
        # Repeat to fill n requests
        return [prompts[i % len(prompts)] for i in range(n)]
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"[WARN] ShareGPT not available ({e}), using synthetic prompts", file=sys.stderr)
        return [LOREM[: DEFAULT_PROMPT_LEN * 4] for _ in range(n)]


# ---------------------------------------------------------------------------
# Per-token latency record
# ---------------------------------------------------------------------------
@dataclass
class TokenRecord:
    timestamp_ns: int   # absolute wall-clock ns since epoch
    request_id: str
    token_idx: int
    itl_ms: float       # latency since previous token (or TTFT for token_idx==0)


# ---------------------------------------------------------------------------
# Core streaming function — records per-token timestamps
# ---------------------------------------------------------------------------
async def stream_request(
    engine: "AsyncLLMEngine",
    request_id: str,
    prompt: str,
    sampling_params: "SamplingParams",
    records: List[TokenRecord],
) -> None:
    """Submit one request and record ITL for every token generated."""
    t_prev = time.perf_counter_ns()
    token_idx = 0
    num_prev_tokens = 0

    async for output in engine.generate(prompt, sampling_params, request_id):
        output: "RequestOutput"
        t_now = time.perf_counter_ns()

        # Count newly generated tokens since last callback
        for completion in output.outputs:
            n_new = len(completion.token_ids) - num_prev_tokens
            if n_new <= 0:
                continue
            for _ in range(n_new):
                itl_ms = (t_now - t_prev) / 1e6 / max(n_new, 1)
                records.append(TokenRecord(
                    timestamp_ns=t_now,
                    request_id=request_id,
                    token_idx=token_idx,
                    itl_ms=itl_ms,
                ))
                token_idx += 1
            num_prev_tokens += n_new
            t_prev = t_now


# ---------------------------------------------------------------------------
# Experiment orchestrator
# ---------------------------------------------------------------------------
async def run_experiment(args: argparse.Namespace) -> None:
    if not VLLM_AVAILABLE:
        print("[FATAL] vllm not installed. Run: pip install vllm", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[PHASE0] Loading prompts (n={args.num_requests})...", flush=True)
    prompts = load_sharegpt_prompts(args.sharegpt, args.num_requests)

    print(f"[PHASE0] Initialising AsyncLLMEngine", flush=True)
    print(f"         model             = {args.model}", flush=True)
    print(f"         gpu_memory_util   = {args.gpu_util}  ← FORCED LOW to trigger eviction", flush=True)
    print(f"         tensor_parallel   = {args.tp}", flush=True)
    print(f"         concurrency       = {args.concurrency}", flush=True)

    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util,
        swap_space=args.swap_space_gb,      # GiB of CPU swap — enables CPU offload path
        max_num_seqs=args.concurrency,
        max_model_len=args.max_model_len,
        disable_log_requests=True,
        enable_chunked_prefill=True,        # mix prefill+decode → more contention
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=1.0,
        top_p=1.0,
    )

    records: List[TokenRecord] = []
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded_request(req_id: str, prompt: str) -> None:
        async with semaphore:
            await stream_request(engine, req_id, prompt, sampling_params, records)

    # Write sync marker so hardware_monitor.sh knows when inference started
    t_start_wall = time.time()
    SYNC_MARKER_FILE.write_text(str(t_start_wall))
    print(f"\n[PHASE0] *** EXPERIMENT START  wall={t_start_wall:.3f} ***", flush=True)
    print(f"[PHASE0] Sync marker written → {SYNC_MARKER_FILE}", flush=True)

    tasks = [
        asyncio.create_task(bounded_request(f"req_{i:04d}", prompts[i]))
        for i in range(args.num_requests)
    ]

    # Progress reporting
    done = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        done += 1
        if done % 10 == 0:
            print(f"[PHASE0]   {done}/{args.num_requests} requests completed, "
                  f"{len(records)} tokens recorded", flush=True)

    t_end_wall = time.time()
    print(f"\n[PHASE0] *** EXPERIMENT END    wall={t_end_wall:.3f} ***", flush=True)
    print(f"[PHASE0] Total duration: {t_end_wall - t_start_wall:.1f}s", flush=True)
    print(f"[PHASE0] Total tokens recorded: {len(records)}", flush=True)

    # ------------------------------------------------------------------
    # Dump ITL CSV
    # ------------------------------------------------------------------
    out_csv = OUTPUT_DIR / "itl_profile.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "request_id", "token_idx", "itl_ms"])
        for r in records:
            writer.writerow([r.timestamp_ns, r.request_id, r.token_idx, f"{r.itl_ms:.4f}"])

    print(f"[PHASE0] ITL CSV written → {out_csv}  ({len(records)} rows)", flush=True)

    # Quick percentile summary
    if records:
        itls = sorted(r.itl_ms for r in records)
        n = len(itls)
        p50 = itls[int(n * 0.50)]
        p95 = itls[int(n * 0.95)]
        p99 = itls[int(n * 0.99)]
        p999 = itls[int(n * 0.999)]
        print(f"\n[PHASE0] ITL Summary (ms):")
        print(f"         P50  = {p50:.2f} ms")
        print(f"         P95  = {p95:.2f} ms")
        print(f"         P99  = {p99:.2f} ms")
        print(f"         P99.9= {p999:.2f} ms")
        print(f"         Max  = {max(itls):.2f} ms")

        # Heuristic: flag eviction-correlated spikes (ITL > 5× P50)
        spike_threshold = p50 * 5.0
        spikes = [r for r in records if r.itl_ms > spike_threshold]
        print(f"\n[PHASE0] Detected {len(spikes)} spike tokens (ITL > {spike_threshold:.1f}ms)")
        if spikes:
            print(f"[PHASE0] First 5 spikes:")
            for s in spikes[:5]:
                print(f"         req={s.request_id} tok={s.token_idx} "
                      f"itl={s.itl_ms:.1f}ms @ t={s.timestamp_ns/1e9:.3f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0: Force KV eviction and measure ITL spikes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",         default=DEFAULT_MODEL,        help="HuggingFace model ID")
    parser.add_argument("--tp",            type=int, default=1,          help="Tensor parallel degree")
    parser.add_argument("--gpu-util",      type=float, default=DEFAULT_GPU_UTIL,
                        help="gpu_memory_utilization — set LOW to force eviction")
    parser.add_argument("--swap-space-gb", type=int, default=16,         help="CPU swap space in GiB")
    parser.add_argument("--concurrency",   type=int, default=DEFAULT_CONCURRENCY,
                        help="Max simultaneous requests (fills HBM faster)")
    parser.add_argument("--num-requests",  type=int, default=DEFAULT_NUM_REQUESTS)
    parser.add_argument("--max-tokens",    type=int, default=DEFAULT_MAX_TOKENS,
                        help="Tokens to generate per request")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max sequence length (longer = more KV memory pressure)")
    parser.add_argument("--sharegpt",      default=SHAREGPT_PATH,        help="Path to ShareGPT JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_experiment(args))
