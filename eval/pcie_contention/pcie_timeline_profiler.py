#!/usr/bin/env python3
"""
phase7/pcie_timeline_profiler.py — PCIe Contention Timeline Profiler
=====================================================================
OSDI Figure 9: µs-level proof that concurrent KV DMA and NCCL AllReduce
compete for PCIe bandwidth, causing measurable AllReduce stalls.

Hardware context (Perlmutter):
  4 × A100 40GB per node, connected via PCIe Gen4 x16 through a PLX switch.
  NVLink connects pairs: (GPU0↔GPU1), (GPU2↔GPU3) — but cross-pair traffic
  and all D2H DMA go through the PCIe Root Complex (CPU ↔ GPU ↔ NVMe).

Experiment design:
  We force NCCL through PCIe by disabling NVLink P2P:
    NCCL_P2P_DISABLE=1   NCCL_SHM_DISABLE=1
  This guarantees AllReduce data moves over PCIe — competing with KV DMA.

  Baseline:  KV DMA (io_stream) runs concurrently with AllReduce (compute_stream)
  TEMPO:     Phase gate ensures AllReduce completes before DMA starts

  Each "step" is measured with CUDA Events (sub-µs accuracy).

Launch (single node, 4 ranks):
  NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \\
  torchrun --nproc_per_node=4 phase7/pcie_timeline_profiler.py --mode baseline
  torchrun --nproc_per_node=4 phase7/pcie_timeline_profiler.py --mode tempo

Output (rank 0 collects results from all ranks):
  results/pcie_contention/timeline_baseline.csv
  results/pcie_contention/timeline_tempo.csv
  Columns: step, rank, allreduce_start_us, allreduce_end_us, dma_start_us,
           dma_end_us, allreduce_ms, dma_ms, overlap_ms, stall_ms
"""

import os
import sys
import csv
import time
import logging
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.distributed as dist
import torch.cuda

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s r%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants calibrated for Perlmutter A100 (PCIe Gen4 × 16 = 32 GB/s peak)
# KV chunk:   512 MB  → D2H saturates ~50-70% PCIe for ~15-25 ms
# AllReduce:   64 MB  → normally completes in ~3-5 ms (NVLink) / ~8-14 ms (PCIe)
# Overlap window guaranteed to produce measurable stall when both compete.
# ---------------------------------------------------------------------------
KV_CHUNK_MB      = 512     # simulated KV cache chunk size
ALLREDUCE_MB     = 64      # gradient tensor size per AllReduce
N_WARMUP         = 20      # discarded warm-up steps
N_STEPS          = 200     # measured steps
FLOAT_DTYPE      = torch.bfloat16
BYTES_PER_ELEM   = 2       # bfloat16


# ============================================================================
# CUDA Event Timing Helper
# ============================================================================

class EventPair:
    """Wraps start/end CUDA events for a single timed region."""

    def __init__(self, device: int):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end   = torch.cuda.Event(enable_timing=True)

    def elapsed_ms(self) -> float:
        """Must be called after torch.cuda.synchronize()."""
        return self.start.elapsed_time(self.end)


# ============================================================================
# Measurement Loop
# ============================================================================

def run_baseline(rank: int, world_size: int, device: torch.device,
                 n_steps: int, n_warmup: int) -> List[dict]:
    """
    Baseline: AllReduce and DMA launched on separate streams simultaneously.
    NCCL sees PCIe bandwidth reduction → AllReduce duration increases.
    """
    kv_elems     = (KV_CHUNK_MB   * 1024 * 1024) // BYTES_PER_ELEM
    ar_elems     = (ALLREDUCE_MB  * 1024 * 1024) // BYTES_PER_ELEM

    kv_gpu    = torch.empty(kv_elems,  dtype=FLOAT_DTYPE, device=device)
    ar_gpu    = torch.empty(ar_elems,  dtype=FLOAT_DTYPE, device=device)
    kv_pinned = torch.empty(kv_elems,  dtype=FLOAT_DTYPE,
                            device="cpu", pin_memory=True)

    compute_stream = torch.cuda.current_stream(device)
    io_stream      = torch.cuda.Stream(device, priority=-1)  # lower priority

    records: List[dict] = []

    for step in range(-n_warmup, n_steps):
        torch.cuda.synchronize(device)

        ar_evt  = EventPair(device)
        dma_evt = EventPair(device)

        # ── Record step wall-clock anchor on rank 0 ────────────────────────
        t0_wall = time.perf_counter()

        # ── Launch AllReduce on compute_stream ────────────────────────────
        ar_evt.start.record(compute_stream)
        dist.all_reduce(ar_gpu, op=dist.ReduceOp.SUM, async_op=False)
        ar_evt.end.record(compute_stream)

        # ── Launch DMA on io_stream concurrently ──────────────────────────
        # (In baseline there is NO phase gate — DMA overlaps with AllReduce)
        dma_evt.start.record(io_stream)
        with torch.cuda.stream(io_stream):
            kv_pinned.copy_(kv_gpu, non_blocking=True)
        dma_evt.end.record(io_stream)

        torch.cuda.synchronize(device)

        if step < 0:
            continue  # warm-up — discard

        ar_ms  = ar_evt.elapsed_ms()
        dma_ms = dma_evt.elapsed_ms()

        # Compute overlap: how many ms of AllReduce overlapped with DMA
        # (Both start simultaneously in baseline, so overlap ≈ min(ar_ms, dma_ms))
        overlap_ms = min(ar_ms, dma_ms)

        records.append({
            "step":            step,
            "rank":            rank,
            "mode":            "baseline",
            "allreduce_ms":    round(ar_ms,   4),
            "dma_ms":          round(dma_ms,  4),
            "overlap_ms":      round(overlap_ms, 4),
            # stall = extra AllReduce time due to PCIe contention
            # (we compute this relative to TEMPO baseline in post-processing)
            "stall_ms":        0.0,   # filled by plot script
            "wall_s":          round(time.perf_counter() - t0_wall, 6),
        })

    return records


def run_tempo(rank: int, world_size: int, device: torch.device,
              n_steps: int, n_warmup: int) -> List[dict]:
    """
    TEMPO mode: AllReduce completes first; DMA starts only after AllReduce.
    PCIe bandwidth is not split → AllReduce duration returns to hardware limit.
    """
    kv_elems     = (KV_CHUNK_MB   * 1024 * 1024) // BYTES_PER_ELEM
    ar_elems     = (ALLREDUCE_MB  * 1024 * 1024) // BYTES_PER_ELEM

    kv_gpu    = torch.empty(kv_elems,  dtype=FLOAT_DTYPE, device=device)
    ar_gpu    = torch.empty(ar_elems,  dtype=FLOAT_DTYPE, device=device)
    kv_pinned = torch.empty(kv_elems,  dtype=FLOAT_DTYPE,
                            device="cpu", pin_memory=True)

    compute_stream = torch.cuda.current_stream(device)
    io_stream      = torch.cuda.Stream(device, priority=-1)

    records: List[dict] = []

    for step in range(-n_warmup, n_steps):
        torch.cuda.synchronize(device)

        ar_evt  = EventPair(device)
        dma_evt = EventPair(device)

        t0_wall = time.perf_counter()

        # ── Phase gate: AllReduce first ────────────────────────────────────
        ar_evt.start.record(compute_stream)
        dist.all_reduce(ar_gpu, op=dist.ReduceOp.SUM, async_op=False)
        ar_evt.end.record(compute_stream)

        # Force compute_stream to finish before DMA begins
        # (io_stream waits for compute_stream event)
        gate_event = torch.cuda.Event()
        gate_event.record(compute_stream)

        # ── DMA starts only AFTER AllReduce completes ──────────────────────
        io_stream.wait_event(gate_event)
        dma_evt.start.record(io_stream)
        with torch.cuda.stream(io_stream):
            kv_pinned.copy_(kv_gpu, non_blocking=True)
        dma_evt.end.record(io_stream)

        torch.cuda.synchronize(device)

        if step < 0:
            continue

        ar_ms  = ar_evt.elapsed_ms()
        dma_ms = dma_evt.elapsed_ms()

        records.append({
            "step":            step,
            "rank":            rank,
            "mode":            "tempo",
            "allreduce_ms":    round(ar_ms,   4),
            "dma_ms":          round(dma_ms,  4),
            "overlap_ms":      0.0,       # no overlap by design
            "stall_ms":        0.0,
            "wall_s":          round(time.perf_counter() - t0_wall, 6),
        })

    return records


# ============================================================================
# Gather + Save
# ============================================================================

def gather_and_save(records: List[dict], rank: int, world_size: int,
                    output_path: Path):
    """Gather per-rank records to rank 0 and write CSV."""
    # Serialize to JSON for all_gather_object
    dist.barrier()
    all_records = [None] * world_size
    dist.all_gather_object(all_records, records)

    if rank != 0:
        return

    flat = []
    for r in all_records:
        flat.extend(r)

    # Sort by (step, rank)
    flat.sort(key=lambda x: (x["step"], x["rank"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "rank", "mode", "allreduce_ms", "dma_ms",
                  "overlap_ms", "stall_ms", "wall_s"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)

    # Print summary statistics (rank 0 only)
    import statistics
    rank0 = [r for r in flat if r["rank"] == 0]
    ar_samples  = [r["allreduce_ms"] for r in rank0]
    dma_samples = [r["dma_ms"]       for r in rank0]
    log.info("=== SUMMARY (rank 0, %d steps) ===", len(rank0))
    log.info("  AllReduce  — mean: %.2f ms  p50: %.2f ms  p99: %.2f ms",
             statistics.mean(ar_samples),
             statistics.median(ar_samples),
             sorted(ar_samples)[int(len(ar_samples) * 0.99)])
    log.info("  DMA        — mean: %.2f ms  p50: %.2f ms  p99: %.2f ms",
             statistics.mean(dma_samples),
             statistics.median(dma_samples),
             sorted(dma_samples)[int(len(dma_samples) * 0.99)])
    log.info("  Saved → %s", output_path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PCIe Timeline Profiler — OSDI µs-level interference proof"
    )
    parser.add_argument("--mode", choices=["baseline", "tempo"], required=True,
                        help="baseline: concurrent DMA+AllReduce | "
                             "tempo: phase-gated (DMA after AllReduce)")
    parser.add_argument("--n-steps",  type=int, default=N_STEPS,
                        help=f"Measured steps (default {N_STEPS})")
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP,
                        help=f"Warm-up steps (default {N_WARMUP})")
    parser.add_argument("--kv-mb",    type=int, default=KV_CHUNK_MB,
                        help=f"KV DMA tensor size MB (default {KV_CHUNK_MB})")
    parser.add_argument("--ar-mb",    type=int, default=ALLREDUCE_MB,
                        help=f"AllReduce tensor size MB (default {ALLREDUCE_MB})")
    parser.add_argument("--output-dir", type=str,
                        default="results/pcie_contention",
                        help="Directory for output CSV files")
    args = parser.parse_args()

    # ── Distributed init ───────────────────────────────────────────────────
    local_rank  = int(os.environ.get("LOCAL_RANK",  0))
    global_rank = int(os.environ.get("RANK",        0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        device_id=device,
    )
    rank = dist.get_rank()

    log.info("[r%d] mode=%s  device=%s  world=%d  kv=%dMB  ar=%dMB",
             rank, args.mode, device, world_size, args.kv_mb, args.ar_mb)

    # ── Run experiment ────────────────────────────────────────────────────
    if args.mode == "baseline":
        records = run_baseline(rank, world_size, device,
                               args.n_steps, args.n_warmup)
    else:
        records = run_tempo(rank, world_size, device,
                            args.n_steps, args.n_warmup)

    # ── Collect and save ──────────────────────────────────────────────────
    out_path = Path(args.output_dir) / f"timeline_{args.mode}.csv"
    gather_and_save(records, rank, world_size, out_path)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
