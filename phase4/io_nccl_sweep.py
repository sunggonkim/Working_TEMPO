#!/usr/bin/env python3
"""
phase4/io_nccl_sweep.py — Lustre I/O Intensity vs. NCCL Bandwidth Sweep
========================================================================
OSDI Figure 10: Quantitative proof that Slingshot-11 cross-traffic degrades
NCCL AllReduce bandwidth as background Lustre I/O intensity increases.

Methodology:
  1. Rank 0 performs background I/O writes to Lustre at a TARGET rate (GB/s)
     controlled by a token-bucket throttle.  Rates are swept: 0, 1, 2, 4,
     8, 16, 32 GB/s — from "quiet" to "near-saturating" the Dragonfly link.
  2. All ranks continuously run NCCL AllReduce over a 256 MB tensor.
  3. Per-AllReduce latency is collected for 60 s at each I/O rate point.
  4. Compute: NCCL effective BW = 2*(N-1)/N * tensor_bytes / latency_s
     (ring-AllReduce formula) for each measurement.

Expected result (Perlmutter Slingshot-11, 200 Gbps / node):
  At 0 GB/s background I/O → NCCL BW ≈ 18–22 GB/s
  At 16+ GB/s background I/O → NCCL BW drops to < 10 GB/s  (-50%)
  This proves the shared-fabric interference motivating TEMPO.

Launch (2+ nodes required for cross-node AllReduce):
  sbatch phase4/run_io_nccl_sweep.slurm

Output:
  results/phase4/io_nccl_sweep/rank0_sweep.csv
  Columns: io_rate_gbs, trial, allreduce_ms, nccl_bw_gbs, rank
"""

import os
import sys
import csv
import time
import logging
import argparse
import threading
import statistics
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s r%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
# I/O rate sweep points in GB/s (0 = no background I/O)
IO_RATES_GBS    = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
MEASURE_SECS    = 30       # seconds to measure at each I/O rate point
WARMUP_SECS     = 5        # seconds to discard at the start of each point
AR_TENSOR_MB    = 256      # AllReduce tensor size per rank (MB)
IO_CHUNK_MB     = 128      # I/O write chunk size (MB)
FLOOD_RANK      = 0        # rank that performs background I/O
FLOAT_DTYPE     = torch.bfloat16
BYTES_PER_ELEM  = 2


# ============================================================================
# Token-Bucket Rate Limiter
# ============================================================================

class TokenBucket:
    """
    Enforces a target bytes/second write rate for the background I/O thread.
    Uses the classical token bucket algorithm:
      - Tokens refill at `rate_bps` bytes/second.
      - A write of `n` bytes consumes `n` tokens; if insufficient, the caller
        sleeps until tokens are available.
    """

    def __init__(self, rate_bps: float):
        self._rate   = rate_bps       # bytes/s
        self._tokens = rate_bps       # start full
        self._last_t = time.monotonic()
        self._lock   = threading.Lock()

    def consume(self, n_bytes: int):
        """Block until `n_bytes` tokens are available, then consume them."""
        if self._rate <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                dt  = now - self._last_t
                self._last_t = now
                self._tokens = min(self._rate, self._tokens + dt * self._rate)
                if self._tokens >= n_bytes:
                    self._tokens -= n_bytes
                    return
            time.sleep(0.001)  # 1 ms sleep — check again

    def set_rate(self, new_rate_bps: float):
        with self._lock:
            self._rate   = new_rate_bps
            self._tokens = new_rate_bps


# ============================================================================
# Background I/O Thread (runs on FLOOD_RANK only)
# ============================================================================

class BackgroundIOThread(threading.Thread):
    """
    Writes large tensors to Lustre at a controlled rate to simulate
    checkpoint flush traffic on the Slingshot-11 interconnect.

    The Lustre client routes file I/O through the same high-speed network
    as NCCL AllReduce — this creates the cross-traffic interference TEMPO
    is designed to handle.
    """

    def __init__(self, lustre_dir: Path, chunk_mb: int, bucket: TokenBucket):
        super().__init__(daemon=True, name="io-flood")
        self._lustre_dir  = lustre_dir
        self._chunk_bytes = chunk_mb * 1024 * 1024
        self._bucket      = bucket
        self._stop_event  = threading.Event()
        self._bytes_written = 0
        self._file_idx      = 0

    def run(self):
        buf = bytearray(self._chunk_bytes)  # pre-allocated write buffer
        while not self._stop_event.is_set():
            # Consume tokens (rate-limit)
            self._bucket.consume(self._chunk_bytes)
            # Write to Lustre
            fpath = self._lustre_dir / f"flood_{self._file_idx % 8}.bin"
            try:
                with open(fpath, "wb") as f:
                    f.write(buf)
                    f.flush()
                    os.fsync(f.fileno())
                self._bytes_written += self._chunk_bytes
                self._file_idx += 1
            except OSError as e:
                log.warning("[io-flood] write error: %s", e)

    def stop(self):
        self._stop_event.set()

    @property
    def bytes_written(self) -> int:
        return self._bytes_written


# ============================================================================
# NCCL AllReduce Probe
# ============================================================================

def measure_allreduce(
    tensor: torch.Tensor,
    device: torch.device,
    duration_s: float,
    warmup_s: float,
) -> List[float]:
    """
    Repeatedly runs AllReduce for `duration_s` seconds (after `warmup_s`
    warm-up) and returns a list of per-AllReduce latency values in ms.
    """
    latencies_ms: List[float] = []
    deadline_warmup = time.monotonic() + warmup_s
    deadline_measure = time.monotonic() + warmup_s + duration_s

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    while time.monotonic() < deadline_measure:
        torch.cuda.synchronize(device)
        start_evt.record()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
        end_evt.record()
        torch.cuda.synchronize(device)

        if time.monotonic() < deadline_warmup:
            continue  # discard warmup

        latencies_ms.append(start_evt.elapsed_time(end_evt))

    return latencies_ms


# ============================================================================
# AllReduce BW Formula
# ============================================================================

def allreduce_bw_gbs(tensor_bytes: int, world_size: int,
                     latency_ms: float) -> float:
    """
    Ring-AllReduce effective bandwidth:
        BW = 2 * (N-1)/N * tensor_bytes / latency_s
    """
    if latency_ms <= 0:
        return 0.0
    return 2.0 * (world_size - 1) / world_size * tensor_bytes / (latency_ms / 1000.0) / 1e9


# ============================================================================
# Main Sweep Loop
# ============================================================================

def run_sweep(args, rank: int, world_size: int, device: torch.device):
    ar_elems   = (AR_TENSOR_MB * 1024 * 1024) // BYTES_PER_ELEM
    ar_tensor  = torch.empty(ar_elems, dtype=FLOAT_DTYPE, device=device)
    ar_bytes   = ar_elems * BYTES_PER_ELEM

    # Only FLOOD_RANK runs background I/O
    io_thread: BackgroundIOThread = None
    bucket: TokenBucket = None
    lustre_dir = Path(args.lustre_dir)

    if rank == FLOOD_RANK:
        lustre_dir.mkdir(parents=True, exist_ok=True)
        bucket    = TokenBucket(rate_bps=0.0)  # start paused
        io_thread = BackgroundIOThread(lustre_dir, IO_CHUNK_MB, bucket)
        io_thread.start()
        log.info("[r%d] I/O flood thread started  lustre_dir=%s", rank, lustre_dir)

    sweep_records: List[Dict] = []

    for io_rate in IO_RATES_GBS:
        # ── Set I/O rate ────────────────────────────────────────────────
        dist.barrier()
        if rank == FLOOD_RANK:
            rate_bps = io_rate * 1e9
            bucket.set_rate(rate_bps)
            log.info("[r%d] I/O rate → %.1f GB/s", rank, io_rate)

        log.info("[r%d] Measuring NCCL BW at I/O=%.1f GB/s ...", rank, io_rate)

        latencies = measure_allreduce(
            ar_tensor, device,
            duration_s=MEASURE_SECS,
            warmup_s=WARMUP_SECS,
        )

        # ── Compute statistics ─────────────────────────────────────────
        if latencies:
            p50_ms  = statistics.median(latencies)
            p99_ms  = sorted(latencies)[int(len(latencies) * 0.99)]
            p999_ms = sorted(latencies)[int(len(latencies) * 0.999)]
            mean_ms = statistics.mean(latencies)
        else:
            p50_ms = p99_ms = p999_ms = mean_ms = float("nan")

        nccl_bw_p50  = allreduce_bw_gbs(ar_bytes, world_size, p50_ms)
        nccl_bw_mean = allreduce_bw_gbs(ar_bytes, world_size, mean_ms)

        log.info(
            "[r%d] io=%.1f GB/s  NCCL BW: mean=%.2f p50=%.2f p99=%.2f p999=%.2f GB/s  "
            "lat: p50=%.2f p99=%.2f ms  n=%d",
            rank, io_rate,
            nccl_bw_mean, nccl_bw_p50,
            allreduce_bw_gbs(ar_bytes, world_size, p99_ms),
            allreduce_bw_gbs(ar_bytes, world_size, p999_ms),
            p50_ms, p99_ms,
            len(latencies),
        )

        # Emit per-sample records (for plotting CDF / scatter at this rate)
        for i, lat in enumerate(latencies):
            sweep_records.append({
                "io_rate_gbs":  io_rate,
                "trial":        i,
                "rank":         rank,
                "allreduce_ms": round(lat, 4),
                "nccl_bw_gbs":  round(allreduce_bw_gbs(ar_bytes, world_size, lat), 4),
            })

    # ── Teardown I/O thread ────────────────────────────────────────────────
    if io_thread is not None:
        bucket.set_rate(0.0)
        io_thread.stop()
        io_thread.join(timeout=5)
        log.info("[r%d] Total bytes written to Lustre: %.2f GB",
                 rank, io_thread.bytes_written / 1e9)

    # ── Gather all ranks to rank 0 and save ───────────────────────────────
    dist.barrier()
    all_records = [None] * world_size
    dist.all_gather_object(all_records, sweep_records)

    if rank == 0:
        flat = []
        for recs in all_records:
            flat.extend(recs)
        flat.sort(key=lambda x: (x["io_rate_gbs"], x["rank"], x["trial"]))

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "io_nccl_sweep.csv"

        fieldnames = ["io_rate_gbs", "trial", "rank",
                      "allreduce_ms", "nccl_bw_gbs"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat)

        log.info("[r0] Sweep complete → %s  (%d rows)", out_path, len(flat))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Lustre I/O vs NCCL BW sweep — OSDI Figure 10"
    )
    parser.add_argument("--io-rates", type=float, nargs="+",
                        default=IO_RATES_GBS,
                        help="I/O rates to sweep (GB/s)")
    parser.add_argument("--measure-secs", type=float, default=MEASURE_SECS)
    parser.add_argument("--warmup-secs",  type=float, default=WARMUP_SECS)
    parser.add_argument("--ar-mb",        type=int,   default=AR_TENSOR_MB)
    parser.add_argument("--io-chunk-mb",  type=int,   default=IO_CHUNK_MB)
    parser.add_argument("--output-dir",   type=str,
                        default="results/phase4/io_nccl_sweep")
    parser.add_argument("--lustre-dir",   type=str,
                        default=os.environ.get("PSCRATCH", "/tmp") + "/tempo_sweep")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK",  0))
    world_size = int(os.environ.get("WORLD_SIZE",  1))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        device_id=device,
    )
    rank = dist.get_rank()

    log.info("[r%d/%d] Starting I/O-NCCL sweep  io_rates=%s",
             rank, world_size, args.io_rates)

    try:
        run_sweep(args, rank, world_size, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
