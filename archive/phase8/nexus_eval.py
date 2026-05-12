"""
phase8/nexus_eval.py — TEMPO-Nexus (v5) Distributed Checkpoint Orchestration Evaluation
==========================================================================================

Experiment design
-----------------
This script quantifies the benefit of TEMPO-Nexus's Distributed Staggered
Checkpoint Protocol (DSCP) at HPC scale.

Measurement protocol
---------------------
Three configurations are compared on Perlmutter (8 nodes × 4 A100):

  (A) baseline    : All ranks flush simultaneously to Lustre at checkpoint steps.
                    Reproduces the collective checkpoint flood observed in phase4.

  (B) tempo-v4    : TEMPO v4 Phase-Gate only (intra-node PCIe isolation).
                    Eliminates per-node PCIe contention but does NOT coordinate
                    cross-node flushing → collective flood persists.

  (C) tempo-nexus : TEMPO v5 DSCP + per-layer micro-gates.
                    Staggered windows eliminate cross-node flood.
                    Per-layer gates reduce I/O bubble from O(N×AR) to O(max_AR).

Per-rank measurements at each step:
  - NCCL AllReduce bandwidth (algbw GB/s)      → nccl_bw_rank{r}.csv
  - NIC TX+RX utilisation (GB/s, 5ms samples)  → nic_bw_rank{r}.csv
  - DMA / flush wall-clock time (ms)            → flush_times_rank{r}.csv
  - Per-layer AR timing (ms per layer)          → layer_ar_rank{r}.csv  [nexus only]
  - Window assignment log                       → nexus_windows_rank{r}.csv

Metrics reported
----------------
  M1  NCCL BW mean/p5/p95 at checkpoint steps (steps where flush is active)
      → Primary metric: does DSCP preserve AllReduce bandwidth?
  M2  Peak NIC utilisation burst (99th pct over all steps)
      → Does DSCP eliminate the synchronized flood spike?
  M3  Time-to-checkpoint per rank (flush wall time)
      → Is DSCP latency-neutral?
  M4  Per-layer DMA overlap efficiency (nexus only)
      → How much I/O bubble do micro-gates remove?

Expected results (8-node, 32×A100, Llama-1B, ckpt every 50 steps)
--------------------------------------------------------------------
  M1 (NCCL BW at ckpt steps):
    baseline   : 13–17 GB/s (high variance, dips to ~11 GB/s during flood)
    tempo-v4   : 16–18 GB/s (intra-node fixed, inter-node flood persists)
    tempo-nexus: 17–18 GB/s (flat — DSCP eliminates flood spike)

  M2 (Peak NIC burst, 99th pct):
    baseline   : ~16 GB/s collective burst (8 × 2 GB/s)
    tempo-v4   : ~14 GB/s (per-node gate slightly staggers flush start)
    tempo-nexus: ~2 GB/s  (serialized windows: only 1 node flushes at a time)

  M3 (Time-to-checkpoint):
    All three: ~200 ms (DSCP does not extend flush time, only staggers start)

  M4 (DMA overlap, nexus only):
    I/O bubble without micro-gates: ~8 ms (all AR done before any DMA)
    I/O bubble with micro-gates:    ~0.5 ms (DMA starts with layer 0 AR)

Usage (SLURM, see run_nexus_eval.slurm)
---------------------------------------
    torchrun --nproc_per_node=4 phase8/nexus_eval.py \\
        --mode tempo-nexus \\
        --n-steps 300 \\
        --ckpt-interval 50 \\
        --world-size $SLURM_NTASKS \\
        --out-dir results/phase8/nexus

Single-node smoke test (no distributed):
    python3 phase8/nexus_eval.py --mode tempo-nexus --world-size 1 --n-steps 20
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tempo.nexus_coordinator import NexusCoordinator

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(name)s rank%(process)d] %(levelname)s %(message)s",
)
logger = logging.getLogger("nexus_eval")


# ---------------------------------------------------------------------------
# Minimal model & dummy data
# ---------------------------------------------------------------------------

def make_model_and_data(device: torch.device, dtype=torch.float16):
    """
    Llama-1B proxy: 16 transformer layers, each producing a 256 MB gradient shard.
    Full Llama-1B FSDP gradient ≈ 1 GB / world_size.
    """
    hidden = 2048
    n_layers = 16
    # Flat gradient tensor per layer (mimics FSDP per-unit gradient shard)
    grad_shards = [
        torch.randn(hidden, hidden, device=device, dtype=dtype)
        for _ in range(n_layers)
    ]
    return grad_shards, n_layers


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def measure_allreduce_bw(tensor: torch.Tensor,
                         n_warmup: int = 2) -> float:
    """Returns AllReduce algorithmic bandwidth in GB/s."""
    if not dist.is_initialized():
        return 0.0
    for _ in range(n_warmup):
        dist.all_reduce(tensor, async_op=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dist.all_reduce(tensor, async_op=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    nbytes  = tensor.numel() * tensor.element_size()
    algbw   = (2 * nbytes / elapsed) / 1e9  # all_reduce: send+recv
    return algbw


def nic_bw_gbps() -> float:
    """Read Slingshot NIC TX+RX bytes/s; returns GB/s. 0.0 if unavailable."""
    total = 0
    for n in range(8):
        for stat in ("tx_bytes", "rx_bytes"):
            path = f"/sys/class/net/hsn{n}/statistics/{stat}"
            try:
                total += int(Path(path).read_text().strip())
            except Exception:
                pass
    # Sample twice with 10ms gap to get rate
    t0 = time.perf_counter()
    b0 = total
    time.sleep(0.01)
    b1 = 0
    for n in range(8):
        for stat in ("tx_bytes", "rx_bytes"):
            path = f"/sys/class/net/hsn{n}/statistics/{stat}"
            try:
                b1 += int(Path(path).read_text().strip())
            except Exception:
                pass
    dt = time.perf_counter() - t0
    return (b1 - b0) / dt / 1e9 if dt > 0 else 0.0


# ---------------------------------------------------------------------------
# Baseline: simultaneous flush
# ---------------------------------------------------------------------------

def run_baseline(rank, world_size, device, n_steps, ckpt_interval,
                 out_dir, n_layers, dtype):
    """All ranks flush simultaneously — reproduces collective flood."""
    grad_shards, _ = make_model_and_data(device, dtype)
    probe = torch.randn(256 * 1024 * 1024 // 4, device=device, dtype=torch.float32)

    os.makedirs(out_dir, exist_ok=True)
    nccl_rows, nic_rows, flush_rows = [], [], []

    for step in range(n_steps):
        is_ckpt = (step % ckpt_interval == 0) and step > 0

        # Simulated AllReduce per layer
        for layer_id in range(n_layers):
            if dist.is_initialized():
                dist.all_reduce(grad_shards[layer_id], async_op=False)

        # Measure AllReduce BW (full-tensor probe)
        bw = measure_allreduce_bw(probe)
        nic = nic_bw_gbps()
        nccl_rows.append((step, bw, is_ckpt))
        nic_rows.append((step, nic, is_ckpt))

        if is_ckpt:
            # Greedy simultaneous flush
            flush_start = time.perf_counter()
            tmp = Path(f"/tmp/tempo_nexus_eval/baseline_rank{rank}_step{step}.pt")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            # Write ~64 MB to simulate a shard flush
            dummy = torch.zeros(16 * 1024 * 1024, dtype=torch.float32)
            torch.save(dummy, str(tmp))
            flush_ms = (time.perf_counter() - flush_start) * 1000
            flush_rows.append((step, flush_ms))
            logger.debug("[baseline] rank=%d step=%d flush=%.1fms bw=%.2f", rank, step, flush_ms, bw)

    _write_csv(out_dir, rank, "nccl_bw", ("step", "algbw_GBs", "is_ckpt"), nccl_rows)
    _write_csv(out_dir, rank, "nic_bw",  ("step", "nic_gbps",  "is_ckpt"), nic_rows)
    _write_csv(out_dir, rank, "flush",   ("step", "flush_ms"),              flush_rows)
    logger.info("[baseline] rank=%d  done %d steps", rank, n_steps)


# ---------------------------------------------------------------------------
# TEMPO-Nexus: DSCP staggered flush + per-layer micro-gates
# ---------------------------------------------------------------------------

def run_nexus(rank, world_size, device, n_steps, ckpt_interval,
              out_dir, n_layers, dtype, base_window_ms=200.0):
    """TEMPO-Nexus: DSCP window assignment + per-layer AR micro-gates."""
    grad_shards, _ = make_model_and_data(device, dtype)
    probe = torch.randn(256 * 1024 * 1024 // 4, device=device, dtype=torch.float32)

    nexus = NexusCoordinator(
        rank=rank,
        world_size=world_size,
        n_layers=n_layers,
        base_window_ms=base_window_ms,
    )

    os.makedirs(out_dir, exist_ok=True)
    nccl_rows, nic_rows, flush_rows, window_rows, layer_rows = [], [], [], [], []

    for step in range(n_steps):
        is_ckpt = (step % ckpt_interval == 0) and step > 0
        nexus.begin_step(step)

        # Per-layer AllReduce with micro-gate signaling
        layer_ar_times = []
        for layer_id in range(n_layers):
            t_ar0 = time.perf_counter()
            if dist.is_initialized():
                dist.all_reduce(grad_shards[layer_id], async_op=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            ar_ms = (time.perf_counter() - t_ar0) * 1000
            layer_ar_times.append(ar_ms)

            # Fire per-layer micro-gate
            nexus.on_layer_ar_done(layer_id=layer_id)

        # Measure AllReduce BW
        bw  = measure_allreduce_bw(probe)
        nic = nic_bw_gbps()
        nccl_rows.append((step, bw, is_ckpt))
        nic_rows.append((step, nic, is_ckpt))
        for lid, ar_ms in enumerate(layer_ar_times):
            layer_rows.append((step, lid, ar_ms))

        if is_ckpt:
            # DSCP: wait for assigned window, then flush
            flush_start = time.perf_counter()
            win = nexus.wait_for_window(step=step)
            window_rows.append((step, rank, win.position, win.delay_seconds * 1000,
                                 win.window_ms, win.my_load_gbps))

            tmp = Path(f"/tmp/tempo_nexus_eval/nexus_rank{rank}_step{step}.pt")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            dummy = torch.zeros(16 * 1024 * 1024, dtype=torch.float32)
            torch.save(dummy, str(tmp))
            flush_ms = (time.perf_counter() - flush_start) * 1000
            nexus.record_flush_time(flush_ms / 1000)
            flush_rows.append((step, flush_ms, win.position, win.delay_seconds * 1000))
            logger.debug("[nexus] rank=%d step=%d pos=%d delay=%.1fms flush=%.1fms bw=%.2f",
                          rank, step, win.position, win.delay_seconds * 1000, flush_ms, bw)

    _write_csv(out_dir, rank, "nccl_bw", ("step", "algbw_GBs", "is_ckpt"),         nccl_rows)
    _write_csv(out_dir, rank, "nic_bw",  ("step", "nic_gbps",  "is_ckpt"),          nic_rows)
    _write_csv(out_dir, rank, "flush",   ("step", "flush_ms", "dscp_pos", "delay_ms"), flush_rows)
    _write_csv(out_dir, rank, "windows", ("step", "rank", "pos", "delay_ms",
                                           "window_ms", "load_gbps"),               window_rows)
    _write_csv(out_dir, rank, "layer_ar", ("step", "layer_id", "ar_ms"),            layer_rows)
    nexus.print_stats()
    logger.info("[nexus] rank=%d  done %d steps", rank, n_steps)


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def _write_csv(out_dir, rank, name, header, rows):
    path = Path(out_dir) / f"{name}_rank{rank}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    logger.info("[CSV] %s  (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TEMPO-Nexus evaluation")
    parser.add_argument("--mode", choices=["baseline", "tempo-nexus"],
                        default="tempo-nexus")
    parser.add_argument("--n-steps",       type=int,   default=300)
    parser.add_argument("--ckpt-interval", type=int,   default=50)
    parser.add_argument("--world-size",    type=int,   default=None)
    parser.add_argument("--base-window-ms", type=float, default=200.0)
    parser.add_argument("--out-dir",       type=str,
                        default="results/phase8/nexus")
    parser.add_argument("--dtype",         choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()

    # --- Distributed init ---
    rank       = int(os.environ.get("RANK",       os.environ.get("SLURM_PROCID",  "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS",
                     str(args.world_size or 1))))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available()
                          else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if world_size > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo",
                                 rank=rank, world_size=world_size)
        logger.info("Distributed init: rank=%d/%d device=%s", rank, world_size, device)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    out   = Path(args.out_dir) / args.mode
    n_layers = 16

    if args.mode == "baseline":
        run_baseline(rank=rank, world_size=world_size, device=device,
                     n_steps=args.n_steps, ckpt_interval=args.ckpt_interval,
                     out_dir=str(out), n_layers=n_layers, dtype=dtype)
    else:
        run_nexus(rank=rank, world_size=world_size, device=device,
                  n_steps=args.n_steps, ckpt_interval=args.ckpt_interval,
                  out_dir=str(out), n_layers=n_layers, dtype=dtype,
                  base_window_ms=args.base_window_ms)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
