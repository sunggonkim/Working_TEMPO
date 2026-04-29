#!/usr/bin/env python3
"""
phase1/train_llm_profiling.py — LLM Training with NCCL Bandwidth Profiling

Demonstrates PCIe Root Complex contention on Perlmutter:
  Background checkpoint flush (NVMe→Lustre via Slingshot NIC) ⟶
  NCCL All-Reduce bandwidth degradation during LLM training.

Two modes controlled by --mode:
  nccl_bench  : Pure NCCL micro-benchmark on a large synthetic tensor.
                Clean measurement, ideal for the "Killer Graph".
  llm_fsdp    : Llama-3-8B (random weights) wrapped in PyTorch FSDP.
                Realistic training loop; NCCL timing via FSDP comm hook.

Output:
  CSV file per rank-0 with columns: step, latency_ms, algbw_GBs, busbw_GBs

Launch on Perlmutter (via SLURM srun — see run_phase1_verification.slurm):
  srun --ntasks-per-node=4 --gpus-per-node=4 \\
       python train_llm_profiling.py --mode nccl_bench --num-steps 300

Environment variables consumed:
  MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK, WORLD_SIZE  (set by torchrun / srun)
  SLURM_PROCID, SLURM_LOCALID                             (set by Slurm srun)
"""

import os
import sys
import csv
import time
import argparse
import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s rank%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="TEMPO Phase 1: NCCL Bandwidth Profiling")

    p.add_argument("--mode", choices=["nccl_bench", "llm_fsdp"],
                   default="nccl_bench",
                   help="nccl_bench: synthetic all_reduce loop; "
                        "llm_fsdp: Llama-3-8B FSDP training")

    # nccl_bench options
    p.add_argument("--tensor-gb", type=float, default=1.0,
                   help="Gradient tensor size in GB for nccl_bench mode (default: 1.0)")
    p.add_argument("--num-steps", type=int, default=300,
                   help="Number of measurement iterations (default: 300)")
    p.add_argument("--warmup-steps", type=int, default=20,
                   help="Warmup iterations (excluded from CSV, default: 20)")

    # llm_fsdp options
    p.add_argument("--model-size", choices=["1b", "7b", "13b", "70b"],
                   default="7b",
                   help="Llama model size for llm_fsdp mode (default: 7b)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Per-GPU micro-batch size (default: 1)")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Sequence length (default: 2048)")

    # Output
    p.add_argument("--output-dir", type=str, default="results",
                   help="Directory for CSV output (default: results/)")
    p.add_argument("--run-tag", type=str, default="baseline",
                   help="Tag for this run: 'baseline' or 'contention' (default: baseline)")

    return p.parse_args()


# ============================================================================
# Distributed initialisation (torchrun / srun compatible)
# ============================================================================

def init_distributed():
    """Initialise process group, handling both torchrun and srun launch styles."""
    # Slurm sets these; torchrun sets RANK / LOCAL_RANK / WORLD_SIZE directly
    rank        = int(os.environ.get("RANK",        os.environ.get("SLURM_PROCID",   0)))
    local_rank  = int(os.environ.get("LOCAL_RANK",  os.environ.get("SLURM_LOCALID",  0)))
    world_size  = int(os.environ.get("WORLD_SIZE",  os.environ.get("SLURM_NTASKS",   1)))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)

    # With CUDA_VISIBLE_DEVICES=0,1,2,3 (set by SLURM header --gpus-per-node=4),
    # all 4 GPUs are visible. Use local_rank to select the correct device.
    # Guard: if SLURM exposed fewer GPUs than expected, fall back to 0.
    num_gpus = torch.cuda.device_count()
    cuda_local = local_rank if local_rank < num_gpus else 0
    torch.cuda.set_device(cuda_local)
    dist.init_process_group(
        backend    = "nccl",
        init_method = "env://",
        rank       = rank,
        world_size = world_size,
    )
    return rank, cuda_local, world_size


# ============================================================================
# NCCL bandwidth measurement helper
# ============================================================================

class NCCLBandwidthMeter:
    """
    Measures NCCL All-Reduce bandwidth using CUDA Events.

    CUDA Events record timestamps on the GPU's internal clock, giving
    sub-microsecond precision without the ~10µs overhead of
    torch.cuda.synchronize() in a tight benchmark loop.
    """

    def __init__(self, world_size: int):
        self.world_size = world_size
        self._start = torch.cuda.Event(enable_timing=True)
        self._end   = torch.cuda.Event(enable_timing=True)

    def measure(self, tensor: torch.Tensor) -> dict:
        """
        Run one all_reduce on `tensor`, return timing and bandwidth.

        Returns dict with:
          latency_ms  — wall time of the collective in milliseconds
          algbw_GBs   — algorithmic bandwidth (tensor_bytes / latency)
          busbw_GBs   — bus bandwidth = algbw * 2*(N-1)/N  (NCCL convention)
        """
        self._start.record()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        self._end.record()
        torch.cuda.synchronize()

        latency_ms = self._start.elapsed_time(self._end)   # ms
        size_GB    = tensor.nbytes / 1e9
        algbw_GBs  = size_GB / (latency_ms * 1e-3)
        busbw_GBs  = algbw_GBs * 2 * (self.world_size - 1) / self.world_size

        return {
            "latency_ms": latency_ms,
            "algbw_GBs":  algbw_GBs,
            "busbw_GBs":  busbw_GBs,
        }


# ============================================================================
# Mode 1: NCCL micro-benchmark
# ============================================================================

def run_nccl_bench(args, rank: int, world_size: int) -> None:
    """
    Tight all_reduce loop on a large synthetic tensor.

    This is the cleanest way to isolate NCCL bandwidth degradation:
    nothing else runs on the GPU except the collective, so any BW drop
    is purely due to PCIe/NIC contention from the background I/O.
    """
    device = torch.device("cuda")

    # Tensor size representing LLM gradient shard (e.g. 1 GB @ fp32)
    num_elements = int(args.tensor_gb * 1e9 / 4)   # float32 = 4 bytes
    tensor = torch.ones(num_elements, dtype=torch.float32, device=device)

    logger = logging.getLogger(str(rank))
    meter  = NCCLBandwidthMeter(world_size)

    output_path = Path(args.output_dir) / args.run_tag
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"nccl_bw_rank{rank}.csv"

    if rank == 0:
        logger.info(f"NCCL Bench: tensor={args.tensor_gb:.1f}GB  "
                    f"steps={args.num_steps}  warmup={args.warmup_steps}  "
                    f"world_size={world_size}")
        logger.info(f"Output CSV: {csv_path}")

    records: List[dict] = []

    total_steps = args.warmup_steps + args.num_steps
    for step in range(total_steps):
        bw = meter.measure(tensor)
        is_warmup = step < args.warmup_steps

        if not is_warmup:
            rec = {"step": step - args.warmup_steps,
                   "latency_ms": round(bw["latency_ms"], 4),
                   "algbw_GBs":  round(bw["algbw_GBs"],  4),
                   "busbw_GBs":  round(bw["busbw_GBs"],  4),
                   "timestamp":  round(time.time(), 3)}
            records.append(rec)

        if rank == 0 and step % 50 == 0:
            tag = "(warmup)" if is_warmup else ""
            logger.info(f"  Step {step:4d} {tag}: "
                        f"latency={bw['latency_ms']:.2f}ms  "
                        f"algbw={bw['algbw_GBs']:.2f}GB/s  "
                        f"busbw={bw['busbw_GBs']:.2f}GB/s")

    # Write CSV (rank 0 only — all ranks see the same traffic)
    if rank == 0 and records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        avg_bw = sum(r["algbw_GBs"] for r in records) / len(records)
        logger.info(f"Done. Avg algbw={avg_bw:.2f} GB/s  → {csv_path}")


# ============================================================================
# Mode 2: Llama FSDP training with NCCL comm hook timing
# ============================================================================

# Llama-3 model configurations keyed by size
LLAMA_CONFIGS = {
    "1b":  dict(hidden_size=2048,  intermediate_size=5632,
                num_hidden_layers=16, num_attention_heads=16,
                num_key_value_heads=8,  vocab_size=128256),
    "7b":  dict(hidden_size=4096,  intermediate_size=14336,
                num_hidden_layers=32, num_attention_heads=32,
                num_key_value_heads=8,  vocab_size=128256),
    "13b": dict(hidden_size=5120,  intermediate_size=13824,
                num_hidden_layers=40, num_attention_heads=40,
                num_key_value_heads=8,  vocab_size=128256),
    "70b": dict(hidden_size=8192,  intermediate_size=28672,
                num_hidden_layers=80, num_attention_heads=64,
                num_key_value_heads=8,  vocab_size=128256),
}


def build_llama_model(size: str, device: torch.device) -> nn.Module:
    """Instantiate a Llama-3-style model with random weights (no download needed)."""
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    except ImportError:
        raise SystemExit(
            "ERROR: transformers not installed.\n"
            "  pip install transformers>=4.40\n"
            "Or use --mode nccl_bench which has no extra dependencies."
        )

    cfg_kwargs = LLAMA_CONFIGS[size]
    config = LlamaConfig(
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        **cfg_kwargs,
    )
    # Initialise with random weights on meta device, then materialise on GPU
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    model = model.to_empty(device=device)
    model.init_weights()
    return model, LlamaDecoderLayer


def wrap_fsdp(model, layer_cls, device):
    """Wrap the model with FSDP using bf16 mixed precision and full sharding."""
    mp_policy = MixedPrecision(
        param_dtype  = torch.bfloat16,
        reduce_dtype = torch.bfloat16,
        buffer_dtype = torch.bfloat16,
    )
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_cls},
    )
    return FSDP(
        model,
        sharding_strategy = ShardingStrategy.FULL_SHARD,
        mixed_precision   = mp_policy,
        auto_wrap_policy  = auto_wrap,
        device_id         = torch.cuda.current_device(),
    )


class _FSDPCommTimingState:
    """State object for the FSDP comm hook that records per-call NCCL timing."""
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.records: List[dict] = []
        self._start_evt = torch.cuda.Event(enable_timing=True)
        self._end_evt   = torch.cuda.Event(enable_timing=True)
        self._step = 0


def timed_fsdp_comm_hook(state: _FSDPCommTimingState, bucket):
    """
    FSDP comm hook that times each gradient all_reduce.
    Appends {step, latency_ms, algbw_GBs, busbw_GBs} to state.records.
    """
    state._start_evt.record()
    tensor = bucket.buffer()
    size_GB = tensor.nbytes / 1e9

    fut = dist.all_reduce(tensor, async_op=True).get_future()

    def on_done(fut):
        state._end_evt.record()
        torch.cuda.synchronize()
        lat_ms   = state._start_evt.elapsed_time(state._end_evt)
        algbw    = size_GB / (lat_ms * 1e-3)
        busbw    = algbw * 2 * (state.world_size - 1) / state.world_size
        state.records.append({
            "step":        state._step,
            "latency_ms":  round(lat_ms, 4),
            "algbw_GBs":   round(algbw,  4),
            "busbw_GBs":   round(busbw,  4),
            "timestamp":   round(time.time(), 3),
        })
        return [fut.value()[0] / state.world_size]

    return fut.then(on_done)


def run_llm_fsdp(args, rank: int, world_size: int) -> None:
    """
    Llama-3 FSDP training loop with per-step NCCL bandwidth measurement.
    Uses random inputs (no real dataset needed) — we measure communication
    patterns, not model accuracy.
    """
    device = torch.device("cuda")
    logger = logging.getLogger(str(rank))

    if rank == 0:
        logger.info(f"Building Llama-{args.model_size.upper()} (random weights)...")
    model, decoder_cls = build_llama_model(args.model_size, device)
    model = wrap_fsdp(model, decoder_cls, device)

    # Register timing comm hook
    timing_state = _FSDPCommTimingState(world_size)
    model.register_comm_hook(timing_state, timed_fsdp_comm_hook)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    vocab_size = LLAMA_CONFIGS[args.model_size]["vocab_size"]
    seq_len    = args.seq_len
    batch_size = args.batch_size

    output_path = Path(args.output_dir) / args.run_tag
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"llm_nccl_bw_rank{rank}.csv"

    if rank == 0:
        logger.info(f"Llama-{args.model_size}: {args.num_steps} steps, "
                    f"bs={batch_size}, seq={seq_len}  →  {csv_path}")

    total_steps = args.warmup_steps + args.num_steps
    for step in range(total_steps):
        timing_state._step = step - args.warmup_steps

        input_ids = torch.randint(0, vocab_size,
                                  (batch_size, seq_len), device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out  = model(input_ids=input_ids, labels=input_ids)
            loss = out.loss

        loss.backward()    # FSDP all_reduce occurs here → comm hook fires
        optimizer.step()
        optimizer.zero_grad()

        if rank == 0 and step % 20 == 0:
            tag = "(warmup)" if step < args.warmup_steps else ""
            logger.info(f"  Step {step:4d} {tag}: loss={loss.item():.4f}")

    # Save CSV (only measurement steps, not warmup)
    if rank == 0:
        good = [r for r in timing_state.records if r["step"] >= 0]
        if good:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(good[0].keys()))
                writer.writeheader()
                writer.writerows(good)
            avg_bw = sum(r["algbw_GBs"] for r in good) / len(good)
            logger.info(f"Done. Avg NCCL algbw={avg_bw:.2f} GB/s  →  {csv_path}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    args = parse_args()
    rank, local_rank, world_size = init_distributed()

    try:
        if args.mode == "nccl_bench":
            run_nccl_bench(args, rank, world_size)
        else:
            run_llm_fsdp(args, rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
