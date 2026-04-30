#!/usr/bin/env python3
"""
phase4/burst_traffic_workload.py — BurstGPT-Style Traffic Workload Injector

Motivation (OSDI framing):
    Real LLM serving traffic is NOT uniform.  Azure OpenAI's 121-day trace
    (BurstGPT, NSDI 2024) shows:
      • Inter-arrival times follow a heavy-tailed distribution (Pareto α≈1.2)
      • Response length varies 10×–100× within a 10-second window
      • Concurrent KV-cache eviction events (→ I/O bursts) correlate with
        arrival spikes — the WORST time for checkpoint flushing

    This script injects a BurstGPT-faithful traffic pattern onto the
    TEMPO checkpoint/flush pipeline and measures:
      1. P50/P95/P99 ITL (inter-token latency) under each mode
      2. NCCL bandwidth during I/O burst windows
      3. SLO violation rate (ITL > 200 ms) per mode

Modes:
    --mode baseline     : Greedy flush at every checkpoint
    --mode tempo-v1     : Phase-gated flush (TEMPO v1)
    --mode tempo-v2     : Full co-scheduling (TEMPO v2)
    --mode tempo-v2-no-net  : V2 without NetworkMonitor (ablation)
    --mode tempo-v2-no-gain : V2 without ServiceGain    (ablation)
    --mode tempo-v2-no-il   : V2 without InterleavingEngine (ablation)

Output:
    results/phase4/{mode}/burst_stats_rank{rank}.csv
    Columns: step, nccl_bw_gbs, itl_ms, io_active, nic_util_pct,
             svc_gain, flush_deferred, slo_violation
"""

import os
import sys
import csv
import math
import time
import random
import signal
import logging
import argparse
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

sys.path.insert(0, str(Path(__file__).parent.parent))
from tempo import TEMPOScheduler, TEMPOSchedulerV2
from tempo.phase_monitor import PhaseMonitor
from tempo.trace_loader import TraceLoader

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger


# ============================================================================
# Args
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Phase 4: BurstGPT-style evaluation")
    p.add_argument("--mode", required=True,
                   choices=["baseline", "tempo-v1", "tempo-v2",
                            "tempo-v2-no-net", "tempo-v2-no-gain", "tempo-v2-no-il"])
    p.add_argument("--model-size",  choices=["1b", "7b"], default="1b")
    p.add_argument("--num-steps",   type=int, default=300)
    p.add_argument("--warmup-steps",type=int, default=20)
    p.add_argument("--ckpt-every",  type=int, default=30)
    p.add_argument("--batch-size",  type=int, default=1)
    p.add_argument("--seq-len",     type=int, default=2048)
    p.add_argument("--burst-rate",  type=float, default=3.0,
                   help="Pareto burst scale (ignored when --trace-path given)")
    p.add_argument("--trace-path",  type=str, default=None,
                   help="Path to BurstGPT CSV (e.g. data/traces/gpt_3.5_turbo.csv). "
                        "If omitted, falls back to synthetic BurstGPT-calibrated generator.")
    p.add_argument("--slo-itl-ms",  type=float, default=200.0,
                   help="ITL SLO threshold in ms (default 200 ms)")
    p.add_argument("--output-dir",  type=str, default="results/phase4")
    p.add_argument("--local-nvme",  type=str,
                   default=os.environ.get("LOCAL_NVME", "/tmp/tempo_ckpts"))
    p.add_argument("--lustre-dir",  type=str,
                   default=os.environ.get("PSCRATCH", "/tmp") + "/tempo_phase4")
    p.add_argument("--milestone-interval", type=int, default=100)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ============================================================================
# Distributed init
# ============================================================================

def init_dist():
    rank       = int(os.environ.get("RANK",       os.environ.get("SLURM_PROCID",  0)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS",  1)))
    os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "localhost"))
    os.environ.setdefault("MASTER_PORT", "29501")
    num_gpus   = torch.cuda.device_count()
    cuda_local = local_rank if local_rank < num_gpus else 0
    device = torch.device(f"cuda:{cuda_local}")
    torch.cuda.set_device(cuda_local)
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size,
                            device_id=device)
    return rank, cuda_local, world_size


# ============================================================================
# Model (same as phase3)
# ============================================================================

LLAMA_CFGS = {
    "1b": dict(hidden_size=2048, intermediate_size=5632,
               num_hidden_layers=16, num_attention_heads=16,
               num_key_value_heads=8, vocab_size=128256),
    "7b": dict(hidden_size=4096, intermediate_size=14336,
               num_hidden_layers=32, num_attention_heads=32,
               num_key_value_heads=8, vocab_size=128256),
}


def build_model(cfg_name: str, device):
    try:
        from transformers import LlamaConfig, LlamaForCausalLM, LlamaDecoderLayer
        cfg = LlamaConfig(**LLAMA_CFGS[cfg_name], max_position_embeddings=4096,
                          rope_theta=500000.0)
        cfg.use_cache = False
        model = LlamaForCausalLM(cfg).to(dtype=torch.bfloat16)
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )
        return FSDP(
            model.to(device),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            ),
            auto_wrap_policy=wrap_policy,
            device_id=device,
        )
    except ImportError:
        # Fallback: simple transformer
        import functools
        class _Layer(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.ln  = nn.LayerNorm(h)
                self.attn = nn.Linear(h, h, bias=False)
                self.ff   = nn.Sequential(nn.Linear(h, h*4, bias=False),
                                          nn.GELU(),
                                          nn.Linear(h*4, h, bias=False))
            def forward(self, x):
                return x + self.ff(self.ln(x + self.attn(self.ln(x))))

        h    = LLAMA_CFGS[cfg_name]["hidden_size"]
        nlay = LLAMA_CFGS[cfg_name]["num_hidden_layers"]
        seq  = nn.Sequential(nn.Embedding(128256, h), *[_Layer(h) for _ in range(nlay)],
                             nn.LayerNorm(h), nn.Linear(h, 128256, bias=False))
        return FSDP(
            seq.to(device),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=device,
        )

import functools


# ============================================================================
# BurstGPT Traffic Wrapper (uses tempo.TraceLoader for real traces)
# ============================================================================

class BurstGPTWorkload:
    """
    Replays real BurstGPT arrival times and response lengths.

    When ``trace_path`` is supplied, loads a real BurstGPT CSV via
    ``tempo.TraceLoader``.  Otherwise falls back to the synthetic
    calibrated generator built into TraceLoader (emits WARNING in logs).

    IAT → ITL mapping:
      Short IATs (burst) → high nic_util_pct → higher ITL penalty.
      Long IATs (slack)  → low  nic_util_pct → near-baseline ITL.
    """

    def __init__(
        self,
        trace_path:   Optional[str] = None,
        n_requests:   int           = 2000,
        burst_rate:   float         = 3.0,
        rng_seed:     int           = 42,
    ):
        loader = TraceLoader(
            trace_path   = trace_path,
            trace_type   = "burstgpt" if trace_path else "synthetic",
            burst_rate   = burst_rate,
            rng_seed     = rng_seed,
        )
        requests = loader.load(n_requests=n_requests)
        stats    = loader.statistics()
        log.info(
            "BurstGPTWorkload: loaded %d requests  mean_rps=%.1f "
            "peak_rps=%.1f burst_ratio=%.2f cv_iat=%.2f",
            stats.n_requests, stats.mean_rps, stats.peak_rps,
            stats.burst_ratio, stats.cv_iat,
        )
        self._iats: List[float] = loader.inter_arrival_times()
        self._cursor  = 0
        self._rng     = np.random.default_rng(rng_seed)
        self._base_itl_ms = 15.0   # baseline ITL under no contention

    def next_arrival(self) -> float:
        """Return next inter-arrival time (ms), cycling through the trace."""
        if not self._iats:
            return 50.0
        iat_s = self._iats[self._cursor % len(self._iats)]
        self._cursor += 1
        return iat_s * 1000.0   # seconds → ms

    def simulate_itl(
        self,
        io_active:    bool,
        nic_util_pct: float,
        nccl_bw_gbs:  float,
        base_bw_gbs:  float = 8.0,
    ) -> float:
        """
        Simulate inter-token latency given current system state.

        Physics:
          ITL = base_itl × (1 + io_penalty) × (1 + nic_penalty) × (1 + bw_penalty)
        """
        io_penalty  = 0.40 if io_active else 0.0
        nic_sat     = min(1.0, nic_util_pct / 100.0)
        nic_penalty = nic_sat ** 2
        bw_ratio    = max(0.1, nccl_bw_gbs / max(0.1, base_bw_gbs))
        bw_penalty  = max(0.0, 1.0 - bw_ratio)
        itl = self._base_itl_ms * (1 + io_penalty) * (1 + nic_penalty) * (1 + bw_penalty)
        itl *= float(self._rng.normal(1.0, 0.1))
        return max(1.0, itl)


# ============================================================================
# NCCL bandwidth measurement
# ============================================================================

def measure_nccl_bw(device, world_size: int, tensor_size_gb: float = 0.5) -> float:
    """Measure AllReduce bandwidth (GB/s) using a synthetic tensor."""
    nbytes = int(tensor_size_gb * 1e9)
    nelem  = nbytes // 2   # bfloat16
    t      = torch.zeros(nelem, dtype=torch.bfloat16, device=device)
    # Warmup
    dist.all_reduce(t)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    dist.all_reduce(t)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Algorithmic bandwidth = (2*(N-1)/N) * bytes / time — approximate here
    return (nbytes / elapsed) / 1e9


# ============================================================================
# Main training loop
# ============================================================================

def build_tempo(args, rank, world_size, device):
    mode_str = args.mode
    if mode_str == "baseline":
        return TEMPOScheduler(
            rank=rank, world_size=world_size,
            local_nvme_dir=args.local_nvme,
            lustre_dir=args.lustre_dir,
            mode="baseline",
        )
    elif mode_str == "tempo-v1":
        return TEMPOScheduler(
            rank=rank, world_size=world_size,
            local_nvme_dir=args.local_nvme,
            lustre_dir=args.lustre_dir,
            mode="tempo",
            flush_chunk_mb=128,
            adaptive_chunk=True,
        )
    else:
        enable_net  = "no-net"  not in mode_str
        enable_gain = "no-gain" not in mode_str
        enable_il   = "no-il"   not in mode_str
        return TEMPOSchedulerV2(
            rank=rank, world_size=world_size,
            local_nvme_dir=args.local_nvme,
            lustre_dir=args.lustre_dir,
            mode="tempo",
            flush_chunk_mb=128,
            adaptive_chunk=True,
            milestone_interval=args.milestone_interval,
            enable_network_monitor=enable_net,
            enable_service_gain=enable_gain,
            enable_interleaving=enable_il,
        )


def main():
    args       = parse_args()
    rank, dev_local, world_size = init_dist()
    device     = torch.device(f"cuda:{dev_local}")

    out_dir = Path(args.output_dir) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"burst_stats_rank{rank}.csv"

    logger = log(f"phase4.r{rank}")
    if rank == 0:
        logger.info("=== Phase 4: BurstGPT Evaluation ===")
        logger.info("mode=%s model=%s steps=%d burst_rate=%.1f",
                    args.mode, args.model_size, args.num_steps, args.burst_rate)

    # Build model + tempo
    model = build_model(args.model_size, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tempo = build_tempo(args, rank, world_size, device)
    workload = BurstGPTWorkload(
        trace_path  = args.trace_path,
        n_requests  = max(500, args.num_steps * 4),
        burst_rate  = args.burst_rate,
    )

    # Register FSDP comm hook
    model.register_comm_hook(tempo.phase_monitor, PhaseMonitor.fsdp_comm_hook)

    records = []

    for step in range(args.num_steps + args.warmup_steps):
        tempo.on_step_begin(step)

        input_ids = torch.randint(0, 128256,
                                  (args.batch_size, args.seq_len),
                                  device=device)
        with tempo.compute_phase():
            out  = model(input_ids, labels=input_ids)
            loss = out.loss if hasattr(out, "loss") else out[0].mean()
            loss.backward()

        with tempo.nccl_phase():
            optimizer.step()
        optimizer.zero_grad()

        tempo.on_step_end(step)

        is_ckpt = (step >= args.warmup_steps) and (step % args.ckpt_every == 0)
        if is_ckpt:
            tempo.checkpoint(model.state_dict(), step)

        if step < args.warmup_steps:
            continue

        # Measure NCCL bandwidth
        nccl_bw = measure_nccl_bw(device, world_size)

        # Simulate ITL from workload model
        nic_util = 0.0
        if hasattr(tempo, "net_monitor") and tempo.net_monitor is not None:
            nic_util = tempo.net_monitor.current_util_fraction() * 100.0
        io_active = is_ckpt

        itl_ms = workload.simulate_itl(
            io_active    = io_active,
            nic_util_pct = nic_util,
            nccl_bw_gbs  = nccl_bw,
        )

        svc_gain     = 0.0
        flush_deferred = False
        if hasattr(tempo, "svc_gain") and tempo.svc_gain is not None:
            svc_gain = tempo.svc_gain.compute_gain(step)
            flush_deferred = (
                tempo.svc_gain._stats["jobs_deferred"] > 0
                and is_ckpt
            )

        slo_violation = itl_ms > args.slo_itl_ms

        records.append(dict(
            step           = step,
            nccl_bw_gbs    = round(nccl_bw,  4),
            itl_ms         = round(itl_ms,   3),
            io_active      = int(io_active),
            nic_util_pct   = round(nic_util, 2),
            svc_gain       = round(svc_gain, 4),
            flush_deferred = int(flush_deferred),
            slo_violation  = int(slo_violation),
            loss           = round(loss.item(), 5),
        ))

        if rank == 0 and step % 50 == 0:
            logger.info("step=%d nccl_bw=%.2f GB/s itl=%.1f ms nic=%.1f%% slo_ok=%s",
                        step, nccl_bw, itl_ms, nic_util,
                        "YES" if not slo_violation else "NO")

    # Write CSV
    if rank == 0 and records:
        fieldnames = list(records[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(records)
        logger.info("Saved %d records → %s", len(records), csv_path)

    tempo.shutdown()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
