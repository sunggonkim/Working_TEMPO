#!/usr/bin/env python3
"""
eval/node_scaling/train_scaling.py — Node Scaling Experiment Training Script
=============================================================================
Purpose: Measure AllReduce bandwidth degradation as node count grows (2→32),
         with and without concurrent checkpoint I/O (baseline vs. TEMPO).

This is the OSDI "scaling curve" figure:
  x-axis: number of nodes (2, 4, 8, 16, 32)
  y-axis: AllReduce mean latency (ms) at checkpoint steps

Differences from train_with_tempo.py:
  - Outputs a CSV with one row per step containing:
      step, n_nodes, world_size, mode, allreduce_ms, algbw_GBs, busbw_GBs,
      is_ckpt_step, timestamp
  - Accepts --output-csv and --n-nodes flags
  - Accepts --scale-exp flag (no-op; documents experiment type in CSV)
"""

import os
import sys
import csv
import time
import logging
import argparse
import functools
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tempo import TEMPOScheduler
from tempo.phase_monitor import PhaseMonitor, TrainingPhase

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s r%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger


# ============================================================================
# Arguments
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",           choices=["baseline", "tempo"], required=True)
    p.add_argument("--model-size",     choices=["1b", "7b", "13b"], default="1b")
    p.add_argument("--num-steps",      type=int, default=80)
    p.add_argument("--warmup-steps",   type=int, default=10)
    p.add_argument("--ckpt-every",     type=int, default=20)
    p.add_argument("--batch-size",     type=int, default=1)
    p.add_argument("--seq-len",        type=int, default=512)
    p.add_argument("--local-nvme",     type=str, default=os.environ.get("LOCAL_NVME", "/tmp/tempo_ckpts"))
    p.add_argument("--lustre-dir",     type=str, default=os.environ.get("PSCRATCH", "/tmp") + "/scaling")
    p.add_argument("--flush-chunk-mb", type=int, default=32)
    p.add_argument("--adaptive-chunk", action="store_true", default=True)
    p.add_argument("--no-adaptive-chunk", dest="adaptive_chunk", action="store_false")
    p.add_argument("--output-csv",     type=str, default="results/node_scaling/scaling.csv")
    p.add_argument("--n-nodes",        type=int, default=None,
                   help="Override node count label in CSV (default: SLURM_NNODES)")
    p.add_argument("--scale-exp",      action="store_true", help="Tag CSV as scaling experiment")
    p.add_argument("--verbose",        action="store_true")
    return p.parse_args()


# ============================================================================
# Distributed init (identical to train_with_tempo.py)
# ============================================================================

def init_dist():
    rank       = int(os.environ.get("RANK",       os.environ.get("SLURM_PROCID",  0)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS",  1)))
    os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "localhost"))
    os.environ.setdefault("MASTER_PORT", "29500")
    num_gpus = torch.cuda.device_count()
    cuda_local = local_rank if local_rank < num_gpus else 0
    torch.cuda.set_device(cuda_local)
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)
    return rank, cuda_local, world_size


# ============================================================================
# Model (same configs as train_with_tempo.py)
# ============================================================================

LLAMA_CFGS = {
    "1b":  dict(hidden_size=2048, intermediate_size=5632,
                num_hidden_layers=16, num_attention_heads=16,
                num_key_value_heads=8, vocab_size=128256),
    "7b":  dict(hidden_size=4096, intermediate_size=14336,
                num_hidden_layers=32, num_attention_heads=32,
                num_key_value_heads=8, vocab_size=128256),
    "13b": dict(hidden_size=5120, intermediate_size=13824,
                num_hidden_layers=40, num_attention_heads=40,
                num_key_value_heads=8, vocab_size=128256),
}


def build_model(size, device):
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    cfg = LlamaConfig(max_position_embeddings=4096, rms_norm_eps=1e-5, **LLAMA_CFGS[size])
    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)
    model = model.to_empty(device=device)
    model.init_weights()
    return model, LlamaDecoderLayer


def fsdp_wrap(model, decoder_cls, device):
    mp = MixedPrecision(param_dtype=torch.bfloat16,
                        reduce_dtype=torch.bfloat16,
                        buffer_dtype=torch.bfloat16)
    policy = functools.partial(transformer_auto_wrap_policy,
                               transformer_layer_cls={decoder_cls})
    return FSDP(model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mp,
                auto_wrap_policy=policy,
                device_id=torch.cuda.current_device())


# ============================================================================
# Comm hook (same as train_with_tempo.py, extended with is_ckpt_step field)
# ============================================================================

class _ScalingCommState:
    def __init__(self, world_size, phase_monitor, n_nodes):
        self.world_size    = world_size
        self.phase_monitor = phase_monitor
        self.n_nodes       = n_nodes
        self.records: List[dict] = []
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)
        self.step = 0
        self.is_ckpt_step = False


def scaling_comm_hook(state: _ScalingCommState,
                      padded_unsharded_grad: torch.Tensor,
                      new_sharded_grad: torch.Tensor) -> None:
    if state.phase_monitor:
        state.phase_monitor.set_phase(TrainingPhase.NCCL_COMM)

    size_GB = padded_unsharded_grad.nbytes / 1e9
    state._s.record()

    dist.reduce_scatter_tensor(
        new_sharded_grad,
        padded_unsharded_grad,
        group=None,
    )

    state._e.record()
    torch.cuda.synchronize()
    lat_ms = state._s.elapsed_time(state._e)
    algbw  = size_GB / (lat_ms * 1e-3)
    busbw  = algbw * (state.world_size - 1) / state.world_size

    if state.phase_monitor:
        state.phase_monitor.set_phase(TrainingPhase.COMPUTE)

    state.records.append({
        "step":         state.step,
        "n_nodes":      state.n_nodes,
        "world_size":   state.world_size,
        "allreduce_ms": round(lat_ms, 4),
        "algbw_GBs":    round(algbw,  4),
        "busbw_GBs":    round(busbw,  4),
        "is_ckpt_step": int(state.is_ckpt_step),
        "timestamp":    round(time.time(), 3),
    })


# ============================================================================
# Main
# ============================================================================

def main():
    args   = parse_args()
    rank, local_rank, world_size = init_dist()
    device = torch.device("cuda")
    logger = log(str(rank))

    n_nodes = args.n_nodes or int(os.environ.get("SLURM_NNODES", world_size // 4))

    if rank == 0:
        logger.info(f"Scaling exp | mode={args.mode} | N={n_nodes} nodes | "
                    f"world_size={world_size} | model={args.model_size}")

    model, decoder_cls = build_model(args.model_size, device)
    model = fsdp_wrap(model, decoder_cls, device)

    tempo = TEMPOScheduler(
        rank           = rank,
        world_size     = world_size,
        local_nvme_dir = args.local_nvme,
        lustre_dir     = args.lustre_dir,
        mode           = args.mode,
        flush_chunk_mb = args.flush_chunk_mb,
        adaptive_chunk = args.adaptive_chunk,
        verbose        = args.verbose,
    )

    comm_state = _ScalingCommState(
        world_size    = world_size,
        phase_monitor = tempo.phase_monitor if args.mode == "tempo" else None,
        n_nodes       = n_nodes,
    )
    model.register_comm_hook(comm_state, scaling_comm_hook)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-4)
    vocab_size = LLAMA_CFGS[args.model_size]["vocab_size"]

    total = args.warmup_steps + args.num_steps
    for step in range(total):
        is_measurement = step >= args.warmup_steps
        mstep          = step - args.warmup_steps
        is_ckpt        = is_measurement and (mstep % args.ckpt_every == 0)

        comm_state.step        = mstep
        comm_state.is_ckpt_step = is_ckpt
        tempo.on_step_begin(step)

        input_ids = torch.randint(0, vocab_size,
                                  (args.batch_size, args.seq_len), device=device)

        with tempo.compute_phase():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out  = model(input_ids=input_ids, labels=input_ids)
                loss = out.loss
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        tempo.on_step_end()

        if is_ckpt:
            tempo.checkpoint(model.state_dict(), step)

        if rank == 0 and step % 20 == 0:
            tag = "(warmup)" if not is_measurement else ""
            logger.info(f"  step {step:3d} {tag}  loss={loss.item():.4f}")

    # Save CSV (rank 0 only)
    if rank == 0:
        good = [r for r in comm_state.records if r["step"] >= 0]
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if good:
            fields = list(good[0].keys())
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(good)
            ckpt_rows = [r for r in good if r["is_ckpt_step"]]
            non_rows  = [r for r in good if not r["is_ckpt_step"]]
            if ckpt_rows:
                mean_ckpt = sum(r["allreduce_ms"] for r in ckpt_rows) / len(ckpt_rows)
                logger.info(f"N={n_nodes} | mode={args.mode} | "
                            f"AllReduce@ckpt mean={mean_ckpt:.3f} ms | "
                            f"rows={len(good)} → {out_path}")
            if non_rows:
                mean_non = sum(r["allreduce_ms"] for r in non_rows) / len(non_rows)
                logger.info(f"N={n_nodes} | mode={args.mode} | "
                            f"AllReduce@non-ckpt mean={mean_non:.3f} ms")

    tempo.shutdown(wait=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
