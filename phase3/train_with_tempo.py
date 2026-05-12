#!/usr/bin/env python3
"""
phase3/train_with_tempo.py — Full Evaluation: BASELINE vs. TEMPO Mode

Runs a Llama-3-1B (random weights) FSDP training loop in one of two modes:

  --mode baseline : Greedy flush — checkpoint is written directly to Lustre
                    while training continues → PCIe contention → BW drop.

  --mode tempo    : Paced flush — checkpoint is staged to local NVMe instantly
                    and flushed to Lustre only during matmul phases, never
                    during NCCL collectives → BW stays flat.

Both modes produce a CSV with per-step NCCL bandwidth, which plot_killer_graph.py
overlays to produce the paper's "Killer Graph" figure.

Launch via run_evaluation.slurm, or directly:
  srun --ntasks-per-node=4 --gpus-per-node=4 \\
       python phase3/train_with_tempo.py --mode tempo --num-steps 400
"""

import os
import sys
import csv
import time
import signal
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

# TEMPO package (workspace root must be on PYTHONPATH)
sys.path.insert(0, str(Path(__file__).parent.parent))
from tempo import TEMPOScheduler
from tempo.phase_monitor import PhaseMonitor

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
    p = argparse.ArgumentParser(description="TEMPO vs Baseline evaluation")
    p.add_argument("--mode", choices=["baseline", "tempo"], required=True,
                   help="'baseline' = greedy flush; 'tempo' = paced flush")
    p.add_argument("--model-size", choices=["1b", "7b", "13b"], default="7b")
    p.add_argument("--num-steps",   type=int, default=400)
    p.add_argument("--warmup-steps",type=int, default=30)
    p.add_argument("--ckpt-every",  type=int, default=50,
                   help="Save checkpoint every N steps (default 50)")
    p.add_argument("--batch-size",  type=int, default=1)
    p.add_argument("--seq-len",     type=int, default=2048)
    p.add_argument("--output-dir",  type=str, default="results")
    p.add_argument("--local-nvme",  type=str,
                   default=os.environ.get("LOCAL_NVME", "/tmp/tempo_ckpts"))
    p.add_argument("--lustre-dir",  type=str,
                   default=os.environ.get("PSCRATCH", "/tmp/lustre_mock") + "/tempo_eval")
    p.add_argument("--flush-chunk-mb", type=int, default=128,
                   help="Chunk size (MB) for paced flush (default 128)")
    p.add_argument("--adaptive-chunk", action="store_true",
                   help="Automatically tune chunk size based on NCCL phase duration")
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
    os.environ.setdefault("MASTER_PORT", "29500")
    # CUDA_VISIBLE_DEVICES=0,1,2,3 set by SLURM --gpus-per-node=4
    # Use local_rank to select the correct GPU; guard against fewer GPUs.
    num_gpus = torch.cuda.device_count()
    cuda_local = local_rank if local_rank < num_gpus else 0
    torch.cuda.set_device(cuda_local)
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)
    return rank, cuda_local, world_size


# ============================================================================
# Model construction (same as phase1)
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


def build_model(size: str, device):
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    cfg = LlamaConfig(max_position_embeddings=4096, rms_norm_eps=1e-5,
                      **LLAMA_CFGS[size])
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
# FSDP comm hook that records NCCL timing AND optionally signals TEMPO
# ============================================================================

class _TimedCommState:
    def __init__(self, world_size: int, phase_monitor: Optional[PhaseMonitor],
                 process_group=None):
        self.world_size    = world_size
        self.phase_monitor = phase_monitor
        self.process_group = process_group   # FSDP process group for reduce_scatter
        self.records: List[dict] = []
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)
        self.step = 0


def timed_pacing_hook(
    state: _TimedCommState,
    padded_unsharded_grad: torch.Tensor,
    new_sharded_grad: torch.Tensor,
) -> None:
    """
    FSDP comm hook (PyTorch 2.11+ API: state, padded_unsharded_grad, new_sharded_grad).
    Dual responsibilities:
      1. Measure NCCL reduce_scatter latency (for CSV output / Killer Graph).
      2. Signal PhaseMonitor to pause background I/O flush (TEMPO mode).
    """
    if state.phase_monitor:
        from tempo.phase_monitor import TrainingPhase
        state.phase_monitor.set_phase(TrainingPhase.NCCL_COMM)

    size_GB = padded_unsharded_grad.nbytes / 1e9
    state._s.record()

    dist.reduce_scatter_tensor(
        new_sharded_grad,
        padded_unsharded_grad,
        group=state.process_group,
    )

    state._e.record()
    torch.cuda.synchronize()
    lat_ms = state._s.elapsed_time(state._e)
    algbw  = size_GB / (lat_ms * 1e-3)
    # reduce_scatter busbw formula: (N-1)/N × algbw
    busbw  = algbw * (state.world_size - 1) / state.world_size

    if state.phase_monitor:
        from tempo.phase_monitor import TrainingPhase
        state.phase_monitor.set_phase(TrainingPhase.COMPUTE)

    state.records.append({
        "step":         state.step,
        "latency_ms":   round(lat_ms, 4),
        "algbw_GBs":    round(algbw,  4),
        "busbw_GBs":    round(busbw,  4),
        "timestamp":    round(time.time(), 3),
    })


# ============================================================================
# Main training loop
# ============================================================================

def main():
    args = parse_args()
    rank, local_rank, world_size = init_dist()
    device = torch.device("cuda")
    logger = log(str(rank))

    # ---- Build model ----
    if rank == 0:
        logger.info(f"Mode={args.mode}  Model=Llama-{args.model_size.upper()}  "
                    f"steps={args.num_steps}  ckpt_every={args.ckpt_every}")
    model, decoder_cls = build_model(args.model_size, device)
    model = fsdp_wrap(model, decoder_cls, device)

    # ---- TEMPO scheduler ----
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

    # ---- FSDP comm hook ----
    # In "tempo" mode: hook signals PhaseMonitor (pauses flush during NCCL).
    # In "baseline" mode: hook only records timing (no gating).
    comm_state = _TimedCommState(
        world_size    = world_size,
        phase_monitor = tempo.phase_monitor if args.mode == "tempo" else None,
        process_group = None,   # None = default process group (dist.group.WORLD)
    )
    model.register_comm_hook(comm_state, timed_pacing_hook)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    vocab_size = LLAMA_CFGS[args.model_size]["vocab_size"]
    out_path   = Path(args.output_dir) / args.mode
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path   = out_path / f"nccl_bw_rank{rank}.csv"

    if rank == 0:
        logger.info(f"CSV output: {csv_path}")

    # ---- Training loop ----
    total = args.warmup_steps + args.num_steps
    for step in range(total):
        comm_state.step = step - args.warmup_steps
        tempo.on_step_begin(step)

        input_ids = torch.randint(
            0, vocab_size,
            (args.batch_size, args.seq_len), device=device
        )

        # Forward + backward (NCCL gradient reduction happens inside backward)
        with tempo.compute_phase():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out  = model(input_ids=input_ids, labels=input_ids)
                loss = out.loss
            loss.backward()   # ← FSDP comm hook fires here

        optimizer.step()
        optimizer.zero_grad()
        step_ms = tempo.on_step_end()

        # ---- VRAM tracking ----
        vram_used_gb = torch.cuda.memory_allocated(device) / 1e9
        # Annotate the last NCCL record for this step with vram + chunk_mb
        if comm_state.records and comm_state.records[-1]["step"] == comm_state.step:
            comm_state.records[-1]["vram_used_gb"] = round(vram_used_gb, 3)
            chunk_mb = tempo.ckpt_manager.chunk_bytes // (1024 * 1024)
            comm_state.records[-1]["chunk_mb"] = chunk_mb

        # ---- Checkpoint ----
        is_measurement = step >= args.warmup_steps
        if is_measurement and ((step - args.warmup_steps) % args.ckpt_every == 0):
            state = model.state_dict()
            if rank == 0:
                logger.info(f"  Step {step}: triggering checkpoint flush "
                            f"(mode={args.mode})")
            tempo.checkpoint(state, step)

        if rank == 0 and step % 20 == 0:
            tag = "(warmup)" if step < args.warmup_steps else ""
            logger.info(f"  Step {step:4d} {tag}  loss={loss.item():.4f}  "
                        f"step_ms={step_ms:.0f}  vram={vram_used_gb:.2f}GB")

    # ---- Save CSV ----
    if rank == 0:
        good = [r for r in comm_state.records if r["step"] >= 0]
        if good:
            # Collect union of all keys (some rows get extra fields like vram_used_gb)
            all_keys: list = list(dict.fromkeys(k for r in good for k in r.keys()))
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore",
                                        restval="")
                writer.writeheader()
                writer.writerows(good)
            avg = sum(r["algbw_GBs"] for r in good) / len(good)
            logger.info(f"Saved {len(good)} measurements → {csv_path}")
            logger.info(f"Avg NCCL algbw = {avg:.3f} GB/s  (mode={args.mode})")

    tempo.shutdown(wait=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
