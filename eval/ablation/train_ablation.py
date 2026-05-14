#!/usr/bin/env python3
"""
eval/ablation/train_ablation.py — TEMPO Ablation Study Training Script
=======================================================================
Purpose: Quantify the contribution of each TEMPO component independently
         (OSDI-style ablation table).

Ablation modes
--------------
  baseline        — greedy Lustre flush, no TEMPO gating
  core            — phase-gate only (V1 PhaseMonitor + CheckpointManager)
  core_p1         — core + GPU-Driven NIC Doorbell (P1, cxi_dry_run=True on
                     non-Perlmutter)
  core_p1_p2      — core + P1 + NVLink PCIe Multipath Routing (P2)
  core_p1_p2_p3   — core + P1 + P2 + libfabric CXI TC Control (P3, full V6)

Metrics collected per step
--------------------------
  allreduce_ms, algbw_GBs, busbw_GBs, is_ckpt_step, ablation_mode

Output: results/ablation/{ablation_mode}/nccl_bw_rank0.csv
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
from tempo.scheduler import TEMPOSchedulerV6
from tempo.phase_monitor import PhaseMonitor, TrainingPhase

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s r%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger

# Valid ablation modes in order from weakest to strongest
ABLATION_MODES = [
    "baseline",
    "core",
    "core_p1",
    "core_p1_p2",
    "core_p1_p2_p3",
]


# ============================================================================
# Arguments
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation-mode", choices=ABLATION_MODES, required=True,
                   help="Which TEMPO components to enable")
    p.add_argument("--model-size",     choices=["1b", "7b", "13b"], default="1b")
    p.add_argument("--num-steps",      type=int, default=80)
    p.add_argument("--warmup-steps",   type=int, default=10)
    p.add_argument("--ckpt-every",     type=int, default=20)
    p.add_argument("--batch-size",     type=int, default=1)
    p.add_argument("--seq-len",        type=int, default=512)
    p.add_argument("--local-nvme",     type=str,
                   default=os.environ.get("LOCAL_NVME", "/tmp/tempo_ablation"))
    p.add_argument("--lustre-dir",     type=str,
                   default=os.environ.get("PSCRATCH", "/tmp") + "/ablation")
    p.add_argument("--output-dir",     type=str, default="results/ablation")
    p.add_argument("--flush-chunk-mb", type=int, default=32)
    p.add_argument("--adaptive-chunk", action="store_true", default=True)
    p.add_argument("--no-adaptive-chunk", dest="adaptive_chunk", action="store_false")
    p.add_argument("--cxi-dry-run",    action="store_true", default=True,
                   help="Dry-run CXI TC control (safe on non-Perlmutter, default True)")
    p.add_argument("--no-cxi-dry-run", dest="cxi_dry_run", action="store_false",
                   help="Enable real fi_setopt CXI TC calls (Perlmutter with CXI only)")
    p.add_argument("--verbose",        action="store_true")
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
    num_gpus   = torch.cuda.device_count()
    cuda_local = local_rank if local_rank < num_gpus else 0
    torch.cuda.set_device(cuda_local)
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)
    return rank, cuda_local, world_size


# ============================================================================
# Model
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
# Scheduler factory — maps ablation mode → TEMPOScheduler instance
# ============================================================================

def build_scheduler(mode: str, rank: int, world_size: int, args) -> TEMPOScheduler:
    """
    Maps ablation mode strings to the appropriate scheduler configuration.

    baseline      → TEMPOScheduler(mode="baseline")          — greedy flush
    core          → TEMPOScheduler(mode="tempo")             — phase-gate only (V1)
    core_p1       → TEMPOSchedulerV6(enable_gpu_doorbell=T)  — + GPU doorbell (PoC)
    core_p1_p2    → TEMPOSchedulerV6(+ enable_nvlink_routing=T) — + NVLink routing
    core_p1_p2_p3 → TEMPOSchedulerV6(all enabled)           — full V6 stack
    """
    common = dict(
        rank           = rank,
        world_size     = world_size,
        local_nvme_dir = args.local_nvme,
        lustre_dir     = args.lustre_dir,
        flush_chunk_mb = args.flush_chunk_mb,
        adaptive_chunk = args.adaptive_chunk,
        verbose        = args.verbose,
    )

    if mode == "baseline":
        return TEMPOScheduler(mode="baseline", **common)

    if mode == "core":
        return TEMPOScheduler(mode="tempo", **common)

    # V6 modes — P1, P1+P2, P1+P2+P3
    enable_p1 = mode in ("core_p1", "core_p1_p2", "core_p1_p2_p3")
    enable_p2 = mode in ("core_p1_p2", "core_p1_p2_p3")
    enable_p3 = mode in ("core_p1_p2_p3",)
    # cxi_dry_run: True by default everywhere except --no-cxi-dry-run on real Perlmutter
    cxi_dry = args.cxi_dry_run if enable_p3 else True

    return TEMPOSchedulerV6(
        mode                  = "tempo",
        enable_gpu_doorbell   = enable_p1,
        enable_nvlink_routing = enable_p2,
        enable_cxi_tc_control = enable_p3,
        cxi_dry_run           = cxi_dry,
        **common,
    )


# ============================================================================
# Comm hook
# ============================================================================

class _AblationCommState:
    def __init__(self, world_size, phase_monitor, ablation_mode):
        self.world_size    = world_size
        self.phase_monitor = phase_monitor
        self.ablation_mode = ablation_mode
        self.records: List[dict] = []
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)
        self.step         = 0
        self.is_ckpt_step = False


def ablation_comm_hook(state: _AblationCommState,
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
        "step":          state.step,
        "ablation_mode": state.ablation_mode,
        "allreduce_ms":  round(lat_ms, 4),
        "algbw_GBs":     round(algbw,  4),
        "busbw_GBs":     round(busbw,  4),
        "is_ckpt_step":  int(state.is_ckpt_step),
        "timestamp":     round(time.time(), 3),
    })


# ============================================================================
# Main
# ============================================================================

def main():
    args   = parse_args()
    rank, local_rank, world_size = init_dist()
    device = torch.device("cuda")
    logger = log(str(rank))

    if rank == 0:
        logger.info(f"Ablation={args.ablation_mode} | model={args.model_size} | "
                    f"world_size={world_size}")

    model, decoder_cls = build_model(args.model_size, device)
    model = fsdp_wrap(model, decoder_cls, device)

    lustre_dir_mode = Path(args.lustre_dir) / args.ablation_mode
    lustre_dir_mode.mkdir(parents=True, exist_ok=True)
    args.lustre_dir = str(lustre_dir_mode)

    tempo = build_scheduler(args.ablation_mode, rank, world_size, args)

    # Phase monitor is always present — baseline just won't gate on it
    comm_state = _AblationCommState(
        world_size    = world_size,
        phase_monitor = tempo.phase_monitor if args.ablation_mode != "baseline" else None,
        ablation_mode = args.ablation_mode,
    )
    model.register_comm_hook(comm_state, ablation_comm_hook)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-4)
    vocab_size = LLAMA_CFGS[args.model_size]["vocab_size"]

    total = args.warmup_steps + args.num_steps
    for step in range(total):
        is_measurement = step >= args.warmup_steps
        mstep          = step - args.warmup_steps
        is_ckpt        = is_measurement and (mstep % args.ckpt_every == 0)

        comm_state.step         = mstep
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
            logger.info(f"  [{args.ablation_mode}] step {step:3d} {tag}  "
                        f"loss={loss.item():.4f}")

    # Save CSV (rank 0 only)
    if rank == 0:
        good = [r for r in comm_state.records if r["step"] >= 0]
        out_path = Path(args.output_dir) / args.ablation_mode / f"nccl_bw_rank{rank}.csv"
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
                m = sum(r["algbw_GBs"] for r in ckpt_rows) / len(ckpt_rows)
                logger.info(f"[{args.ablation_mode}] ckpt-step algbw mean={m:.3f} GB/s → {out_path}")
            if non_rows:
                m = sum(r["algbw_GBs"] for r in non_rows) / len(non_rows)
                logger.info(f"[{args.ablation_mode}] non-ckpt  algbw mean={m:.3f} GB/s")

    tempo.shutdown(wait=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
