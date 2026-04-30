#!/usr/bin/env python3
"""
phase5/topology_qos_eval.py
============================
TEMPO v3 evaluation: Dragonfly topology-aware placement + Slingshot-11 QoS.

Two sub-experiments
-------------------
Exp A — Topology-Aware Placement (run with --exp placement)
  Measures NCCL AllReduce BW while a competing Lustre I/O flood is active,
  comparing three placement strategies:
    1. naive      : I/O goes to Lustre with no quota / topology awareness
    2. tempo-v2   : I/O gated by NCCL phase (InterleavingEngine)
    3. tempo-v3   : v2 + topology-aware slicing (TopologyRouter)

  Key metric: AllReduce BW (GB/s) at each step.  We expect:
    naive    → sawtooth (−40–50 % drop during flood)
    tempo-v2 → flat but with idle periods
    tempo-v3 → flat AND higher utilisation (no idle, just sliced)

Exp B — Hardware QoS Traffic Class (run with --exp qos)
  Measures P99 tail latency of NCCL AllReduce under I/O flood with:
    1. no-qos     : I/O uses same TC as NCCL (TC default / BE)
    2. qos-soft   : I/O marked TC0 (DSCP 0), NCCL uses default
    3. qos-hard   : I/O TC0 + NCCL explicitly TC3 (EF, DSCP 46)

  Key metric: P50 / P95 / P99 AllReduce latency (ms).  We expect:
    no-qos    → high P99 variance (tail inflation up to 5×)
    qos-soft  → moderate improvement (−25–35 % P99)
    qos-hard  → near-baseline P99 even at 100 % I/O flood (−55–65 %)

Output CSVs
-----------
  results/phase5/placement/{mode}/probe_rank{rank}.csv
  results/phase5/qos/{mode}/probe_rank{rank}.csv

Columns (both experiments):
  step, allreduce_bw_gbs, allreduce_lat_ms, io_flood_active,
  nic_util_pct, placement_tier, tc_name, svc_gain
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist

# ── TEMPO imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tempo.topology_router import TopologyRouter, PlacementTier
from tempo.qos_mapper       import QoSMapper, TC
from tempo.network_monitor  import NetworkMonitor
from tempo.interleaving_engine import InterleavingEngine
from tempo.service_gain     import ServiceGainScheduler


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

ALLREDUCE_STEPS   = 400
FLOOD_START_STEP  = 80
FLOOD_END_STEP    = 279
TENSOR_SIZE_GB    = 0.25        # 256 MB allreduce tensor
FLOOD_CHUNK_MB    = 128         # MB per I/O write
LUSTRE_PATH       = os.environ.get("PSCRATCH", "/tmp") + "/tempo_phase5_flood"


# ═══════════════════════════════════════════════════════════════════════════
# I/O flood thread
# ═══════════════════════════════════════════════════════════════════════════

class IOFloodThread(threading.Thread):
    """Continuously writes large chunks to Lustre to saturate the NIC."""

    def __init__(
        self,
        path:        str,
        chunk_mb:    int   = FLOOD_CHUNK_MB,
        qos:         Optional[QoSMapper] = None,
        topo_router: Optional[TopologyRouter] = None,
        ie:          Optional[InterleavingEngine] = None,
        mode:        str   = "naive",
    ) -> None:
        super().__init__(daemon=True)
        self.path        = path
        self.chunk_bytes = chunk_mb * 1024 * 1024
        self.qos         = qos
        self.topo_router = topo_router
        self.ie          = ie
        self.mode        = mode
        self._stop_ev    = threading.Event()
        self._active_ev  = threading.Event()
        self.bytes_written = 0

    def start_flood(self)  -> None: self._active_ev.set()
    def stop_flood(self)   -> None: self._active_ev.clear()
    def stop(self)         -> None:
        self._stop_ev.set()
        self._active_ev.set()   # unblock any wait

    def run(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        chunk = bytes(self.chunk_bytes)
        seq   = 0
        while not self._stop_ev.is_set():
            if not self._active_ev.wait(timeout=0.05):
                continue

            # --- topology-aware placement decision ---
            if self.topo_router is not None:
                safe_ms = self.ie.get_safe_window_ms() if self.ie else None
                decision = self.topo_router.route_kv_placement(
                    kv_size_bytes=self.chunk_bytes,
                    nccl_window_ms_remaining=safe_ms,
                )
                if decision.tier == PlacementTier.DEFERRED:
                    time.sleep(0.004)   # wait one NCCL window (~8 ms / 2)
                    continue

            # --- interleaving gate (v2+) ---
            if self.ie is not None and self.mode in ("tempo-v2", "tempo-v3"):
                self.ie.wait_for_io_window()

            # --- QoS marking (v3 qos modes) ---
            gain = 0.05   # background I/O has near-zero service gain
            if self.qos is not None:
                self.qos.apply_fd_priority(1, gain, urgency=0.0)

            fpath = os.path.join(self.path, f"chunk_{seq % 8}.bin")
            try:
                t0 = time.perf_counter()
                with open(fpath, "wb") as f:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                self.bytes_written += self.chunk_bytes
            except OSError:
                pass
            seq += 1


# ═══════════════════════════════════════════════════════════════════════════
# Measurement loop (shared)
# ═══════════════════════════════════════════════════════════════════════════

def _run_probe(
    rank:        int,
    world_size:  int,
    mode:        str,
    sub_exp:     str,
    out_dir:     Path,
) -> None:
    """
    Run ALLREDUCE_STEPS AllReduce ops, activating I/O flood on steps
    [FLOOD_START_STEP, FLOOD_END_STEP).  Record timing and BW per step.
    """
    n_elems = int(TENSOR_SIZE_GB * 1024**3 / 4)   # float32 elements
    tensor  = torch.randn(n_elems, dtype=torch.float32, device="cuda")

    # ── TEMPO v3 components ────────────────────────────────────────────
    net_mon  = NetworkMonitor() if mode != "naive" else None
    svc_gain = ServiceGainScheduler() if mode != "naive" else None
    ie       = InterleavingEngine()   if mode != "naive" else None

    topo_router = None
    if mode == "tempo-v3" and sub_exp == "placement":
        topo_router = TopologyRouter(world_size=world_size, rank=rank)
        # Register simulated peer groups (in real deployment: all-gather nids)
        rank_to_group = {r: r // 64 for r in range(world_size)}
        topo_router.register_peer_groups(rank_to_group)

    qos = None
    if sub_exp == "qos":
        if mode == "qos-soft":
            qos = QoSMapper(enabled=True, dry_run=True)   # classify only
        elif mode == "qos-hard":
            qos = QoSMapper(enabled=True, dry_run=False)  # apply DSCP marks

    # ── I/O flood thread (rank 0 only) ─────────────────────────────────
    flood_thread: Optional[IOFloodThread] = None
    if rank == 0:
        flood_thread = IOFloodThread(
            path=LUSTRE_PATH,
            chunk_mb=FLOOD_CHUNK_MB,
            qos=qos,
            topo_router=topo_router,
            ie=ie,
            mode=mode,
        )
        flood_thread.start()

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"probe_rank{rank}.csv"
    rows: List[dict] = []

    dist.barrier()

    for step in range(ALLREDUCE_STEPS):
        is_flood = FLOOD_START_STEP <= step < FLOOD_END_STEP

        # Start/stop flood
        if flood_thread is not None:
            if is_flood:
                flood_thread.start_flood()
            else:
                flood_thread.stop_flood()

        # InterleavingEngine hooks
        if ie is not None:
            ie.on_compute_start()

        # ServiceGain update
        if svc_gain is not None:
            svc_gain.update_step(step)

        # NCCL AllReduce measurement
        dist.barrier()
        t0 = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        lat_ms  = (t1 - t0) * 1000
        bw_gbs  = (TENSOR_SIZE_GB * 2 * (world_size - 1) / world_size) / (t1 - t0)

        if ie is not None:
            ie.on_compute_end()
            ie.on_nccl_end(lat_ms)

        # NIC utilisation
        nic_pct = 0.0
        if net_mon is not None:
            nic_pct = net_mon.current_util_fraction() * 100

        # Placement tier
        tier_name = "N/A"
        if topo_router is not None:
            d = topo_router.route_kv_placement(256 * 1024**2)
            tier_name = d.tier.name

        # QoS TC name
        tc_name = "N/A"
        gain_val = 0.0
        if qos is not None and svc_gain is not None:
            gain_val = svc_gain.compute_gain(step)
            tc       = qos.classify(gain_val, traffic_type="kv_cache", urgency=0.5)
            tc_name  = tc.name
        elif svc_gain is not None:
            gain_val = svc_gain.compute_gain(step)

        rows.append(dict(
            step               = step,
            allreduce_bw_gbs   = round(bw_gbs, 4),
            allreduce_lat_ms   = round(lat_ms, 4),
            io_flood_active    = int(is_flood),
            nic_util_pct       = round(nic_pct, 2),
            placement_tier     = tier_name,
            tc_name            = tc_name,
            svc_gain           = round(gain_val, 4),
        ))

        if step % 50 == 0 and rank == 0:
            print(f"  [phase5/{mode}] step={step:4d}  "
                  f"bw={bw_gbs:.2f} GB/s  lat={lat_ms:.2f} ms  "
                  f"flood={is_flood}  nic={nic_pct:.0f}%")

    # ── Clean up ────────────────────────────────────────────────────────
    if flood_thread is not None:
        flood_thread.stop()
        flood_thread.join(timeout=5)
    if net_mon is not None:
        net_mon.stop()

    # ── Write CSV ───────────────────────────────────────────────────────
    if rank == 0:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[phase5/{mode}] saved {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp",  choices=["placement", "qos"], required=True,
                    help="Which sub-experiment to run")
    ap.add_argument("--mode", required=True,
                    help="Mode name (naive / tempo-v2 / tempo-v3 / "
                         "no-qos / qos-soft / qos-hard)")
    ap.add_argument("--out",  default=str(ROOT / "results" / "phase5"),
                    help="Output directory root")
    args = ap.parse_args()

    # ── Distributed init ────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    out_dir = Path(args.out) / args.exp / args.mode

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  TEMPO v3  |  exp={args.exp}  mode={args.mode}")
        print(f"  world_size={world_size}  rank={rank}")
        print(f"  output → {out_dir}")
        print(f"{'='*60}\n")

    _run_probe(rank, world_size, args.mode, args.exp, out_dir)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
