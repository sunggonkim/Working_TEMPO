"""
tempo/nvlink_router.py — NVLink-Mediated PCIe Multipath Router
==============================================================

**Problem:** Each GPU on a Perlmutter node has a dedicated PCIe Gen4 x16 lane
to one Slingshot-11 NIC (GPU0→hsn0, GPU1→hsn1, GPU2→hsn2, GPU3→hsn3).
A single rank doing a 64 GB checkpoint flush saturates its PCIe lane (32 GB/s
unidirectional) and — through the shared AMD EPYC PCIe I/O die — causes
AllReduce latency to spike +50% on the other ranks too.

**Solution:** Exploit the NVLink fabric that connects the four A100 GPUs on
each Perlmutter node.  When GPU_i's PCIe is saturated, move data via NVLink
to GPU_j (whose PCIe is idle), then flush through GPU_j's NIC.

**Topology (Perlmutter node):**

    GPU0 ──NVLink─── GPU1 ──NVLink─── GPU2 ──NVLink─── GPU3
     │                │                │                │
    PCIe             PCIe             PCIe             PCIe
     │ (32 GB/s)      │                │                │
    hsn0             hsn1             hsn2             hsn3
     └───────────────────────────────────────────────────┘
                  Slingshot-11 fabric (200 Gbps)

**Decision logic (deterministic, no ML):**

    1. Poll per-NIC ``tx_bytes`` via sysfs every 5 ms (EMA α=0.3).
    2. Compute utilisation fraction = ema_tx / (200 Gbps / 8).
    3. If primary NIC util > ``saturation_threshold`` (default 0.80):
         a. Select the NIC with the lowest util as egress.
         b. Move checkpoint tensor: ``torch.cuda.device(egress_gpu)`` +
            ``tensor.to(egress_device, non_blocking=True)``.
         c. Hand off to GpuDrivenEndpoint on egress NIC.
    4. If ALL NICs > threshold → fall back to Lustre (TopologyRouter).

**Why this works without hardware changes:**
    NVLink P2P copy is 600 GB/s aggregate (NVLink 3.0) — 19× faster than a
    single Slingshot link.  Copying 128 MB across NVLink takes < 0.5 ms,
    while the alternative (waiting for PCIe congestion to clear) costs ~10 ms
    stall inside the NCCL AllReduce window.  Net effect: flush completes
    through an idle NIC **before** the AllReduce even notices.

**Integration:**

    router = NVLinkRouter(n_gpus=4, saturation_threshold=0.80)
    router.start()

    # On each checkpoint chunk:
    egress_gpu = router.select_egress_gpu(primary_gpu=dist.get_local_rank())
    if egress_gpu != primary_gpu:
        chunk = chunk.to(f"cuda:{egress_gpu}", non_blocking=True)
        # pipeline flush through hsn{egress_gpu} NIC

    router.stop()

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Slingshot-11 link capacity in bytes/second
_LINK_BPS: float = 200e9 / 8.0         # 25 GB/s per NIC

# NVLink 3.0 aggregate bandwidth per direction (A100 ×4 node)
# 12 links × 50 GB/s = 600 GB/s; per-pair logical ~150 GB/s
_NVLINK_PAIR_BPS: float = 150e9

# AMD EPYC PCIe ceiling: 4 × Gen4 x16 = 4 × 32 GB/s unidirectional
_PCIE_UNIDIRECTIONAL_BPS: float = 32e9

# EMA decay factor for per-NIC utilisation smoothing
_EMA_ALPHA: float = 0.3

# sysfs paths for Slingshot NIC tx bytes
_SYSFS_TX_TMPL = "/sys/class/net/{nic}/statistics/tx_bytes"
_SYSFS_RX_TMPL = "/sys/class/net/{nic}/statistics/rx_bytes"


# ---------------------------------------------------------------------------
# Per-NIC utilisation snapshot
# ---------------------------------------------------------------------------

@dataclass
class NICStats:
    name: str
    gpu_idx: int
    ema_tx_bps: float = 0.0
    ema_rx_bps: float = 0.0
    _prev_tx: int = field(default=0, init=False, repr=False)
    _prev_rx: int = field(default=0, init=False, repr=False)
    _last_poll_ts: float = field(default=0.0, init=False, repr=False)

    @property
    def util_fraction(self) -> float:
        """Fraction of Slingshot link capacity used (TX direction)."""
        return min(1.0, self.ema_tx_bps / _LINK_BPS)

    @property
    def is_idle(self) -> bool:
        return self.ema_tx_bps < _LINK_BPS * 0.10


# ---------------------------------------------------------------------------
# NVLinkRouter
# ---------------------------------------------------------------------------

class NVLinkRouter:
    """
    Routes checkpoint flush traffic through the least-loaded Slingshot NIC,
    using NVLink to move tensor data between GPUs as needed.

    Parameters
    ----------
    n_gpus : int
        Number of GPUs per node (default 4).  Each GPU i maps to NIC hsn{i}.
    nic_names : list of str or None
        Override NIC names; defaults to ["hsn0", "hsn1", "hsn2", "hsn3"].
    saturation_threshold : float
        NIC utilisation fraction above which a NIC is considered saturated
        and traffic should be rerouted.  Default 0.80 (80% of 25 GB/s).
    poll_interval_s : float
        How often to update sysfs utilisation readings.  Default 5 ms.
    torch_available : bool or None
        If None, auto-detect PyTorch.  Set False to suppress import.

    Attributes
    ----------
    nic_stats : dict[str, NICStats]
        Live utilisation snapshot for each NIC.

    Usage
    -----
    ::

        router = NVLinkRouter(n_gpus=4)
        router.start()

        # Inside flush loop (per 128 MB chunk):
        primary = local_rank        # e.g. 0
        egress  = router.select_egress_gpu(primary)
        relay   = router.need_nvlink_relay(primary, egress)

        if relay:
            chunk = router.nvlink_transfer(chunk, src_gpu=primary, dst_gpu=egress)
            # chunk is now on cuda:{egress}

        # proceed with GpuDrivenEndpoint on hsn{egress}

        # After checkpoint:
        router.record_flush_bytes(egress, n_bytes)
        router.stop()
    """

    def __init__(
        self,
        n_gpus:               int              = 4,
        nic_names:            Optional[List[str]] = None,
        saturation_threshold: float            = 0.80,
        poll_interval_s:      float            = 0.005,
        torch_available:      Optional[bool]   = None,
    ) -> None:
        self.n_gpus               = n_gpus
        self.saturation_threshold = saturation_threshold
        self.poll_interval_s      = poll_interval_s

        if nic_names is None:
            nic_names = [f"hsn{i}" for i in range(n_gpus)]
        self.nic_names = nic_names

        # Build NICStats map
        self.nic_stats: Dict[str, NICStats] = {
            name: NICStats(name=name, gpu_idx=i)
            for i, name in enumerate(nic_names)
        }

        # GPU → NIC mapping
        self._gpu_to_nic: Dict[int, str] = {
            i: name for i, name in enumerate(nic_names)
        }
        self._nic_to_gpu: Dict[str, int] = {
            name: i for i, name in enumerate(nic_names)
        }

        # Torch
        if torch_available is None:
            try:
                import torch
                torch_available = True
            except ImportError:
                torch_available = False
        self._torch_ok = torch_available

        # Background polling thread
        self._stop_evt = threading.Event()
        self._thread:   Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Stats
        self._relays_done   = 0
        self._bytes_relayed = 0
        self._reroutes      = 0

        log.info(
            "[NVLinkRouter] %d GPUs  NICs=%s  sat_thresh=%.0f%%  poll=%dms  torch=%s",
            n_gpus, nic_names,
            saturation_threshold * 100,
            int(poll_interval_s * 1000),
            torch_available,
        )

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def start(self) -> "NVLinkRouter":
        """Start background polling thread."""
        # Seed initial readings
        for stat in self.nic_stats.values():
            self._read_nic(stat)
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="nvlink-router-poll",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop background polling thread."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ----------------------------------------------------------------
    # Routing decisions
    # ----------------------------------------------------------------

    def select_egress_gpu(self, primary_gpu: int) -> int:
        """
        Return the GPU index whose NIC should be used for the next flush chunk.

        Algorithm:
          1. If primary NIC util ≤ threshold → use primary (no relay needed).
          2. Else find the NIC with the lowest utilisation.
          3. If all NICs saturated → return primary anyway (Lustre fallback
             will handle actual throttling via TopologyRouter).

        Parameters
        ----------
        primary_gpu : int
            The GPU index that owns the data to be sent.

        Returns
        -------
        int
            GPU index whose NIC to use.  May differ from ``primary_gpu``
            if rerouting is needed.
        """
        primary_nic = self._gpu_to_nic.get(primary_gpu)
        if primary_nic is None:
            return primary_gpu

        with self._lock:
            primary_util = self.nic_stats[primary_nic].util_fraction

        if primary_util <= self.saturation_threshold:
            return primary_gpu   # no reroute needed

        # Find least-utilised NIC
        with self._lock:
            ranked = sorted(
                self.nic_stats.values(),
                key=lambda s: s.ema_tx_bps,
            )

        best = ranked[0]
        if best.util_fraction < primary_util - 0.05:  # meaningful improvement
            egress = best.gpu_idx
            with self._lock:
                self._reroutes += 1
            log.debug(
                "[NVLinkRouter] reroute GPU%d→GPU%d  "
                "primary_util=%.0f%%  egress_util=%.0f%%",
                primary_gpu, egress,
                primary_util * 100, best.util_fraction * 100,
            )
            return egress

        return primary_gpu   # all saturated; return original

    def need_nvlink_relay(self, src_gpu: int, egress_gpu: int) -> bool:
        """Return True if src_gpu and egress_gpu differ (NVLink relay needed)."""
        return src_gpu != egress_gpu

    def nvlink_transfer(
        self,
        tensor: object,
        src_gpu: int,
        dst_gpu: int,
        non_blocking: bool = True,
    ) -> object:
        """
        Copy ``tensor`` from ``cuda:{src_gpu}`` to ``cuda:{dst_gpu}`` via
        NVLink using PyTorch.

        NVLink 3.0 on Perlmutter A100 nodes: 600 GB/s aggregate bandwidth.
        A 128 MiB chunk transfers in ≈ 0.85 ms — far less than the
        PCIe saturation penalty (~10 ms stall) it avoids.

        Parameters
        ----------
        tensor : torch.Tensor
            Source tensor (must already be on ``cuda:{src_gpu}``).
        src_gpu / dst_gpu : int
            CUDA device indices.
        non_blocking : bool
            If True, copy is queued on the default stream of ``dst_gpu``
            and returns immediately (caller must sync before NIC trigger).

        Returns
        -------
        torch.Tensor
            A new tensor resident on ``cuda:{dst_gpu}``.
        """
        if not self._torch_ok:
            log.debug("[NVLinkRouter] torch unavailable — returning identity")
            return tensor

        import torch

        dst_device = torch.device(f"cuda:{dst_gpu}")
        t_start = time.perf_counter()

        # Enable P2P access between the two devices if not already enabled.
        with torch.cuda.device(src_gpu):
            if not torch.cuda.can_device_access_peer(src_gpu, dst_gpu):
                try:
                    torch.cuda.set_device(dst_gpu)
                    torch.cuda.enable_peer_access(src_gpu, flags=0)
                    torch.cuda.set_device(src_gpu)
                    torch.cuda.enable_peer_access(dst_gpu, flags=0)
                except RuntimeError:
                    pass  # Already enabled or not supported

        dst_tensor = tensor.to(dst_device, non_blocking=non_blocking)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        n_bytes = tensor.numel() * tensor.element_size()

        with self._lock:
            self._relays_done   += 1
            self._bytes_relayed += n_bytes

        log.debug(
            "[NVLinkRouter] NVLink relay  GPU%d→GPU%d  %.1f MiB  %.2f ms",
            src_gpu, dst_gpu, n_bytes / 1024**2, elapsed_ms,
        )
        return dst_tensor

    def record_flush_bytes(self, gpu_idx: int, n_bytes: int) -> None:
        """
        Account for bytes flushed through a given GPU's NIC.
        Used to opportunistically adjust EMA without waiting for sysfs poll.
        """
        nic = self._gpu_to_nic.get(gpu_idx)
        if nic is None:
            return
        with self._lock:
            stat = self.nic_stats[nic]
            # Instant injection: assume transferred at link speed over poll window
            estimated_bps = n_bytes / max(self.poll_interval_s, 1e-6)
            stat.ema_tx_bps = (
                _EMA_ALPHA * estimated_bps
                + (1 - _EMA_ALPHA) * stat.ema_tx_bps
            )

    # ----------------------------------------------------------------
    # Live utilisation queries
    # ----------------------------------------------------------------

    def get_nic_util(self, gpu_idx: int) -> float:
        """Return current utilisation fraction [0,1] for the NIC of ``gpu_idx``."""
        nic = self._gpu_to_nic.get(gpu_idx)
        if nic is None:
            return 0.0
        with self._lock:
            return self.nic_stats[nic].util_fraction

    def get_all_utils(self) -> Dict[int, float]:
        """Return {gpu_idx: util_fraction} for all NICs."""
        with self._lock:
            return {
                stat.gpu_idx: stat.util_fraction
                for stat in self.nic_stats.values()
            }

    def count_saturated_nics(self) -> int:
        """Return number of NICs with util > saturation_threshold."""
        with self._lock:
            return sum(
                1 for s in self.nic_stats.values()
                if s.util_fraction > self.saturation_threshold
            )

    def all_saturated(self) -> bool:
        """True when every NIC exceeds the saturation threshold."""
        return self.count_saturated_nics() >= self.n_gpus

    def estimate_relay_overhead_ms(
        self,
        n_bytes: int,
        src_gpu: int,
        dst_gpu: int,
    ) -> float:
        """
        O(1) estimate: how long does an NVLink relay take?

        Returns time in milliseconds.  Intended for the PCIePressurePredictor
        look-ahead to decide whether a relay is worth it.
        """
        if src_gpu == dst_gpu:
            return 0.0
        return (n_bytes / _NVLINK_PAIR_BPS) * 1000.0

    def estimate_reroute_gain_ms(
        self,
        n_bytes: int,
        primary_gpu: int,
    ) -> float:
        """
        Estimate time *saved* (in ms) by routing through an idle NIC
        instead of waiting behind a saturated primary NIC.

        Positive values mean rerouting wins.
        negative values mean it is not worth the NVLink overhead.
        """
        primary_util = self.get_nic_util(primary_gpu)
        if primary_util <= self.saturation_threshold:
            return 0.0   # No saturation — no benefit

        # Time to flush n_bytes through saturated NIC
        available_bps = _LINK_BPS * max(0.01, 1.0 - primary_util)
        wait_ms = (n_bytes / available_bps) * 1000.0

        # Best egress util
        with self._lock:
            best_util = min(s.util_fraction for s in self.nic_stats.values())
        egress_avail = _LINK_BPS * max(0.01, 1.0 - best_util)
        flush_ms = (n_bytes / egress_avail) * 1000.0

        egress_gpu = self.select_egress_gpu(primary_gpu)
        relay_overhead = self.estimate_relay_overhead_ms(n_bytes, primary_gpu, egress_gpu)

        return wait_ms - (flush_ms + relay_overhead)

    # ----------------------------------------------------------------
    # Background polling
    # ----------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop_evt.is_set():
            time.sleep(self.poll_interval_s)
            for stat in self.nic_stats.values():
                self._read_nic(stat)

    def _read_nic(self, stat: NICStats) -> None:
        """Read sysfs tx_bytes and update EMA."""
        tx_path = _SYSFS_TX_TMPL.format(nic=stat.name)
        try:
            with open(tx_path) as f:
                tx_bytes = int(f.read().strip())
        except (OSError, ValueError):
            return

        now = time.monotonic()
        if stat._last_poll_ts > 0:
            dt = max(1e-6, now - stat._last_poll_ts)
            delta_bytes = max(0, tx_bytes - stat._prev_tx)
            inst_bps = delta_bytes / dt
            with self._lock:
                stat.ema_tx_bps = (
                    _EMA_ALPHA * inst_bps
                    + (1 - _EMA_ALPHA) * stat.ema_tx_bps
                )
        stat._prev_tx      = tx_bytes
        stat._last_poll_ts = now

    # ----------------------------------------------------------------
    # Stats
    # ----------------------------------------------------------------

    def get_stats(self) -> dict:
        with self._lock:
            nic_utils = {
                s.name: {
                    "gpu_idx":      s.gpu_idx,
                    "ema_tx_gbps":  s.ema_tx_bps / 1e9,
                    "util_pct":     s.util_fraction * 100,
                    "is_idle":      s.is_idle,
                }
                for s in self.nic_stats.values()
            }
        return {
            "relays_done":       self._relays_done,
            "bytes_relayed":     self._bytes_relayed,
            "reroutes":          self._reroutes,
            "saturated_nics":    self.count_saturated_nics(),
            "all_saturated":     self.all_saturated(),
            "nic_utils":         nic_utils,
        }

    def print_stats(self) -> None:
        s = self.get_stats()
        print(f"\n{'='*60}")
        print(f"  NVLinkRouter Statistics")
        print(f"  Relays done     : {s['relays_done']}")
        print(f"  Bytes relayed   : {s['bytes_relayed']/1024**3:.2f} GB")
        print(f"  Reroutes        : {s['reroutes']}")
        print(f"  Saturated NICs  : {s['saturated_nics']} / {self.n_gpus}")
        for nic, u in s["nic_utils"].items():
            bar = "█" * int(u["util_pct"] / 5)
            print(f"  {nic} (GPU{u['gpu_idx']})  {u['ema_tx_gbps']:.1f} GB/s  "
                  f"[{bar:<20}] {u['util_pct']:.0f}%")
        print(f"{'='*60}\n")
