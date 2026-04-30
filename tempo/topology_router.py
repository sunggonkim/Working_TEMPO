"""
tempo/topology_router.py — Dragonfly Topology-Aware KV Cache Placement Router
==============================================================================

Perlmutter (NERSC) topology overview
--------------------------------------
  Network   : HPE Slingshot-11, 200 Gbps per port
  Topology  : Dragonfly+ (Dragonfly with intra-group all-to-all)
  Nodes     : ~3456 total (1536 GPU nodes)
  Group size: 64 nodes per local group (copper/copper-optical backplane)
  Global links: limited optical inter-group cables — shared resource

Key insight (OSDI motivation)
-------------------------------
Existing disaggregated LLM systems (LMCache, Mooncake, DistServe) treat the
network as an infinite-bandwidth pipe and place KV caches on whichever node
has free memory.  On a Dragonfly fabric at Perlmutter scale this is fatally
wrong: a single rank doing 64 GB/s of KV I/O can saturate ALL global links
into/out of its Dragonfly group, causing a >40% AllReduce BW collapse on
every other rank in the job — even those with no I/O activity of their own.

This module implements topology-aware placement that:
  1. Prefers intra-group peer GPU memory  (zero global-link consumption)
  2. Falls back to Lustre with transfer slicing timed to NCCL-free windows
  3. Defers cross-group transfer entirely when global links are saturated

Mathematical model
-------------------
For a Dragonfly group with g global links each of capacity B_link:
  - NCCL ring-AllReduce consumes ~ 2(G-1)/G × R_model of global BW
  - KV I/O at rate R_io adds congestion term: R_io / (g × B_link)
  - Conflict probability: P_conflict = 1 − (1 − R_io/(g·B−R_nccl))^n
where n = active I/O ranks, G = total groups, g·B = total global BW capacity.

Typical Perlmutter numbers (per group):
  g·B  ≈ 8 global links × 200 Gbps = 1.6 Tbps max
  But shared across all groups → effective quota per active job is ~80 Gbps
  A single rank doing 64 GB/s KV I/O uses 80% of that quota → P_conflict ≈ 1
"""

from __future__ import annotations

import os
import re
import socket
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Perlmutter Dragonfly+ local group size (nodes per group)
_DRAGONFLY_GROUP_SIZE: int = 64

# Maximum fraction of estimated global link BW to use for KV transfers
# 20% leaves 80% for NCCL AllReduce — keeps P_conflict < 5%
_GLOBAL_LINK_QUOTA: float = 0.20

# Slingshot-11 physical link speed
_LINK_BPS: int = 200 * 10**9  # 200 Gbps

# Target per-window slice size for global transfers (bytes)
# Window ≈ 8 ms, quota BW = 0.20 × 25 GB/s → 40 MB per window
_MAX_SLICE_BYTES: int = 40 * 1024 * 1024


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class PlacementTier(Enum):
    LOCAL_PEER    = auto()   # Same Dragonfly group, peer GPU/CPU memory
    LUSTRE_LOCAL  = auto()   # Lustre path with local-group routing
    LUSTRE_REMOTE = auto()   # Lustre path, may cross global links
    DEFERRED      = auto()   # Hold; retry after NCCL window closes


@dataclass
class PlacementDecision:
    tier: PlacementTier
    target_rank: Optional[int]           # for LOCAL_PEER, else None
    slice_size_bytes: int = _MAX_SLICE_BYTES
    estimated_latency_ms: float = 0.0
    crosses_global_link: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# Hostname parsing helpers
# ---------------------------------------------------------------------------

def _parse_nid(hostname: str) -> Optional[int]:
    """Parse Perlmutter node-ID from hostname ('nid001234' → 1234)."""
    m = re.match(r"nid0*(\d+)", hostname)
    return int(m.group(1)) if m else None


def _nid_to_group(nid: int, group_size: int = _DRAGONFLY_GROUP_SIZE) -> int:
    """Map a Perlmutter node-ID to its Dragonfly local-group index."""
    return nid // group_size


# ---------------------------------------------------------------------------
# TopologyRouter
# ---------------------------------------------------------------------------

class TopologyRouter:
    """
    Routes KV-cache placement to minimise Dragonfly global-link consumption.

    Usage
    -----
    In a distributed setup, each rank creates a TopologyRouter, then calls
    ``register_peer_groups()`` after an all-gather of group IDs so every rank
    knows which of its peers share the same local group.

    Outside distributed contexts (unit tests, simulation) the router falls
    back to the simulated group topology based on rank arithmetic.

    Example
    -------
    ::

        router = TopologyRouter(world_size=8, rank=dist.get_rank())
        # After all-gather:
        router.register_peer_groups({r: group_ids[r] for r in range(8)})

        decision = router.route_kv_placement(kv_size_bytes=2 * 2**30)
        if decision.tier == PlacementTier.DEFERRED:
            time.sleep(0.008)   # wait one NCCL window
            decision = router.route_kv_placement(kv_size_bytes=...)
    """

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        group_size: int = _DRAGONFLY_GROUP_SIZE,
        global_link_quota: float = _GLOBAL_LINK_QUOTA,
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.group_size = group_size
        self.global_link_quota = global_link_quota

        self._hostname: str = socket.gethostname()
        self._my_nid: Optional[int] = _parse_nid(self._hostname)
        self._my_group: int = (
            _nid_to_group(self._my_nid, group_size)
            if self._my_nid is not None
            else rank // group_size   # fallback for non-Perlmutter hosts
        )

        # Populated by register_peer_groups()
        self._rank_to_group: Dict[int, int] = {rank: self._my_group}
        self._local_peers: List[int] = []

        # Runtime state
        self._global_link_saturated: bool = False
        self._total_decisions: int = 0
        self._local_decisions: int = 0
        self._deferred_decisions: int = 0

        log.info(
            "TopologyRouter: rank=%d host=%s nid=%s group=%d "
            "world_size=%d group_size=%d",
            rank, self._hostname, self._my_nid, self._my_group,
            world_size, group_size,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def register_peer_groups(self, rank_to_group: Dict[int, int]) -> None:
        """
        Register the mapping of all ranks to their Dragonfly group IDs.
        Must be called after an all-gather so the router knows which peers
        are local (intra-group) vs remote (cross-group).
        """
        self._rank_to_group.update(rank_to_group)
        self._local_peers = [
            r for r, g in rank_to_group.items()
            if g == self._my_group and r != self.rank
        ]
        log.info(
            "TopologyRouter: registered %d peers; %d in local group %d",
            len(rank_to_group), len(self._local_peers), self._my_group,
        )

    # ------------------------------------------------------------------
    # Placement API
    # ------------------------------------------------------------------

    def route_kv_placement(
        self,
        kv_size_bytes: int,
        available_peers: Optional[List[int]] = None,
        nccl_window_ms_remaining: Optional[float] = None,
    ) -> PlacementDecision:
        """
        Decide where to place a KV-cache chunk.

        Placement priority
        ------------------
        1. LOCAL_PEER    — same-group peer GPU memory: zero global-link use
        2. LUSTRE_LOCAL  — Lustre write with traffic staying intra-group
        3. LUSTRE_REMOTE — Lustre write crossing global links (sliced)
        4. DEFERRED      — global links saturated AND NCCL is active

        Args
        ----
        kv_size_bytes:
            Size of the KV block to transfer.
        available_peers:
            List of candidate peer ranks with free GPU memory.
            If None, uses all known local peers.
        nccl_window_ms_remaining:
            Milliseconds until the next NCCL phase starts (from
            InterleavingEngine.get_safe_window_ms()).  If close to zero and
            global link is saturated, defer the transfer.
        """
        self._total_decisions += 1
        peers = available_peers if available_peers is not None else self._local_peers

        # --- Tier 1: local peer memory (intra-group) --------------------
        local = [p for p in peers if self._is_same_group(p)]
        if local:
            target = min(local)
            self._local_decisions += 1
            return PlacementDecision(
                tier=PlacementTier.LOCAL_PEER,
                target_rank=target,
                slice_size_bytes=min(kv_size_bytes, 128 * 1024 * 1024),
                estimated_latency_ms=self._local_latency_ms(kv_size_bytes),
                crosses_global_link=False,
                reason="intra-group peer memory",
            )

        # --- Tier 2 / 3: Lustre — check if we must defer ----------------
        must_defer = (
            self._global_link_saturated
            and nccl_window_ms_remaining is not None
            and nccl_window_ms_remaining < 4.0   # < 4 ms remaining → defer
        )
        if must_defer:
            self._deferred_decisions += 1
            return PlacementDecision(
                tier=PlacementTier.DEFERRED,
                target_rank=None,
                estimated_latency_ms=(nccl_window_ms_remaining or 0) + 8.0,
                crosses_global_link=True,
                reason="global link saturated, NCCL window closing",
            )

        # --- Sliced Lustre transfer (quota-aware) -----------------------
        slice_bytes = self._compute_slice(kv_size_bytes)
        lat_ms = self._lustre_latency_ms(kv_size_bytes)
        return PlacementDecision(
            tier=PlacementTier.LUSTRE_REMOTE,
            target_rank=None,
            slice_size_bytes=slice_bytes,
            estimated_latency_ms=lat_ms,
            crosses_global_link=True,
            reason="no local peers; sliced Lustre transfer",
        )

    # ------------------------------------------------------------------
    # State setters
    # ------------------------------------------------------------------

    def set_global_link_saturated(self, saturated: bool) -> None:
        """Called by NetworkMonitor when global link utilisation > quota."""
        self._global_link_saturated = saturated

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def my_group(self) -> int:
        return self._my_group

    def get_local_peers(self) -> List[int]:
        return list(self._local_peers)

    def is_cross_group(self, rank: int) -> bool:
        return self._rank_to_group.get(rank, -1) != self._my_group

    def get_stats(self) -> dict:
        total = max(1, self._total_decisions)
        return {
            "my_group":            self._my_group,
            "my_nid":              self._my_nid,
            "local_peer_count":    len(self._local_peers),
            "global_link_sat":     self._global_link_saturated,
            "total_decisions":     self._total_decisions,
            "local_pct":           100 * self._local_decisions / total,
            "deferred_pct":        100 * self._deferred_decisions / total,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_same_group(self, rank: int) -> bool:
        return self._rank_to_group.get(rank, -1) == self._my_group

    def _local_latency_ms(self, size_bytes: int) -> float:
        """Intra-group transfer: 200 Gbps = 25 GB/s, ~1 µs base."""
        bw = 25.0 * 1024**3 / 1000  # bytes per ms
        return 0.001 + size_bytes / bw

    def _lustre_latency_ms(self, size_bytes: int) -> float:
        """Lustre (PSCRATCH) sequential write: ~10 GB/s aggregate."""
        bw = 10.0 * 1024**3 / 1000  # bytes per ms
        return 0.5 + size_bytes / bw

    def _compute_slice(self, total_bytes: int) -> int:
        """
        Compute per-NCCL-window transfer slice size.
        Budget = quota_fraction × link_BW × window_duration
        At 20% quota, 25 GB/s, 8 ms window → 40 MB/window.
        """
        bw_bytes_per_ms = _LINK_BPS * self.global_link_quota / 8 / 1000
        window_ms = 8.0
        max_per_window = int(bw_bytes_per_ms * window_ms)
        return min(max_per_window, total_bytes, 128 * 1024 * 1024)
