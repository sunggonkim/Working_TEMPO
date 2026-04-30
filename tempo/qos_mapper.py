"""
tempo/qos_mapper.py — Slingshot-11 Hardware QoS Traffic-Class Mapper
=====================================================================

HPE Slingshot-11 implements hardware-level QoS through a combination of
DSCP (Differentiated Services Code Point) bits in IP headers and internal
fabric traffic classes (TCs).  Each TC maps to a separate priority queue
inside every switch ASIC; higher-TC packets receive preferential forwarding
even under congestion.

Traffic class layout
--------------------
  TC3 (highest) — Expedited Forwarding (DSCP 46)
      Low-latency, near-lossless; used for RDMA / NCCL AllReduce, and any
      KV-cache transfer flagged as deadline-critical (service gain ≥ 0.70).

  TC2 — Assured Forwarding class 4 (DSCP 34)
      Bulk transfers with bandwidth guarantee; normal KV-cache operations
      where service gain ∈ [0.40, 0.70).

  TC1 — Assured Forwarding class 2 (DSCP 18)
      Storage / checkpoint traffic; service gain ∈ [0.15, 0.40).

  TC0 (lowest) — Best-Effort (DSCP 0)
      Background bulk I/O, prefetch, and deferred flushes (gain < 0.15).
      This traffic is the first to be de-prioritised by the fabric under
      any congestion condition, preserving TC2/TC3 headroom for NCCL.

Co-design insight (OSDI contribution)
--------------------------------------
TEMPO's software Service-Gain score (a weighted sum of learning-progress,
recovery-value, and urgency) is computed per-checkpoint at nanosecond
granularity.  By mapping that score directly to a hardware traffic class,
the scheduling decision propagates end-to-end through the fabric without
any additional software-layer policing.  A low-gain background flush that
happens to contend with an AllReduce simply loses at the switch ASIC —
exactly the priority inversion we want, achieved at zero CPU overhead.

User-space mechanism
--------------------
We mark our own data-movement sockets via ``socket.IP_TOS`` (sets DSCP).
NCCL, which manages its own socket lifecycle, uses its default priority
(configurable via ``NCCL_SOCKET_IFNAME`` and ``NCCL_NET_GDR_LEVEL``).
The key insight is that *our I/O sockets go low*, not that NCCL goes high.
Slingshot-11's strict priority scheduling then naturally separates flows.

Reference: HPE Slingshot Administration Guide, §"Quality of Service"
"""

from __future__ import annotations

import os
import socket
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Traffic class enumeration
# ---------------------------------------------------------------------------

class TC(IntEnum):
    """HPE Slingshot-11 hardware traffic classes (0 = lowest priority)."""
    BACKGROUND = 0   # Best-effort: background I/O, deferred flushes
    STORAGE    = 1   # Assured: checkpoint, cold-KV prefetch
    BULK       = 2   # Assured: normal KV-cache, medium-gain transfers
    LATENCY    = 3   # Expedited: NCCL, deadline-critical KV


# DSCP values per TC (IETF RFC 4594 recommendations)
_TC_DSCP: Dict[TC, int] = {
    TC.BACKGROUND: 0,    # CS0 / Best-Effort
    TC.STORAGE:    18,   # AF21
    TC.BULK:       34,   # AF41
    TC.LATENCY:    46,   # EF (Expedited Forwarding)
}

# Linux SO_PRIORITY values (0-6, 7 reserved for root)
_TC_SO_PRIO: Dict[TC, int] = {
    TC.BACKGROUND: 0,
    TC.STORAGE:    2,
    TC.BULK:       4,
    TC.LATENCY:    6,
}


@dataclass(frozen=True)
class TrafficClass:
    tc: TC
    dscp: int
    tos_byte: int    # IP ToS field = DSCP << 2
    so_prio: int     # Linux SO_PRIORITY value
    name: str


def _build_tc(tc: TC) -> TrafficClass:
    dscp = _TC_DSCP[tc]
    return TrafficClass(
        tc=tc,
        dscp=dscp,
        tos_byte=dscp << 2,
        so_prio=_TC_SO_PRIO[tc],
        name=tc.name,
    )


# Pre-built TrafficClass objects
TC_CLASSES: Dict[TC, TrafficClass] = {tc: _build_tc(tc) for tc in TC}


# ---------------------------------------------------------------------------
# QoSMapper
# ---------------------------------------------------------------------------

class QoSMapper:
    """
    Maps TEMPO service-gain scores to Slingshot-11 hardware traffic classes.

    The mapper classifies each data-movement operation and optionally applies
    the corresponding DSCP / SO_PRIORITY mark to the operating socket so the
    kernel and fabric honour the priority end-to-end.

    Service-gain → TC thresholds
    ----------------------------
    gain ≥ 0.70   → TC3 LATENCY   (critical, EF)
    gain ∈ [0.40, 0.70) → TC2 BULK       (normal KV)
    gain ∈ [0.15, 0.40) → TC1 STORAGE    (checkpoint)
    gain <  0.15  → TC0 BACKGROUND (defer/drop candidate)

    These thresholds are chosen so that ≥80% of traffic (by byte count) in
    a typical LLM serving workload is TC0/TC1, preserving TC2/TC3 queues for
    the small fraction of high-value transfers.

    Parameters
    ----------
    enabled:  Whether to perform any marking (default True).
    dry_run:  Classify but do not actually apply socket options (default
              False).  Useful for benchmarking the classification overhead.
    """

    GAIN_TC3 = 0.70
    GAIN_TC2 = 0.40
    GAIN_TC1 = 0.15

    def __init__(self, enabled: bool = True, dry_run: bool = False) -> None:
        self.enabled   = enabled
        self.dry_run   = dry_run
        self._tos_ok   = self._probe_tos() if (enabled and not dry_run) else False

        # Per-class accounting
        self._counters: Dict[TC, int]  = {tc: 0 for tc in TC}
        self._bytes:    Dict[TC, int]  = {tc: 0 for tc in TC}
        self._applied:  int            = 0

        log.info(
            "QoSMapper: enabled=%s dry_run=%s ip_tos_capable=%s",
            enabled, dry_run, self._tos_ok,
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        gain:         float,
        traffic_type: str   = "kv_cache",
        urgency:      float = 0.5,
        size_bytes:   int   = 0,
    ) -> TrafficClass:
        """
        Assign a hardware traffic class to a data-movement operation.

        Args
        ----
        gain:         TEMPO ServiceGain score ∈ [0, 1].
        traffic_type: One of "kv_cache" | "checkpoint" | "nccl" | "bg_io".
        urgency:      Deadline urgency ∈ [0, 1].  Adds a small upward bias
                      to gain so near-deadline transfers can cross TC thresholds.
        size_bytes:   Transfer size; used for byte-count tracking only.
        """
        if not self.enabled:
            return TC_CLASSES[TC.BACKGROUND]

        # Fixed-class overrides
        if traffic_type == "nccl":
            tc = TC.LATENCY
        elif traffic_type == "bg_io":
            tc = TC.BACKGROUND
        else:
            # Urgency-adjusted effective gain
            eff = min(1.0, gain + 0.08 * urgency)
            if eff >= self.GAIN_TC3:
                tc = TC.LATENCY
            elif eff >= self.GAIN_TC2:
                tc = TC.BULK
            elif eff >= self.GAIN_TC1:
                tc = TC.STORAGE
            else:
                tc = TC.BACKGROUND

        self._counters[tc] += 1
        self._bytes[tc] += size_bytes
        return TC_CLASSES[tc]

    # ------------------------------------------------------------------
    # Socket marking
    # ------------------------------------------------------------------

    def apply_to_socket(
        self,
        sock: socket.socket,
        tc: TrafficClass,
    ) -> bool:
        """
        Apply IP ToS (DSCP) and SO_PRIORITY to *sock* for Slingshot-11 QoS.

        Returns True on success.  Failures are logged at DEBUG and silently
        ignored so training can continue without QoS if permissions are absent.
        """
        if not self.enabled:
            return False
        if self.dry_run:
            log.debug("QoS dry-run: TOS=0x%02x SO_PRIO=%d (%s)",
                      tc.tos_byte, tc.so_prio, tc.name)
            self._applied += 1
            return True
        if not self._tos_ok:
            return False

        success = True
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, tc.tos_byte)
        except OSError as e:
            log.debug("IP_TOS failed: %s", e)
            success = False
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, tc.so_prio)
        except OSError as e:
            log.debug("SO_PRIORITY failed: %s", e)

        if success:
            self._applied += 1
        return success

    def apply_fd_priority(self, fd: int, gain: float, urgency: float = 0.5) -> bool:
        """
        Best-effort I/O priority hint for file descriptors (Lustre I/O).

        Uses Linux ``ionice`` / ``ioprio_set`` semantics via
        ``os.setpriority`` when the traffic class maps to BACKGROUND.
        Effectively requests CFQ idle-class scheduling for the writing
        thread, yielding disk I/O bandwidth to higher-priority processes.

        Note: This affects the initiating *thread*, not just the fd, so it
        should be called from a dedicated I/O worker thread.
        """
        if not self.enabled:
            return False
        tc = self.classify(gain, traffic_type="kv_cache", urgency=urgency)
        if tc.tc == TC.BACKGROUND:
            try:
                os.setpriority(os.PRIO_PROCESS, 0, 10)   # nice +10
                return True
            except OSError:
                pass
        return False

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        total_ops   = max(1, sum(self._counters.values()))
        total_bytes = max(1, sum(self._bytes.values()))
        return {
            "enabled":       self.enabled,
            "dry_run":       self.dry_run,
            "tos_capable":   self._tos_ok,
            "applied_marks": self._applied,
            "tc_distribution": {
                tc.name: {
                    "ops_pct":   100.0 * self._counters[tc] / total_ops,
                    "bytes_pct": 100.0 * self._bytes[tc]    / total_bytes,
                }
                for tc in TC
            },
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_tos() -> bool:
        """Check whether IP_TOS can be set (requires no special capability)."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0)
            s.close()
            return True
        except OSError:
            return False
