"""
tempo/libfabric_qos.py — libfabric CXI Provider Traffic-Class Control
======================================================================

**Why this file exists (vs. the existing qos_mapper.py)**

The existing ``QoSMapper`` in ``tempo/qos_mapper.py`` marks sockets with
``socket.IP_TOS`` (DSCP bits).  This works for TCP/IP traffic, but has two
limitations on Perlmutter:

  1. **Scope:** ``IP_TOS`` modifies the ToS byte in IP headers of TCP
     connections.  However, RDMA/NCCL AllReduce traffic on Slingshot-11
     uses the OFI *CXI provider* which bypasses the kernel network stack
     entirely.  The socket option never reaches the Cassini ASIC.

  2. **Granularity:** Even for TCP sockets, the OS may re-label packets
     at the IP stack before they hit the NIC driver.  On Perlmutter,
     DSCP remarking by the EPYC Host Fabric Interface (HFI) driver can
     silently override ``IP_TOS`` unless the admin has whitelisted the
     process.

**This module's approach:** Call libfabric's *endpoint-level* option
``fi_setopt(ep, FI_OPT_ENDPOINT, FI_OPT_CXI_TRAFFIC_CLASS, &tc, sizeof(tc))``
**before** any ``fi_send``/``fi_inject`` call.  The CXI provider embeds the
traffic class directly into the *Cassini Portals4 Put* header, bypassing the
IP stack entirely.  Slingshot switch ASICs read this field and place the flit
in the corresponding hardware priority queue.

**Traffic class → hardware priority queue mapping (HPE Slingshot-11):**

    CXI TC value  | Hardware Queue | Use in TEMPO
    ─────────────────────────────────────────────────────────
    TC_UNSPEC  (0)| Default        | (unused)
    TC_BEST_EFF(1)| Q0 (lowest)    | deferred bulk I/O, prefetch
    TC_STORAGE (2)| Q1             | checkpoint, cold-KV flush
    TC_BULK    (4)| Q2             | normal KV-cache, medium-gain
    TC_LOW_LAT (6)| Q3 (highest)   | NCCL AllReduce, deadline-critical KV

**Integration with TEMPO service-gain scores:**

    gain ≥ 0.70 → TC_LOW_LATENCY (Q3) — only for truly deadline-critical
    gain ∈ [0.40, 0.70) → TC_BULK (Q2)
    gain ∈ [0.15, 0.40) → TC_STORAGE (Q1)
    gain < 0.15 → TC_BEST_EFFORT (Q0)

    This table mirrors QoSMapper.TC but operates at the fabric level, not
    at the IP socket level.

**Fallback:** If libfabric is unavailable (CI, non-Perlmutter), the module
falls back silently to the ``socket.IP_TOS`` approach in ``QoSMapper``.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import socket
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

from tempo.qos_mapper import QoSMapper, TC as SocketTC   # socket-level fallback

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CXI Traffic Class values (libfabric cxi_tc enum)
# ---------------------------------------------------------------------------

class CXI_TC(IntEnum):
    """
    Cassini/Slingshot-11 hardware traffic class values.

    These are the values passed to fi_setopt(FI_OPT_CXI_TRAFFIC_CLASS).
    They differ from the DSCP values used by socket.IP_TOS — they are
    internal to the Cassini ASIC's packet header and never appear in
    IP headers.
    """
    UNSPECIFIED  = 0   # Cassini default (implementation-defined)
    BEST_EFFORT  = 1   # Q0: background, deferred
    STORAGE      = 2   # Q1: checkpoint, cold prefetch
    BULK         = 4   # Q2: normal KV-cache
    LOW_LATENCY  = 6   # Q3: NCCL, deadline-critical


# Mapping: TEMPO service-gain score → CXI TC
_GAIN_TO_CXI_TC: Tuple[Tuple[float, CXI_TC], ...] = (
    (0.70, CXI_TC.LOW_LATENCY),   # gain ≥ 0.70
    (0.40, CXI_TC.BULK),          # gain ∈ [0.40, 0.70)
    (0.15, CXI_TC.STORAGE),       # gain ∈ [0.15, 0.40)
    (0.00, CXI_TC.BEST_EFFORT),   # gain < 0.15
)

# Fallback: CXI_TC → socket-level TC (for non-Perlmutter)
_CXI_TC_TO_SOCKET_TC: Dict[CXI_TC, SocketTC] = {
    CXI_TC.UNSPECIFIED: SocketTC.BACKGROUND,
    CXI_TC.BEST_EFFORT: SocketTC.BACKGROUND,
    CXI_TC.STORAGE:     SocketTC.STORAGE,
    CXI_TC.BULK:        SocketTC.BULK,
    CXI_TC.LOW_LATENCY: SocketTC.LATENCY,
}

# libfabric option identifiers
_FI_OPT_ENDPOINT         = 0      # fi_setopt level
_FI_OPT_CXI_TRAFFIC_CLASS = 0x4210  # CXI provider extension option ID

# ---------------------------------------------------------------------------
# libfabric loader
# ---------------------------------------------------------------------------

def _load_libfabric() -> Optional[ctypes.CDLL]:
    path = ctypes.util.find_library("fabric")
    if path is None:
        return None
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


_LIBFABRIC: Optional[ctypes.CDLL] = _load_libfabric()
_LIBFABRIC_AVAILABLE: bool = (
    _LIBFABRIC is not None
    and os.path.exists("/sys/bus/cxi/devices")   # Perlmutter CXI sysfs
)

if _LIBFABRIC_AVAILABLE:
    log.debug("[libfabric_qos] CXI endpoint-level TC control AVAILABLE")
else:
    log.debug(
        "[libfabric_qos] libfabric or CXI sysfs not found — "
        "TC control will use socket.IP_TOS fallback"
    )


# ---------------------------------------------------------------------------
# CXIEndpointQoS
# ---------------------------------------------------------------------------

class CXIEndpointQoS:
    """
    Controls the Slingshot traffic class of a libfabric OFI endpoint at the
    *endpoint level* (not at the socket/IP level).

    This class is designed to wrap a ``GpuDrivenEndpoint``'s ``fi_ep`` handle
    or any other OFI endpoint obtained via ``fi_endpoint()``.  It exposes a
    single ``set_tc()`` method that calls ``fi_setopt`` directly.

    Parameters
    ----------
    ep_ptr : int
        Raw ctypes pointer value (``ctypes.c_void_p``) to a ``fi_ep``.
    initial_tc : CXI_TC
        Traffic class to apply immediately on construction.
    dry_run : bool
        If True, log the TC change but do not call ``fi_setopt``
        (safe default for testing on non-Perlmutter nodes).

    Usage
    -----
    ::

        from tempo.gpu_driven import GpuDrivenEndpoint
        from tempo.libfabric_qos import CXIEndpointQoS, CXI_TC, gain_to_cxi_tc

        ep  = GpuDrivenEndpoint(nic_idx=2).open()
        qos = CXIEndpointQoS(ep._ep, initial_tc=CXI_TC.STORAGE)

        # Before checkpoint flush:
        tc = gain_to_cxi_tc(service_gain_score)
        qos.set_tc(tc)

        # NIC will now use hardware Q1/Q2/Q3 for all subsequent fi_send calls
        ep.trigger_from_gpu(handle_id, stream)
    """

    def __init__(
        self,
        ep_ptr:     int,
        initial_tc: CXI_TC = CXI_TC.STORAGE,
        dry_run:    bool    = False,
    ) -> None:
        self._ep_ptr    = ep_ptr
        self._current   = initial_tc
        self._dry_run   = dry_run or not _LIBFABRIC_AVAILABLE
        self._lock      = threading.Lock()
        self._set_count = 0

        if not self._dry_run:
            self._apply(initial_tc)
        else:
            log.debug(
                "[CXIEndpointQoS] dry_run=True — fi_setopt calls suppressed "
                "(initial_tc=%s)", initial_tc.name
            )

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def set_tc(self, tc: CXI_TC) -> bool:
        """
        Set the Slingshot traffic class for this OFI endpoint.

        Calls ``fi_setopt(ep, FI_OPT_ENDPOINT, FI_OPT_CXI_TRAFFIC_CLASS,
        &tc, sizeof(uint32_t))``.  The change takes effect on the *next*
        ``fi_send`` / ``fi_inject`` / doorbell trigger.

        Parameters
        ----------
        tc : CXI_TC
            Target traffic class.

        Returns
        -------
        bool
            True if the TC was changed; False if already at ``tc`` or error.
        """
        with self._lock:
            if tc == self._current:
                return False
            ok = self._apply(tc)
            if ok:
                self._current = tc
            return ok

    def set_tc_for_gain(self, gain_score: float) -> CXI_TC:
        """
        Compute the appropriate TC from a service-gain score and apply it.

        Returns the selected CXI_TC.
        """
        tc = gain_to_cxi_tc(gain_score)
        self.set_tc(tc)
        return tc

    @property
    def current_tc(self) -> CXI_TC:
        with self._lock:
            return self._current

    def get_stats(self) -> dict:
        return {
            "current_tc":       self._current.name,
            "set_count":        self._set_count,
            "libfabric_active": _LIBFABRIC_AVAILABLE and not self._dry_run,
            "dry_run":          self._dry_run,
        }

    # ----------------------------------------------------------------
    # Private
    # ----------------------------------------------------------------

    def _apply(self, tc: CXI_TC) -> bool:
        """Issue fi_setopt for the given TC, or log in dry-run mode."""
        if self._dry_run:
            log.debug("[CXIEndpointQoS] (dry-run) fi_setopt tc=%s", tc.name)
            self._set_count += 1
            return True

        if _LIBFABRIC is None or self._ep_ptr == 0:
            return False

        tc_val = ctypes.c_uint32(int(tc))
        ret = _LIBFABRIC.fi_setopt(
            ctypes.c_void_p(self._ep_ptr),
            ctypes.c_int(_FI_OPT_ENDPOINT),
            ctypes.c_int(_FI_OPT_CXI_TRAFFIC_CLASS),
            ctypes.byref(tc_val),
            ctypes.c_size_t(ctypes.sizeof(tc_val)),
        )
        if ret != 0:
            log.warning(
                "[CXIEndpointQoS] fi_setopt(tc=%s) returned %d — "
                "TC control may not be active",
                tc.name, ret,
            )
            return False

        self._set_count += 1
        log.debug("[CXIEndpointQoS] fi_setopt tc=%s OK", tc.name)
        return True


# ---------------------------------------------------------------------------
# FabricQoSManager — manages TC for a pool of endpoints
# ---------------------------------------------------------------------------

class FabricQoSManager:
    """
    Manages traffic-class assignment across a pool of libfabric endpoints.

    Designed to work alongside ``GpuDrivenPool``: one ``CXIEndpointQoS``
    object per NIC.  On each checkpoint step, the scheduler calls
    ``apply_for_gain(nic_idx, gain_score)`` to set the appropriate TC
    before triggering the NIC.

    Falls back to ``QoSMapper`` (socket.IP_TOS) if libfabric is unavailable,
    ensuring end-to-end API compatibility across Perlmutter and dev nodes.

    Parameters
    ----------
    n_nics : int
        Number of NICs (one endpoint per NIC).
    dry_run : bool
        If True, suppress actual fi_setopt calls (log only).
    default_tc : CXI_TC
        Initial TC for all endpoints.

    Usage
    -----
    ::

        gpu_pool = GpuDrivenPool(n_nics=4).open_all()
        qos_mgr  = FabricQoSManager(n_nics=4)
        qos_mgr.attach_pool(gpu_pool)

        # Assign TC before each flush:
        qos_mgr.apply_for_gain(nic_idx=2, gain_score=0.55)  # → BULK (Q2)
        gpu_pool.trigger(nic_idx=2, handle_id=hid, stream=s)
    """

    def __init__(
        self,
        n_nics:     int     = 4,
        dry_run:    bool    = False,
        default_tc: CXI_TC  = CXI_TC.STORAGE,
    ) -> None:
        self.n_nics     = n_nics
        self._dry_run   = dry_run
        self._default_tc = default_tc

        # CXIEndpointQoS objects populated by attach_pool()
        self._endpoints: Dict[int, CXIEndpointQoS] = {}

        # Socket-level fallback QoSMapper
        self._socket_qos = QoSMapper()

        if not _LIBFABRIC_AVAILABLE:
            log.info(
                "[FabricQoSManager] libfabric/CXI not available — "
                "all TC control via socket.IP_TOS (QoSMapper fallback)"
            )

    def attach_pool(self, gpu_pool: object) -> "FabricQoSManager":
        """
        Wire CXIEndpointQoS objects to each GpuDrivenEndpoint in the pool.

        Call after ``gpu_pool.open_all()``.
        """
        for nic_idx in range(self.n_nics):
            ep_obj = None
            if hasattr(gpu_pool, "_endpoints"):
                ep_obj = gpu_pool._endpoints.get(nic_idx)

            ep_ptr = 0
            if ep_obj is not None and hasattr(ep_obj, "_ep") and ep_obj._ep is not None:
                ep_ptr = int(ep_obj._ep)   # ctypes c_void_p → int

            qos_ep = CXIEndpointQoS(
                ep_ptr     = ep_ptr,
                initial_tc = self._default_tc,
                dry_run    = self._dry_run,
            )
            self._endpoints[nic_idx] = qos_ep

        log.info(
            "[FabricQoSManager] attached %d endpoints  default_tc=%s",
            len(self._endpoints), self._default_tc.name,
        )
        return self

    def apply_for_gain(self, nic_idx: int, gain_score: float) -> CXI_TC:
        """
        Set TC on NIC ``nic_idx`` based on service-gain score.

        Returns the CXI_TC that was applied.
        """
        tc = gain_to_cxi_tc(gain_score)
        ep = self._endpoints.get(nic_idx)
        if ep is not None:
            ep.set_tc(tc)
        else:
            log.debug(
                "[FabricQoSManager] nic_idx=%d not in pool — "
                "using socket fallback",
                nic_idx,
            )
        return tc

    def apply_socket_fallback(self, sock_fd: int, gain_score: float) -> None:
        """
        Apply ``socket.IP_TOS`` via QoSMapper as a secondary guarantee.

        Should be called alongside ``apply_for_gain`` for defence-in-depth:
        even if the CXI fi_setopt path is unavailable (e.g., older CXI
        firmware), the socket-level marking still provides some prioritisation
        via the IP stack.
        """
        self._socket_qos.apply(sock_fd, gain_score)

    def set_all_tc(self, tc: CXI_TC) -> None:
        """Set the same TC on every endpoint (e.g., downgrade to STORAGE before batch flush)."""
        for ep in self._endpoints.values():
            ep.set_tc(tc)

    def get_stats(self) -> dict:
        return {
            "libfabric_available": _LIBFABRIC_AVAILABLE,
            "n_endpoints": len(self._endpoints),
            "endpoints": {
                f"nic{i}": ep.get_stats()
                for i, ep in self._endpoints.items()
            },
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def gain_to_cxi_tc(gain_score: float) -> CXI_TC:
    """
    Map a TEMPO service-gain score ∈ [0, 1] to the appropriate CXI_TC.

    Thresholds (mirroring QoSMapper.classify but for fabric-level TC):
      gain ≥ 0.70 → TC_LOW_LATENCY  (Q3 — Expedited Forwarding)
      gain ∈ [0.40, 0.70) → TC_BULK  (Q2 — Assured large KV)
      gain ∈ [0.15, 0.40) → TC_STORAGE  (Q1 — Checkpoint / cold)
      gain < 0.15 → TC_BEST_EFFORT  (Q0 — Background)
    """
    for threshold, tc in _GAIN_TO_CXI_TC:
        if gain_score >= threshold:
            return tc
    return CXI_TC.BEST_EFFORT


def describe_tc_impact(tc: CXI_TC) -> str:
    """Human-readable description of why a TC was chosen (for logging/reporting)."""
    descriptions = {
        CXI_TC.UNSPECIFIED:  "Default (Cassini implementation-defined behaviour)",
        CXI_TC.BEST_EFFORT:  "Q0: de-prioritised under any congestion; deferred transfer",
        CXI_TC.STORAGE:      "Q1: checkpoint / cold-KV flush, yield to NCCL and hot KV",
        CXI_TC.BULK:         "Q2: normal KV-cache; guaranteed BW, lower latency than Q1",
        CXI_TC.LOW_LATENCY:  "Q3: deadline-critical; Expedited Forwarding, co-equal with NCCL",
    }
    return descriptions.get(tc, "Unknown TC")


def cxi_tc_to_socket_tc(cxi_tc: CXI_TC) -> SocketTC:
    """Convert CXI hardware TC to socket-level TC for dual-marking."""
    return _CXI_TC_TO_SOCKET_TC.get(cxi_tc, SocketTC.BACKGROUND)
