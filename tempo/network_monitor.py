"""
tempo/network_monitor.py — Slingshot-11 / HSN NIC Bandwidth Monitor

Perlmutter-specific design:
  Each node exposes HPE Slingshot-11 (HSN) NICs via sysfs at:
    /sys/class/net/hsn{0,1}/statistics/{rx,tx}_bytes

  The monitor polls these counters at configurable intervals to compute
  instantaneous NIC utilization.  When I/O traffic is about to saturate
  the shared NIC bandwidth, we signal the CheckpointManager to PAUSE
  flushing — even if the training loop is in COMPUTE phase.

  Key insight (Dragonfly topology):
    On Dragonfly networks, a single node flooding the network with KV-cache
    I/O can cause global congestion that affects *unrelated* NCCL collectives
    on other nodes.  Therefore, we cannot rely solely on per-process phase
    gating — we need a global bandwidth budget that accounts for all traffic
    on the node's NICs.

  Fallback:
    If Slingshot sysfs paths are not available (non-Perlmutter systems), the
    monitor falls back to per-process socket statistics via /proc/net/dev.

Architecture:
    NetworkMonitor runs as a daemon thread (10 ms poll interval).
    It maintains:
      - Exponential moving average (EMA) of TX+RX bytes/sec per NIC
      - Rolling max (last 32 samples) for burst detection
      - A "congestion" Event that CheckpointManager respects

    CheckpointManager calls:
        nm.wait_for_bw_headroom(needed_bytes_per_sec)
    which blocks until the EMA utilization drops below
        LINK_SPEED_BPS * CONGESTION_THRESHOLD (default 0.80)

OSDI framing:
    We prove that CheckpointManager's flush pressure routinely exceeds
    0.80 × 200 Gbps = 160 Gbps per node, causing global Dragonfly
    congestion.  This is the "root cause" observation that motivates the
    entire TEMPO v2 design.
"""

import os
import re
import time
import threading
import logging
import socket
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sysfs paths for HPE Slingshot-11 / generic HSN
# ---------------------------------------------------------------------------
_HSN_SYSFS_PATTERN   = "/sys/class/net/hsn{n}/statistics/{stat}"
_PROC_NET_DEV        = "/proc/net/dev"
_LINK_SPEED_GBPS     = 200.0           # Slingshot-11 per-link bandwidth
_LINK_SPEED_BPS      = _LINK_SPEED_GBPS * 1e9
_CONGESTION_THRESH   = 0.75            # Pause I/O when util > 75 % of link bw
_EMA_ALPHA           = 0.25            # EMA smoothing factor (faster = more reactive)
_POLL_INTERVAL_S     = 0.005           # 5 ms poll (practical lower bound for sysfs)
_WINDOW_SIZE         = 64              # Rolling window for burst detection


def _read_sysfs_counter(path: str) -> Optional[int]:
    """Read a single integer from sysfs, return None on failure."""
    try:
        return int(Path(path).read_text().strip())
    except (FileNotFoundError, ValueError, PermissionError):
        return None


def _discover_slingshot_nics() -> List[str]:
    """
    Enumerate Slingshot NICs by looking for hsn* under /sys/class/net/.
    Falls back to checking /proc/net/dev for any hs* or eth* interfaces.
    """
    candidates = []
    net_root = Path("/sys/class/net")
    if net_root.exists():
        for iface in sorted(net_root.iterdir()):
            name = iface.name
            if name.startswith("hsn") or name.startswith("ib") or name.startswith("hfi"):
                tx_path = iface / "statistics" / "tx_bytes"
                if tx_path.exists():
                    candidates.append(name)
    if not candidates:
        # Fallback: parse /proc/net/dev for any high-speed interface
        try:
            with open(_PROC_NET_DEV) as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) == 2:
                        iface = parts[0].strip()
                        if any(iface.startswith(p) for p in ("ens", "eth", "bond", "ib")):
                            candidates.append(iface)
        except OSError:
            pass
    return candidates


# ---------------------------------------------------------------------------
class NicSample:
    """Snapshot of a single NIC's byte counters."""
    __slots__ = ("ts", "tx", "rx")

    def __init__(self, ts: float, tx: int, rx: int):
        self.ts = ts
        self.tx = tx
        self.rx = rx


# ---------------------------------------------------------------------------
class NetworkMonitor:
    """
    Slingshot-11 NIC bandwidth monitor for TEMPO v2.

    Parameters
    ----------
    congestion_threshold : float
        Fraction of link bandwidth above which I/O is gated (default 0.75).
    poll_interval_s : float
        Sysfs polling period in seconds (default 5 ms).
    link_speed_bps : float
        Per-NIC link speed in bits/sec (default 200 Gbps for Slingshot-11).
    """

    def __init__(
        self,
        congestion_threshold: float = _CONGESTION_THRESH,
        poll_interval_s:      float = _POLL_INTERVAL_S,
        link_speed_bps:       float = _LINK_SPEED_BPS,
    ):
        self.congestion_threshold = congestion_threshold
        self.poll_interval_s      = poll_interval_s
        self.link_speed_bps       = link_speed_bps

        self._nics     = _discover_slingshot_nics()
        self._simulated = len(self._nics) == 0   # no real hardware

        if self._simulated:
            logger.warning("[NetworkMonitor] No Slingshot/HSN NICs found — "
                           "running in simulation mode (always IO-allowed)")
        else:
            logger.info("[NetworkMonitor] Monitoring NICs: %s", self._nics)

        # Per-NIC: (last_sample, EMA_bytes_per_sec, rolling_window)
        self._nic_state: Dict[str, Tuple[Optional[NicSample], float, deque]] = {
            nic: (None, 0.0, deque(maxlen=_WINDOW_SIZE))
            for nic in self._nics
        }

        # Aggregated state
        self._total_bps_ema: float = 0.0           # EMA of total BW (bytes/s)
        self._total_bps_peak: float = 0.0          # Rolling max of total BW

        # Congestion gate: clear = congested (IO blocked), set = IO allowed
        self._io_allowed_event = threading.Event()
        self._io_allowed_event.set()               # start as allowed

        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Statistics counters
        self._stats = dict(
            poll_count=0,
            congestion_events=0,
            total_congestion_ms=0.0,
            peak_bps=0.0,
        )
        self._congestion_start: Optional[float] = None

        # Simulated congestion state (for tests/non-Slingshot nodes)
        self._sim_bps: float = 0.0   # externally set for simulation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start the background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="NetworkMonitor"
        )
        self._thread.start()
        logger.debug("[NetworkMonitor] started (poll=%.0f ms, threshold=%.0f%%)",
                     self.poll_interval_s * 1e3,
                     self.congestion_threshold * 100)

    def stop(self):
        """Stop the background polling thread."""
        self._stop.set()
        self._io_allowed_event.set()   # unblock any waiting caller
        if self._thread:
            self._thread.join(timeout=1.0)

    def is_congested(self) -> bool:
        """Return True if current NIC utilisation exceeds threshold."""
        return not self._io_allowed_event.is_set()

    def wait_for_bw_headroom(self, timeout: Optional[float] = None) -> bool:
        """
        Block until NIC utilisation drops below congestion_threshold.
        Returns True if headroom is available, False if timeout expired.
        """
        return self._io_allowed_event.wait(timeout=timeout)

    def current_util_fraction(self) -> float:
        """Return EMA NIC utilisation as a fraction of link speed (0–1+)."""
        if self._simulated:
            bps = self._sim_bps
        else:
            bps = self._total_bps_ema
        # convert bytes/s → bits/s → fraction
        return (bps * 8.0) / self.link_speed_bps

    def set_simulated_bps(self, bps: float):
        """Inject a simulated bandwidth value (for unit tests)."""
        with self._lock:
            self._sim_bps = bps
            self._update_congestion_state(bps)

    def get_stats(self) -> dict:
        with self._lock:
            util = self.current_util_fraction()
            s    = dict(self._stats)
        s["nic_count"]       = len(self._nics)
        s["simulated"]       = self._simulated
        s["util_pct"]        = round(util * 100, 1)
        s["ema_bps_gbps"]    = round(self._total_bps_ema * 8 / 1e9, 3)
        s["peak_bps_gbps"]   = round(self._stats["peak_bps"] * 8 / 1e9, 3)
        return s

    # ------------------------------------------------------------------
    # Internal polling
    # ------------------------------------------------------------------

    def _read_nic_bytes(self, nic: str) -> Optional[Tuple[int, int]]:
        """Read (tx_bytes, rx_bytes) from sysfs for one NIC."""
        tx = _read_sysfs_counter(f"/sys/class/net/{nic}/statistics/tx_bytes")
        rx = _read_sysfs_counter(f"/sys/class/net/{nic}/statistics/rx_bytes")
        if tx is None or rx is None:
            return None
        return tx, rx

    def _poll_one_nic(self, nic: str, now: float) -> float:
        """
        Read current NIC counters, compute instantaneous bytes/sec.
        Update EMA and rolling window for this NIC.
        Returns EMA bytes/sec for this NIC.
        """
        raw = self._read_nic_bytes(nic)
        if raw is None:
            return 0.0
        tx, rx = raw
        total = tx + rx

        prev_sample, ema, window = self._nic_state[nic]
        if prev_sample is None:
            new_ema = 0.0
        else:
            dt = now - prev_sample.ts
            if dt < 1e-6:
                new_ema = ema
            else:
                inst_bps = (total - (prev_sample.tx + prev_sample.rx)) / dt
                inst_bps = max(inst_bps, 0.0)
                new_ema = _EMA_ALPHA * inst_bps + (1 - _EMA_ALPHA) * ema

        window.append(new_ema)
        self._nic_state[nic] = (NicSample(now, tx, rx), new_ema, window)
        return new_ema

    def _update_congestion_state(self, total_bps: float):
        """Flip the IO-allowed gate based on aggregate bandwidth."""
        threshold_bps = self.link_speed_bps * self.congestion_threshold / 8.0  # bps→Bps
        congested     = total_bps >= threshold_bps
        now           = time.perf_counter()

        if congested and self._io_allowed_event.is_set():
            # Transition: allowed → congested
            self._io_allowed_event.clear()
            self._congestion_start = now
            self._stats["congestion_events"] += 1
            logger.debug("[NetworkMonitor] CONGESTED %.1f Gbps > %.1f Gbps threshold",
                         total_bps * 8 / 1e9,
                         threshold_bps * 8 / 1e9)

        elif not congested and not self._io_allowed_event.is_set():
            # Transition: congested → allowed
            self._io_allowed_event.set()
            if self._congestion_start:
                dur_ms = (now - self._congestion_start) * 1e3
                self._stats["total_congestion_ms"] += dur_ms
                self._congestion_start = None
            logger.debug("[NetworkMonitor] CLEAR")

        if total_bps > self._stats["peak_bps"]:
            self._stats["peak_bps"] = total_bps

    def _poll_loop(self):
        """Background daemon: poll NICs, update EMA, update gate."""
        logger.debug("[NetworkMonitor] poll loop started")
        while not self._stop.is_set():
            now = time.perf_counter()
            with self._lock:
                self._stats["poll_count"] += 1

                if self._simulated:
                    # Simulated mode: use externally set _sim_bps
                    total_bps = self._sim_bps
                else:
                    total_bps = sum(
                        self._poll_one_nic(nic, now) for nic in self._nics
                    )

                self._total_bps_ema = (
                    _EMA_ALPHA * total_bps + (1 - _EMA_ALPHA) * self._total_bps_ema
                )
                if total_bps > self._total_bps_peak:
                    self._total_bps_peak = total_bps

                self._update_congestion_state(self._total_bps_ema)

            time.sleep(self.poll_interval_s)

        logger.debug("[NetworkMonitor] poll loop stopped")


# ============================================================================
# CassiniHWCounters — HPE Slingshot-11 Cassini NIC Performance Counters
# ============================================================================
#
# HPE Slingshot-11 (Cassini ASIC) exposes per-NIC hardware performance
# counters via sysfs at:
#   /sys/class/net/hsn{n}/device/
#   /sys/bus/cxi/devices/cxi{n}/ports/p1/
#
# The counters relevant to TEMPO routing decisions are:
#
#   CxiCongestion       — number of NACK/ECN congestion signals received
#                         from the fabric in the last polling window.
#                         Non-zero → global links are saturated.
#
#   CxiLinkReliability  — raw retransmit count.  >0 indicates fabric stress.
#
#   CxiHWMark           — High-watermark counter for the FLIT buffer.
#                         Near-full (>90%) means the NIC's inbound queue is
#                         backing up — a leading indicator of congestion.
#
#   tx_stall_ns         — Nanoseconds the NIC TX engine was stalled waiting
#                         for fabric credits.  >0 is the first symptom of
#                         global link pressure.
#
# Sysfs path convention (varies by Cassini firmware version):
#   /sys/bus/cxi/devices/cxi{n}/stats/
#   /sys/class/net/hsn{n}/device/infiniband/... (Mellanox-style symlink)
#
# libfabric telemetry alternative:
#   fi_cxi provider exposes per-CQ stats via fi_getopt(FI_OPT_CXI_*).
#   We do NOT use the libfabric path here because it requires the process
#   to have an open endpoint, which conflicts with background monitor
#   threads.  The sysfs path is read-only and available to any UID.
#
# Perlmutter firmware note (as of 2025Q4):
#   Cassini firmware >= 2.3.0 exposes additional congestion counters under
#   /sys/bus/cxi/devices/cxi{n}/stats/flit_cntr/
#   Earlier firmware only exposes tx/rx_bytes (handled by NetworkMonitor).
#
# OSDI artifact note:
#   On Perlmutter nodes allocated via SLURM, we typically see 2 Cassini NICs
#   (cxi0, cxi1) per node mapped to hsn0, hsn1.  Both should be monitored
#   and their congestion counters OR'd.

_CXI_SYSFS_ROOTS = [
    "/sys/bus/cxi/devices/cxi{n}/stats",          # Cassini >= 2.3.0
    "/sys/class/net/hsn{n}/device/stats",          # symlink fallback
]

_CASSINI_COUNTERS = {
    # counter_name           : (sysfs_subpath, scale_factor, description)
    "congestion_flits":       ("flit_cntr/congestion",    1,   "NACK/ECN flit count"),
    "reliability_retx":       ("link/reliability_retx",   1,   "retransmit count"),
    "tx_stall_ns":            ("performance/tx_stall_ns", 1,   "TX stall time (ns)"),
    "hwmark_pct":             ("buffer/hwmark_pct",       0.01, "HW buffer watermark %"),
}


def _read_cassini_counter(cxi_idx: int, subpath: str) -> Optional[int]:
    """
    Attempt to read a Cassini hardware counter for NIC cxi{cxi_idx}.
    Tries all known sysfs root patterns in order.  Returns None if not
    accessible (no Cassini firmware, no permission, or older kernel).
    """
    for root_tmpl in _CXI_SYSFS_ROOTS:
        root = root_tmpl.format(n=cxi_idx)
        full = os.path.join(root, subpath)
        val  = _read_sysfs_counter(full)
        if val is not None:
            return val
    return None


class CassiniHWCounters:
    """
    Reader for HPE Slingshot-11 (Cassini ASIC) hardware performance counters.

    This is the ground-truth congestion signal for TEMPO's TopologyRouter.
    When CxiCongestion > 0 (even a single NACK from the fabric), the
    TopologyRouter should immediately switch from LUSTRE_REMOTE to
    LUSTRE_LOCAL or DEFERRED placement to stop injecting traffic into the
    already-saturated global links.

    Parameters
    ----------
    n_nics : int
        Number of Cassini NICs to monitor (default: auto-detect up to 4).
    poll_interval_s : float
        How often to refresh the counters (default 10 ms — counters are
        monotonically increasing so we diff them between polls).
    congestion_threshold : int
        Number of new congestion flits per polling interval that triggers
        the congestion flag (default 1 — any congestion is flagged).

    Usage
    -----
        hw = CassiniHWCounters()
        hw.start()
        ...
        if hw.is_fabric_congested():
            router.set_global_link_saturated(True)
        stats = hw.get_stats()
        hw.stop()
    """

    def __init__(
        self,
        n_nics:               int   = 0,    # 0 = auto-detect
        poll_interval_s:      float = 0.010,
        congestion_threshold: int   = 1,
    ) -> None:
        self.poll_interval_s      = poll_interval_s
        self.congestion_threshold = congestion_threshold

        # Auto-detect CXI device count
        if n_nics == 0:
            n_nics = self._detect_cxi_count()
        self.n_nics = n_nics

        self._available = (n_nics > 0)
        if not self._available:
            logger.warning(
                "[CassiniHWCounters] No CXI devices found under /sys/bus/cxi/. "
                "Hardware congestion counters unavailable — routing will rely "
                "on sysfs BW counters only (NetworkMonitor)."
            )
        else:
            logger.info(
                "[CassiniHWCounters] Found %d CXI NIC(s). "
                "Monitoring Cassini hardware congestion counters.", n_nics
            )

        # Per-NIC previous counter values (for delta computation)
        self._prev: Dict[int, Dict[str, int]] = {
            i: {k: 0 for k in _CASSINI_COUNTERS} for i in range(n_nics)
        }
        # Per-NIC deltas (counts in last interval)
        self._delta: Dict[int, Dict[str, int]] = {
            i: {k: 0 for k in _CASSINI_COUNTERS} for i in range(n_nics)
        }

        self._congested    = False
        self._lock         = threading.Lock()
        self._stop_evt     = threading.Event()
        self._thread:       Optional[threading.Thread] = None

        # Cumulative stats
        self._total_congestion_events = 0
        self._total_retx              = 0
        self._total_stall_ns          = 0

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def start(self) -> "CassiniHWCounters":
        """Start the background counter-polling thread."""
        if not self._available:
            return self
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="cassini-hw-ctr",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2)

    # ----------------------------------------------------------------
    # Public queries
    # ----------------------------------------------------------------

    def is_fabric_congested(self) -> bool:
        """
        Return True if ANY Cassini NIC reported congestion in the last
        polling interval.

        This is the authoritative signal for TopologyRouter to stop using
        global Dragonfly links.  Unlike sysfs BW counters (which are
        reactive), CxiCongestion is a *proactive* signal — the fabric
        ECN/NACK mechanism fires before the link is fully saturated.
        """
        with self._lock:
            return self._congested

    def congestion_flit_rate(self) -> float:
        """
        Return total congestion flit count per second across all NICs
        (averaged over the last polling interval).
        """
        with self._lock:
            total = sum(
                self._delta[i].get("congestion_flits", 0)
                for i in range(self.n_nics)
            )
        return total / max(1e-6, self.poll_interval_s)

    def retransmit_rate(self) -> float:
        """Return retransmit events per second (leading indicator of link stress)."""
        with self._lock:
            total = sum(
                self._delta[i].get("reliability_retx", 0)
                for i in range(self.n_nics)
            )
        return total / max(1e-6, self.poll_interval_s)

    def get_stats(self) -> dict:
        """Return a snapshot of all hardware counter statistics."""
        with self._lock:
            deltas = {
                f"cxi{i}": dict(self._delta[i])
                for i in range(self.n_nics)
            }
        return {
            "available":                  self._available,
            "n_nics":                     self.n_nics,
            "congested":                  self._congested,
            "total_congestion_events":    self._total_congestion_events,
            "total_retx":                 self._total_retx,
            "total_stall_ns":             self._total_stall_ns,
            "per_nic_deltas":             deltas,
            "congestion_flit_rate":       self.congestion_flit_rate(),
            "retransmit_rate":            self.retransmit_rate(),
        }

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _detect_cxi_count() -> int:
        """Count available /sys/bus/cxi/devices/cxi{n} entries."""
        base = Path("/sys/bus/cxi/devices")
        if not base.exists():
            return 0
        return sum(1 for d in base.iterdir()
                   if d.name.startswith("cxi") and d.name[3:].isdigit())

    def _read_nic_counters(self, cxi_idx: int) -> Dict[str, int]:
        """Read all tracked counters for one NIC; return 0 for unavailable ones."""
        result = {}
        for name, (subpath, _scale, _desc) in _CASSINI_COUNTERS.items():
            val = _read_cassini_counter(cxi_idx, subpath)
            result[name] = val if val is not None else self._prev[cxi_idx].get(name, 0)
        return result

    def _poll_loop(self) -> None:
        """Background thread: read hardware counters, compute deltas."""
        # Initialise previous values
        for i in range(self.n_nics):
            self._prev[i] = self._read_nic_counters(i)

        while not self._stop_evt.is_set():
            time.sleep(self.poll_interval_s)
            total_congestion_flits = 0

            with self._lock:
                for i in range(self.n_nics):
                    curr = self._read_nic_counters(i)
                    for name in _CASSINI_COUNTERS:
                        delta = max(0, curr.get(name, 0) - self._prev[i].get(name, 0))
                        self._delta[i][name] = delta
                        if name == "congestion_flits":
                            total_congestion_flits += delta
                        elif name == "reliability_retx":
                            self._total_retx += delta
                        elif name == "tx_stall_ns":
                            self._total_stall_ns += delta
                    self._prev[i] = curr

                new_congested = total_congestion_flits >= self.congestion_threshold
                if new_congested and not self._congested:
                    self._total_congestion_events += 1
                    logger.debug(
                        "[CassiniHW] Fabric congestion detected: "
                        "%d congestion flits across %d NICs",
                        total_congestion_flits, self.n_nics,
                    )
                self._congested = new_congested
