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
