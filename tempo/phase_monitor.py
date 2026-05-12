"""
tempo/phase_monitor.py — Training Phase Detection for TEMPO Pacing Scheduler

Tracks whether the current training step is in a:
  - COMPUTE phase  (forward pass, backward matmul)  → I/O flush ALLOWED
  - NCCL_COMM phase (All-Reduce, Reduce-Scatter, All-Gather) → I/O flush PAUSED
  - CHECKPOINT phase (saving state dict to local NVMe)      → I/O flush PAUSED

Thread-safety guarantee:
  All state transitions use an RLock.  The background flush thread polls
  `wait_for_io_allowed()` before each write chunk — no busy-spin.

Integration options (in order of invasiveness):
  1. Manual context managers in training loop (simplest, recommended):
        with monitor.nccl_phase():
            dist.all_reduce(tensor)

  2. FSDP comm hook (automatic gradient reduction timing):
        model.register_comm_hook(monitor, PhaseMonitor.fsdp_comm_hook)

  3. DDP comm hook:
        model.register_comm_hook(None, monitor.make_ddp_comm_hook())
"""

import time
import threading
import logging
from enum import Enum, auto
from contextlib import contextmanager
from typing import Optional, Callable

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
class TrainingPhase(Enum):
    """Enumeration of training pipeline phases."""
    IDLE       = auto()   # Between steps
    COMPUTE    = auto()   # Forward/backward matmul
    NCCL_COMM  = auto()   # NCCL collective (All-Reduce / Reduce-Scatter / All-Gather)
    CHECKPOINT = auto()   # Checkpoint save (local NVMe write)


# ---------------------------------------------------------------------------
class PhaseMonitor:
    """
    Thread-safe monitor for the current training phase.

    The background I/O flush thread (in CheckpointManager) calls
    `wait_for_io_allowed()` before each write chunk.  The training thread
    calls `nccl_phase()` / `compute_phase()` context managers to signal
    phase transitions, which internally set/clear the `_io_allowed` Event.

    This is the core mechanism that gives TEMPO its "pacing" behaviour:
    flush throughput is gated behind an asyncio-style event, with zero
    overhead on the critical training path.
    """

    def __init__(self, rank: int = 0, verbose: bool = False):
        self._lock = threading.RLock()
        self._phase: TrainingPhase = TrainingPhase.IDLE
        self._rank = rank
        self._verbose = verbose

        # ---- Timing accumulators ----
        self._nccl_enter_time: float = 0.0
        self._nccl_total_s: float = 0.0
        self._compute_enter_time: float = 0.0
        self._compute_total_s: float = 0.0
        self._step: int = 0

        # ---- Rolling NCCL duration window (for adaptive chunk sizing) ----
        # Stores the last N observed NCCL phase durations in milliseconds.
        self._nccl_window_size: int = 16
        self._nccl_durations_ms: list = []   # circular buffer (deque semantics)
        # EMA-smoothed estimate (α=0.30); more noise-robust than raw mean.
        # Updated each time the NCCL phase ends.
        self._ema_nccl_ms: float = 0.0

        # ---- I/O gating Event ----
        # SET   = background I/O is allowed to proceed
        # CLEAR = background I/O must pause (NCCL is active)
        self._io_allowed = threading.Event()
        self._io_allowed.set()   # default: allow I/O

        # ---- Optional callbacks (e.g., for logging / metrics) ----
        self.on_nccl_start: Optional[Callable] = None
        self.on_nccl_end: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Public read-only accessors
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> TrainingPhase:
        with self._lock:
            return self._phase

    def is_nccl_active(self) -> bool:
        """Returns True if an NCCL collective is currently in flight."""
        return self._phase == TrainingPhase.NCCL_COMM

    def is_io_allowed(self) -> bool:
        """Non-blocking check: can the flush thread write right now?"""
        return self._io_allowed.is_set()

    def wait_for_io_allowed(self, timeout: float = 0.05) -> bool:
        """
        Block the calling thread until I/O is allowed (or timeout expires).
        Returns True if I/O is now allowed, False if it timed out while blocked.
        Called by CheckpointManager's flush thread before each write chunk.
        """
        return self._io_allowed.wait(timeout=timeout)

    def get_dynamic_flush_rate(
        self,
        requested_bps:     float,
        nccl_bw_bps:       float = 0.0,
        pcie_ceiling_bps:  float = 64e9,
        safety_margin:     float = 0.10,
    ) -> float:
        """
        Continuous (non-binary) bandwidth budget for the next flush chunk.

        Unlike wait_for_io_allowed() which fully blocks during NCCL phases,
        this returns a float target rate that CheckpointManager feeds into
        its token bucket.  The effect is smooth rate adaptation instead of
        stop-go oscillation:

          NCCL active          → 0.0  (must not write during AllReduce)
          NCCL idle, low load  → requested_bps (full rate)
          NCCL idle, high load → pcie_ceiling × (1 − nccl_fraction − margin)

        Parameters
        ----------
        requested_bps    : ideal flush rate (bytes/sec) from token bucket
        nccl_bw_bps      : current NCCL AllReduce bandwidth on PCIe (bytes/sec);
                           0 = caller doesn't know; use EMA estimate.
        pcie_ceiling_bps : total PCIe bandwidth (DMA + NCCL share, default 64 GB/s)
        safety_margin    : fractional headroom to preserve above NCCL estimate
                           (default 10% → stops flush before ceiling is reached)

        Returns
        -------
        float  allocated bytes/sec for the next write chunk (0.0 = do not write)
        """
        with self._lock:
            if self._phase == TrainingPhase.NCCL_COMM:
                return 0.0   # hard block during active AllReduce

        # Estimate NCCL PCIe consumption from EMA phase duration
        effective_nccl_bps = nccl_bw_bps
        if effective_nccl_bps == 0.0 and self._ema_nccl_ms > 0.0:
            # Heuristic: typical AllReduce tensor size / NCCL phase duration
            # For 1B model (256 MB AllReduce tensor per phase):
            _TYPICAL_ALLREDUCE_BYTES = 256 * 1024 * 1024
            effective_nccl_bps = _TYPICAL_ALLREDUCE_BYTES / (self._ema_nccl_ms * 1e-3)

        nccl_fraction     = effective_nccl_bps / max(pcie_ceiling_bps, 1.0)
        available_fraction = max(0.0, 1.0 - nccl_fraction - safety_margin)
        allocated_bps      = pcie_ceiling_bps * available_fraction
        return min(requested_bps, allocated_bps)

    @property
    def estimated_nccl_bps(self) -> float:
        """
        Rough estimate of NCCL PCIe bandwidth consumption (bytes/sec).

        Returns 0.0 when in COMPUTE phase (NCCL not active).
        Used by PCIePressurePredictor to compute real-time look-ahead.
        """
        with self._lock:
            if self._phase != TrainingPhase.NCCL_COMM:
                return 0.0
            if self._ema_nccl_ms <= 0.0:
                return 0.0
            _TYPICAL_ALLREDUCE_BYTES = 256 * 1024 * 1024
            return _TYPICAL_ALLREDUCE_BYTES / (self._ema_nccl_ms * 1e-3)

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def set_phase(self, new_phase: TrainingPhase) -> None:
        """Atomic phase transition with I/O gating and timing."""
        with self._lock:
            old_phase = self._phase
            if old_phase == new_phase:
                return

            now = time.perf_counter()

            # --- Leaving COMPUTE ---
            if old_phase == TrainingPhase.COMPUTE:
                self._compute_total_s += now - self._compute_enter_time

            # --- Leaving NCCL_COMM ---
            if old_phase == TrainingPhase.NCCL_COMM:
                elapsed = now - self._nccl_enter_time
                self._nccl_total_s += elapsed
                elapsed_ms = elapsed * 1e3
                # Record for adaptive chunk sizing (circular buffer)
                self._nccl_durations_ms.append(elapsed_ms)
                if len(self._nccl_durations_ms) > self._nccl_window_size:
                    self._nccl_durations_ms.pop(0)
                # EMA-smoothed estimate (α=0.30, bias-corrected on first sample)
                if self._ema_nccl_ms == 0.0:
                    self._ema_nccl_ms = elapsed_ms
                else:
                    self._ema_nccl_ms = 0.30 * elapsed_ms + 0.70 * self._ema_nccl_ms
                self._io_allowed.set()   # ← Resume I/O flush
                if self.on_nccl_end:
                    self.on_nccl_end()
                if self._verbose and self._rank == 0:
                    logger.debug(f"[PhaseMonitor] Step {self._step}: "
                                 f"NCCL phase ended ({elapsed*1000:.2f} ms)")

            # --- Entering NCCL_COMM ---
            if new_phase == TrainingPhase.NCCL_COMM:
                self._nccl_enter_time = now
                self._io_allowed.clear()  # ← Pause I/O flush
                if self.on_nccl_start:
                    self.on_nccl_start()
                if self._verbose and self._rank == 0:
                    logger.debug(f"[PhaseMonitor] Step {self._step}: NCCL phase started")

            # --- Entering COMPUTE ---
            if new_phase == TrainingPhase.COMPUTE:
                self._compute_enter_time = now

            self._phase = new_phase

    def increment_step(self) -> None:
        with self._lock:
            self._step += 1

    # ------------------------------------------------------------------
    # Context managers (Option 1: manual annotation)
    # ------------------------------------------------------------------

    @contextmanager
    def nccl_phase(self):
        """
        Mark a block of code as an NCCL communication phase.
        Background I/O flush is paused for the duration.

        Example:
            with monitor.nccl_phase():
                dist.all_reduce(grad_tensor)
        """
        self.set_phase(TrainingPhase.NCCL_COMM)
        try:
            yield
        finally:
            self.set_phase(TrainingPhase.COMPUTE)

    @contextmanager
    def compute_phase(self):
        """
        Mark a block of code as a compute (matmul) phase.
        Background I/O flush is allowed.
        """
        self.set_phase(TrainingPhase.COMPUTE)
        try:
            yield
        finally:
            self.set_phase(TrainingPhase.IDLE)

    @contextmanager
    def checkpoint_phase(self):
        """
        Mark a checkpoint-save block.
        Background I/O flush paused during local NVMe write.
        """
        self.set_phase(TrainingPhase.CHECKPOINT)
        try:
            yield
        finally:
            self.set_phase(TrainingPhase.IDLE)

    # ------------------------------------------------------------------
    # Option 2: FSDP Communication Hook
    # ------------------------------------------------------------------

    @staticmethod
    def fsdp_comm_hook(state: "PhaseMonitor", bucket):
        """
        FSDP communication hook — automatically times gradient reduction.

        Register via:
            model.register_comm_hook(monitor, PhaseMonitor.fsdp_comm_hook)

        The hook wraps FSDP's default all_reduce with NCCL phase transitions,
        so the PhaseMonitor tracks communication automatically without any
        manual annotations in the training loop.
        """
        state.set_phase(TrainingPhase.NCCL_COMM)
        tensor = bucket.buffer()
        world_size = dist.get_world_size()

        # All-reduce with sum, then divide (matches FSDP default behaviour)
        fut = dist.all_reduce(tensor, async_op=True).get_future()

        def mark_done(fut):
            state.set_phase(TrainingPhase.COMPUTE)
            return [fut.value()[0] / world_size]

        return fut.then(mark_done)

    # ------------------------------------------------------------------
    # Option 3: DDP Communication Hook
    # ------------------------------------------------------------------

    def make_ddp_comm_hook(self):
        """
        Creates a DDP gradient hook that marks NCCL phases.

        Register via:
            model.register_comm_hook(None, monitor.make_ddp_comm_hook())
        """
        monitor = self

        def ddp_hook(state, bucket):
            monitor.set_phase(TrainingPhase.NCCL_COMM)
            fut = dist.all_reduce(bucket.buffer(), async_op=True).get_future()

            def on_done(fut):
                monitor.set_phase(TrainingPhase.COMPUTE)
                return [fut.value()[0]]

            return fut.then(on_done)

        return ddp_hook

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        with self._lock:
            total = self._nccl_total_s + self._compute_total_s + 1e-12
            avg_nccl = (sum(self._nccl_durations_ms) / len(self._nccl_durations_ms)
                        if self._nccl_durations_ms else 0.0)
            return {
                "step":               self._step,
                "phase":              self._phase.name,
                "nccl_total_s":       round(self._nccl_total_s, 4),
                "compute_total_s":    round(self._compute_total_s, 4),
                "nccl_fraction":      round(self._nccl_total_s / total, 4),
                "io_allowed":         self._io_allowed.is_set(),
                "avg_nccl_ms":        round(avg_nccl, 2),
                "ema_nccl_ms":        round(self._ema_nccl_ms, 2),
            }

    def get_avg_nccl_duration_ms(self) -> float:
        """
        Returns the rolling average of observed NCCL phase durations (ms).
        Used by CheckpointManager for adaptive chunk sizing.
        Returns 0.0 if no NCCL phases have been observed yet.
        """
        with self._lock:
            if not self._nccl_durations_ms:
                return 0.0
            return sum(self._nccl_durations_ms) / len(self._nccl_durations_ms)

    @property
    def nccl_phase_duration_ms(self) -> float:
        """
        EMA-smoothed estimate of NCCL phase duration (ms), α=0.30.
        Preferred over ``get_avg_nccl_duration_ms`` for adaptive chunk sizing
        because it down-weights old observations and reacts faster to changes
        in NCCL phase length (e.g., gradient accumulation schedule changes).
        Returns 0.0 before the first NCCL phase completes.
        """
        with self._lock:
            return self._ema_nccl_ms

    def reset_stats(self) -> None:
        with self._lock:
            self._nccl_total_s = 0.0
            self._compute_total_s = 0.0
