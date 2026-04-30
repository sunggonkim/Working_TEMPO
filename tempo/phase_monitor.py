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
                # Record for adaptive chunk sizing
                self._nccl_durations_ms.append(elapsed * 1e3)
                if len(self._nccl_durations_ms) > self._nccl_window_size:
                    self._nccl_durations_ms.pop(0)
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
            return {
                "step":               self._step,
                "phase":              self._phase.name,
                "nccl_total_s":       round(self._nccl_total_s, 4),
                "compute_total_s":    round(self._compute_total_s, 4),
                "nccl_fraction":      round(self._nccl_total_s / total, 4),
                "io_allowed":         self._io_allowed.is_set(),
                "avg_nccl_ms":        round(self.get_avg_nccl_duration_ms(), 2),
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

    def reset_stats(self) -> None:
        with self._lock:
            self._nccl_total_s = 0.0
            self._compute_total_s = 0.0
