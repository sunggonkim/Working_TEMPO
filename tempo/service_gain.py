"""
tempo/service_gain.py — Service-Gain-Based Differential I/O Bandwidth Allocator

Core Insight (OSDI framing):
  Existing systems treat all checkpoint flushes equally — they write every
  model snapshot to Lustre with the same urgency.  This is wrong:

    • A flush for step N+1 (model barely changed) provides negligible recovery
      value compared to a flush for step N+1000 (large gradient accumulation).
    • During a micro-burst of NCCL traffic, it's rational to DEFER a low-value
      flush rather than cause global Dragonfly congestion.
    • Conversely, a flush protecting a milestone checkpoint (e.g., every 500
      steps) should ALWAYS be expedited regardless of network pressure.

Design:
  Each FlushJob is assigned a ServiceGain score ∈ [0, 1]:

      gain = α·learning_progress + β·recovery_value + γ·urgency
      
  where:
    learning_progress = (steps_since_last_committed_ckpt / horizon) ∈ [0,1]
    recovery_value    = 1 − e^{−λ·steps_since_last_ckpt}  (diminishing returns)
    urgency           = max(0, 1 − remaining_budget / slo_deadline)

  The scheduler maintains a priority heap sorted by (−gain, enqueue_time).
  High-gain jobs get bandwidth-guaranteed flush windows.
  Low-gain jobs below DEFERRAL_THRESHOLD may be:
    (a) deferred until network is clear, or
    (b) marked for "recompute" (the training loop reconstructs the state
        from a lower-frequency base checkpoint + saved activations).

Bandwidth Allocation:
  The scheduler exposes a token-bucket interface.  The CheckpointManager
  calls `acquire_tokens(nbytes)` before each write chunk.  The bucket is
  refilled at a rate equal to:
      allocated_bps = base_bps × job_gain × priority_multiplier
  Highest-gain active job wins the bandwidth.

Integration with NetworkMonitor:
  When NetworkMonitor signals congestion, the scheduler HALVES the token
  refill rate for all low-priority (gain < DEFERRAL_THRESHOLD) jobs.
  High-priority jobs maintain full rate.
"""

import heapq
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_ALPHA             = 0.45   # weight for learning progress
_BETA              = 0.35   # weight for recovery value
_GAMMA             = 0.20   # weight for urgency
_LAMBDA            = 0.01   # exponential decay rate for recovery value
_DEFERRAL_THRESH   = 0.30   # jobs with gain < 0.30 are deferrable
_MILESTONE_GAIN    = 0.95   # milestone checkpoints always get high gain
_DEFAULT_HORIZON   = 1000   # steps before next "milestone" if not set
_DEFAULT_SLO_STEPS = 100    # step budget for best-effort jobs


# ---------------------------------------------------------------------------
@dataclass(order=True)
class _JobEntry:
    """Priority heap entry: (−gain, enqueue_time, job_id)."""
    neg_gain:     float
    enqueue_time: float
    job_id:       int
    flush_job:    object = field(compare=False)   # _FlushJob or similar


@dataclass
class FlushPriority:
    """Per-job priority metadata returned to CheckpointManager."""
    job_id:             int
    gain:               float        # ∈ [0,1]
    deferrable:         bool         # True → OK to defer under congestion
    allocated_bps:      float        # bytes/sec bandwidth budget
    recompute_fallback: bool         # True → caller may skip flush & recompute


# ---------------------------------------------------------------------------
class ServiceGainScheduler:
    """
    Priority-based checkpoint flush scheduler.

    Parameters
    ----------
    base_bps : float
        Baseline NIC throughput budget per rank (bytes/sec).
        Default: 200 Gbps / 8 → 25 GB/s (Slingshot-11 per-port).
    milestone_interval : int
        Every N steps a flush is treated as a milestone (always high-priority).
    slo_deadline_steps : int
        How many steps a best-effort job can wait before becoming urgent.
    congestion_bw_factor : float
        Bandwidth multiplier applied to deferrable jobs under congestion.
    """

    def __init__(
        self,
        base_bps:             float = 25e9,    # 25 GB/s (200 Gbps / 8)
        milestone_interval:   int   = 500,
        slo_deadline_steps:   int   = _DEFAULT_SLO_STEPS,
        congestion_bw_factor: float = 0.25,    # reduce to 25 % under congestion
    ):
        self.base_bps             = base_bps
        self.milestone_interval   = milestone_interval
        self.slo_deadline_steps   = slo_deadline_steps
        self.congestion_bw_factor = congestion_bw_factor

        self._lock           = threading.Lock()
        self._job_counter    = 0
        self._heap: List[_JobEntry] = []

        # State for gain computation
        self._last_committed_step: int   = 0
        self._last_ckpt_step:      int   = 0
        self._current_step:        int   = 0
        self._horizon:             int   = _DEFAULT_HORIZON

        # Congestion flag (set by NetworkMonitor integration)
        self._congested: bool = False

        # Statistics
        self._stats = dict(
            jobs_submitted=0,
            jobs_deferred=0,
            jobs_expedited=0,
            recompute_recommended=0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_step(self, step: int):
        """Inform the scheduler of the current training step."""
        with self._lock:
            self._current_step = step

    def mark_committed(self, step: int):
        """Called when a flush fully lands on Lustre."""
        with self._lock:
            self._last_committed_step = max(self._last_committed_step, step)

    def set_congested(self, flag: bool):
        """Called by NetworkMonitor when congestion state changes."""
        with self._lock:
            self._congested = flag

    def compute_gain(self, step: int) -> float:
        """
        Compute Service Gain score ∈ [0, 1] for a checkpoint at `step`.

        Components:
          learning_progress: fraction of recovery horizon consumed since
                             last committed checkpoint.
          recovery_value:    marginal value of this checkpoint vs prior;
                             models diminishing returns — early steps matter more.
          urgency:           proximity to SLO deadline.
        """
        with self._lock:
            steps_since_commit = max(0, step - self._last_committed_step)
            steps_in_budget    = max(0, step - self._last_ckpt_step)
            remaining_budget   = max(1, self.slo_deadline_steps - steps_in_budget)

            learning_progress = min(1.0, steps_since_commit / max(1, self._horizon))
            recovery_value    = 1.0 - math.exp(-_LAMBDA * steps_since_commit)
            urgency           = max(0.0, 1.0 - remaining_budget / self.slo_deadline_steps)

            # Milestone override
            if step > 0 and step % self.milestone_interval == 0:
                return _MILESTONE_GAIN

            gain = (
                _ALPHA * learning_progress
                + _BETA  * recovery_value
                + _GAMMA * urgency
            )
            return min(1.0, max(0.0, gain))

    def submit_job(self, flush_job, step: int) -> FlushPriority:
        """
        Register a flush job and return its priority descriptor.

        The CheckpointManager passes this descriptor to the token-bucket
        refill logic to set per-job bandwidth.
        """
        with self._lock:
            self._last_ckpt_step = step
            self._job_counter   += 1
            job_id               = self._job_counter
            self._stats["jobs_submitted"] += 1

        gain       = self.compute_gain(step)
        deferrable = gain < _DEFERRAL_THRESH and not self._congested
        # Under congestion, even moderately low-priority jobs become deferrable
        if self._congested and gain < 0.6:
            deferrable = True

        allocated_bps      = self.base_bps * gain
        if self._congested and deferrable:
            allocated_bps *= self.congestion_bw_factor

        # Recommend recompute for very low-gain jobs under heavy congestion
        recompute_fallback = (
            self._congested and gain < (_DEFERRAL_THRESH * 0.5)
        )

        with self._lock:
            entry = _JobEntry(
                neg_gain     = -gain,
                enqueue_time = time.perf_counter(),
                job_id       = job_id,
                flush_job    = flush_job,
            )
            heapq.heappush(self._heap, entry)

            if recompute_fallback:
                self._stats["recompute_recommended"] += 1
            elif deferrable:
                self._stats["jobs_deferred"] += 1
            else:
                self._stats["jobs_expedited"] += 1

        prio = FlushPriority(
            job_id             = job_id,
            gain               = gain,
            deferrable         = deferrable,
            allocated_bps      = allocated_bps,
            recompute_fallback = recompute_fallback,
        )
        logger.debug(
            "[ServiceGain] step=%d gain=%.3f defer=%s recompute=%s bps=%.1f GB/s",
            step, gain, deferrable, recompute_fallback,
            allocated_bps / 1e9,
        )
        return prio

    def get_stats(self) -> dict:
        with self._lock:
            return {
                **self._stats,
                "last_committed_step": self._last_committed_step,
                "current_step":        self._current_step,
                "queue_depth":         len(self._heap),
                "congested":           self._congested,
            }


# ---------------------------------------------------------------------------
# Token Bucket for per-job bandwidth throttling
# ---------------------------------------------------------------------------

class TokenBucket:
    """
    Thread-safe token bucket for bandwidth throttling.

    Tokens represent bytes.  The bucket is refilled at `rate_bps` bytes/sec.
    `acquire(n)` blocks until n bytes are available.

    Under congestion, the caller can call `set_rate(new_rate)` to reduce
    the refill rate dynamically.
    """

    def __init__(self, rate_bps: float, capacity_bytes: Optional[float] = None):
        self.rate_bps  = rate_bps
        self.capacity  = capacity_bytes or (rate_bps * 0.1)  # 100 ms burst
        self._tokens   = self.capacity
        self._last_ts  = time.perf_counter()
        self._lock     = threading.Lock()
        self._cond     = threading.Condition(self._lock)

    def set_rate(self, rate_bps: float):
        with self._cond:
            self.rate_bps = rate_bps
            self._cond.notify_all()

    def acquire(self, n_bytes: float, timeout: Optional[float] = 5.0) -> bool:
        """
        Block until n_bytes tokens are available.
        Returns False if timeout exceeded (caller should retry later).
        """
        deadline = time.perf_counter() + (timeout or 1e9)
        with self._cond:
            while True:
                self._refill()
                if self._tokens >= n_bytes:
                    self._tokens -= n_bytes
                    return True
                # Calculate how long until we have enough
                deficit   = n_bytes - self._tokens
                wait_s    = deficit / max(self.rate_bps, 1.0)
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=min(wait_s, remaining, 0.005))

    def _refill(self):
        now = time.perf_counter()
        dt  = now - self._last_ts
        self._tokens = min(self.capacity, self._tokens + dt * self.rate_bps)
        self._last_ts = now
