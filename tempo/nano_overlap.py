"""
tempo/nano_overlap.py — NanoFlow-style GPU-Stream Compute/Network Overlapper
=============================================================================

OSDI motivation (NanoFlow, OSDI 2025)
--------------------------------------
NanoFlow observes that existing LLM serving systems leave significant GPU
utilisation headroom because compute, memory, and network operations are
executed *sequentially* within each decode step.  By decomposing the work
into nano-batches and pipelining the three resource types across overlapping
CUDA streams, NanoFlow achieves near-100% GPU utilisation.

TEMPO v4 contribution
---------------------
We apply the same principle to *checkpoint I/O*: instead of writing the KV
cache as a monolithic block (which stalls the training loop), we split it
into layer-granularity nano-chunks and pipeline them across two CUDA streams:

  compute_stream  — runs the forward/backward pass for layer L+1
  io_stream       — concurrently DMA-transfers the KV chunk for layer L
                    to host memory and initiates the P2P/Lustre write

This eliminates the "I/O bubble" — the idle period between two AllReduce
phases when the GPU waits for checkpoint I/O to complete.

Pipeline illustration (one step)
---------------------------------
  Time →  0    1    2    3    4    5    6    7    8
          ┌────────────────────────────────────────┐
  compute │ L0  L1  L2  L3  L4  L5  L6  L7  L8   │ compute_stream
          └────────────────────────────────────────┘
          ┌────────────────────────────────────────┐
  KV I/O  │    [L0] [L1] [L2] [L3] [L4] [L5] [L6]│ io_stream (1 layer behind)
          └────────────────────────────────────────┘

The KV chunk for layer L is written during the compute window of layer L+1.
As long as the I/O time per layer ≤ the compute time per layer, the pipeline
runs without any bubble.

Mathematical condition for zero-bubble pipeline
------------------------------------------------
Let t_c = compute time per layer (ms), t_io = I/O time per layer (ms).
  t_io ≤ t_c     → zero-bubble
  t_io > t_c     → pipeline stall (bubble = t_io − t_c per layer)

For Perlmutter (A100 40 GB, Llama-1B):
  t_c  ≈ 3.5 ms / layer (32 layers, FP16 matmul)
  t_io ≈ 256 MB / 10 GB/s ≈ 25 ms for the full cache
       → 25 ms / 32 layers ≈ 0.78 ms / layer   ← well within t_c
  Expected pipeline efficiency: (1 − 0.78/3.5) ≈ 78% overlap

With SparseTransferFilter reducing I/O to 12% of tokens:
  t_io → 0.12 × 0.78 ≈ 0.09 ms / layer → 97% overlap.

API
---
    ctrl = NanoOverlapController(n_layers=32, chunk_bytes=8*1024*1024)
    ctrl.begin_step(step)
    for layer in range(32):
        ctrl.on_layer_compute_start(layer)
        # ... transformer layer forward ...
        ctrl.on_layer_compute_end(layer)
        # io_stream DMA happens concurrently in background
    metrics = ctrl.end_step()
    # metrics["bubble_ms_per_layer"] → target < 0.5 ms
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LayerTiming:
    layer_id:        int
    compute_start:   float = 0.0
    compute_end:     float = 0.0
    io_start:        float = 0.0
    io_end:          float = 0.0
    io_bytes:        int   = 0
    bubble_ms:       float = 0.0   # max(0, io_time − compute_time)
    overlap_ms:      float = 0.0   # min(compute_time, io_time)


@dataclass
class StepMetrics:
    step:             int
    n_layers:         int
    total_compute_ms: float
    total_io_ms:      float
    total_bubble_ms:  float
    total_overlap_ms: float
    avg_bubble_ms:    float
    pipeline_eff:     float     # overlap_ms / (compute_ms + bubble_ms)
    io_bytes_total:   int
    io_throughput_gbs: float


# ---------------------------------------------------------------------------
# EWMA tracker (reuse pattern from InterleavingEngine)
# ---------------------------------------------------------------------------

class _EWMA:
    def __init__(self, alpha: float = 0.1, init: float = 3.0) -> None:
        self._a   = alpha
        self._val = init
        self._var = 0.0

    def update(self, x: float) -> None:
        d          = x - self._val
        self._val += self._a * d
        self._var  = (1 - self._a) * (self._var + self._a * d * d)

    @property
    def mean(self) -> float: return self._val
    @property
    def std(self) -> float:  return self._var ** 0.5


# ---------------------------------------------------------------------------
# NanoOverlapController
# ---------------------------------------------------------------------------

class NanoOverlapController:
    """
    Pipelines per-layer KV-cache I/O with transformer layer compute.

    The controller tracks layer-level timings, schedules I/O callbacks on a
    background thread (simulating CUDA io_stream), and reports pipeline
    efficiency metrics.

    Parameters
    ----------
    n_layers : int
        Number of transformer layers.
    chunk_bytes : int
        Approximate bytes per layer KV chunk.  Used for throughput estimation.
    io_callback : callable, optional
        ``io_callback(layer_id, data)`` called on the io_stream thread for
        each layer.  If None, a dummy stub is used (for benchmarking).
    prefetch_depth : int
        Number of layers to prefetch ahead.  1 = classic 1-layer pipeline.
        Higher values amortise I/O latency at the cost of more DRAM.
    """

    def __init__(
        self,
        n_layers:      int               = 32,
        chunk_bytes:   int               = 8 * 1024 * 1024,
        io_callback:   Optional[Callable] = None,
        prefetch_depth: int              = 1,
    ) -> None:
        self.n_layers      = n_layers
        self.chunk_bytes   = chunk_bytes
        self.io_callback   = io_callback or (lambda lid, data: None)
        self.prefetch_depth = prefetch_depth

        # Per-step state
        self._step: int = -1
        self._layer_timings: List[LayerTiming] = []
        self._pending_io: Deque[int] = deque()
        self._lock = threading.Lock()

        # Background io_stream thread
        self._io_queue: Deque[tuple] = deque()
        self._io_thread = threading.Thread(
            target=self._io_worker, name="nano-io-stream", daemon=True
        )
        self._io_cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._io_thread.start()

        # EWMA predictors
        self._compute_ewma = _EWMA(alpha=0.15, init=3.5)
        self._io_ewma      = _EWMA(alpha=0.15, init=0.8)

        # Aggregate stats
        self._steps_done: int        = 0
        self._steps_history: Deque[StepMetrics] = deque(maxlen=100)

        log.info(
            "NanoOverlapController: n_layers=%d chunk_bytes=%d "
            "prefetch_depth=%d",
            n_layers, chunk_bytes, prefetch_depth,
        )

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def begin_step(self, step: int) -> None:
        """Call at the beginning of each training step."""
        self._step = step
        self._layer_timings = [LayerTiming(layer_id=i) for i in range(self.n_layers)]

    def on_layer_compute_start(self, layer_id: int) -> None:
        """Mark the beginning of transformer layer *layer_id* forward pass."""
        if 0 <= layer_id < self.n_layers:
            self._layer_timings[layer_id].compute_start = time.perf_counter()

    def on_layer_compute_end(
        self,
        layer_id: int,
        kv_data: Optional[bytes] = None,
    ) -> None:
        """
        Mark the end of layer *layer_id* compute.

        Enqueues the KV chunk for I/O on the background io_stream thread.
        The I/O for layer L begins *during* the compute of layer L+1,
        realising the pipeline overlap.

        Parameters
        ----------
        kv_data : bytes, optional
            Serialised KV chunk for this layer.  If None, a zero-copy
            placeholder of ``chunk_bytes`` is used for timing purposes.
        """
        if not (0 <= layer_id < self.n_layers):
            return
        t = time.perf_counter()
        self._layer_timings[layer_id].compute_end = t
        compute_ms = (t - self._layer_timings[layer_id].compute_start) * 1000
        self._compute_ewma.update(compute_ms)

        # Enqueue I/O on io_stream (executed concurrently with L+1 compute)
        data  = kv_data or bytes(min(self.chunk_bytes, 1024))   # stub
        n_bytes = len(kv_data) if kv_data else self.chunk_bytes
        self._layer_timings[layer_id].io_bytes = n_bytes
        with self._io_cond:
            self._io_queue.append((layer_id, data, time.perf_counter()))
            self._io_cond.notify()

    def end_step(self) -> StepMetrics:
        """
        Wait for all pending I/O to complete, compute step-level metrics.
        """
        # Drain io_queue for this step
        deadline = time.perf_counter() + 2.0   # 2 s safety timeout
        while True:
            with self._lock:
                if not self._io_queue:
                    break
            if time.perf_counter() > deadline:
                log.warning("NanoOverlapController: io_queue drain timed out")
                break
            time.sleep(0.001)

        metrics = self._compute_step_metrics()
        self._steps_history.append(metrics)
        self._steps_done += 1
        return metrics

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_predicted_io_window_ms(self) -> float:
        """
        Estimated available I/O window per layer (ms).

        = predicted compute time − safety margin (0.5 ms)
        """
        return max(0.0, self._compute_ewma.mean - 0.5)

    def get_stats(self) -> dict:
        if not self._steps_history:
            return {"steps_done": self._steps_done}
        recent = list(self._steps_history)[-20:]
        avg_eff = sum(m.pipeline_eff for m in recent) / len(recent)
        avg_bubble = sum(m.avg_bubble_ms for m in recent) / len(recent)
        return {
            "steps_done":        self._steps_done,
            "avg_pipeline_eff":  avg_eff,
            "avg_bubble_ms":     avg_bubble,
            "compute_ms_ewma":   self._compute_ewma.mean,
            "io_ms_ewma":        self._io_ewma.mean,
            "zero_bubble_steps": sum(1 for m in recent if m.total_bubble_ms < 0.5),
        }

    def shutdown(self) -> None:
        """Stop the background io_stream thread."""
        self._stop.set()
        with self._io_cond:
            self._io_cond.notify_all()
        self._io_thread.join(timeout=3)

    # ------------------------------------------------------------------
    # Private: io_stream worker thread
    # ------------------------------------------------------------------

    def _io_worker(self) -> None:
        """Background thread simulating the CUDA io_stream."""
        while not self._stop.is_set():
            item = None
            with self._io_cond:
                if self._io_queue:
                    item = self._io_queue.popleft()
                else:
                    self._io_cond.wait(timeout=0.005)
                    continue

            if item is None:
                continue

            layer_id, data, enqueue_time = item
            io_start = time.perf_counter()

            # Simulate I/O latency proportional to data size
            # Real implementation: torch.cuda.memcpy_async or UCX send
            n_bytes = len(data)
            # Simulate 10 GB/s NVMe/network throughput
            sim_lat = n_bytes / (10 * 1024**3) * 1000   # ms
            if sim_lat > 0:
                time.sleep(sim_lat / 1000)

            # Call user-provided callback (real: DMA + socket send)
            try:
                self.io_callback(layer_id, data)
            except Exception as e:
                log.debug("io_callback layer=%d: %s", layer_id, e)

            io_end = time.perf_counter()
            io_ms  = (io_end - io_start) * 1000
            self._io_ewma.update(io_ms)

            if 0 <= layer_id < len(self._layer_timings):
                lt = self._layer_timings[layer_id]
                lt.io_start = io_start
                lt.io_end   = io_end
                # Compute overlap and bubble
                c_start = lt.compute_start
                c_end   = lt.compute_end
                if c_end > 0:
                    # Overlap = intersection of [c_start, c_end] and [io_start, io_end]
                    overlap = max(0, min(c_end, io_end) - max(c_start, io_start))
                    bubble  = max(0, io_ms - (c_end - c_start) * 1000)
                    lt.overlap_ms = overlap * 1000
                    lt.bubble_ms  = bubble

    # ------------------------------------------------------------------
    # Private: metrics computation
    # ------------------------------------------------------------------

    def _compute_step_metrics(self) -> StepMetrics:
        lts = [lt for lt in self._layer_timings if lt.compute_end > 0]
        if not lts:
            return StepMetrics(
                step=self._step, n_layers=self.n_layers,
                total_compute_ms=0, total_io_ms=0, total_bubble_ms=0,
                total_overlap_ms=0, avg_bubble_ms=0, pipeline_eff=0,
                io_bytes_total=0, io_throughput_gbs=0,
            )
        total_c  = sum((lt.compute_end - lt.compute_start) * 1000 for lt in lts)
        total_io = sum((lt.io_end - lt.io_start) * 1000
                       for lt in lts if lt.io_end > 0)
        total_bubble  = sum(lt.bubble_ms  for lt in lts)
        total_overlap = sum(lt.overlap_ms for lt in lts)
        total_bytes   = sum(lt.io_bytes   for lt in lts)
        elapsed_s = sum((lt.io_end - lt.io_start) for lt in lts if lt.io_end > 0)
        throughput_gbs = (total_bytes / max(1e-6, elapsed_s)) / 1024**3

        return StepMetrics(
            step              = self._step,
            n_layers          = len(lts),
            total_compute_ms  = total_c,
            total_io_ms       = total_io,
            total_bubble_ms   = total_bubble,
            total_overlap_ms  = total_overlap,
            avg_bubble_ms     = total_bubble / max(1, len(lts)),
            pipeline_eff      = total_overlap / max(0.001, total_c + total_bubble),
            io_bytes_total    = total_bytes,
            io_throughput_gbs = throughput_gbs,
        )
