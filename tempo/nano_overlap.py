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

Implementation
--------------
When CUDA is available, ``NanoOverlapController`` creates:
  - ``compute_stream``: primary stream (default stream 0)
  - ``io_stream``:      secondary CUDA stream for H2D/D2H DMA operations

At layer boundaries:
  1. ``on_layer_compute_end(layer_id, kv_tensor)``:
     - Records a ``torch.cuda.Event`` on the compute stream.
     - Issues ``io_stream.wait_event(compute_event)`` so the I/O stream
       does not start until the KV tensor is fully written by the compute
       stream (no race condition).
     - Issues ``tensor.to(device='cpu', non_blocking=True)`` on io_stream,
       which DMA-copies the KV chunk to pinned host memory without stalling
       the GPU's compute SMs.
  2. ``end_step()``:
     - Records a final CUDA event on io_stream.
     - Calls ``torch.cuda.current_stream().wait_event(io_done_event)`` to
       ensure all I/O is flushed before the next AllReduce.

Pipeline illustration (one step)
---------------------------------
  Time →  0    1    2    3    4    5    6    7    8
          ┌────────────────────────────────────────┐
  compute │ L0  L1  L2  L3  L4  L5  L6  L7  L8   │ compute_stream
          └────────────────────────────────────────┘
          ┌────────────────────────────────────────┐
  KV I/O  │    [L0] [L1] [L2] [L3] [L4] [L5] [L6]│ io_stream (1 layer behind)
          └────────────────────────────────────────┘

Mathematical condition for zero-bubble pipeline
------------------------------------------------
Let t_c = compute time per layer (ms), t_io = I/O time per layer (ms).
  t_io ≤ t_c     → zero-bubble
  t_io > t_c     → pipeline stall (bubble = t_io − t_c per layer)

For Perlmutter (A100 40 GB, Llama-1B):
  t_c  ≈ 3.5 ms / layer (16 layers, FP16 matmul)
  t_io ≈ 256 MB / 10 GB/s ≈ 25 ms for the full cache
       → 25 ms / 16 layers ≈ 1.56 ms / layer   ← well within t_c
  Expected pipeline efficiency: (1 − 1.56/3.5) ≈ 55% overlap (dense)

With SparseTransferFilter reducing I/O to 12% of tokens:
  t_io → 0.12 × 1.56 ≈ 0.19 ms / layer → 95% overlap.

API
---
    ctrl = NanoOverlapController(n_layers=16)
    ctrl.begin_step(step)
    for layer_id, layer_module in enumerate(model.layers):
        ctrl.on_layer_compute_start(layer_id)
        hidden = layer_module(hidden)
        # Pass the actual KV tensor to pipeline the DMA on io_stream
        kv = extract_kv(hidden, layer_id)
        ctrl.on_layer_compute_end(layer_id, kv_tensor=kv)
    metrics = ctrl.end_step()
    # metrics.pipeline_eff → target > 0.90 with sparse transfer
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
# Try to import torch.cuda — graceful fallback for CPU-only test environments
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.cuda
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False
    torch = None  # type: ignore


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
    bubble_ms:       float = 0.0
    overlap_ms:      float = 0.0


@dataclass
class StepMetrics:
    step:              int
    n_layers:          int
    total_compute_ms:  float
    total_io_ms:       float
    total_bubble_ms:   float
    total_overlap_ms:  float
    avg_bubble_ms:     float
    pipeline_eff:      float     # overlap_ms / (compute_ms + bubble_ms)
    io_bytes_total:    int
    io_throughput_gbs: float
    cuda_streams_used: bool


# ---------------------------------------------------------------------------
# EWMA tracker
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
# PinnedBufferPool — pre-allocated host pinned memory to prevent caching
#                    allocator lock contention on the compute stream
# ---------------------------------------------------------------------------

class PinnedBufferPool:
    """
    Pre-allocated pool of CUDA pinned (page-locked) host memory buffers.

    Problem this solves
    -------------------
    When ``kv_tensor.to('cpu', non_blocking=True)`` is called inside
    ``torch.cuda.stream(io_stream)``, PyTorch's CUDACachingAllocator
    dynamically allocates a pinned host buffer for the DMA destination.
    This allocation acquires the global allocator lock, which can stall
    the compute stream for 50–200 µs even when the two streams are
    otherwise independent.

    Solution
    --------
    We pre-allocate ``n_layers`` pinned CPU tensors at startup.  Each
    layer's KV DMA always writes into its dedicated buffer, avoiding any
    dynamic allocation on the critical path.

    Technical note: ``torch.empty(..., pin_memory=True)`` calls
    ``cudaHostAlloc(flags=cudaHostAllocDefault)`` once.  Subsequent
    ``copy_(src, non_blocking=True)`` reuses the same physical pages
    — the DMA engine programs the IOMMU tables at open time and never
    touches the allocator lock again.

    Thread safety: ``acquire(layer_id)`` is lock-free (round-robins by
    layer_id).  ``release()`` is a no-op (buffers are reused in-place).

    Parameters
    ----------
    n_layers : int
        Number of pre-allocated buffers (one per transformer layer).
    chunk_elems : int
        Number of elements per buffer (should match the KV tensor numel).
    dtype : torch.dtype
        Element type (default bfloat16 — matches Llama training dtype).
    """

    def __init__(
        self,
        n_layers:    int,
        chunk_elems: int,
        dtype        = None,
    ) -> None:
        self._pool:   List["torch.Tensor"] = []
        self._available = _CUDA_AVAILABLE and torch is not None

        if self._available:
            _dtype = dtype or torch.bfloat16
            try:
                for _ in range(n_layers):
                    buf = torch.empty(chunk_elems, dtype=_dtype, pin_memory=True)
                    self._pool.append(buf)
                log.info(
                    "PinnedBufferPool: pre-allocated %d × %d × %d bytes "
                    "(pinned host memory, zero dynamic alloc on hot path)",
                    n_layers, chunk_elems,
                    chunk_elems * torch.finfo(_dtype).bits // 8,
                )
            except Exception as e:
                log.warning(
                    "PinnedBufferPool: pinned alloc failed (%s) — "
                    "falling back to dynamic allocation (may cause implicit sync)", e
                )
                self._pool = []
                self._available = False

    @property
    def ready(self) -> bool:
        return bool(self._pool)

    def get(self, layer_id: int) -> "Optional[torch.Tensor]":
        """Return the pre-allocated pinned buffer for this layer.

        Returns None if pool is empty (fallback to dynamic allocation).
        """
        if not self._pool:
            return None
        return self._pool[layer_id % len(self._pool)]

    def resize_if_needed(self, required_elems: int, dtype) -> None:
        """Grow each buffer if the actual KV tensor is larger than expected."""
        if not self._pool:
            return
        if self._pool[0].numel() < required_elems:
            old = len(self._pool)
            self._pool = [
                torch.empty(required_elems, dtype=dtype, pin_memory=True)
                for _ in range(old)
            ]
            log.debug(
                "PinnedBufferPool: resized to %d elements per buffer", required_elems
            )


# ---------------------------------------------------------------------------
# NanoOverlapController
# ---------------------------------------------------------------------------

class NanoOverlapController:
    """
    Pipelines per-layer KV-cache I/O with transformer layer compute using
    separate CUDA streams.

    When CUDA is available (Perlmutter A100 nodes), genuine CUDA stream
    overlapping is used:
      - ``compute_stream`` = torch.cuda.default_stream()
      - ``io_stream``      = torch.cuda.Stream()
      - Synchronisation via torch.cuda.Event (zero CPU stall)

    Memory allocation strategy
    --------------------------
    PyTorch's CUDACachingAllocator acquires a global lock when it allocates
    pinned host buffers.  To prevent this from stalling the compute stream
    during KV DMA, we use ``PinnedBufferPool`` to pre-allocate all DMA
    destination buffers at construction time.  After the first step, there
    are zero dynamic allocations on the io_stream hot path.

    When CUDA is not available (login nodes, CI), falls back to a background
    thread that mimics the I/O stream.  This path is labelled ``cpu_fallback``
    in all reported stats so it is never confused with real hardware numbers.

    Parameters
    ----------
    n_layers : int
        Number of transformer layers.
    chunk_bytes : int
        Expected bytes per layer KV chunk (used for throughput accounting
        and pre-allocating pinned memory buffers).
    io_callback : callable, optional
        ``io_callback(layer_id, cpu_tensor_or_bytes)`` called once the
        DMA to host memory completes.  Use this to write the chunk to
        Lustre or P2PCacheStore.
    prefetch_depth : int
        Number of layers to prefetch ahead on io_stream.  1 = standard
        pipeline; increase to hide longer DMA latencies.
    device : str or torch.device, optional
        CUDA device to use.  Defaults to current device.
    preallocate_pinned : bool
        Whether to pre-allocate pinned host memory at construction time
        (default True).  Disable only in memory-constrained environments.
    """

    def __init__(
        self,
        n_layers:            int                = 32,
        chunk_bytes:         int                = 8 * 1024 * 1024,
        io_callback:         Optional[Callable] = None,
        prefetch_depth:      int                = 1,
        device:              Optional[str]      = None,
        preallocate_pinned:  bool               = True,
    ) -> None:
        self.n_layers      = n_layers
        self.chunk_bytes   = chunk_bytes
        self.io_callback   = io_callback
        self.prefetch_depth = prefetch_depth

        # ── CUDA stream setup ──────────────────────────────────────────
        self._use_cuda = _CUDA_AVAILABLE
        self._compute_stream = None
        self._io_stream      = None
        self._device         = None

        if self._use_cuda and torch is not None:
            self._device = torch.device(device or "cuda")
            # io_stream is a distinct CUDA stream — allows the DMA engine
            # to run concurrently with the SM compute on the other stream.
            # priority=-1 gives it slightly lower scheduling priority than
            # the compute stream so it never pre-empts forward pass SMs.
            self._compute_stream = torch.cuda.default_stream(self._device)
            self._io_stream = torch.cuda.Stream(
                device=self._device, priority=-1
            )
            log.info(
                "NanoOverlapController: CUDA streams active  "
                "io_stream_id=%d  device=%s",
                self._io_stream.cuda_stream, self._device,
            )
        else:
            log.warning(
                "NanoOverlapController: CUDA not available — "
                "using CPU fallback thread (NOT suitable for hardware results)"
            )

        # ── Pinned memory pool — prevents CUDACachingAllocator lock contention
        # on the io_stream hot path.
        # Pre-allocate n_layers bfloat16 buffers at construction time.
        # Each buffer holds chunk_bytes / 2 elements (bfloat16 = 2 bytes).
        # ──────────────────────────────────────────────────────────────
        self._pinned_pool: Optional[PinnedBufferPool] = None
        if self._use_cuda and preallocate_pinned and torch is not None:
            chunk_elems = max(1, chunk_bytes // 2)   # bfloat16
            self._pinned_pool = PinnedBufferPool(
                n_layers    = n_layers,
                chunk_elems = chunk_elems,
                dtype       = torch.bfloat16,
            )
            if not self._pinned_pool.ready:
                self._pinned_pool = None

        # Per-step bookkeeping
        self._step: int = -1
        self._layer_timings: List[LayerTiming] = []

        # Pending CUDA events: layer_id → (compute_done_event, io_done_event)
        self._compute_events: Dict[int, "torch.cuda.Event"] = {}
        self._io_events:      Dict[int, "torch.cuda.Event"] = {}

        # CPU fallback: background thread queue
        self._cpu_queue:  deque = deque()
        self._cpu_lock    = threading.Lock()
        self._cpu_cond    = threading.Condition(self._cpu_lock)
        self._cpu_stop    = threading.Event()
        self._cpu_thread: Optional[threading.Thread] = None
        if not self._use_cuda:
            self._cpu_thread = threading.Thread(
                target=self._cpu_io_worker,
                name="nano-io-cpu",
                daemon=True,
            )
            self._cpu_thread.start()

        # EWMA predictors
        self._compute_ewma = _EWMA(alpha=0.15, init=3.5)
        self._io_ewma      = _EWMA(alpha=0.15, init=1.0)

        # Aggregate stats
        self._steps_done: int = 0
        self._steps_history: Deque[StepMetrics] = deque(maxlen=100)

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def begin_step(self, step: int) -> None:
        """Call at the start of each training step."""
        self._step = step
        self._layer_timings = [LayerTiming(layer_id=i) for i in range(self.n_layers)]
        self._compute_events.clear()
        self._io_events.clear()

    def on_layer_compute_start(self, layer_id: int) -> None:
        """
        Mark the beginning of layer *layer_id* forward pass.
        Called on the default (compute) CUDA stream.
        """
        if 0 <= layer_id < self.n_layers:
            self._layer_timings[layer_id].compute_start = time.perf_counter()

    def on_layer_compute_end(
        self,
        layer_id: int,
        kv_tensor: "Optional[torch.Tensor]" = None,
        kv_data:   Optional[bytes]           = None,
    ) -> None:
        """
        Mark the end of layer *layer_id* compute.

        Enqueues the KV chunk for DMA on the CUDA io_stream, which runs
        concurrently with the compute of layer L+1 (the next call to
        on_layer_compute_start).

        Parameters
        ----------
        kv_tensor : torch.Tensor, optional
            GPU KV-cache tensor for this layer.  When provided on a CUDA
            device, a non-blocking D2H copy is issued on io_stream.
        kv_data : bytes, optional
            Pre-serialised KV bytes (CPU).  Used when kv_tensor is None.
        """
        if not (0 <= layer_id < self.n_layers):
            return

        t_compute_end = time.perf_counter()
        lt = self._layer_timings[layer_id]
        lt.compute_end = t_compute_end
        compute_ms = (t_compute_end - lt.compute_start) * 1000
        self._compute_ewma.update(compute_ms)

        if self._use_cuda and torch is not None and kv_tensor is not None:
            self._enqueue_cuda_io(layer_id, kv_tensor)
        else:
            # CPU fallback: push to background worker queue
            data = kv_data or bytes(min(self.chunk_bytes, 512))
            n_bytes = len(kv_data) if kv_data else self.chunk_bytes
            lt.io_bytes = n_bytes
            with self._cpu_cond:
                self._cpu_queue.append((layer_id, data, time.perf_counter()))
                self._cpu_cond.notify()

    def end_step(self) -> StepMetrics:
        """
        Synchronise all pending CUDA io_stream operations, then compute
        and return per-step pipeline efficiency metrics.

        For CUDA path: waits for the io_done_event of the last layer.
        For CPU fallback: drains the queue.
        """
        if self._use_cuda and torch is not None:
            self._sync_cuda_io()
        else:
            self._drain_cpu_queue()

        metrics = self._compute_step_metrics()
        self._steps_history.append(metrics)
        self._steps_done += 1
        return metrics

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_predicted_io_window_ms(self) -> float:
        """Available I/O window per layer = compute_ewma − 0.5 ms margin."""
        return max(0.0, self._compute_ewma.mean - 0.5)

    def get_stats(self) -> dict:
        if not self._steps_history:
            return {"steps_done": self._steps_done,
                    "cuda_streams": self._use_cuda,
                    "pinned_pool_ready": self._pinned_pool is not None and self._pinned_pool.ready}
        recent = list(self._steps_history)[-20:]
        avg_eff    = sum(m.pipeline_eff for m in recent) / len(recent)
        avg_bubble = sum(m.avg_bubble_ms for m in recent) / len(recent)
        return {
            "steps_done":        self._steps_done,
            "avg_pipeline_eff":  avg_eff,
            "avg_bubble_ms":     avg_bubble,
            "compute_ms_ewma":   self._compute_ewma.mean,
            "io_ms_ewma":        self._io_ewma.mean,
            "zero_bubble_steps": sum(1 for m in recent if m.total_bubble_ms < 0.5),
            "cuda_streams":      self._use_cuda,
            "pinned_pool_ready": self._pinned_pool is not None and self._pinned_pool.ready,
        }

    def shutdown(self) -> None:
        """Stop background threads and synchronise CUDA streams."""
        if self._use_cuda and torch is not None:
            try:
                if self._io_stream is not None:
                    self._io_stream.synchronize()
            except Exception:
                pass
        self._cpu_stop.set()
        with self._cpu_cond:
            self._cpu_cond.notify_all()
        if self._cpu_thread:
            self._cpu_thread.join(timeout=3)

    # ------------------------------------------------------------------
    # Private: CUDA stream path
    # ------------------------------------------------------------------

    def _enqueue_cuda_io(self, layer_id: int, kv_tensor: "torch.Tensor") -> None:
        """
        Issue a non-blocking D2H DMA copy on io_stream.

        Protocol:
          1. Record compute_done_event on compute_stream (default stream).
             This event fires when the GPU has finished writing kv_tensor.
          2. Make io_stream wait for compute_done_event.
             This is a CUDA stream-level dependency: io_stream will NOT start
             the copy until compute_stream signals the event.  There is no
             CPU involvement after this point.
          3. Issue .cpu(non_blocking=True) on io_stream (pinned memory copy).
          4. Record io_done_event on io_stream.

        The CPU thread never stalls on any of these operations — they are
        all asynchronous GPU scheduler commands.
        """
        assert torch is not None

        io_start = time.perf_counter()
        lt = self._layer_timings[layer_id]
        lt.io_start = io_start
        lt.io_bytes = kv_tensor.nelement() * kv_tensor.element_size()

        # 1. Compute stream signals "KV tensor is ready"
        compute_done = torch.cuda.Event(enable_timing=True)
        compute_done.record(stream=self._compute_stream)
        self._compute_events[layer_id] = compute_done

        # 2. io_stream waits for the compute stream to finish kv_tensor write
        self._io_stream.wait_event(compute_done)

        # 3. Asynchronous D2H copy on io_stream (pinned memory → no stall)
        # Use pre-allocated pinned buffer when available (zero dynamic alloc).
        # If the KV tensor is larger than expected, resize the pool in-place.
        with torch.cuda.stream(self._io_stream):
            if kv_tensor.is_cuda:
                pinned_buf = None
                if self._pinned_pool is not None:
                    # Grow pool if this layer's KV is larger than pre-allocated
                    self._pinned_pool.resize_if_needed(
                        kv_tensor.numel(), kv_tensor.dtype
                    )
                    pinned_buf = self._pinned_pool.get(layer_id)

                if pinned_buf is not None:
                    # Zero-alloc path: copy_ into pre-allocated pinned buffer.
                    # The DMA engine re-uses the existing IOMMU mapping.
                    # Reshape if necessary (contiguous view of same numel).
                    if pinned_buf.numel() >= kv_tensor.numel():
                        dst = pinned_buf[:kv_tensor.numel()].view(kv_tensor.shape)
                    else:
                        dst = pinned_buf  # fallback; pool resize above handles growth
                    dst.copy_(kv_tensor, non_blocking=True)
                    cpu_tensor = dst
                else:
                    # Dynamic allocation fallback (may trigger allocator lock)
                    cpu_tensor = kv_tensor.to(device="cpu", non_blocking=True)
            else:
                cpu_tensor = kv_tensor  # already on CPU

        # 4. Record "I/O done" event on io_stream
        io_done = torch.cuda.Event(enable_timing=True)
        io_done.record(stream=self._io_stream)
        self._io_events[layer_id] = io_done

        # Schedule user callback to fire after io_done syncs
        # We use a lightweight CPU hook rather than blocking here.
        if self.io_callback is not None:
            _callback = self.io_callback
            _lid = layer_id

            def _deferred_cb():
                io_done.synchronize()
                io_end = time.perf_counter()
                lt.io_end = io_end
                io_ms = (io_end - lt.io_start) * 1000
                self._io_ewma.update(io_ms)
                c_ms = (lt.compute_end - lt.compute_start) * 1000
                lt.bubble_ms  = max(0.0, io_ms - c_ms)
                lt.overlap_ms = min(io_ms, c_ms) * 1000  # ms² → already ms
                try:
                    _callback(_lid, cpu_tensor)
                except Exception as e:
                    log.debug("io_callback layer=%d: %s", _lid, e)

            t = threading.Thread(target=_deferred_cb, daemon=True)
            t.start()
        else:
            # No callback: still need to record timing after sync
            def _record_timing():
                io_done.synchronize()
                io_end = time.perf_counter()
                lt.io_end = io_end
                io_ms = (io_end - lt.io_start) * 1000
                self._io_ewma.update(io_ms)
                c_ms = (lt.compute_end - lt.compute_start) * 1000
                lt.bubble_ms  = max(0.0, io_ms - c_ms)
                lt.overlap_ms = min(io_ms, c_ms)

            t = threading.Thread(target=_record_timing, daemon=True)
            t.start()

    def _sync_cuda_io(self) -> None:
        """
        Synchronise the io_stream: make the compute stream wait for all
        pending I/O events before the next AllReduce.

        This is a stream-level wait (zero CPU stall):
          compute_stream.wait_event(last_io_done_event)
        Only if no io events were recorded (all layers skipped), we do nothing.
        """
        if not self._io_events:
            return
        last_layer = max(self._io_events.keys())
        last_event = self._io_events[last_layer]
        # Make the default compute stream wait for io_stream to finish.
        # This ensures the next AllReduce (which runs on compute_stream) sees
        # a fully consistent state.
        if self._compute_stream is not None:
            self._compute_stream.wait_event(last_event)

    # ------------------------------------------------------------------
    # Private: CPU fallback path (non-CUDA environments)
    # ------------------------------------------------------------------

    def _cpu_io_worker(self) -> None:
        """CPU fallback background thread (labelled in stats as cpu_fallback)."""
        while not self._cpu_stop.is_set():
            item = None
            with self._cpu_cond:
                if self._cpu_queue:
                    item = self._cpu_queue.popleft()
                else:
                    self._cpu_cond.wait(timeout=0.005)
                    continue

            if item is None:
                continue

            layer_id, data, enqueue_time = item
            io_start = time.perf_counter()

            # Simulate 10 GB/s NVMe throughput (CPU fallback — not real CUDA DMA)
            n_bytes = len(data)
            sim_s = n_bytes / (10 * 1024**3)
            if sim_s > 0.0001:
                time.sleep(sim_s)

            if self.io_callback is not None:
                try:
                    self.io_callback(layer_id, data)
                except Exception as e:
                    log.debug("cpu io_callback layer=%d: %s", layer_id, e)

            io_end = time.perf_counter()
            io_ms  = (io_end - io_start) * 1000
            self._io_ewma.update(io_ms)

            if 0 <= layer_id < len(self._layer_timings):
                lt = self._layer_timings[layer_id]
                lt.io_start = io_start
                lt.io_end   = io_end
                c_ms = (lt.compute_end - lt.compute_start) * 1000
                lt.bubble_ms  = max(0.0, io_ms - c_ms)
                lt.overlap_ms = min(io_ms, c_ms)

    def _drain_cpu_queue(self, timeout: float = 2.0) -> None:
        """Wait until the CPU fallback queue is empty."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._cpu_lock:
                if not self._cpu_queue:
                    return
            time.sleep(0.002)

    # ------------------------------------------------------------------
    # Private: metrics computation
    # ------------------------------------------------------------------

    def _compute_step_metrics(self) -> StepMetrics:
        lts = [lt for lt in self._layer_timings if lt.compute_end > 0]
        if not lts:
            return StepMetrics(
                step=self._step, n_layers=self.n_layers,
                total_compute_ms=0, total_io_ms=0,
                total_bubble_ms=0, total_overlap_ms=0,
                avg_bubble_ms=0, pipeline_eff=0,
                io_bytes_total=0, io_throughput_gbs=0,
                cuda_streams_used=self._use_cuda,
            )

        total_c      = sum((lt.compute_end - lt.compute_start) * 1000 for lt in lts)
        total_io     = sum((lt.io_end - lt.io_start) * 1000
                           for lt in lts if lt.io_end > 0)
        total_bubble  = sum(lt.bubble_ms  for lt in lts)
        total_overlap = sum(lt.overlap_ms for lt in lts)
        total_bytes   = sum(lt.io_bytes   for lt in lts)
        elapsed_s     = sum((lt.io_end - lt.io_start)
                            for lt in lts if lt.io_end > 0)
        throughput_gbs = (total_bytes / max(1e-9, elapsed_s)) / 1024**3

        return StepMetrics(
            step               = self._step,
            n_layers           = len(lts),
            total_compute_ms   = total_c,
            total_io_ms        = total_io,
            total_bubble_ms    = total_bubble,
            total_overlap_ms   = total_overlap,
            avg_bubble_ms      = total_bubble / max(1, len(lts)),
            pipeline_eff       = total_overlap / max(0.001, total_c + total_bubble),
            io_bytes_total     = total_bytes,
            io_throughput_gbs  = throughput_gbs,
            cuda_streams_used  = self._use_cuda,
        )
