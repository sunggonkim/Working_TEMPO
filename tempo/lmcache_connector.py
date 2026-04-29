# SPDX-License-Identifier: Apache-2.0
# TEMPO: Harmonious Burst Buffer for Jitter-Free LLM Systems
# tempo/lmcache_connector.py
#
# TEMPOStorageBackend — a drop-in LMCache storage backend that wraps any
# existing backend (CPU, Disk, NIXL, Mooncake) with Attention-Phase-Aware
# pacing.
#
# How it integrates:
#
#   LMCache config (lmcache_config.yaml):
#     local_cpu: cpu
#     remote_url: redis://...
#     # NEW: wrap with TEMPO pacing
#     storage_backend: tempo
#     tempo_backing_backend: cpu
#
#   Or programmatically:
#     from tempo.lmcache_connector import TEMPOStorageBackend
#     backend = TEMPOStorageBackend(backing=CpuMemoryBackend(cfg), cfg=tempo_cfg)
#     lmcache.use_backend(backend)
#
# Under the hood:
#   1. put(key, tensor) → SpikeAbsorber.absorb() (O(1), no PCIe I/O)
#   2. PacingDaemon (C++ thread) drains ring → backing backend.put()
#      ONLY during FFN windows detected by AttentionPhaseMonitor (CUPTI)
#   3. get(key) → backing backend.get() as normal (reads are not paced)

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to load libtempo.so — the C++ pacing library.
# Falls back to a pure-Python soft implementation when the .so is not built.
# ---------------------------------------------------------------------------

_libtempo: Optional[ctypes.CDLL] = None

def _load_libtempo() -> Optional[ctypes.CDLL]:
    search = [
        os.path.join(os.path.dirname(__file__), "../build/libtempo.so"),
        "/usr/local/lib/libtempo.so",
        os.environ.get("LIBTEMPO_PATH", ""),
    ]
    for path in search:
        if path and os.path.exists(path):
            try:
                lib = ctypes.CDLL(path)
                log.info("TEMPO: loaded libtempo.so from %s", path)
                return lib
            except OSError as e:
                log.warning("TEMPO: failed to load %s: %s", path, e)
    log.warning("TEMPO: libtempo.so not found — using Python soft implementation. "
                "Build with: cmake .. && make -j$(nproc)")
    return None


# ---------------------------------------------------------------------------
# TEMPOConfig
# ---------------------------------------------------------------------------

@dataclass
class TEMPOConfig:
    rate_gbps: float       = float(os.environ.get("TEMPO_RATE_GBPS",   "5.0"))
    burst_mb: int          = int(os.environ.get("TEMPO_BURST_MB",     "256"))
    ffn_wait_us: int       = int(os.environ.get("TEMPO_FFN_WAIT_US",  "200"))
    strict_gate: bool      = bool(int(os.environ.get("TEMPO_STRICT_GATE", "1")))
    staging_dir: str       = os.environ.get("TEMPO_STAGE_DIR",
                                             f"/tmp/tempo_{os.getpid()}")
    lustre_dir: str        = os.environ.get("TEMPO_LUSTRE_DIR",
                                             os.path.join(
                                                 os.environ.get("PSCRATCH", "/tmp"),
                                                 "tempo_kvcache"))
    verbose: bool          = bool(int(os.environ.get("TEMPO_VERBOSE", "0")))


# ---------------------------------------------------------------------------
# Soft (Python) implementation of the pacing absorber
#
# Used when libtempo.so is not available (CI, login nodes without CUDA).
# Semantics match the C++ version but use threading.Event instead of CUPTI.
# ---------------------------------------------------------------------------

class _SoftAbsorber:
    """Pure-Python fallback absorber with background flush thread."""

    def __init__(self, backing, cfg: TEMPOConfig) -> None:
        self._backing  = backing
        self._cfg      = cfg
        self._queue: list = []
        self._lock     = threading.Lock()
        self._cond     = threading.Condition(self._lock)
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._flush_worker,
                                           daemon=True, name="tempo-soft-flush")
        self._thread.start()
        if cfg.verbose:
            log.info("TEMPO soft absorber started (rate=%.1f GB/s)", cfg.rate_gbps)

    def absorb(self, key, tensor: torch.Tensor) -> None:
        """Enqueue a KV tensor for deferred flush. O(1) — returns immediately."""
        with self._cond:
            self._queue.append((key, tensor.cpu() if tensor.is_cuda else tensor))
            self._cond.notify()

    def _flush_worker(self) -> None:
        """Background thread: drain queue → backing backend."""
        rate_bps  = self._cfg.rate_gbps * 1e9
        burst_b   = self._cfg.burst_mb * 1024 * 1024
        tokens    = float(burst_b)
        last_t    = time.monotonic()

        while not self._stop.is_set():
            with self._cond:
                while not self._queue and not self._stop.is_set():
                    self._cond.wait(timeout=0.001)
                if self._stop.is_set() and not self._queue:
                    break
                if not self._queue:
                    continue
                key, tensor = self._queue.pop(0)

            # Token bucket refill
            now    = time.monotonic()
            tokens = min(tokens + (now - last_t) * rate_bps, burst_b)
            last_t = now

            size = tensor.nbytes
            if tokens < size:
                time.sleep(size / rate_bps)  # wait for bucket to fill
                tokens = min(tokens + (time.monotonic() - last_t) * rate_bps, burst_b)

            tokens -= size
            self._backing.put(key, tensor)

    def shutdown(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# TEMPOStorageBackend  — public interface
# ---------------------------------------------------------------------------

class TEMPOStorageBackend:
    """
    TEMPO-paced LMCache storage backend.

    Wraps any LMCache StorageBackend with Attention-Phase-Aware pacing:
    - put() is non-blocking (O(1) spike absorption)
    - Actual writes to the backing store happen only during FFN windows
    - get() is passed through directly (reads must be synchronous)

    Example (vLLM + LMCache):
        cfg = LMCacheEngineConfig.from_defaults()
        backing = CpuMemoryBackend(cfg)
        backend = TEMPOStorageBackend(backing, TEMPOConfig())
        engine = LMCacheEngine(cfg, backend)
    """

    def __init__(self, backing, cfg: Optional[TEMPOConfig] = None) -> None:
        self._backing = backing
        self._cfg     = cfg or TEMPOConfig()

        global _libtempo
        if _libtempo is None:
            _libtempo = _load_libtempo()

        if _libtempo is not None:
            self._absorber = self._init_c_absorber()
            self._using_c  = True
            log.info("TEMPO: C++ backend active (CUPTI phase-gated pacing)")
        else:
            self._absorber = _SoftAbsorber(backing, self._cfg)
            self._using_c  = False
            log.info("TEMPO: Python soft backend active (no CUPTI)")

    # ------------------------------------------------------------------
    # LMCache StorageBackend interface
    # ------------------------------------------------------------------

    def put(self, key, kv_tensor: torch.Tensor) -> None:
        """
        Non-blocking KV eviction spike absorber.

        Called by vLLM/LMCache when GPU HBM is full and a KV block must be
        evicted. TEMPO immediately stages the tensor in the lock-free ring
        buffer and returns, freeing the GPU scheduler in O(1).

        The actual Lustre write happens asynchronously, gated by the C++
        PacingDaemon which monitors CUPTI events to ensure we never write
        during an ATTENTION window.
        """
        if self._using_c:
            self._c_absorb(key, kv_tensor)
        else:
            self._absorber.absorb(key, kv_tensor)

    def get(self, key) -> Optional[torch.Tensor]:
        """Pass-through read (not paced — reads are latency-critical)."""
        return self._backing.get(key)

    def contains(self, key) -> bool:
        return self._backing.contains(key)

    def delete(self, key) -> None:
        self._backing.delete(key)

    def close(self) -> None:
        if not self._using_c and hasattr(self._absorber, "shutdown"):
            self._absorber.shutdown()

    # ------------------------------------------------------------------
    # C++ backend initialisation helpers (called once at construction)
    # ------------------------------------------------------------------

    def _init_c_absorber(self):
        """
        Initialise the C++ SpikeAbsorber + AttentionPhaseMonitor + PacingDaemon
        via ctypes bindings to libtempo.so.
        """
        lib = _libtempo
        assert lib is not None

        # tempo_create_engine(rate_bps, burst_bytes, ffn_wait_us,
        #                     staging_dir, lustre_dir, strict_gate, verbose)
        lib.tempo_create_engine.restype  = ctypes.c_void_p
        lib.tempo_create_engine.argtypes = [
            ctypes.c_uint64,   # rate_bytes_per_sec
            ctypes.c_uint64,   # burst_bytes
            ctypes.c_uint64,   # ffn_wait_us
            ctypes.c_char_p,   # staging_dir
            ctypes.c_char_p,   # lustre_dir
            ctypes.c_int,      # strict_gate
            ctypes.c_int,      # verbose
        ]
        lib.tempo_destroy_engine.argtypes = [ctypes.c_void_p]
        lib.tempo_destroy_engine.restype  = None
        lib.tempo_absorb.argtypes = [
            ctypes.c_void_p,   # engine handle
            ctypes.c_uint64,   # block_id
            ctypes.c_void_p,   # host_ptr
            ctypes.c_size_t,   # size_bytes
        ]
        lib.tempo_absorb.restype = ctypes.c_int  # 1=ok, 0=overflow

        os.makedirs(self._cfg.staging_dir, exist_ok=True)
        os.makedirs(self._cfg.lustre_dir,  exist_ok=True)

        handle = lib.tempo_create_engine(
            int(self._cfg.rate_gbps * 1e9),
            self._cfg.burst_mb * 1024 * 1024,
            self._cfg.ffn_wait_us,
            self._cfg.staging_dir.encode(),
            self._cfg.lustre_dir.encode(),
            int(self._cfg.strict_gate),
            int(self._cfg.verbose),
        )
        if not handle:
            raise RuntimeError("tempo_create_engine returned NULL")
        return handle

    def _c_absorb(self, key, tensor: torch.Tensor) -> None:
        """Submit a KV block to the C++ SpikeAbsorber (O(1))."""
        # Ensure data is in pinned CPU memory for DMA (vLLM already does this)
        if tensor.is_cuda:
            tensor = tensor.to("cpu", non_blocking=True)

        # Use a simple content-addressed integer key
        block_id = hash(key) & 0xFFFF_FFFF_FFFF_FFFF

        ret = _libtempo.tempo_absorb(  # type: ignore[union-attr]
            self._absorber,
            ctypes.c_uint64(block_id),
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_size_t(tensor.nbytes),
        )
        if ret == 0:
            # Ring overflow: fall back to direct backing write
            log.warning("TEMPO: absorber ring overflow — direct fallback for block %x",
                        block_id)
            self._backing.put(key, tensor)

    def __del__(self) -> None:
        if self._using_c and _libtempo and hasattr(self, "_absorber"):
            _libtempo.tempo_destroy_engine(self._absorber)
