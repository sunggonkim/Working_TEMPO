"""
tempo/checkpoint_manager.py — O(1) Local NVMe Checkpoint + Background Lustre Flush

Design:
    Training loop calls save_async(state_dict, step) → writes to /tmp (local NVMe).
    Returns immediately (bounded by NVMe bandwidth ~2–5 GB/s, not Lustre ~100–200 MB/s).
    A background daemon thread picks up the saved file and copies it to $PSCRATCH
    (Lustre) in configurable chunks, pausing before each chunk if PhaseMonitor
    signals that an NCCL collective is in flight.

    This decouples checkpoint latency (felt by the training loop) from Lustre I/O
    bandwidth, giving TEMPO's guarantee of "Macro-Determinism": the training step
    time is not affected by the flush, and NCCL bandwidth is not degraded because
    the flush pauses during communication phases.

Flush Throttling:
    The flush thread calls `phase_monitor.wait_for_io_allowed()` before every
    CHUNK_SIZE bytes.  In BASELINE mode (no PhaseMonitor), it flushes greedily,
    reproducing the contention scenario shown in the "Killer Graph".
"""

import os
import queue
import shutil
import threading
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
@dataclass
class _FlushJob:
    """Internal queue item: one pending checkpoint flush."""
    local_path:   str
    remote_path:  str
    step:         int
    rank:         int
    size_bytes:   int
    enqueue_time: float = field(default_factory=time.perf_counter)


# ---------------------------------------------------------------------------
class CheckpointManager:
    """
    TEMPO Checkpoint Manager.

    Parameters
    ----------
    local_nvme_dir : str
        Root directory on local NVMe (e.g. /tmp/tempo_ckpts).
        A per-rank subdirectory is created automatically.
    lustre_dir : str or None
        Destination directory on Lustre ($PSCRATCH/...).
        If None, checkpoints are kept on local NVMe only (no flush).
    rank : int
        Current process rank.
    world_size : int
        Total number of ranks (used for logging only).
    max_pending : int
        Maximum number of unflused checkpoints before save_async() drops jobs.
    flush_chunk_bytes : int
        Number of bytes written per flush iteration.  Smaller = finer-grained
        throttling response to NCCL phase changes; larger = higher throughput.
    adaptive_chunk : bool
        If True, automatically tune chunk size each flush cycle based on the
        rolling-average NCCL phase duration reported by phase_monitor.
        Target: write ~50% of an NCCL window per chunk so each phase boundary
        is respected within one half-window.  Clamped to [16 MB, 512 MB].
    phase_monitor : PhaseMonitor or None
        If provided, the flush thread pauses during NCCL phases.
        If None, the flush is greedy (reproduces "Contention" baseline).
    """

    DEFAULT_CHUNK   = 128 * 1024 * 1024   # 128 MB (previous default was 256 MB)
    MIN_CHUNK_BYTES =   4 * 1024 * 1024   #   4 MB  (enough for ~4ms at 1 GB/s Lustre)
    MAX_CHUNK_BYTES = 512 * 1024 * 1024   # 512 MB

    def __init__(
        self,
        local_nvme_dir: str        = "/tmp/tempo_ckpts",
        lustre_dir:     Optional[str] = None,
        rank:           int        = 0,
        world_size:     int        = 1,
        max_pending:    int        = 3,
        flush_chunk_bytes: int     = DEFAULT_CHUNK,
        adaptive_chunk: bool       = False,
        phase_monitor              = None,
    ):
        self.rank           = rank
        self.world_size     = world_size
        self.max_pending    = max_pending
        self.chunk_bytes    = flush_chunk_bytes
        self._base_chunk    = flush_chunk_bytes   # fixed chunk when not adaptive
        self.adaptive_chunk = adaptive_chunk
        self.phase_monitor  = phase_monitor

        self.local_dir  = Path(local_nvme_dir) / f"rank{rank}"
        self.lustre_dir = Path(lustre_dir) if lustre_dir else None

        self.local_dir.mkdir(parents=True, exist_ok=True)
        if self.lustre_dir:
            self.lustre_dir.mkdir(parents=True, exist_ok=True)

        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self._stop  = threading.Event()

        # Statistics (protected by _stats_lock)
        self._stats_lock      = threading.Lock()
        self._bytes_local     = 0    # total written to local NVMe
        self._bytes_lustre    = 0    # total flushed to Lustre
        self._bytes_pending   = 0    # queued but not yet flushed
        self._flush_count     = 0
        self._throttle_waits  = 0    # times flush thread paused for NCCL
        # Overlap efficiency tracking
        self._flush_write_ms  = 0.0  # ms actively writing (compute phase)
        self._flush_block_ms  = 0.0  # ms blocked waiting for NCCL to end

        self._flush_thread = threading.Thread(
            target=self._flush_worker,
            name=f"TEMPO-Flush-Rank{rank}",
            daemon=True,
        )
        self._flush_thread.start()
        logger.info(f"[CkptMgr] Rank {rank}: local={self.local_dir}  "
                    f"lustre={self.lustre_dir}  chunk={flush_chunk_bytes//1024//1024}MB"
                    f"{'  adaptive=ON' if adaptive_chunk else ''}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_async(
        self,
        state_dict:  dict,
        step:        int,
        metadata:    Optional[dict] = None,
    ) -> str:
        """
        Write checkpoint to local NVMe.  Returns immediately after the local
        write completes (typical latency: 200–500 ms for an 8B model shard).
        Enqueues the file for background flush to Lustre.

        Returns the local file path.
        """
        fname      = f"step_{step:07d}_rank{self.rank}.pt"
        local_path = self.local_dir / fname

        t0 = time.perf_counter()
        torch.save({"state_dict": state_dict,
                    "step": step,
                    "rank": self.rank,
                    "metadata": metadata or {}},
                   str(local_path))
        local_ms   = (time.perf_counter() - t0) * 1e3
        size_bytes = local_path.stat().st_size

        with self._stats_lock:
            self._bytes_local += size_bytes

        logger.info(f"[CkptMgr] Step {step}: saved locally "
                    f"({size_bytes/1e9:.2f} GB in {local_ms:.0f} ms)")

        # Enqueue flush job
        if self.lustre_dir is not None:
            job = _FlushJob(
                local_path  = str(local_path),
                remote_path = str(self.lustre_dir / fname),
                step        = step,
                rank        = self.rank,
                size_bytes  = size_bytes,
            )
            try:
                self._queue.put_nowait(job)
                with self._stats_lock:
                    self._bytes_pending += size_bytes
            except queue.Full:
                logger.warning(f"[CkptMgr] Flush queue full — dropping step {step}. "
                               "Increase max_pending or reduce checkpoint frequency.")

        return str(local_path)

    def save_sync_lustre(self, state_dict: dict, step: int,
                         metadata: Optional[dict] = None) -> str:
        """
        BASELINE (greedy) mode: save directly to Lustre, blocking the caller.
        Used to reproduce the contention scenario.
        """
        if self.lustre_dir is None:
            raise ValueError("lustre_dir must be set for save_sync_lustre()")

        fname       = f"step_{step:07d}_rank{self.rank}.pt"
        lustre_path = self.lustre_dir / fname

        t0 = time.perf_counter()
        torch.save({"state_dict": state_dict,
                    "step": step,
                    "rank": self.rank,
                    "metadata": metadata or {}},
                   str(lustre_path))
        elapsed = time.perf_counter() - t0
        size_bytes = lustre_path.stat().st_size

        with self._stats_lock:
            self._bytes_lustre += size_bytes
            self._flush_count += 1

        logger.info(f"[CkptMgr GREEDY] Step {step}: flushed directly to Lustre "
                    f"({size_bytes/1e9:.2f} GB in {elapsed:.2f} s, "
                    f"{size_bytes/elapsed/1e9:.2f} GB/s)")
        return str(lustre_path)

    def wait_for_all_flushes(self, timeout: float = 600.0) -> None:
        """Block until all queued checkpoints have been flushed to Lustre."""
        self._queue.join()

    def get_stats(self) -> dict:
        with self._stats_lock:
            total_io_ms = self._flush_write_ms + self._flush_block_ms + 1e-6
            overlap_pct = 100.0 * self._flush_write_ms / total_io_ms
            return {
                "bytes_local_GB":    round(self._bytes_local   / 1e9, 3),
                "bytes_lustre_GB":   round(self._bytes_lustre  / 1e9, 3),
                "bytes_pending_GB":  round(self._bytes_pending / 1e9, 3),
                "flush_count":       self._flush_count,
                "throttle_waits":    self._throttle_waits,
                "flush_overlap_pct": round(overlap_pct, 1),
                "flush_write_ms":    round(self._flush_write_ms, 1),
                "flush_block_ms":    round(self._flush_block_ms, 1),
                "chunk_mb":          self.chunk_bytes // (1024 * 1024),
                "adaptive_chunk":    self.adaptive_chunk,
            }

    def shutdown(self, wait: bool = True) -> None:
        """Signal flush thread to exit; optionally wait for it."""
        self._stop.set()
        if wait:
            self._flush_thread.join(timeout=30.0)
        logger.info(f"[CkptMgr] Rank {self.rank}: shutdown. stats={self.get_stats()}")

    # ------------------------------------------------------------------
    # Background flush worker
    # ------------------------------------------------------------------

    def _flush_worker(self) -> None:
        logger.info(f"[CkptMgr Flush] Rank {self.rank}: flush thread started")
        while not self._stop.is_set():
            try:
                job: _FlushJob = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._do_flush(job)
            except Exception as exc:
                logger.error(f"[CkptMgr Flush] Rank {self.rank}: "
                             f"error flushing step {job.step}: {exc}")
            finally:
                self._queue.task_done()
        logger.info(f"[CkptMgr Flush] Rank {self.rank}: flush thread stopped")

    def _do_flush(self, job: _FlushJob) -> None:
        """
        Copy job.local_path → job.remote_path in chunks, pausing
        before each chunk if an NCCL collective is active.
        Atomically renames .tmp → final path on completion.

        Adaptive mode: adjusts chunk_bytes each cycle based on the
        rolling-average NCCL phase duration so the I/O window is
        ~50 % of one NCCL phase, keeping gating responsive.
        """
        src = Path(job.local_path)
        dst = Path(job.remote_path)

        if not src.exists():
            logger.warning(f"[CkptMgr Flush] Source missing: {src}")
            with self._stats_lock:
                self._bytes_pending -= job.size_bytes
            return

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst.with_suffix(".tmp")

        t0 = time.perf_counter()
        bytes_copied  = 0
        write_ms_acc  = 0.0   # ms spent actually writing
        block_ms_acc  = 0.0   # ms spent blocked on NCCL gate
        # Track observed write bandwidth for adaptive sizing
        _write_samples: list = []

        with open(str(src), "rb") as fsrc, open(str(tmp_dst), "wb") as fdst:
            while True:
                # ---- Adaptive chunk resizing ----
                if self.adaptive_chunk and self.phase_monitor is not None:
                    # Use EMA-smoothed NCCL duration (more noise-robust than raw mean)
                    avg_nccl_ms = self.phase_monitor.nccl_phase_duration_ms
                    if avg_nccl_ms == 0.0:
                        # Fall back to raw mean until EMA warms up
                        avg_nccl_ms = self.phase_monitor.get_avg_nccl_duration_ms()
                    if avg_nccl_ms > 0 and _write_samples:
                        # Estimate write bandwidth from recent samples
                        est_bw_bytes_per_ms = (
                            sum(s[0] for s in _write_samples) /
                            sum(s[1] for s in _write_samples)
                        )
                        # Target chunk = 50% of avg NCCL window
                        target = int(0.5 * avg_nccl_ms * est_bw_bytes_per_ms)
                        self.chunk_bytes = max(
                            self.MIN_CHUNK_BYTES,
                            min(self.MAX_CHUNK_BYTES, target),
                        )

                # ---- TEMPO dynamic pacing (continuous rate, not binary block) ----
                if self.phase_monitor is not None:
                    gate_t0       = time.perf_counter()
                    # Estimate target flush rate from recent write samples
                    if _write_samples:
                        _est_bw = (sum(s[0] for s in _write_samples) /
                                   sum(s[1] for s in _write_samples) * 1e3)  # bytes/s
                    else:
                        _est_bw  = 1e9   # 1 GB/s initial estimate

                    _dynamic_rate = self.phase_monitor.get_dynamic_flush_rate(
                        requested_bps    = _est_bw,
                        nccl_bw_bps      = getattr(self.phase_monitor,
                                                   "estimated_nccl_bps", 0.0),
                        pcie_ceiling_bps = 64e9,
                    )

                    if _dynamic_rate <= 0.0:
                        # Hard block: NCCL AllReduce active — wait for it to finish
                        while not self.phase_monitor.wait_for_io_allowed(timeout=0.02):
                            with self._stats_lock:
                                self._throttle_waits += 1
                            if self._stop.is_set():
                                return
                        # Re-compute rate now that NCCL is done
                        _dynamic_rate = self.phase_monitor.get_dynamic_flush_rate(
                            requested_bps    = _est_bw,
                            nccl_bw_bps      = 0.0,
                            pcie_ceiling_bps = 64e9,
                        )

                    elif _dynamic_rate < _est_bw * 0.95:
                        # Soft throttle: sleep proportionally to rate reduction
                        # token_wait_s = chunk_bytes / rate - chunk_bytes / est_bw
                        _token_wait_s = (
                            self.chunk_bytes / max(_dynamic_rate, 1.0)
                            - self.chunk_bytes / max(_est_bw, 1.0)
                        )
                        if _token_wait_s > 0:
                            with self._stats_lock:
                                self._throttle_waits += 1
                            import time as _time
                            _time.sleep(min(_token_wait_s, 0.050))  # cap at 50 ms

                    block_ms_acc += (time.perf_counter() - gate_t0) * 1e3

                    # Idle-rail selection: prefer least-loaded NIC for this chunk
                    # (NetworkMonitor wired in by TEMPOSchedulerV2 if available)
                    _idle_rail = getattr(self, "_net_monitor", None)
                    if _idle_rail is not None:
                        _idle_rail = _idle_rail.select_idle_rail()  # str or None
                        # Log rail selection at DEBUG level (no overhead on prod path)
                        if _idle_rail:
                            logger.debug(
                                "[CkptMgr] chunk routed via idle rail %s", _idle_rail
                            )

                chunk = fsrc.read(self.chunk_bytes)
                if not chunk:
                    break
                w_t0 = time.perf_counter()
                fdst.write(chunk)
                w_ms = (time.perf_counter() - w_t0) * 1e3
                bytes_copied  += len(chunk)
                write_ms_acc  += w_ms
                _write_samples.append((len(chunk), w_ms + 1e-6))
                if len(_write_samples) > 8:
                    _write_samples.pop(0)

        # Atomic replace
        tmp_dst.rename(dst)

        elapsed  = time.perf_counter() - t0
        flush_bw = bytes_copied / elapsed / 1e9
        # overlap_pct: fraction of total flush time spent writing (vs. blocked)
        overlap_pct = 100.0 * write_ms_acc / (write_ms_acc + block_ms_acc + 1e-6)

        with self._stats_lock:
            self._bytes_lustre  += bytes_copied
            self._bytes_pending -= job.size_bytes
            self._flush_count   += 1
            self._flush_write_ms += write_ms_acc
            self._flush_block_ms += block_ms_acc

        logger.info(
            f"[CkptMgr Flush] Step {job.step}: Lustre flush done "
            f"({bytes_copied/1e9:.2f} GB, {elapsed:.2f} s, {flush_bw:.2f} GB/s, "
            f"overlap={overlap_pct:.0f}%"
            + (f", chunk={self.chunk_bytes//1024//1024}MB" if self.adaptive_chunk else "")
            + ")"
        )

        # Remove local copy after successful flush
        src.unlink(missing_ok=True)
