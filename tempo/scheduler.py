"""
tempo/scheduler.py — TEMPO Pacing Scheduler (Main Orchestrator)

TEMPOScheduler integrates PhaseMonitor + CheckpointManager and exposes
a clean API to the training loop.

Two operating modes:
  - mode="tempo"    : Paced flush.  Checkpoint is saved to local NVMe instantly;
                      background flush to Lustre is paused during NCCL phases.
                      → "flat" NCCL bandwidth in the Killer Graph.

  - mode="baseline" : Greedy flush.  Checkpoint is saved directly to Lustre,
                      blocking the training loop and saturating the NIC.
                      → "sawtooth" NCCL bandwidth degradation in the Killer Graph.

Recommended training loop pattern (with manual phase annotation):

    tempo = TEMPOScheduler(rank=rank, world_size=ws,
                           lustre_dir=os.environ["PSCRATCH"] + "/checkpoints",
                           mode="tempo")

    for step in range(num_steps):
        tempo.on_step_begin(step)

        with tempo.compute_phase():           # matmul phase
            outputs = model(input_ids)
            loss    = outputs.loss
            loss.backward()

        with tempo.nccl_phase():              # NCCL phase — flush paused here
            optimizer.step()                  # DDP all_reduce inside

        if step % ckpt_interval == 0:
            tempo.checkpoint(model.state_dict(), step)

    tempo.shutdown()

For FSDP (automatic phase detection via comm hook):

    tempo = TEMPOScheduler(...)
    model.register_comm_hook(tempo.phase_monitor,
                             PhaseMonitor.fsdp_comm_hook)
"""

import os
import logging
import time
from typing import Optional

from tempo.phase_monitor import PhaseMonitor, TrainingPhase
from tempo.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class TEMPOScheduler:
    """
    TEMPO Pacing Scheduler.

    Parameters
    ----------
    rank : int
        Process rank.
    world_size : int
        Total process count.
    local_nvme_dir : str
        Local NVMe staging directory (typically /tmp/tempo_ckpts).
    lustre_dir : str or None
        Lustre destination.  Required for mode="baseline" (direct flush).
    mode : {"tempo", "baseline"}
        "tempo"    → paced flush (paper contribution)
        "baseline" → greedy flush (reproduces contention for Killer Graph)
    flush_chunk_mb : int
        Chunk size for paced flush (smaller = finer throttling granularity).
    verbose : bool
        Enable per-step phase logging.
    """

    def __init__(
        self,
        rank:           int  = 0,
        world_size:     int  = 1,
        local_nvme_dir: str  = "/tmp/tempo_ckpts",
        lustre_dir:     Optional[str] = None,
        mode:           str  = "tempo",
        flush_chunk_mb: int  = 128,
        adaptive_chunk: bool = False,
        verbose:        bool = False,
    ):
        if mode not in ("tempo", "baseline"):
            raise ValueError(f"mode must be 'tempo' or 'baseline', got '{mode}'")

        self.rank       = rank
        self.world_size = world_size
        self.mode       = mode
        self.verbose    = verbose

        # Default lustre_dir from environment
        if lustre_dir is None:
            pscratch = os.environ.get("PSCRATCH")
            lustre_dir = f"{pscratch}/tempo_checkpoints" if pscratch else None

        self.lustre_dir = lustre_dir

        # ---- Build sub-components ----
        self.phase_monitor = PhaseMonitor(rank=rank, verbose=verbose)

        self.ckpt_manager = CheckpointManager(
            local_nvme_dir    = local_nvme_dir,
            lustre_dir        = lustre_dir,
            rank              = rank,
            world_size        = world_size,
            flush_chunk_bytes = flush_chunk_mb * 1024 * 1024,
            adaptive_chunk    = adaptive_chunk,
            # In "tempo" mode, pass the monitor so flush is paced.
            # In "baseline" mode, pass None so flush is greedy.
            phase_monitor     = self.phase_monitor if mode == "tempo" else None,
        )

        self._step_start: float = 0.0
        self._step_times: list  = []

        logger.info(f"[TEMPO] Rank {rank}: mode={mode}  "
                    f"lustre={lustre_dir}  chunk={flush_chunk_mb}MB")

    # ------------------------------------------------------------------
    # Training loop API
    # ------------------------------------------------------------------

    def on_step_begin(self, step: int) -> None:
        """Call at the start of every training step."""
        self.phase_monitor.increment_step()
        self._step_start = time.perf_counter()

    def on_step_end(self) -> float:
        """
        Call at the end of every training step.
        Returns the wall-clock step time in milliseconds.
        """
        elapsed_ms = (time.perf_counter() - self._step_start) * 1e3
        self._step_times.append(elapsed_ms)
        return elapsed_ms

    # Context managers delegated to PhaseMonitor
    def nccl_phase(self):
        """Context manager: marks an NCCL collective block (flush paused inside)."""
        return self.phase_monitor.nccl_phase()

    def compute_phase(self):
        """Context manager: marks a compute/matmul block (flush allowed inside)."""
        return self.phase_monitor.compute_phase()

    # ------------------------------------------------------------------
    # Checkpoint API
    # ------------------------------------------------------------------

    def checkpoint(
        self,
        state_dict: dict,
        step:       int,
        metadata:   Optional[dict] = None,
    ) -> str:
        """
        Save a checkpoint according to the current mode.

        "tempo"    → O(1) local NVMe save + schedule background flush
        "baseline" → Blocking Lustre write (greedy, causes PCIe contention)

        Returns the path where the checkpoint was written.
        """
        if self.mode == "tempo":
            with self.phase_monitor.checkpoint_phase():
                path = self.ckpt_manager.save_async(state_dict, step, metadata)
            if self.rank == 0:
                logger.info(f"[TEMPO] Step {step}: checkpoint staged locally "
                            f"(async flush to Lustre scheduled)")
            return path

        else:  # "baseline" — greedy flush
            if self.lustre_dir is None:
                raise RuntimeError("lustre_dir required for baseline mode checkpoint")
            path = self.ckpt_manager.save_sync_lustre(state_dict, step, metadata)
            if self.rank == 0:
                logger.info(f"[TEMPO BASELINE] Step {step}: checkpoint flushed "
                            f"greedily to Lustre (PCIe contention occurring)")
            return path

    # ------------------------------------------------------------------
    # Statistics & diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Returns a unified stats dict covering all TEMPO sub-components."""
        phase_stats = self.phase_monitor.get_stats()
        ckpt_stats  = self.ckpt_manager.get_stats()
        step_times  = self._step_times

        avg_step_ms = (sum(step_times) / len(step_times)) if step_times else 0.0

        return {
            "mode":           self.mode,
            "rank":           self.rank,
            "avg_step_ms":    round(avg_step_ms, 2),
            "phase":          phase_stats,
            "checkpoint":     ckpt_stats,
        }

    def print_stats(self) -> None:
        """Pretty-print statistics (rank 0 only)."""
        if self.rank != 0:
            return
        s = self.get_stats()
        print(f"\n{'='*60}")
        print(f"  TEMPO Statistics (mode={s['mode']})")
        print(f"{'='*60}")
        print(f"  Avg step time      : {s['avg_step_ms']:.2f} ms")
        print(f"  NCCL total time    : {s['phase']['nccl_total_s']:.2f} s  "
              f"({s['phase']['nccl_fraction']*100:.1f}% of runtime)")
        print(f"  Ckpt bytes (local) : {s['checkpoint']['bytes_local_GB']:.2f} GB")
        print(f"  Ckpt bytes (Lustre): {s['checkpoint']['bytes_lustre_GB']:.2f} GB")
        print(f"  Flush count        : {s['checkpoint']['flush_count']}")
        print(f"  Throttle waits     : {s['checkpoint']['throttle_waits']}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def wait_for_flushes(self, timeout: float = 600.0) -> None:
        """Block until all pending checkpoint flushes complete."""
        self.ckpt_manager.wait_for_all_flushes(timeout=timeout)

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully stop all background threads."""
        if wait:
            self.ckpt_manager.wait_for_all_flushes()
        self.ckpt_manager.shutdown(wait=wait)
        self.print_stats()
