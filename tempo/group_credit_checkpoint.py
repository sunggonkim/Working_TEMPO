"""Group-coordinated D2H staging for distributed checkpoints.

This module is deliberately small. It implements the data path exercised by
``eval/sota_4node`` instead of layering another policy on top of the removed
local-NVMe checkpoint manager. TEMPO snapshots sharded state into pinned host
memory and persists it with PyTorch Distributed Checkpoint (DCP); the harness
compares that path with each baseline's official checkpoint implementation.

The implementation currently uses two private PyTorch state-dict helpers.
They are the same helpers used by ``dcp.async_save`` in PyTorch 2.8, but their
private status is recorded as a prototype limitation in the paper artifact.
"""

from __future__ import annotations

import json
import os
import copy
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.tensor import DTensor


Policy = Literal["greedy", "local", "group"]


@dataclass
class CheckpointMetrics:
    step: int
    policy: str
    trigger_unix_ns: int
    state_bytes: int = 0
    chunks: int = 0
    credit_mb: float = 0.0
    d2h_ms: float = 0.0
    d2h_active_ms: float = 0.0
    d2h_wait_ms: float = 0.0
    dcp_write_ms: float = 0.0
    durable_ms: float = 0.0
    deadline_ms: float = 0.0
    deadline_met: bool = False
    forced_credits: int = 0
    tail_pauses: int = 0
    commit_path: str = ""
    error: str = ""


class TrainingPhase:
    """Thread-safe local signal used by the staging worker."""

    def __init__(self) -> None:
        self._compute = threading.Event()
        self._compute.set()

    def set_compute(self) -> None:
        self._compute.set()

    def set_collective(self) -> None:
        self._compute.clear()

    def is_compute(self) -> bool:
        return self._compute.is_set()


def _tensor_pairs(source: Any, destination: Any) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield corresponding local tensor leaves without gathering shards."""

    if isinstance(source, DTensor):
        if not isinstance(destination, DTensor):
            raise TypeError("DTensor destination mismatch")
        yield source.to_local(), destination.to_local()
        return
    if isinstance(source, ShardedTensor):
        if not isinstance(destination, ShardedTensor):
            raise TypeError("ShardedTensor destination mismatch")
        src_shards = source.local_shards()
        dst_shards = destination.local_shards()
        if len(src_shards) != len(dst_shards):
            raise ValueError("local shard count mismatch")
        for src_shard, dst_shard in zip(src_shards, dst_shards):
            yield src_shard.tensor, dst_shard.tensor
        return
    if isinstance(source, torch.Tensor):
        if not isinstance(destination, torch.Tensor):
            raise TypeError("Tensor destination mismatch")
        yield source, destination
        return
    if isinstance(source, dict):
        if not isinstance(destination, dict) or source.keys() != destination.keys():
            raise ValueError("state-dict key mismatch")
        for key in source:
            yield from _tensor_pairs(source[key], destination[key])
        return
    if isinstance(source, (list, tuple)):
        if not isinstance(destination, (list, tuple)) or len(source) != len(destination):
            raise ValueError("state-dict sequence mismatch")
        for src_value, dst_value in zip(source, destination):
            yield from _tensor_pairs(src_value, dst_value)


def _empty_pinned_like(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return torch.empty_like(tensor, device="cpu")
    return torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)


def _create_pinned_state_dict(value: Any) -> Any:
    """Clone checkpoint structure without reading a CUDA tensor.

    PyTorch 2.8's private ``_create_cpu_state_dict`` first calls ``.to(cpu)``
    for DTensor and ShardedTensor, then replaces the copied local storage with
    an empty tensor.  That eager transfer bypasses admission control.  Here we
    retain the distributed metadata and replace only the local storage.
    """

    if isinstance(value, DTensor):
        result = copy.copy(value)
        result._local_tensor = _empty_pinned_like(value.to_local())
        return result
    if isinstance(value, ShardedTensor):
        result = copy.copy(value)
        result._local_shards = []
        for shard in value.local_shards():
            new_shard = copy.copy(shard)
            new_shard.tensor = _empty_pinned_like(shard.tensor)
            result._local_shards.append(new_shard)
        return result
    if isinstance(value, torch.Tensor):
        return _empty_pinned_like(value)
    if isinstance(value, dict):
        return {key: _create_pinned_state_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_create_pinned_state_dict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_create_pinned_state_dict(item) for item in value)
    return copy.deepcopy(value)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class GroupCreditCheckpointer:
    """One-checkpoint-at-a-time FSDP/DCP checkpointer.

    ``start`` returns after the GPU state references and pinned destination are
    prepared.  D2H and DCP persistence run in a worker.  The training loop must
    call ``before_optimizer_step``; this is the copy-on-update boundary that
    prevents a checkpoint from observing a later optimizer update.
    """

    def __init__(
        self,
        *,
        policy: Policy,
        checkpoint_root: str | os.PathLike[str],
        rank: int,
        world_size: int,
        control_group: dist.ProcessGroup,
        dcp_group: dist.ProcessGroup,
        credit_bytes: int = 4 * 1024 * 1024,
        target_slowdown: float = 1.10,
        deadline_seconds: float = 120.0,
        assumed_d2h_gbps: float = 8.0,
        tail_window: int = 32,
        max_tail_pause_ms: float = 2.0,
        group_credit_interval_us: float = 100.0,
    ) -> None:
        if policy not in ("greedy", "local", "group"):
            raise ValueError(f"unsupported policy: {policy}")
        if credit_bytes <= 0:
            raise ValueError("credit_bytes must be positive")
        self.policy = policy
        self.root = Path(checkpoint_root)
        self.rank = rank
        self.world_size = world_size
        self.control_group = control_group
        self.dcp_group = dcp_group
        self.credit_bytes = credit_bytes
        self.target_slowdown = target_slowdown
        self.deadline_seconds = deadline_seconds
        self.assumed_d2h_gbps = assumed_d2h_gbps
        self.max_tail_pause_ms = max_tail_pause_ms
        self.group_credit_interval_us = group_credit_interval_us
        self.phase = TrainingPhase()

        self._baseline_ms: Optional[float] = None
        self._baseline_samples: list[float] = []
        self._collective_ms: deque[float] = deque(maxlen=tail_window)
        self._metric_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._staging_done = threading.Event()
        self._durable_done = threading.Event()
        self._staging_done.set()
        self._durable_done.set()
        self._worker: Optional[threading.Thread] = None
        self._metrics: Optional[CheckpointMetrics] = None
        self._error: Optional[BaseException] = None
        self._optimizer_sync_pending = False

    def observe_collective(self, latency_ms: float, baseline_sample: bool = False) -> None:
        with self._metric_lock:
            if baseline_sample:
                self._baseline_samples.append(latency_ms)
                ordered = sorted(self._baseline_samples)
                self._baseline_ms = ordered[len(ordered) // 2]
            else:
                if self._baseline_ms is None:
                    self._baseline_ms = latency_ms
                self._collective_ms.append(latency_ms)

    def _tail_ratio(self) -> float:
        with self._metric_lock:
            if not self._collective_ms or not self._baseline_ms:
                return 1.0
            ordered = sorted(self._collective_ms)
            index = max(0, min(len(ordered) - 1, int(0.99 * len(ordered))))
            return ordered[index] / self._baseline_ms

    @property
    def active(self) -> bool:
        return not self._durable_done.is_set()

    @property
    def metrics(self) -> Optional[CheckpointMetrics]:
        return self._metrics

    def start(self, state_dict: dict[str, Any], step: int) -> None:
        if not self._durable_done.is_set():
            raise RuntimeError("checkpoint backpressure: previous save is not durable")
        if self._worker is not None:
            self._worker.join()
        self._error = None
        self._staging_done.clear()
        self._durable_done.clear()
        self._optimizer_sync_pending = self.policy == "group"
        trigger_ns = time.time_ns()
        self._metrics = CheckpointMetrics(
            step=step,
            policy=self.policy,
            trigger_unix_ns=trigger_ns,
            credit_mb=self.credit_bytes / (1024 * 1024),
            deadline_ms=self.deadline_seconds * 1000.0,
        )

        # Allocation occurs before the worker so pinned-memory cost is explicit
        # trigger overhead, not accidentally charged to D2H scheduling.
        staged = _create_pinned_state_dict(state_dict)
        pairs = list(_tensor_pairs(state_dict, staged))
        self._metrics.state_bytes = sum(src.numel() * src.element_size() for src, _ in pairs)
        self._worker = threading.Thread(
            target=self._run,
            args=(pairs, staged, trigger_ns),
            name=f"tempo-credit-rank{self.rank}",
            daemon=False,
        )
        self._worker.start()

    def before_optimizer_step(self) -> float:
        """Enforce snapshot consistency and return blocking milliseconds."""
        begin = time.perf_counter()
        if not self._staging_done.is_set():
            self._staging_done.wait()
        self._raise_if_failed()
        return (time.perf_counter() - begin) * 1000.0

    def after_optimizer_step(self) -> float:
        """Align the first collective re-entry after a group snapshot."""
        if not self._optimizer_sync_pending:
            return 0.0
        begin = time.perf_counter()
        # Place the one-time rendezvous after optimizer work; otherwise
        # rank-dependent optimizer duration can recreate the skew just removed.
        dist.barrier(group=self.control_group)
        self._optimizer_sync_pending = False
        return (time.perf_counter() - begin) * 1000.0

    def wait_durable(self) -> CheckpointMetrics:
        self._durable_done.wait()
        if self._worker is not None:
            self._worker.join()
        self._raise_if_failed()
        assert self._metrics is not None
        return self._metrics

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("checkpoint worker failed") from self._error

    def _urgent(self, remaining_bytes: int, trigger_ns: int) -> bool:
        elapsed = (time.time_ns() - trigger_ns) / 1e9
        time_left = self.deadline_seconds - elapsed
        if time_left <= 0:
            return True
        required_gbps = remaining_bytes / time_left / 1e9
        return required_gbps >= 0.8 * self.assumed_d2h_gbps

    def _group_admit(
        self,
        local_safe: bool,
        urgent: bool,
        tail_ratio: float,
        tail_cooldown_complete: bool = False,
    ) -> tuple[bool, bool]:
        # One all-gather is both the credit-epoch rendezvous and the signal
        # exchange.  Entering the next epoch also implies every rank finished
        # its previous chunk, so no extra foreground barriers are needed.
        signal = torch.tensor(
            [int(local_safe), int(urgent), int(tail_ratio * 1_000_000)],
            dtype=torch.int64,
        )
        gathered = [torch.empty_like(signal) for _ in range(self.world_size)]
        dist.all_gather(gathered, signal, group=self.control_group)
        group_urgent = any(bool(item[1].item()) for item in gathered)
        group_safe = all(bool(item[0].item()) for item in gathered)
        max_tail = max(item[2].item() for item in gathered) / 1_000_000
        group_tail_ok = (
            max_tail <= self.target_slowdown
            or tail_cooldown_complete
        )
        return group_urgent or (group_safe and group_tail_ok), group_urgent

    def _group_plan(
        self,
        *,
        local_chunks: int,
        trigger_ns: int,
    ) -> tuple[int, int, int, bool]:
        """Agree once on an aligned open-loop credit plan.

        Per-credit rendezvous is intentionally avoided: on 16 ranks its Gloo
        latency can exceed the D2H work being controlled and can itself create
        arrival skew.  A conservative short delay relative to the all-gather
        return gives every worker nearly
        identical credit boundaries without assuming synchronized host clocks.
        """
        tail_ratio = self._tail_ratio()
        urgent = self._urgent(self._metrics.state_bytes, trigger_ns)  # type: ignore[union-attr]
        signal = torch.tensor(
            [
                local_chunks,
                int(self.phase.is_compute()),
                int(urgent),
                int(tail_ratio * 1_000_000),
            ],
            dtype=torch.int64,
        )
        gathered = [torch.empty_like(signal) for _ in range(self.world_size)]
        dist.all_gather(gathered, signal, group=self.control_group)
        max_chunks = max(int(item[0].item()) for item in gathered)
        all_safe = all(bool(item[1].item()) for item in gathered)
        group_urgent = any(bool(item[2].item()) for item in gathered)
        max_tail = max(item[3].item() for item in gathered) / 1_000_000
        start_ns = time.monotonic_ns() + 5_000_000
        if not all_safe and not group_urgent:
            start_ns += int(self.max_tail_pause_ms * 1e6)
        interval_us = self.group_credit_interval_us
        if max_tail > self.target_slowdown and not group_urgent:
            interval_us *= 2.0
            self._metrics.tail_pauses += 1  # type: ignore[union-attr]
        interval_ns = max(1, int(interval_us * 1000.0))
        return max_chunks, start_ns, interval_ns, group_urgent

    def _run(
        self,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
        staged: dict[str, Any],
        trigger_ns: int,
    ) -> None:
        assert self._metrics is not None
        try:
            d2h_begin = time.perf_counter()
            active_seconds = 0.0
            stream = torch.cuda.Stream()
            work: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
            for source, destination in pairs:
                if source.numel() == 0:
                    continue
                src = source.reshape(-1)
                dst = destination.reshape(-1)
                elements_per_credit = max(1, self.credit_bytes // source.element_size())
                for offset in range(0, source.numel(), elements_per_credit):
                    count = min(elements_per_credit, source.numel() - offset)
                    work.append((src, dst, offset, count))

            local_chunks = len(work)
            if self.policy == "group":
                max_chunks, plan_start_ns, interval_ns, plan_forced = self._group_plan(
                    local_chunks=local_chunks,
                    trigger_ns=trigger_ns,
                )
                self._metrics.forced_credits += int(plan_forced)
            else:
                max_chunks = local_chunks
                plan_start_ns = 0
                interval_ns = 0

            for chunk_index in range(max_chunks):
                remaining = max(0, self._metrics.state_bytes - chunk_index * self.credit_bytes)
                urgent = self._urgent(remaining, trigger_ns)
                local_safe = self.phase.is_compute()
                tail_ratio = self._tail_ratio()

                if self.policy == "greedy":
                    admitted = True
                    forced = False
                elif self.policy == "local":
                    forced = urgent
                    admitted = urgent or (local_safe and tail_ratio <= self.target_slowdown)
                    pause_begin = time.perf_counter()
                    while not admitted:
                        self._metrics.tail_pauses += int(tail_ratio > self.target_slowdown)
                        time.sleep(0.0005)
                        urgent = self._urgent(remaining, trigger_ns)
                        local_safe = self.phase.is_compute()
                        tail_ratio = self._tail_ratio()
                        forced = urgent
                        cooldown_complete = (
                            (time.perf_counter() - pause_begin) * 1000.0
                            >= self.max_tail_pause_ms
                        )
                        admitted = urgent or (
                            local_safe
                            and (tail_ratio <= self.target_slowdown or cooldown_complete)
                        )
                else:
                    forced = plan_forced
                    target_ns = plan_start_ns + chunk_index * interval_ns
                    while time.monotonic_ns() < target_ns:
                        time.sleep(0.00002)
                    admitted = True

                if forced:
                    self._metrics.forced_credits += 1
                if chunk_index < local_chunks:
                    src, dst, offset, count = work[chunk_index]
                    copy_begin = time.perf_counter()
                    with torch.cuda.stream(stream):
                        dst[offset : offset + count].copy_(
                            src[offset : offset + count], non_blocking=True
                        )
                    stream.synchronize()
                    active_seconds += time.perf_counter() - copy_begin
                    self._metrics.chunks += 1

            self._metrics.d2h_ms = (time.perf_counter() - d2h_begin) * 1000.0
            self._metrics.d2h_active_ms = active_seconds * 1000.0
            self._metrics.d2h_wait_ms = self._metrics.d2h_ms - self._metrics.d2h_active_ms
            self._staging_done.set()

            checkpoint_dir = self.root / f"step_{self._metrics.step:07d}"
            self._metrics.commit_path = str(checkpoint_dir / "TEMPO_COMMITTED.json")
            write_begin = time.perf_counter()
            writer = dcp.FileSystemWriter(
                checkpoint_dir,
                single_file_per_rank=True,
                sync_files=True,
                thread_count=1,
                overwrite=True,
            )
            dcp.save(staged, storage_writer=writer, process_group=self.dcp_group)
            self._metrics.dcp_write_ms = (time.perf_counter() - write_begin) * 1000.0

            dist.barrier(group=self.dcp_group)
            if self.rank == 0:
                commit = checkpoint_dir / "TEMPO_COMMITTED.json"
                _atomic_json(
                    commit,
                    {
                        "format": "tempo-dcp-v1",
                        "step": self._metrics.step,
                        "world_size": self.world_size,
                        "policy": self.policy,
                        "state_bytes_per_rank_rank0": self._metrics.state_bytes,
                        "dcp_metadata": ".metadata",
                    },
                )
            dist.barrier(group=self.dcp_group)
            self._metrics.durable_ms = (time.time_ns() - trigger_ns) / 1e6
            self._metrics.deadline_met = self._metrics.durable_ms <= self._metrics.deadline_ms
        except BaseException as exc:
            self._error = exc
            self._metrics.error = repr(exc)
            self._staging_done.set()
        finally:
            self._durable_done.set()

    def write_rank_metrics(self, output_path: str | os.PathLike[str]) -> None:
        metrics = self.wait_durable()
        _atomic_json(Path(output_path), asdict(metrics))
