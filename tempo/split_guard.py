"""Stage-split admission primitives for the next TEMPO experiment.

This module deliberately contains no PyTorch, CUDA, distributed, or filesystem
code.  It is the small state machine that the runtime adapter must preserve:

* D2H admission is causal and local to a real CUDA execution interval.
* A submitted D2H grant is the only non-preemptible residual (one quantum).
* Host-ready PFS work is not tied to an FSDP phase and is scheduled by a
  work-conserving node pool with deadline priority and byte fairness.

The runtime integration is expected to call these objects from the existing
admission mutex / stream-token path.  Keeping the arithmetic here independent
of the runtime makes the dangerous policy regressions testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class D2HGuardState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"


@dataclass(frozen=True)
class D2HSnapshot:
    state: D2HGuardState
    admitted_bytes: int
    completed_bytes: int
    inflight_bytes: int
    residual_bytes: int
    grants: int


class D2HCausalGuard:
    """One-rank causal D2H admission with a bounded non-preemptible tail.

    ``open_interval`` and ``close_interval`` correspond to stream-ordered
    tokens.  A new grant is accepted only while open and only when no previous
    grant is in flight.  The latter is intentionally conservative: it makes
    the residual bound explicit instead of hiding multiple outstanding copies
    behind a phase budget.
    """

    def __init__(self, quantum_bytes: int = 1 << 20) -> None:
        if quantum_bytes <= 0:
            raise ValueError("quantum_bytes must be positive")
        self.quantum_bytes = int(quantum_bytes)
        self._state = D2HGuardState.CLOSED
        self._admitted = 0
        self._completed = 0
        self._inflight = 0
        self._grants = 0

    @property
    def state(self) -> D2HGuardState:
        return self._state

    def open_interval(self) -> None:
        if self._state is D2HGuardState.OPEN:
            raise RuntimeError("D2H interval is already open")
        if self._inflight:
            raise RuntimeError("cannot open D2H interval with an in-flight grant")
        self._state = D2HGuardState.OPEN
        self._check()

    def close_interval(self) -> None:
        if self._state is D2HGuardState.CLOSED:
            raise RuntimeError("D2H interval is already closed")
        self._state = D2HGuardState.CLOSED
        self._check()

    def admit(self, remaining_bytes: int) -> int:
        if remaining_bytes < 0:
            raise ValueError("remaining_bytes must be nonnegative")
        if self._state is D2HGuardState.CLOSED or self._inflight:
            return 0
        grant = min(self.quantum_bytes, int(remaining_bytes))
        if grant <= 0:
            return 0
        self._inflight = grant
        self._admitted += grant
        self._grants += 1
        self._check()
        return grant

    def complete(self, completed_bytes: int | None = None) -> int:
        if not self._inflight:
            raise RuntimeError("D2H completion has no matching grant")
        amount = self._inflight if completed_bytes is None else int(completed_bytes)
        if amount != self._inflight:
            raise ValueError("partial D2H completion must be retried before final completion")
        self._completed += amount
        self._inflight = 0
        self._check()
        return amount

    def snapshot(self) -> D2HSnapshot:
        return D2HSnapshot(
            state=self._state,
            admitted_bytes=self._admitted,
            completed_bytes=self._completed,
            inflight_bytes=self._inflight,
            residual_bytes=self._inflight,
            grants=self._grants,
        )

    def _check(self) -> None:
        if self._admitted < self._completed:
            raise AssertionError("D2H completed bytes exceed admitted bytes")
        if self._inflight < 0 or self._inflight > self.quantum_bytes:
            raise AssertionError("D2H residual exceeds one quantum")
        if self._completed + self._inflight != self._admitted:
            raise AssertionError("D2H byte conservation violated")


@dataclass(frozen=True)
class PFSRequest:
    request_id: int
    rank: int
    bytes: int
    deadline_ns: int
    enqueued_ns: int


@dataclass(frozen=True)
class PFSGrant:
    request_id: int
    rank: int
    bytes: int
    deadline_ns: int


@dataclass(frozen=True)
class PFSLaneSnapshot:
    queued_bytes: int
    inflight_bytes: int
    inflight_requests: int
    completed_bytes: int
    completed_requests: int
    per_rank_completed_bytes: tuple[tuple[int, int], ...]


class NodePFSLane:
    """Work-conserving node-local PFS grant scheduler.

    The lane accounts the shared node cap, but keeps rank identity in the
    queue.  Earliest laxity is the primary key; completed-byte deficit is the
    tie-breaker so a rank cannot be starved by a stream of requests from a
    peer.  No FSDP phase or predicted gap is consulted.
    """

    def __init__(
        self,
        *,
        quantum_bytes: int = 4 << 20,
        max_inflight_bytes: int = 64 << 20,
        max_inflight_requests: int = 16,
    ) -> None:
        if quantum_bytes <= 0:
            raise ValueError("quantum_bytes must be positive")
        if max_inflight_bytes < quantum_bytes or max_inflight_bytes % quantum_bytes:
            raise ValueError("max_inflight_bytes must contain whole quantums")
        if max_inflight_requests <= 0:
            raise ValueError("max_inflight_requests must be positive")
        self.quantum_bytes = int(quantum_bytes)
        self.max_inflight_bytes = int(max_inflight_bytes)
        self.max_inflight_requests = int(max_inflight_requests)
        self._next_id = 1
        self._queue: list[PFSRequest] = []
        self._inflight: dict[int, PFSGrant] = {}
        self._completed_bytes = 0
        self._completed_requests = 0
        self._per_rank_completed: dict[int, int] = {}

    def submit(self, *, rank: int, bytes: int, deadline_ns: int, now_ns: int) -> int:
        size = int(bytes)
        if rank < 0:
            raise ValueError("rank must be nonnegative")
        if size <= 0 or size > self.quantum_bytes:
            raise ValueError("PFS request must be in (0, quantum_bytes]")
        if deadline_ns < now_ns:
            raise ValueError("deadline must not precede enqueue time")
        request = PFSRequest(self._next_id, int(rank), size, int(deadline_ns), int(now_ns))
        self._next_id += 1
        self._queue.append(request)
        self._check()
        return request.request_id

    def grant_ready(self, *, now_ns: int, limit: int | None = None) -> tuple[PFSGrant, ...]:
        """Grant as many queued requests as current node capacity allows."""

        grants: list[PFSGrant] = []
        target = len(self._queue) if limit is None else max(0, int(limit))
        while self._queue and len(grants) < target:
            if len(self._inflight) >= self.max_inflight_requests:
                break
            used = sum(item.bytes for item in self._inflight.values())
            available = self.max_inflight_bytes - used
            eligible = [item for item in self._queue if item.bytes <= available]
            if not eligible:
                break
            selected = min(
                eligible,
                key=lambda item: (
                    max(0, item.deadline_ns - int(now_ns)),
                    self._per_rank_completed.get(item.rank, 0),
                    item.enqueued_ns,
                    item.request_id,
                ),
            )
            self._queue.remove(selected)
            grant = PFSGrant(
                selected.request_id,
                selected.rank,
                selected.bytes,
                selected.deadline_ns,
            )
            self._inflight[grant.request_id] = grant
            grants.append(grant)
        self._check()
        return tuple(grants)

    def complete(self, request_id: int) -> PFSGrant:
        try:
            grant = self._inflight.pop(int(request_id))
        except KeyError as exc:
            raise RuntimeError("unknown PFS completion") from exc
        self._completed_bytes += grant.bytes
        self._completed_requests += 1
        self._per_rank_completed[grant.rank] = (
            self._per_rank_completed.get(grant.rank, 0) + grant.bytes
        )
        self._check()
        return grant

    def snapshot(self) -> PFSLaneSnapshot:
        return PFSLaneSnapshot(
            queued_bytes=sum(item.bytes for item in self._queue),
            inflight_bytes=sum(item.bytes for item in self._inflight.values()),
            inflight_requests=len(self._inflight),
            completed_bytes=self._completed_bytes,
            completed_requests=self._completed_requests,
            per_rank_completed_bytes=tuple(sorted(self._per_rank_completed.items())),
        )

    def _check(self) -> None:
        snapshot = self.snapshot()
        if snapshot.inflight_bytes > self.max_inflight_bytes:
            raise AssertionError("PFS node byte cap exceeded")
        if snapshot.inflight_requests > self.max_inflight_requests:
            raise AssertionError("PFS node request cap exceeded")
        if any(item.bytes <= 0 or item.bytes > self.quantum_bytes for item in self._queue):
            raise AssertionError("queued PFS request exceeds physical quantum")
        if any(item.bytes <= 0 or item.bytes > self.quantum_bytes for item in self._inflight.values()):
            raise AssertionError("inflight PFS request exceeds physical quantum")


def drain_requests(lane: NodePFSLane, request_ids: Iterable[int]) -> int:
    """Complete a known request sequence and return completed bytes."""

    total = 0
    for request_id in request_ids:
        total += lane.complete(request_id).bytes
    return total


def transition_deltas(
    *,
    phase_id: int,
    first_phase_id: int,
    is_compute: bool,
    d2h_quantum_bytes: int,
    event_pfs_bytes: int,
) -> tuple[int, int]:
    """Return the stream-token deltas for the minimal SplitGuard path.

    Phase zero is a preparation boundary, not a predicted compute gap.  It
    opens the continuous PFS ceiling, while every later real compute token
    gets one causal D2H quantum.  Collective tokens get no new D2H credit.
    """

    if phase_id < first_phase_id:
        raise ValueError("phase_id must not precede first_phase_id")
    if d2h_quantum_bytes <= 0 or event_pfs_bytes <= 0:
        raise ValueError("SplitGuard byte sizes must be positive")
    pfs_delta = event_pfs_bytes if phase_id == first_phase_id else 0
    d2h_delta = (
        d2h_quantum_bytes
        if phase_id > first_phase_id and is_compute
        else 0
    )
    return d2h_delta, pfs_delta
