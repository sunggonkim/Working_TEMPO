"""Deadline-feasible admission planning for TEMPO v4.

This module is intentionally independent of PyTorch and DataStates.  It turns
group-wide checkpoint progress plus a predicted sequence of training windows
into deterministic, per-rank, per-phase D2H and persistence byte budgets.

The planner controls *admission*, not completion.  A DataStates worker may
already have one D2H chunk and a bounded amount of persistence I/O in flight.
Those residuals must be reported separately when interpreting a plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math
from typing import Iterable, Sequence


NS_PER_SECOND = 1_000_000_000
PPM = 1_000_000
UNLIMITED_BUDGET = (1 << 64) - 1


class AdmissionMode(str, Enum):
    """Controller states that are safe to persist in experiment artifacts."""

    PROFILE = "PROFILE"
    PROTECT = "PROTECT"
    BALANCED = "BALANCED"
    DRAIN = "DRAIN"
    FINALIZE = "FINALIZE"


class WindowKind(str, Enum):
    COMPUTE = "compute"
    COLLECTIVE = "collective"


class Stage(str, Enum):
    D2H = "d2h"
    PFS = "pfs"


def _require_nonnegative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (numerator + denominator - 1) // denominator


def _floor_quantum(value: int, quantum: int) -> int:
    return value // quantum * quantum


def _ceil_quantum(value: int, quantum: int) -> int:
    if value <= 0:
        return 0
    return _ceil_div(value, quantum) * quantum


def _request_aligned_target(unadmitted: int, target: int, quantum: int) -> int:
    """Round a horizon target to requests without crossing the event tail.

    Credit is checked against a whole request by the worker.  An arbitrary
    sub-quantum horizon target therefore cannot admit useful work unless it is
    the event's exact final partial request.  Keep intermediate targets in
    whole request quanta and authorize the exact remainder only when the plan
    reaches all currently unadmitted bytes.
    """

    if target <= 0 or unadmitted <= 0:
        return 0
    if target >= unadmitted:
        return unadmitted
    return min(unadmitted, _ceil_quantum(target, quantum))


def _request_aligned_limit(unadmitted: int, available: int, quantum: int) -> int:
    """Return the useful request bytes wholly covered by ``available``.

    Unlike a horizon target, an availability limit must never round up.  The
    one exception is the event's exact final partial request, which is useful
    only when every byte of that tail is already available.
    """

    if unadmitted <= 0 or available <= 0:
        return 0
    if available >= unadmitted:
        return unadmitted
    return min(unadmitted, _floor_quantum(available, quantum))


def _service_ns(byte_count: int, rate_bytes_per_second: int) -> int:
    if byte_count <= 0:
        return 0
    if rate_bytes_per_second <= 0:
        return math.inf  # type: ignore[return-value]
    return _ceil_div(byte_count * NS_PER_SECOND, rate_bytes_per_second)


class _SuffixMinHeadroom:
    """Range-add/suffix-min tracker for lagged producer prefix credit."""

    def __init__(self, values: Sequence[int]) -> None:
        size = 1
        while size < len(values):
            size *= 2
        self._size = size
        self._minimum = [UNLIMITED_BUDGET] * (2 * size)
        self._lazy = [0] * (2 * size)
        for index, value in enumerate(values):
            self._minimum[size + index] = int(value)
        for index in range(size - 1, 0, -1):
            self._minimum[index] = min(
                self._minimum[2 * index], self._minimum[2 * index + 1]
            )

    def _apply(self, node: int, delta: int) -> None:
        self._minimum[node] += delta
        self._lazy[node] += delta

    def _push(self, node: int) -> None:
        delta = self._lazy[node]
        if delta:
            self._apply(2 * node, delta)
            self._apply(2 * node + 1, delta)
            self._lazy[node] = 0

    def _add(
        self,
        node: int,
        left: int,
        right: int,
        query_left: int,
        query_right: int,
        delta: int,
    ) -> None:
        if query_left <= left and right <= query_right:
            self._apply(node, delta)
            return
        self._push(node)
        middle = (left + right) // 2
        if query_left <= middle:
            self._add(2 * node, left, middle, query_left, query_right, delta)
        if query_right > middle:
            self._add(2 * node + 1, middle + 1, right, query_left, query_right, delta)
        self._minimum[node] = min(
            self._minimum[2 * node], self._minimum[2 * node + 1]
        )

    def _query(
        self,
        node: int,
        left: int,
        right: int,
        query_left: int,
        query_right: int,
    ) -> int:
        if query_left <= left and right <= query_right:
            return self._minimum[node]
        self._push(node)
        middle = (left + right) // 2
        result = UNLIMITED_BUDGET
        if query_left <= middle:
            result = min(
                result,
                self._query(2 * node, left, middle, query_left, query_right),
            )
        if query_right > middle:
            result = min(
                result,
                self._query(
                    2 * node + 1, middle + 1, right, query_left, query_right
                ),
            )
        return result

    def available_from(self, index: int) -> int:
        return max(
            0,
            self._query(1, 0, self._size - 1, index, self._size - 1),
        )

    def consume_from(self, index: int, byte_count: int) -> None:
        if byte_count:
            self._add(
                1,
                0,
                self._size - 1,
                index,
                self._size - 1,
                -byte_count,
            )


@dataclass(frozen=True)
class StageProgress:
    """Checkpoint-event-relative counters for one asynchronous stage.

    DataStates exposes process-lifetime cumulative counters.  The integration
    layer must subtract the event-start snapshot before constructing this
    object.  Gauges (``queued_bytes``, ``ready_bytes``, and ``inflight_bytes``)
    are already instantaneous and must not be subtracted.
    """

    total_bytes: int
    queued_bytes: int
    ready_bytes: int
    admitted_bytes: int
    completed_bytes: int
    inflight_bytes: int
    last_progress_monotonic_ns: int = 0

    def __post_init__(self) -> None:
        for name in (
            "total_bytes",
            "queued_bytes",
            "ready_bytes",
            "admitted_bytes",
            "completed_bytes",
            "inflight_bytes",
            "last_progress_monotonic_ns",
        ):
            _require_nonnegative(name, int(getattr(self, name)))
        if self.completed_bytes > self.admitted_bytes:
            raise ValueError("completed_bytes cannot exceed admitted_bytes")
        if self.admitted_bytes > self.total_bytes:
            raise ValueError("admitted_bytes cannot exceed total_bytes")
        if self.inflight_bytes > self.admitted_bytes - self.completed_bytes:
            raise ValueError("inflight_bytes exceeds admitted-but-incomplete bytes")

    @property
    def remaining_bytes(self) -> int:
        return self.total_bytes - self.completed_bytes

    @property
    def unadmitted_bytes(self) -> int:
        return self.total_bytes - self.admitted_bytes


@dataclass(frozen=True)
class RankProgress:
    """One rank's progress and conservative service envelope.

    ``now_ns`` and ``deadline_ns`` must use the same corrected clock domain.
    They may differ between ranks; taking the minimum rank slack makes the
    projection conservative without pretending that local monotonic clocks are
    synchronized across nodes.
    """

    rank: int
    now_ns: int
    deadline_ns: int
    d2h: StageProgress
    pfs: StageProgress
    d2h_rate_bytes_per_second: int
    pfs_rate_bytes_per_second: int
    finalization_reserve_ns: int
    pipeline_reserve_ns: int
    host_ready_bytes: int
    watchdog_fail_open: bool = False
    progress_stalled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "rank",
            "now_ns",
            "deadline_ns",
            "d2h_rate_bytes_per_second",
            "pfs_rate_bytes_per_second",
            "finalization_reserve_ns",
            "pipeline_reserve_ns",
            "host_ready_bytes",
        ):
            _require_nonnegative(name, int(getattr(self, name)))
        # The durable file includes alignment padding and metadata, so its PFS
        # byte total need not equal the tensor-only D2H total.  Likewise,
        # metadata may be host-ready before any GPU payload completes.

    @property
    def finished(self) -> bool:
        return self.d2h.remaining_bytes == 0 and self.pfs.remaining_bytes == 0

    def service_requirement_ns(self) -> int:
        d2h_ns = _service_ns(self.d2h.remaining_bytes, self.d2h_rate_bytes_per_second)
        pfs_ns = _service_ns(self.pfs.remaining_bytes, self.pfs_rate_bytes_per_second)
        if math.isinf(d2h_ns) or math.isinf(pfs_ns):
            return math.inf  # type: ignore[return-value]
        return max(d2h_ns, pfs_ns) + self.pipeline_reserve_ns

    def horizon_ns(self, deadline_margin_ns: int) -> int:
        return self.deadline_ns - self.now_ns - self.finalization_reserve_ns - deadline_margin_ns

    def slack_ns(self, deadline_margin_ns: int) -> int:
        requirement = self.service_requirement_ns()
        if math.isinf(requirement):
            return -UNLIMITED_BUDGET
        return self.horizon_ns(deadline_margin_ns) - requirement


def _protect_pfs_limits(
    ranks: Sequence[RankProgress], quantum: int
) -> dict[int, int]:
    """Build group-fair PFS limits from snapshot-start host inventory.

    Each local limit contains only whole requests already host-ready, except
    for an entirely ready final tail.  The least host-ready feasible fraction
    then limits every unfinished rank, so a rank with more inventory cannot
    advance past the group's normalized frontier.
    """

    local_limits = {
        rank.rank: _request_aligned_limit(
            rank.pfs.unadmitted_bytes,
            rank.host_ready_bytes,
            quantum,
        )
        for rank in ranks
    }
    unfinished = [rank for rank in ranks if rank.pfs.unadmitted_bytes]
    if not unfinished:
        return local_limits
    group_fraction = min(
        Fraction(local_limits[rank.rank], rank.pfs.unadmitted_bytes)
        for rank in unfinished
    )
    limits: dict[int, int] = {}
    for rank in ranks:
        unadmitted = rank.pfs.unadmitted_bytes
        normalized_bytes = (
            unadmitted * group_fraction.numerator // group_fraction.denominator
        )
        limits[rank.rank] = min(
            local_limits[rank.rank],
            _request_aligned_limit(unadmitted, normalized_bytes, quantum),
        )
    return limits


@dataclass(frozen=True)
class WindowSpec:
    """A chronological admission window within one receding-horizon plan.

    Capacity fractions are parts per million of the rank's conservative stage
    rate.  ``safe_*`` is the tail-preserving preferred capacity; ``hard_*`` is
    the maximum capacity used only when the deadline projection requires spill.
    """

    phase_id: int
    signature: str
    kind: WindowKind
    duration_ns: int
    d2h_risk_ppm: int
    pfs_risk_ppm: int
    safe_d2h_capacity_ppm: int = PPM
    safe_pfs_capacity_ppm: int = PPM
    hard_d2h_capacity_ppm: int = PPM
    hard_pfs_capacity_ppm: int = PPM
    eligible_ranks: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _require_nonnegative("phase_id", self.phase_id)
        if not self.signature:
            raise ValueError("window signature must be nonempty")
        if self.duration_ns <= 0:
            raise ValueError("window duration_ns must be positive")
        for name in (
            "d2h_risk_ppm",
            "pfs_risk_ppm",
            "safe_d2h_capacity_ppm",
            "safe_pfs_capacity_ppm",
            "hard_d2h_capacity_ppm",
            "hard_pfs_capacity_ppm",
        ):
            value = int(getattr(self, name))
            if not 0 <= value <= PPM:
                raise ValueError(f"{name} must be in [0, {PPM}], got {value}")
        if self.safe_d2h_capacity_ppm > self.hard_d2h_capacity_ppm:
            raise ValueError("safe D2H capacity cannot exceed hard capacity")
        if self.safe_pfs_capacity_ppm > self.hard_pfs_capacity_ppm:
            raise ValueError("safe PFS capacity cannot exceed hard capacity")
        if self.eligible_ranks is not None and len(set(self.eligible_ranks)) != len(self.eligible_ranks):
            raise ValueError("eligible_ranks contains duplicates")

    def eligible(self, rank: int) -> bool:
        return self.eligible_ranks is None or rank in self.eligible_ranks

    def capacity_bytes(self, rank: RankProgress, stage: Stage, *, safe: bool) -> int:
        if not self.eligible(rank.rank):
            return 0
        if stage is Stage.D2H:
            rate = rank.d2h_rate_bytes_per_second
            fraction = self.safe_d2h_capacity_ppm if safe else self.hard_d2h_capacity_ppm
        else:
            rate = rank.pfs_rate_bytes_per_second
            fraction = self.safe_pfs_capacity_ppm if safe else self.hard_pfs_capacity_ppm
        return rate * self.duration_ns * fraction // (NS_PER_SECOND * PPM)

    def risk(self, stage: Stage) -> int:
        return self.d2h_risk_ppm if stage is Stage.D2H else self.pfs_risk_ppm


@dataclass(frozen=True)
class PlannerInput:
    checkpoint_id: str
    generation: int
    step: int
    ranks: tuple[RankProgress, ...]
    windows: tuple[WindowSpec, ...]
    active: bool = True
    signatures_valid: bool = True
    observer_healthy: bool = True
    external_fail_open_reason: str = ""
    _canonical_digest_cache: str = field(
        default="", init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must be nonempty")
        _require_nonnegative("generation", self.generation)
        _require_nonnegative("step", self.step)
        if not self.ranks:
            raise ValueError("at least one rank is required")
        rank_ids = [rank.rank for rank in self.ranks]
        if len(rank_ids) != len(set(rank_ids)):
            raise ValueError("rank IDs must be unique")
        phase_ids = [window.phase_id for window in self.windows]
        if phase_ids != sorted(phase_ids) or len(phase_ids) != len(set(phase_ids)):
            raise ValueError("phase IDs must be unique and strictly increasing")

    def canonical_digest(self) -> str:
        if self._canonical_digest_cache:
            return self._canonical_digest_cache
        def stage_values(progress: StageProgress) -> tuple[int, ...]:
            return (
                progress.total_bytes,
                progress.queued_bytes,
                progress.ready_bytes,
                progress.admitted_bytes,
                progress.completed_bytes,
                progress.inflight_bytes,
                progress.last_progress_monotonic_ns,
            )

        rank_values = sorted(
            (
                rank.rank,
                rank.now_ns,
                rank.deadline_ns,
                stage_values(rank.d2h),
                stage_values(rank.pfs),
                rank.d2h_rate_bytes_per_second,
                rank.pfs_rate_bytes_per_second,
                rank.finalization_reserve_ns,
                rank.pipeline_reserve_ns,
                rank.host_ready_bytes,
                rank.watchdog_fail_open,
                rank.progress_stalled,
            )
            for rank in self.ranks
        )
        window_values = [
            (
                window.phase_id,
                window.signature,
                window.kind.value,
                window.duration_ns,
                window.d2h_risk_ppm,
                window.pfs_risk_ppm,
                window.safe_d2h_capacity_ppm,
                window.safe_pfs_capacity_ppm,
                window.hard_d2h_capacity_ppm,
                window.hard_pfs_capacity_ppm,
                window.eligible_ranks,
            )
            for window in self.windows
        ]
        # A positional primitive-only encoding is both canonical and much
        # cheaper than recursively materializing hundreds of dataclass dicts
        # on every controller step.
        payload = (
            self.checkpoint_id,
            self.generation,
            self.step,
            rank_values,
            window_values,
            self.active,
            self.signatures_valid,
            self.observer_healthy,
            self.external_fail_open_reason,
        )
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        object.__setattr__(self, "_canonical_digest_cache", digest)
        return digest


@dataclass(frozen=True)
class WindowCredit:
    phase_id: int
    signature: str
    kind: WindowKind
    d2h_budget_bytes: int
    pfs_budget_bytes: int
    d2h_spill_bytes: int
    pfs_spill_bytes: int
    max_pfs_inflight_bytes: int


@dataclass(frozen=True)
class RankCreditPlan:
    rank: int
    target_d2h_bytes: int
    target_pfs_bytes: int
    planned_d2h_bytes: int
    planned_pfs_bytes: int
    windows: tuple[WindowCredit, ...]

    def window(self, phase_id: int) -> WindowCredit:
        for credit in self.windows:
            if credit.phase_id == phase_id:
                return credit
        raise KeyError(f"unknown phase_id {phase_id}")


@dataclass(frozen=True)
class CreditPlan:
    checkpoint_id: str
    plan_version: int
    generation: int
    step: int
    input_digest: str
    mode: AdmissionMode
    global_slack_ns: int
    projected_completion_ns: int
    deadline_feasible: bool
    force_drain: bool
    reason: str
    rank_plans: tuple[RankCreditPlan, ...]

    def for_rank(self, rank: int) -> RankCreditPlan:
        for plan in self.rank_plans:
            if plan.rank == rank:
                return plan
        raise KeyError(f"unknown rank {rank}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ControllerConfig:
    d2h_quantum_bytes: int = 1 << 20
    pfs_quantum_bytes: int = 4 << 20
    max_pfs_inflight_bytes: int = 16 << 20
    # Retain a finite structural ceiling, but do not force a large horizon
    # target across nearly every collective.  The 5/1-request experiment did
    # exactly that and increased controlled tail while still missing the
    # deadline projection.  Sixteen D2H requests cover the largest 15 MiB
    # phase observed live; four PFS requests match the existing 16 MiB
    # non-preemptible residual.  Rate-derived window capacity remains the
    # tighter bound in normal operation.
    max_collective_d2h_requests: int = 16
    max_collective_pfs_requests: int = 4
    low_slack_ns: int = 50_000_000
    bounded_recovery_slack_ns: int = 50_000_000
    high_slack_ns: int = 200_000_000
    deadline_margin_ns: int = 25_000_000
    max_plan_staleness_steps: int = 1
    watchdog_timeout_ns: int = 250_000_000
    minimum_progress_quanta: int = 1

    def __post_init__(self) -> None:
        for name in (
            "d2h_quantum_bytes",
            "pfs_quantum_bytes",
            "max_pfs_inflight_bytes",
            "max_collective_d2h_requests",
            "max_collective_pfs_requests",
            "low_slack_ns",
            "bounded_recovery_slack_ns",
            "high_slack_ns",
            "deadline_margin_ns",
            "max_plan_staleness_steps",
            "watchdog_timeout_ns",
            "minimum_progress_quanta",
        ):
            _require_nonnegative(name, int(getattr(self, name)))
        if self.d2h_quantum_bytes == 0 or self.pfs_quantum_bytes == 0:
            raise ValueError("stage quantums must be positive")
        if self.max_pfs_inflight_bytes == 0:
            raise ValueError("max_pfs_inflight_bytes must be positive")
        if self.max_pfs_inflight_bytes < self.pfs_quantum_bytes:
            raise ValueError("max_pfs_inflight_bytes must cover at least one PFS quantum")
        if self.max_pfs_inflight_bytes % self.pfs_quantum_bytes:
            raise ValueError("max_pfs_inflight_bytes must be a whole number of PFS quantums")
        if self.max_collective_d2h_requests == 0:
            raise ValueError("max_collective_d2h_requests must be positive")
        if self.max_collective_pfs_requests == 0:
            raise ValueError("max_collective_pfs_requests must be positive")
        if self.watchdog_timeout_ns == 0:
            raise ValueError("watchdog_timeout_ns must be positive")
        if self.high_slack_ns <= self.low_slack_ns:
            raise ValueError("high_slack_ns must exceed low_slack_ns")
        if not (
            self.low_slack_ns
            <= self.bounded_recovery_slack_ns
            < self.high_slack_ns
        ):
            raise ValueError(
                "bounded_recovery_slack_ns must satisfy low <= recovery < high"
            )
        if self.minimum_progress_quanta == 0:
            raise ValueError("minimum_progress_quanta must be positive")


@dataclass
class TailFeedback:
    """One-step-delayed risk feedback for FSDP window signatures."""

    beta: float = 0.8
    latency_tolerance: float = 0.10
    skew_tolerance: float = 0.10
    gain_ppm: int = 250_000
    max_debt: float = 4.0
    _debt: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if self.latency_tolerance < 0.0 or self.skew_tolerance < 0.0:
            raise ValueError("tail tolerances must be nonnegative")
        if self.gain_ppm < 0 or self.max_debt <= 0.0:
            raise ValueError("gain_ppm and max_debt must be positive")

    def observe(
        self,
        signature: str,
        *,
        latency_ms: float,
        baseline_latency_ms: float,
        skew_ms: float,
        baseline_skew_ms: float,
        clock_uncertainty_ms: float,
    ) -> float:
        if not signature:
            raise ValueError("signature must be nonempty")
        if baseline_latency_ms <= 0.0:
            raise ValueError("baseline_latency_ms must be positive")
        skew_denom = baseline_skew_ms + clock_uncertainty_ms
        if skew_denom <= 0.0:
            raise ValueError("skew reference plus clock uncertainty must be positive")
        latency_excess = latency_ms / baseline_latency_ms - (1.0 + self.latency_tolerance)
        skew_excess = skew_ms / skew_denom - (1.0 + self.skew_tolerance)
        error = max(0.0, latency_excess, skew_excess)
        previous = self._debt.get(signature, 0.0)
        debt = min(self.max_debt, max(0.0, self.beta * previous + (1.0 - self.beta) * error))
        self._debt[signature] = debt
        return debt

    def adjusted_risk_ppm(self, signature: str, base_risk_ppm: int) -> int:
        if not 0 <= base_risk_ppm <= PPM:
            raise ValueError(f"base_risk_ppm must be in [0, {PPM}]")
        adjusted = base_risk_ppm + round(self.gain_ppm * self._debt.get(signature, 0.0))
        return min(PPM, adjusted)

    def debt(self, signature: str) -> float:
        return self._debt.get(signature, 0.0)


class TempoV4Controller:
    """Deterministic group planner with fail-open event liveness."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        # Runtime adapter may opt into a compute-only D2H lane for the
        # scheduled policy.  Keep the controller's offline/default contract
        # unchanged so callers that explicitly study unavoidable collective
        # spill retain the previous behavior.
        self.compute_only_d2h = False
        self._next_plan_version = 1
        self._last_generation: dict[str, int] = {}
        self._last_step: dict[str, int] = {}
        self._draining_events: dict[str, str] = {}
        # A phase-local ceiling can expire before the corresponding worker is
        # scheduled.  Receding-horizon fair share must carry that *unissued*
        # byte debt into the next plan; otherwise every revision assumes its
        # full target was consumed and repeatedly under-admits until DRAIN.
        self._previous_admission: dict[
            tuple[str, int, Stage], tuple[int, int]
        ] = {}

    def plan(self, snapshot: PlannerInput) -> CreditPlan:
        input_digest = snapshot.canonical_digest()
        version = self._next_plan_version
        self._next_plan_version += 1

        failure = self._validate_liveness(snapshot)
        slack_by_rank = {
            rank.rank: rank.slack_ns(self.config.deadline_margin_ns) for rank in snapshot.ranks
        }
        global_slack = min(slack_by_rank.values())
        projected_completion = max(
            self._projected_completion_ns(rank) for rank in snapshot.ranks
        )
        deadline_feasible = all(
            rank.service_requirement_ns() <= rank.horizon_ns(self.config.deadline_margin_ns)
            for rank in snapshot.ranks
        )

        if not snapshot.active:
            return self._open_plan(
                snapshot,
                version,
                input_digest,
                AdmissionMode.PROFILE,
                global_slack,
                projected_completion,
                True,
                "checkpoint inactive; profile with open admission",
            )
        if all(rank.finished for rank in snapshot.ranks):
            return self._open_plan(
                snapshot,
                version,
                input_digest,
                AdmissionMode.FINALIZE,
                global_slack,
                projected_completion,
                deadline_feasible,
                "all rank-local stages complete",
            )

        minimum_physical_horizon_ns = min(
            rank.deadline_ns - rank.now_ns
            for rank in snapshot.ranks
        )
        maximum_terminal_reserve_ns = max(
            rank.pipeline_reserve_ns + rank.finalization_reserve_ns
            for rank in snapshot.ranks
        )
        bounded_last_chance = False
        if failure:
            self._draining_events[snapshot.checkpoint_id] = failure
        elif global_slack <= self.config.bounded_recovery_slack_ns:
            # A pessimistic service projection crossing the low-slack
            # watermark does not mean the physical deadline has passed.  If
            # finalization plus pipeline residual still fits, continue issuing
            # a finite, phase-capped last-chance plan.  Credits can be consumed
            # in the early phases; requiring the *entire* next phase schedule
            # to fit needlessly drained a live 12--24 MiB PFS tail with
            # 97--112 ms still left before the physical deadline.
            # This preserves the byte/request residual bounds while avoiding
            # the large uncontrolled tail caused by premature fail-open.
            bounded_last_chance = (
                minimum_physical_horizon_ns
                > maximum_terminal_reserve_ns
            )
            if not bounded_last_chance:
                failure = (
                    f"global slack {global_slack}ns is at or below bounded recovery "
                    f"watermark {self.config.bounded_recovery_slack_ns}ns and no "
                    "bounded phase horizon remains"
                )
                self._draining_events[snapshot.checkpoint_id] = failure
        elif snapshot.checkpoint_id in self._draining_events:
            failure = self._draining_events[snapshot.checkpoint_id]

        if failure:
            self._record_generation(snapshot)
            return self._open_plan(
                snapshot,
                version,
                input_digest,
                AdmissionMode.DRAIN,
                global_slack,
                projected_completion,
                deadline_feasible,
                failure,
            )

        mode = (
            AdmissionMode.PROTECT
            if global_slack >= self.config.high_slack_ns
            else AdmissionMode.BALANCED
        )
        normal_targets = self._group_targets(snapshot, mode)
        targets = (
            {
                rank.rank: (
                    rank.d2h.unadmitted_bytes,
                    rank.pfs.unadmitted_bytes,
                )
                for rank in snapshot.ranks
            }
            if bounded_last_chance
            else normal_targets
        )

        def allocate_targets(
            selected: dict[int, tuple[int, int]],
        ) -> tuple[list[RankCreditPlan], str]:
            rank_plans: list[RankCreditPlan] = []
            allocation_failure = ""
            # Per-rank progress is normally group-symmetric.  Allocation
            # depends only on the fields in this key when every window is
            # group-eligible; reuse the immutable template and replace only
            # its rank ID.  Rank-specific eligibility disables this fast path.
            allocation_templates: dict[
                tuple[int, int, int, int, int, int, int],
                tuple[RankCreditPlan, str],
            ] = {}
            group_eligible = all(
                window.eligible_ranks is None for window in snapshot.windows
            )
            for rank in sorted(snapshot.ranks, key=lambda item: item.rank):
                target_d2h, target_pfs = selected[rank.rank]
                cache_key = (
                    target_d2h,
                    target_pfs,
                    rank.d2h.unadmitted_bytes,
                    rank.pfs.unadmitted_bytes,
                    rank.host_ready_bytes,
                    rank.d2h_rate_bytes_per_second,
                    rank.pfs_rate_bytes_per_second,
                )
                cached = allocation_templates.get(cache_key) if group_eligible else None
                if cached is None:
                    rank_plan, reason = self._allocate_rank(
                        snapshot.windows, rank, target_d2h, target_pfs
                    )
                    if group_eligible:
                        allocation_templates[cache_key] = (rank_plan, reason)
                else:
                    template, reason = cached
                    rank_plan = replace(template, rank=rank.rank)
                if reason:
                    allocation_failure = f"rank {rank.rank}: {reason}"
                    break
                rank_plans.append(rank_plan)
            return rank_plans, allocation_failure

        def clamp_pfs_targets(
            selected: dict[int, tuple[int, int]],
        ) -> dict[int, tuple[int, int]]:
            """Return the maximum feasible request-aligned PFS target per rank."""

            clamped: dict[int, tuple[int, int]] = {}
            for rank in sorted(snapshot.ranks, key=lambda item: item.rank):
                target_d2h, target_pfs = selected[rank.rank]
                quantum = self.config.pfs_quantum_bytes
                high_requests = _ceil_div(target_pfs, quantum)
                low_requests = 0
                best = 0
                while low_requests <= high_requests:
                    middle = (low_requests + high_requests) // 2
                    candidate = min(target_pfs, middle * quantum)
                    _, reason = self._allocate_rank(
                        snapshot.windows,
                        rank,
                        target_d2h,
                        candidate,
                    )
                    if reason:
                        high_requests = middle - 1
                    else:
                        best = candidate
                        low_requests = middle + 1
                clamped[rank.rank] = (target_d2h, best)
            return clamped

        bounded_producer_fallback = False
        bounded_normal_fallback = False
        prefix_clamped_fallback = False
        rank_plans, allocation_failure = allocate_targets(targets)
        if allocation_failure and "PFS" in allocation_failure:
            # Consume the largest feasible producer prefix before falling back
            # to a smaller fair-share target.  Doing this after the fallback
            # left 40--80 ms of usable PFS service idle in live recovery.
            targets = clamp_pfs_targets(targets)
            rank_plans, allocation_failure = allocate_targets(targets)
            prefix_clamped_fallback = not allocation_failure
        if bounded_last_chance and allocation_failure:
            # A full PFS tail can be temporarily infeasible under the strict
            # one-window producer prefix even though leading all D2H is useful.
            # Preserve the normal PFS target for this revision, lead all D2H,
            # and retry the full finite tail at the next common snapshot.
            targets = {
                rank.rank: (
                    rank.d2h.unadmitted_bytes,
                    normal_targets[rank.rank][1],
                )
                for rank in snapshot.ranks
            }
            rank_plans, allocation_failure = allocate_targets(targets)
            bounded_producer_fallback = not allocation_failure
        if bounded_last_chance and allocation_failure:
            # If even the D2H lead cannot satisfy this revision's lagged
            # prefix geometry, retain the ordinary finite plan rather than
            # converting a conservative optimization miss into DRAIN.
            targets = normal_targets
            rank_plans, allocation_failure = allocate_targets(targets)
            bounded_normal_fallback = not allocation_failure
        if allocation_failure and "PFS" in allocation_failure:
            targets = clamp_pfs_targets(targets)
            rank_plans, allocation_failure = allocate_targets(targets)
            prefix_clamped_fallback = not allocation_failure

        self._record_generation(snapshot)
        if allocation_failure:
            self._draining_events[snapshot.checkpoint_id] = allocation_failure
            return self._open_plan(
                snapshot,
                version,
                input_digest,
                AdmissionMode.DRAIN,
                global_slack,
                projected_completion,
                deadline_feasible,
                allocation_failure,
            )

        for rank in snapshot.ranks:
            target_d2h, target_pfs = targets[rank.rank]
            self._previous_admission[
                (snapshot.checkpoint_id, rank.rank, Stage.D2H)
            ] = (rank.d2h.admitted_bytes, target_d2h)
            self._previous_admission[
                (snapshot.checkpoint_id, rank.rank, Stage.PFS)
            ] = (rank.pfs.admitted_bytes, target_pfs)

        return CreditPlan(
            checkpoint_id=snapshot.checkpoint_id,
            plan_version=version,
            generation=snapshot.generation,
            step=snapshot.step,
            input_digest=input_digest,
            mode=mode,
            global_slack_ns=global_slack,
            projected_completion_ns=projected_completion,
            deadline_feasible=deadline_feasible,
            force_drain=False,
            reason=(
                (
                    "finite PFS target was clamped to the maximum request-aligned "
                    "one-window producer prefix"
                    if prefix_clamped_fallback
                    else (
                        "bounded last-chance retained the ordinary finite plan; the full "
                        "tail and D2H lead were deferred by the one-window prefix"
                        if bounded_normal_fallback
                        else (
                            "bounded last-chance D2H producer lead with a normal PFS target; "
                            "the full PFS tail was deferred by the one-window prefix"
                            if bounded_producer_fallback
                            else (
                                "bounded last-chance admission of all remaining bytes before the "
                                "physical deadline; phase byte/request caps remain enforced"
                            )
                        )
                    )
                )
                if bounded_last_chance
                else (
                    "deadline-projected compute-first admission; high-slack producer "
                    "lead uses a group-fair snapshot-host-ready cap"
                    if mode is AdmissionMode.PROTECT
                    else (
                        "deadline-projected compute-first admission; remaining service "
                        "is risk-ordered under the lagged producer prefix"
                    )
                )
            ),
            rank_plans=tuple(rank_plans),
        )

    def _validate_liveness(self, snapshot: PlannerInput) -> str:
        if snapshot.external_fail_open_reason:
            return snapshot.external_fail_open_reason
        if snapshot.checkpoint_id in self._draining_events:
            return self._draining_events[snapshot.checkpoint_id]
        if not snapshot.signatures_valid:
            return "collective signature mismatch"
        if not snapshot.observer_healthy:
            return "collective observer unhealthy"
        if any(rank.watchdog_fail_open for rank in snapshot.ranks):
            return "DataStates admission watchdog failed open"
        if any(rank.progress_stalled for rank in snapshot.ranks):
            return "zero-progress watchdog detected a stalled rank"
        if not snapshot.windows:
            return "no future admission windows"
        previous_generation = self._last_generation.get(snapshot.checkpoint_id)
        previous_step = self._last_step.get(snapshot.checkpoint_id)
        if previous_generation is not None and snapshot.generation <= previous_generation:
            return "stale or duplicate controller generation"
        if previous_step is not None and snapshot.step - previous_step > self.config.max_plan_staleness_steps:
            return "controller plan exceeded maximum staleness"
        for rank in snapshot.ranks:
            if rank.d2h.remaining_bytes and rank.d2h_rate_bytes_per_second <= 0:
                return f"rank {rank.rank} has no conservative D2H service rate"
            if rank.pfs.remaining_bytes and rank.pfs_rate_bytes_per_second <= 0:
                return f"rank {rank.rank} has no conservative PFS service rate"
        return ""

    def _record_generation(self, snapshot: PlannerInput) -> None:
        self._last_generation[snapshot.checkpoint_id] = snapshot.generation
        self._last_step[snapshot.checkpoint_id] = snapshot.step

    def _projected_completion_ns(self, rank: RankProgress) -> int:
        requirement = rank.service_requirement_ns()
        if math.isinf(requirement):
            return UNLIMITED_BUDGET
        return rank.now_ns + requirement + rank.finalization_reserve_ns + self.config.deadline_margin_ns

    def _stage_target(
        self,
        rank: RankProgress,
        stage: Stage,
        plan_duration_ns: int,
    ) -> int:
        progress = rank.d2h if stage is Stage.D2H else rank.pfs
        rate = (
            rank.d2h_rate_bytes_per_second
            if stage is Stage.D2H
            else rank.pfs_rate_bytes_per_second
        )
        unadmitted = progress.unadmitted_bytes
        if unadmitted <= 0:
            return 0
        horizon = max(1, rank.horizon_ns(self.config.deadline_margin_ns))
        active_plan_ns = min(plan_duration_ns, horizon)
        fair_share = _ceil_div(unadmitted * active_plan_ns, horizon)
        # ``plan()`` irreversibly fails open once the projected service slack
        # reaches ``low_slack_ns``.  Pipeline drain is part of that projection
        # too, but neither reserve can carry stage bytes in a later plan.  Do
        # not count those intervals as future service capacity: doing so lets
        # the receding-horizon target defer useful work until the next gather,
        # then hit the fail-open watermark with bytes still unadmitted.
        #
        # Keep the original horizon for ``fair_share``.  Shrinking its
        # denominator would front-load every plan, even when the last-chance
        # bound is inactive, and creates avoidable collective spill.  The
        # reserve belongs only in the feasibility/last-chance calculation.
        future_horizon_ns = max(
            0,
            horizon
            - plan_duration_ns
            - rank.pipeline_reserve_ns
            - self.config.low_slack_ns,
        )
        future_capacity = rate * future_horizon_ns // NS_PER_SECOND
        last_chance = max(0, unadmitted - future_capacity)
        quantum = self.config.d2h_quantum_bytes if stage is Stage.D2H else self.config.pfs_quantum_bytes
        minimum = min(unadmitted, quantum * self.config.minimum_progress_quanta)
        return min(unadmitted, max(fair_share, last_chance, minimum))

    def _unissued_target_debt(
        self,
        checkpoint_id: str,
        rank: RankProgress,
        stage: Stage,
    ) -> int:
        previous = self._previous_admission.get(
            (checkpoint_id, rank.rank, stage)
        )
        if previous is None:
            return 0
        previous_admitted, previous_target = previous
        progress = rank.d2h if stage is Stage.D2H else rank.pfs
        newly_admitted = max(0, progress.admitted_bytes - previous_admitted)
        return max(0, previous_target - newly_admitted)

    def _group_targets(
        self,
        snapshot: PlannerInput,
        mode: AdmissionMode = AdmissionMode.BALANCED,
    ) -> dict[int, tuple[int, int]]:
        plan_duration_ns = sum(window.duration_ns for window in snapshot.windows)
        raw: dict[int, dict[Stage, int]] = {}
        maximum_fraction: dict[Stage, Fraction] = {Stage.D2H: Fraction(0), Stage.PFS: Fraction(0)}
        for rank in snapshot.ranks:
            raw[rank.rank] = {}
            for stage, progress in ((Stage.D2H, rank.d2h), (Stage.PFS, rank.pfs)):
                target = min(
                    progress.unadmitted_bytes,
                    self._stage_target(rank, stage, plan_duration_ns)
                    + self._unissued_target_debt(
                        snapshot.checkpoint_id, rank, stage
                    ),
                )
                raw[rank.rank][stage] = target
                if progress.unadmitted_bytes:
                    maximum_fraction[stage] = max(
                        maximum_fraction[stage], Fraction(target, progress.unadmitted_bytes)
                    )

        protect_pfs_limits = (
            _protect_pfs_limits(snapshot.ranks, self.config.pfs_quantum_bytes)
            if mode is AdmissionMode.PROTECT
            else {}
        )
        targets: dict[int, tuple[int, int]] = {}
        for rank in snapshot.ranks:
            stage_targets: dict[Stage, int] = {}
            for stage, progress in ((Stage.D2H, rank.d2h), (Stage.PFS, rank.pfs)):
                fraction = maximum_fraction[stage]
                fair_target = _ceil_div(
                    progress.unadmitted_bytes * fraction.numerator,
                    fraction.denominator,
                )
                quantum = (
                    self.config.d2h_quantum_bytes
                    if stage is Stage.D2H
                    else self.config.pfs_quantum_bytes
                )
                stage_target = _request_aligned_target(
                    progress.unadmitted_bytes,
                    max(raw[rank.rank][stage], fair_target),
                    quantum,
                )
                if stage is Stage.PFS and mode is AdmissionMode.PROTECT:
                    stage_target = min(stage_target, protect_pfs_limits[rank.rank])
                stage_targets[stage] = stage_target
            targets[rank.rank] = (stage_targets[Stage.D2H], stage_targets[Stage.PFS])
        return targets

    def _capacities(
        self,
        windows: Sequence[WindowSpec],
        rank: RankProgress,
        stage: Stage,
        quantum: int,
    ) -> tuple[list[int], list[int]]:
        safe: list[int] = []
        hard: list[int] = []
        for window in windows:
            safe_bytes = _floor_quantum(window.capacity_bytes(rank, stage, safe=True), quantum)
            # FSDP phases on the archived Perlmutter trace are usually shorter
            # than the service time of one stage request.  Rounding every phase
            # down independently would therefore report zero capacity for an
            # otherwise serviceable horizon and force an artificial DRAIN.
            #
            # A hard admission may start one whole request in such a window;
            # its unfinished portion is precisely the non-preemptible residual
            # already covered by the one-D2H-request / bounded-PFS-inflight
            # invariant.  The horizon target remains rate-derived, so this
            # ceiling does not authorize more aggregate work than the
            # receding-horizon projection requires.
            hard_bytes = _ceil_quantum(
                window.capacity_bytes(rank, stage, safe=False), quantum
            )
            if window.kind is WindowKind.COLLECTIVE:
                collective_cap = (
                    self.config.max_collective_d2h_requests * quantum
                    if stage is Stage.D2H
                    else self.config.max_collective_pfs_requests * quantum
                )
                hard_bytes = min(hard_bytes, collective_cap)
                safe_bytes = min(safe_bytes, hard_bytes)
            safe.append(safe_bytes)
            hard.append(max(safe_bytes, hard_bytes))
        return safe, hard

    @staticmethod
    def _allocate_by_risk(
        windows: Sequence[WindowSpec],
        stage: Stage,
        target: int,
        allocations: list[int],
        capacities: Sequence[int],
        quantum: int,
        prefix_headroom: _SuffixMinHeadroom | None = None,
        *,
        order: Sequence[int] | None = None,
        index_limit: int | None = None,
    ) -> int:
        remaining = target
        if order is None:
            order = sorted(
                range(len(windows)),
                key=lambda index: (
                    0 if windows[index].kind is WindowKind.COMPUTE else 1,
                    windows[index].risk(stage),
                    windows[index].phase_id,
                ),
            )
        for index in order:
            if index_limit is not None and index >= index_limit:
                continue
            available = max(0, capacities[index] - allocations[index])
            if available <= 0:
                continue
            if prefix_headroom is not None:
                available = min(available, prefix_headroom.available_from(index))
                if available <= 0:
                    continue
            # Capacities are conservative whole-quantum envelopes, but the
            # event's final request may be smaller than a quantum.  Horizon
            # targets are request-aligned before reaching this routine, so a
            # sub-quantum ``remaining`` is necessarily that exact event tail,
            # not an unusable partial allowance for a full request.
            if 0 < remaining < quantum and available >= remaining:
                grant = remaining
            else:
                grant = min(remaining, _floor_quantum(available, quantum))
            if grant <= 0:
                continue
            allocations[index] += grant
            if prefix_headroom is not None:
                prefix_headroom.consume_from(index, grant)
            remaining -= grant
            if remaining == 0:
                break
        return remaining

    def _allocate_rank(
        self,
        windows: Sequence[WindowSpec],
        rank: RankProgress,
        target_d2h: int,
        target_pfs: int,
    ) -> tuple[RankCreditPlan, str]:
        q_d = self.config.d2h_quantum_bytes
        q_p = self.config.pfs_quantum_bytes
        target_d2h = _request_aligned_target(
            rank.d2h.unadmitted_bytes, target_d2h, q_d
        )
        target_pfs = _request_aligned_target(
            rank.pfs.unadmitted_bytes, target_pfs, q_p
        )
        # Persistence includes serialized metadata in addition to the GPU
        # payload.  Ensure this horizon plans enough producer progress to back
        # its PFS target after accounting for bytes already host-ready.
        target_d2h = _request_aligned_target(
            rank.d2h.unadmitted_bytes,
            max(target_d2h, max(0, target_pfs - rank.host_ready_bytes)),
            q_d,
        )
        d_safe, d_hard = self._capacities(windows, rank, Stage.D2H, q_d)
        p_safe, p_hard = self._capacities(windows, rank, Stage.PFS, q_p)

        # Never spill D2H into a collective when the current horizon has
        # enough compute-only capacity to make useful producer progress.  A
        # full receding-horizon target can exceed the immediately installable
        # compute prefix; allowing the remainder to start in a collective is
        # exactly the interference pattern the scheduled policy is meant to
        # avoid (and is unnecessary because the next common plan can carry
        # the unadmitted debt forward).  The cap is request-aligned and is a
        # no-op for v4_open's smaller target, while preserving its matched
        # open-path semantics.
        compute_d2h_capacity = sum(
            capacity
            for window, capacity in zip(windows, d_hard)
            if str(getattr(window.kind, "value", window.kind)) == WindowKind.COMPUTE.value
        )
        if self.compute_only_d2h and target_d2h > compute_d2h_capacity + q_d:
            target_d2h = _request_aligned_target(
                rank.d2h.unadmitted_bytes,
                compute_d2h_capacity + q_d,
                q_d,
            )
        p_alloc = [0] * len(windows)

        if sum(d_hard) < target_d2h:
            empty = RankCreditPlan(rank.rank, target_d2h, target_pfs, 0, 0, ())
            return empty, f"D2H hard capacity is short by {target_d2h - sum(d_hard)} bytes"

        # Select consumer credit against the *maximum feasible* producer
        # prefix, capped by this horizon's D2H target.  This avoids committing
        # D2H to a cheap late phase and then repairing the tandem plan by
        # spilling into the chronologically earliest collective.
        producer_capacity = 0
        headrooms: list[int] = []
        for capacity in d_hard:
            headrooms.append(
                rank.host_ready_bytes + min(target_d2h, producer_capacity)
            )
            producer_capacity += capacity
        prefix_headroom = _SuffixMinHeadroom(headrooms)
        compute_producer_prefix: list[int] = []
        compute_capacity = 0
        for index, capacity in enumerate(d_hard):
            compute_producer_prefix.append(compute_capacity)
            if windows[index].kind is WindowKind.COMPUTE:
                compute_capacity += capacity
        p_risk_order = sorted(
            range(len(windows)),
            key=lambda index: (
                0 if windows[index].kind is WindowKind.COMPUTE else 1,
                windows[index].risk(Stage.PFS),
                windows[index].phase_id,
            ),
        )
        p_order = sorted(
            range(len(windows)),
            key=lambda index: (
                0 if windows[index].kind is WindowKind.COMPUTE else 1,
                -min(target_d2h, compute_producer_prefix[index]),
                windows[index].risk(Stage.PFS),
                windows[index].phase_id,
            ),
        )
        # Spend already-host-ready inventory by tail risk first; it needs no
        # producer placement.  The remaining coupled bytes prefer consumer
        # windows with more preceding compute producer capacity, avoiding a
        # low-risk early choice that would force otherwise-unnecessary D2H in
        # a collective.
        host_backed_target = (
            target_pfs
            if rank.host_ready_bytes >= target_pfs
            else min(target_pfs, _floor_quantum(rank.host_ready_bytes, q_p))
        )
        if host_backed_target:
            host_headroom = _SuffixMinHeadroom(
                [rank.host_ready_bytes] * len(windows)
            )
            remaining_host = self._allocate_by_risk(
                windows,
                Stage.PFS,
                host_backed_target,
                p_alloc,
                p_safe,
                q_p,
                host_headroom,
                order=p_risk_order,
            )
            if remaining_host:
                remaining_host = self._allocate_by_risk(
                    windows,
                    Stage.PFS,
                    remaining_host,
                    p_alloc,
                    p_hard,
                    q_p,
                    host_headroom,
                    order=p_risk_order,
                )
            if remaining_host:
                empty = RankCreditPlan(rank.rank, target_d2h, target_pfs, 0, 0, ())
                return empty, f"PFS host-ready capacity is short by {remaining_host} bytes"
            for index, allocation in enumerate(p_alloc):
                prefix_headroom.consume_from(index, allocation)

        remaining_p = target_pfs - sum(p_alloc)
        if remaining_p:
            remaining_p = self._allocate_by_risk(
                windows,
                Stage.PFS,
                remaining_p,
                p_alloc,
                p_safe,
                q_p,
                prefix_headroom,
                order=p_order,
            )
        if remaining_p:
            remaining_p = self._allocate_by_risk(
                windows,
                Stage.PFS,
                remaining_p,
                p_alloc,
                p_hard,
                q_p,
                prefix_headroom,
                order=p_order,
            )
        p_safe_alloc = [
            min(allocation, safe_capacity)
            for allocation, safe_capacity in zip(p_alloc, p_safe)
        ]
        if remaining_p:
            empty = RankCreditPlan(rank.rank, target_d2h, target_pfs, 0, 0, ())
            return empty, f"PFS hard capacity or D2H prefix is short by {remaining_p} bytes"

        # Given the selected PFS windows, satisfy each nested producer
        # deadline with the lowest-risk safe D2H slots available before it.
        # Only then use hard capacity, and allocate producer credit unrelated
        # to this PFS target independently by risk.  The nested deadlines make
        # this deterministic greedy complete for the selected consumer plan.
        d_alloc = [0] * len(windows)
        d_order = sorted(
            range(len(windows)),
            key=lambda index: (
                0 if windows[index].kind is WindowKind.COMPUTE else 1,
                windows[index].risk(Stage.D2H),
                windows[index].phase_id,
            ),
        )
        allocated_d = 0
        cumulative_p = 0
        for index, pfs_bytes in enumerate(p_alloc):
            cumulative_p += pfs_bytes
            required_d = max(0, cumulative_p - rank.host_ready_bytes)
            deficit = required_d - allocated_d
            if deficit <= 0:
                continue
            needed = min(target_d2h - allocated_d, _ceil_quantum(deficit, q_d))
            remaining_needed = self._allocate_by_risk(
                windows,
                Stage.D2H,
                needed,
                d_alloc,
                d_safe,
                q_d,
                order=d_order,
                index_limit=index,
            )
            if remaining_needed:
                remaining_needed = self._allocate_by_risk(
                    windows,
                    Stage.D2H,
                    remaining_needed,
                    d_alloc,
                    d_hard,
                    q_d,
                    order=d_order,
                    index_limit=index,
                )
            granted = needed - remaining_needed
            allocated_d += granted
            if remaining_needed:
                empty = RankCreditPlan(
                    rank.rank, target_d2h, target_pfs, allocated_d, 0, ()
                )
                return empty, (
                    "D2H prefix scheduler is short by "
                    f"{remaining_needed} bytes before phase {windows[index].phase_id}"
                )

        remaining_d = target_d2h - allocated_d
        if remaining_d:
            remaining_d = self._allocate_by_risk(
                windows,
                Stage.D2H,
                remaining_d,
                d_alloc,
                d_safe,
                q_d,
                order=d_order,
            )
        if remaining_d:
            remaining_d = self._allocate_by_risk(
                windows,
                Stage.D2H,
                remaining_d,
                d_alloc,
                d_hard,
                q_d,
                order=d_order,
            )
        if remaining_d:
            empty = RankCreditPlan(rank.rank, target_d2h, target_pfs, sum(d_alloc), 0, ())
            return empty, f"D2H hard capacity is short by {remaining_d} bytes"

        d_safe_alloc = [
            min(allocation, safe_capacity)
            for allocation, safe_capacity in zip(d_alloc, d_safe)
        ]

        credits = tuple(
            WindowCredit(
                phase_id=window.phase_id,
                signature=window.signature,
                kind=window.kind,
                d2h_budget_bytes=d_alloc[index],
                pfs_budget_bytes=p_alloc[index],
                d2h_spill_bytes=max(0, d_alloc[index] - d_safe_alloc[index]),
                pfs_spill_bytes=max(0, p_alloc[index] - p_safe_alloc[index]),
                max_pfs_inflight_bytes=self.config.max_pfs_inflight_bytes,
            )
            for index, window in enumerate(windows)
        )
        return (
            RankCreditPlan(
                rank=rank.rank,
                target_d2h_bytes=target_d2h,
                target_pfs_bytes=target_pfs,
                planned_d2h_bytes=sum(d_alloc),
                planned_pfs_bytes=sum(p_alloc),
                windows=credits,
            ),
            "",
        )

    def _open_plan(
        self,
        snapshot: PlannerInput,
        version: int,
        input_digest: str,
        mode: AdmissionMode,
        global_slack: int,
        projected_completion: int,
        deadline_feasible: bool,
        reason: str,
    ) -> CreditPlan:
        # DRAIN is an irreversible fail-open and therefore authorizes unlimited
        # admission. FINALIZE is a normal, already-drained state: giving it the
        # same force_drain bit makes the runtime misclassify successful events
        # as failures on every later step before the event is recorded.
        window_budget = (
            UNLIMITED_BUDGET
            if mode in (AdmissionMode.PROFILE, AdmissionMode.DRAIN)
            else 0
        )
        rank_plans = tuple(
            RankCreditPlan(
                rank=rank.rank,
                target_d2h_bytes=rank.d2h.unadmitted_bytes,
                target_pfs_bytes=rank.pfs.unadmitted_bytes,
                planned_d2h_bytes=rank.d2h.unadmitted_bytes,
                planned_pfs_bytes=rank.pfs.unadmitted_bytes,
                windows=tuple(
                    WindowCredit(
                        phase_id=window.phase_id,
                        signature=window.signature,
                        kind=window.kind,
                        d2h_budget_bytes=window_budget,
                        pfs_budget_bytes=window_budget,
                        d2h_spill_bytes=0,
                        pfs_spill_bytes=0,
                        max_pfs_inflight_bytes=self.config.max_pfs_inflight_bytes,
                    )
                    for window in snapshot.windows
                ),
            )
            for rank in sorted(snapshot.ranks, key=lambda item: item.rank)
        )
        return CreditPlan(
            checkpoint_id=snapshot.checkpoint_id,
            plan_version=version,
            generation=snapshot.generation,
            step=snapshot.step,
            input_digest=input_digest,
            mode=mode,
            global_slack_ns=global_slack,
            projected_completion_ns=projected_completion,
            deadline_feasible=deadline_feasible,
            force_drain=mode is AdmissionMode.DRAIN,
            reason=reason,
            rank_plans=rank_plans,
        )

    def close_event(self, checkpoint_id: str) -> None:
        """Forget per-event liveness state after a durable global commit."""

        self._last_generation.pop(checkpoint_id, None)
        self._last_step.pop(checkpoint_id, None)
        self._draining_events.pop(checkpoint_id, None)
        for key in tuple(self._previous_admission):
            if key[0] == checkpoint_id:
                self._previous_admission.pop(key, None)


def apply_tail_feedback(windows: Iterable[WindowSpec], feedback: TailFeedback) -> tuple[WindowSpec, ...]:
    """Return windows with feedback-adjusted risk and unchanged capacities."""

    adjusted = []
    for window in windows:
        adjusted.append(
            WindowSpec(
                phase_id=window.phase_id,
                signature=window.signature,
                kind=window.kind,
                duration_ns=window.duration_ns,
                d2h_risk_ppm=feedback.adjusted_risk_ppm(window.signature, window.d2h_risk_ppm),
                pfs_risk_ppm=feedback.adjusted_risk_ppm(window.signature, window.pfs_risk_ppm),
                safe_d2h_capacity_ppm=window.safe_d2h_capacity_ppm,
                safe_pfs_capacity_ppm=window.safe_pfs_capacity_ppm,
                hard_d2h_capacity_ppm=window.hard_d2h_capacity_ppm,
                hard_pfs_capacity_ppm=window.hard_pfs_capacity_ppm,
                eligible_ranks=window.eligible_ranks,
            )
        )
    return tuple(adjusted)


def validate_plan(snapshot: PlannerInput, plan: CreditPlan, config: ControllerConfig) -> None:
    """Raise if a serialized or broadcast plan violates admission invariants."""

    if plan.checkpoint_id != snapshot.checkpoint_id:
        raise ValueError("plan checkpoint_id does not match snapshot")
    if plan.generation != snapshot.generation or plan.step != snapshot.step:
        raise ValueError("plan generation/step does not match snapshot")
    if plan.input_digest != snapshot.canonical_digest():
        raise ValueError("plan input digest mismatch")
    expected_ranks = sorted(rank.rank for rank in snapshot.ranks)
    actual_ranks = sorted(rank_plan.rank for rank_plan in plan.rank_plans)
    if actual_ranks != expected_ranks:
        raise ValueError("plan rank set mismatch")
    if plan.force_drain:
        if plan.mode is not AdmissionMode.DRAIN:
            raise ValueError("force_drain is only valid for DRAIN")
        return
    if plan.mode is AdmissionMode.FINALIZE:
        if not snapshot.active or not all(rank.finished for rank in snapshot.ranks):
            raise ValueError("FINALIZE requires every active rank stage to be complete")
        for rank_plan in plan.rank_plans:
            if (
                rank_plan.target_d2h_bytes
                or rank_plan.target_pfs_bytes
                or rank_plan.planned_d2h_bytes
                or rank_plan.planned_pfs_bytes
                or any(
                    credit.d2h_budget_bytes or credit.pfs_budget_bytes
                    for credit in rank_plan.windows
                )
            ):
                raise ValueError("FINALIZE must not authorize new stage work")
        return
    if plan.mode is AdmissionMode.PROFILE:
        if snapshot.active:
            raise ValueError("PROFILE open plan is only valid without an active checkpoint")
        return

    ranks = {rank.rank: rank for rank in snapshot.ranks}
    windows = {window.phase_id: window for window in snapshot.windows}
    group_eligible = all(window.eligible_ranks is None for window in snapshot.windows)
    protect_pfs_limits = (
        _protect_pfs_limits(snapshot.ranks, config.pfs_quantum_bytes)
        if plan.mode is AdmissionMode.PROTECT
        else {}
    )
    validated_templates: dict[
        tuple[int, int, int, int, int], RankCreditPlan
    ] = {}
    for rank_plan in plan.rank_plans:
        rank = ranks[rank_plan.rank]
        progress_key = (
            rank.d2h.unadmitted_bytes,
            rank.pfs.unadmitted_bytes,
            rank.host_ready_bytes,
            rank.d2h_rate_bytes_per_second,
            rank.pfs_rate_bytes_per_second,
        )
        template = validated_templates.get(progress_key) if group_eligible else None
        if template is not None and (
            rank_plan.target_d2h_bytes == template.target_d2h_bytes
            and rank_plan.target_pfs_bytes == template.target_pfs_bytes
            and rank_plan.planned_d2h_bytes == template.planned_d2h_bytes
            and rank_plan.planned_pfs_bytes == template.planned_pfs_bytes
            # Planner memoization intentionally shares this immutable tuple.
            # A deserialized/non-shared rank plan is validated in full.
            and rank_plan.windows is template.windows
        ):
            continue
        if [credit.phase_id for credit in rank_plan.windows] != [
            window.phase_id for window in snapshot.windows
        ]:
            raise ValueError(f"rank {rank.rank} window sequence differs from the snapshot")
        if sum(credit.d2h_budget_bytes for credit in rank_plan.windows) != rank_plan.planned_d2h_bytes:
            raise ValueError(f"rank {rank.rank} D2H window budgets do not match planned bytes")
        if sum(credit.pfs_budget_bytes for credit in rank_plan.windows) != rank_plan.planned_pfs_bytes:
            raise ValueError(f"rank {rank.rank} PFS window budgets do not match planned bytes")
        if rank_plan.planned_d2h_bytes < rank_plan.target_d2h_bytes:
            raise ValueError(f"rank {rank.rank} D2H plan does not meet its target")
        if rank_plan.planned_pfs_bytes < rank_plan.target_pfs_bytes:
            raise ValueError(f"rank {rank.rank} PFS plan does not meet its target")
        if plan.mode is AdmissionMode.PROTECT and (
            rank_plan.target_pfs_bytes > protect_pfs_limits[rank.rank]
            or rank_plan.planned_pfs_bytes > protect_pfs_limits[rank.rank]
        ):
            raise ValueError(
                f"rank {rank.rank} PROTECT PFS plan exceeds its group-fair "
                "snapshot host-ready limit"
            )
        if rank_plan.planned_d2h_bytes > rank.d2h.unadmitted_bytes:
            raise ValueError(f"rank {rank.rank} D2H plan exceeds unadmitted bytes")
        if rank_plan.planned_pfs_bytes > rank.pfs.unadmitted_bytes:
            raise ValueError(f"rank {rank.rank} PFS plan exceeds unadmitted bytes")
        cumulative_d2h = rank.host_ready_bytes
        cumulative_pfs = 0
        for credit in rank_plan.windows:
            window = windows[credit.phase_id]
            if credit.d2h_budget_bytes < 0 or credit.pfs_budget_bytes < 0:
                raise ValueError("credit budgets must be nonnegative")
            d2h_safe = _floor_quantum(
                window.capacity_bytes(rank, Stage.D2H, safe=True), config.d2h_quantum_bytes
            )
            d2h_hard = _ceil_quantum(
                window.capacity_bytes(rank, Stage.D2H, safe=False), config.d2h_quantum_bytes
            )
            pfs_safe = _floor_quantum(
                window.capacity_bytes(rank, Stage.PFS, safe=True), config.pfs_quantum_bytes
            )
            pfs_hard = _ceil_quantum(
                window.capacity_bytes(rank, Stage.PFS, safe=False), config.pfs_quantum_bytes
            )
            if window.kind is WindowKind.COLLECTIVE:
                d2h_hard = min(
                    d2h_hard,
                    config.max_collective_d2h_requests
                    * config.d2h_quantum_bytes,
                )
                pfs_hard = min(
                    pfs_hard,
                    config.max_collective_pfs_requests
                    * config.pfs_quantum_bytes,
                )
                d2h_safe = min(d2h_safe, d2h_hard)
                pfs_safe = min(pfs_safe, pfs_hard)
            if credit.d2h_budget_bytes > d2h_hard or credit.pfs_budget_bytes > pfs_hard:
                raise ValueError(f"rank {rank.rank} credit exceeds a hard window capacity")
            if credit.d2h_spill_bytes != max(0, credit.d2h_budget_bytes - d2h_safe):
                raise ValueError(f"rank {rank.rank} D2H spill accounting mismatch")
            if credit.pfs_spill_bytes != max(0, credit.pfs_budget_bytes - pfs_safe):
                raise ValueError(f"rank {rank.rank} PFS spill accounting mismatch")
            cumulative_pfs += credit.pfs_budget_bytes
            if cumulative_pfs > cumulative_d2h:
                raise ValueError(f"rank {rank.rank} PFS credit exceeds the D2H-ready prefix")
            cumulative_d2h += credit.d2h_budget_bytes
            if credit.max_pfs_inflight_bytes != config.max_pfs_inflight_bytes:
                raise ValueError("PFS in-flight cap differs from controller configuration")
        if group_eligible:
            validated_templates.setdefault(progress_key, rank_plan)


__all__ = [
    "AdmissionMode",
    "ControllerConfig",
    "CreditPlan",
    "NS_PER_SECOND",
    "PPM",
    "PlannerInput",
    "RankCreditPlan",
    "RankProgress",
    "Stage",
    "StageProgress",
    "TailFeedback",
    "TempoV4Controller",
    "UNLIMITED_BUDGET",
    "WindowCredit",
    "WindowKind",
    "WindowSpec",
    "apply_tail_feedback",
    "validate_plan",
]
