"""Pure ingress admission state for cache-aware elastic P/D serving.

The controller commits LOCAL or REMOTE before either execution path starts.
It never changes a route inside a KV connector.  Local work is limited by a
compute-time budget, remote work by missing-KV bytes, and requests for which
neither path is safe remain explicitly queued.

Arrival gaps are only a fast regime signal.  Per-request conservative latency
bounds, cache residency, deadline feasibility, and credits still gate every
decision.  A high-load epoch can recover through one bounded remote probe.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
import math
import statistics
import threading


POLICY_ID = "tempo-elastic-pd-ingress-dual-credit-442"


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _finite_nonnegative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


class ElasticRoute(str, Enum):
    LOCAL = "decoder_local_chunked_prefill"
    REMOTE = "official_lmcache_remote_prefill"
    QUEUE = "bounded_ingress_queue"


class ElasticRegime(str, Enum):
    OBSERVING = "observing"
    REMOTE_STABLE = "remote_stable"
    DEFLECT_ACTIVE = "deflect_active"
    RECOVERY_PROBE = "recovery_probe"


class CacheResidency(str, Enum):
    UNKNOWN = "unknown"
    MISS = "confirmed_miss"
    P_ONLY = "prefill_only"
    D_ONLY = "decode_only"
    BOTH = "prefill_and_decode"


class ElasticPhase(str, Enum):
    QUEUED = "queued"
    LOCAL_RESERVED = "local_reserved"
    REMOTE_RESERVED = "remote_reserved"
    STARTED = "started"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class ElasticConfig:
    local_compute_budget_us: int
    remote_kv_budget_bytes: int
    arrival_window: int = 4
    enter_high_gap_ns: int = 39_000_000
    exit_high_gap_ns: int = 78_000_000
    exit_consecutive_windows: int = 2
    route_margin_ms: float = 5.0
    spill_regression_budget_ms: float = 5.0

    def __post_init__(self) -> None:
        _positive_int("local_compute_budget_us", self.local_compute_budget_us)
        _positive_int("remote_kv_budget_bytes", self.remote_kv_budget_bytes)
        if type(self.arrival_window) is not int or self.arrival_window < 2:
            raise ValueError("arrival_window must be at least two")
        _positive_int("enter_high_gap_ns", self.enter_high_gap_ns)
        _positive_int("exit_high_gap_ns", self.exit_high_gap_ns)
        if self.exit_high_gap_ns <= self.enter_high_gap_ns:
            raise ValueError("exit gap must exceed enter gap")
        _positive_int("exit_consecutive_windows", self.exit_consecutive_windows)
        _finite_nonnegative("route_margin_ms", self.route_margin_ms)
        _finite_nonnegative(
            "spill_regression_budget_ms", self.spill_regression_budget_ms
        )


@dataclass(frozen=True)
class ElasticRequest:
    request_id: str
    arrival_ns: int
    cache_residency: CacheResidency
    local_compute_cost_us: int
    remote_kv_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be nonempty")
        _nonnegative_int("arrival_ns", self.arrival_ns)
        if not isinstance(self.cache_residency, CacheResidency):
            raise TypeError("cache_residency must be CacheResidency")
        _positive_int("local_compute_cost_us", self.local_compute_cost_us)
        _nonnegative_int("remote_kv_bytes", self.remote_kv_bytes)


@dataclass(frozen=True)
class ElasticEstimate:
    local_upper_bound_ms: float
    remote_upper_bound_ms: float
    uncertainty_ms: float
    remaining_deadline_ms: float
    local_tbt_safe: bool
    remote_backend_available: bool
    remote_evidence_valid: bool

    def __post_init__(self) -> None:
        for name in (
            "local_upper_bound_ms",
            "remote_upper_bound_ms",
            "uncertainty_ms",
            "remaining_deadline_ms",
        ):
            _finite_nonnegative(name, getattr(self, name))
        for name in (
            "local_tbt_safe",
            "remote_backend_available",
            "remote_evidence_valid",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")


@dataclass(frozen=True)
class ElasticDecision:
    request_id: str
    route: ElasticRoute
    reason: str
    phase: ElasticPhase
    regime: ElasticRegime
    attempt: int
    median_gap_ns: int | None
    cache_residency: CacheResidency
    local_score_ms: float
    remote_score_ms: float
    local_compute_used_before_us: int
    local_compute_budget_us: int
    remote_kv_used_before_bytes: int
    remote_kv_budget_bytes: int
    local_credit_available: bool
    remote_credit_available: bool
    local_deadline_feasible: bool
    remote_deadline_feasible: bool
    remote_probe: bool
    policy_id: str = POLICY_ID


@dataclass(frozen=True)
class _Entry:
    request: ElasticRequest
    estimate: ElasticEstimate
    decision: ElasticDecision
    phase: ElasticPhase


class ElasticPDController:
    """Thread-safe ingress route ledger with exact dual-credit ownership."""

    def __init__(self, config: ElasticConfig) -> None:
        if not isinstance(config, ElasticConfig):
            raise TypeError("config must be ElasticConfig")
        self.config = config
        self._gaps: deque[int] = deque(maxlen=config.arrival_window)
        self._last_arrival_ns: int | None = None
        self._median_gap_ns: int | None = None
        self._regime = ElasticRegime.OBSERVING
        self._low_windows = 0
        self._remote_probe_request: str | None = None
        self._local_owned: dict[str, int] = {}
        self._remote_owned: dict[str, int] = {}
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    @property
    def regime(self) -> ElasticRegime:
        with self._lock:
            return self._regime

    @property
    def local_compute_used_us(self) -> int:
        with self._lock:
            return sum(self._local_owned.values())

    @property
    def remote_kv_used_bytes(self) -> int:
        with self._lock:
            return sum(self._remote_owned.values())

    def submit(
        self, request: ElasticRequest, estimate: ElasticEstimate
    ) -> ElasticDecision:
        if not isinstance(request, ElasticRequest):
            raise TypeError("request must be ElasticRequest")
        if not isinstance(estimate, ElasticEstimate):
            raise TypeError("estimate must be ElasticEstimate")
        with self._lock:
            prior = self._entries.get(request.request_id)
            if prior is not None:
                if prior.request != request:
                    raise ValueError("request identity changed")
                return prior.decision
            self._observe_arrival(request.arrival_ns)
            decision = self._evaluate(request, estimate, attempt=1)
            self._entries[request.request_id] = _Entry(
                request, estimate, decision, decision.phase
            )
            return decision

    def retry(self, request_id: str, estimate: ElasticEstimate) -> ElasticDecision:
        if not isinstance(estimate, ElasticEstimate):
            raise TypeError("estimate must be ElasticEstimate")
        with self._lock:
            entry = self._get(request_id)
            if entry.phase is not ElasticPhase.QUEUED:
                raise ValueError("only queued requests can be retried")
            decision = self._evaluate(
                entry.request, estimate, attempt=entry.decision.attempt + 1
            )
            self._entries[request_id] = replace(
                entry, estimate=estimate, decision=decision, phase=decision.phase
            )
            return decision

    def mark_started(self, request_id: str) -> None:
        with self._lock:
            entry = self._get(request_id)
            if entry.phase not in {
                ElasticPhase.LOCAL_RESERVED,
                ElasticPhase.REMOTE_RESERVED,
            }:
                raise ValueError("only a reserved request can start")
            self._entries[request_id] = replace(
                entry, phase=ElasticPhase.STARTED,
                decision=replace(entry.decision, phase=ElasticPhase.STARTED),
            )

    def fallback_remote_before_start(
        self, request_id: str, estimate: ElasticEstimate
    ) -> ElasticDecision:
        """Release a failed remote reservation and try local before P starts."""
        if not isinstance(estimate, ElasticEstimate):
            raise TypeError("estimate must be ElasticEstimate")
        with self._lock:
            entry = self._get(request_id)
            if entry.phase is not ElasticPhase.REMOTE_RESERVED:
                raise ValueError("remote fallback is allowed only before start")
            self._release_remote(request_id)
            if entry.decision.remote_probe:
                self._remote_probe_request = None
                self._regime = ElasticRegime.DEFLECT_ACTIVE
            local_only = replace(
                estimate,
                remote_backend_available=False,
                remote_evidence_valid=False,
            )
            decision = self._evaluate(
                entry.request, local_only, attempt=entry.decision.attempt + 1
            )
            decision = replace(
                decision,
                reason=("remote_prestart_failure_to_local"
                        if decision.route is ElasticRoute.LOCAL
                        else "remote_prestart_failure_queued"),
            )
            self._entries[request_id] = replace(
                entry, estimate=local_only, decision=decision, phase=decision.phase
            )
            return decision

    def complete(self, request_id: str, *, remote_probe_success: bool | None = None) -> None:
        with self._lock:
            entry = self._get(request_id)
            if entry.phase is not ElasticPhase.STARTED:
                raise ValueError("only a started request can complete")
            if entry.decision.remote_probe:
                if type(remote_probe_success) is not bool:
                    raise ValueError("remote probe completion requires success bool")
                self._remote_probe_request = None
                self._regime = (
                    ElasticRegime.REMOTE_STABLE
                    if remote_probe_success
                    else ElasticRegime.DEFLECT_ACTIVE
                )
            elif remote_probe_success is not None:
                raise ValueError("non-probe completion must not report probe success")
            self._release_credit(entry)
            self._entries[request_id] = replace(
                entry, phase=ElasticPhase.COMPLETE,
                decision=replace(entry.decision, phase=ElasticPhase.COMPLETE),
            )

    def fail(self, request_id: str) -> None:
        with self._lock:
            entry = self._get(request_id)
            if entry.phase in {ElasticPhase.COMPLETE, ElasticPhase.FAILED}:
                raise ValueError("terminal request cannot fail again")
            if entry.decision.remote_probe:
                self._remote_probe_request = None
                self._regime = ElasticRegime.DEFLECT_ACTIVE
            self._release_credit(entry)
            self._entries[request_id] = replace(
                entry, phase=ElasticPhase.FAILED,
                decision=replace(entry.decision, phase=ElasticPhase.FAILED),
            )

    def decision(self, request_id: str) -> ElasticDecision:
        with self._lock:
            return self._get(request_id).decision

    def _observe_arrival(self, now_ns: int) -> None:
        if self._last_arrival_ns is not None:
            gap = now_ns - self._last_arrival_ns
            if gap <= 0:
                raise ValueError("arrival clock must advance")
            self._gaps.append(gap)
        self._last_arrival_ns = now_ns
        if len(self._gaps) < self.config.arrival_window:
            return
        self._median_gap_ns = int(statistics.median(self._gaps))
        if self._median_gap_ns <= self.config.enter_high_gap_ns:
            self._regime = ElasticRegime.DEFLECT_ACTIVE
            self._low_windows = 0
            return
        if self._median_gap_ns >= self.config.exit_high_gap_ns:
            if self._regime is ElasticRegime.DEFLECT_ACTIVE:
                self._low_windows += 1
                if self._low_windows >= self.config.exit_consecutive_windows:
                    self._regime = ElasticRegime.RECOVERY_PROBE
            elif self._regime is ElasticRegime.OBSERVING:
                self._regime = ElasticRegime.REMOTE_STABLE
            return
        self._low_windows = 0
        if self._regime is ElasticRegime.OBSERVING:
            self._regime = ElasticRegime.REMOTE_STABLE

    def _evaluate(
        self, request: ElasticRequest, estimate: ElasticEstimate, *, attempt: int
    ) -> ElasticDecision:
        local_used = sum(self._local_owned.values())
        remote_used = sum(self._remote_owned.values())
        local_score = estimate.local_upper_bound_ms + estimate.uncertainty_ms
        remote_score = estimate.remote_upper_bound_ms + estimate.uncertainty_ms
        local_credit = (
            local_used + request.local_compute_cost_us
            <= self.config.local_compute_budget_us
        )
        remote_credit = (
            remote_used + request.remote_kv_bytes
            <= self.config.remote_kv_budget_bytes
        )
        local_deadline = local_score <= estimate.remaining_deadline_ms
        remote_deadline = remote_score <= estimate.remaining_deadline_ms
        local_intrinsic = estimate.local_tbt_safe and local_deadline
        remote_intrinsic = (
            estimate.remote_backend_available
            and estimate.remote_evidence_valid
            and remote_deadline
        )
        local_feasible = local_intrinsic and local_credit
        remote_feasible = remote_intrinsic and remote_credit
        remote_advantage = local_score - remote_score

        preferred = ElasticRoute.LOCAL
        if request.cache_residency is CacheResidency.P_ONLY:
            if remote_score <= local_score + self.config.spill_regression_budget_ms:
                preferred = ElasticRoute.REMOTE
        elif request.cache_residency in {CacheResidency.D_ONLY, CacheResidency.BOTH}:
            preferred = ElasticRoute.LOCAL
        elif remote_advantage >= self.config.route_margin_ms:
            preferred = ElasticRoute.REMOTE

        if self._regime is ElasticRegime.DEFLECT_ACTIVE:
            if remote_advantage < self.config.route_margin_ms:
                preferred = ElasticRoute.LOCAL
        remote_probe = False
        if self._regime is ElasticRegime.RECOVERY_PROBE:
            if (
                self._remote_probe_request is None
                and remote_intrinsic
                and remote_score
                <= local_score + self.config.spill_regression_budget_ms
            ):
                preferred = ElasticRoute.REMOTE
                remote_probe = True
            else:
                preferred = ElasticRoute.LOCAL

        route = ElasticRoute.QUEUE
        reason = "neither_path_admissible"
        if preferred is ElasticRoute.LOCAL and local_feasible:
            route, reason = ElasticRoute.LOCAL, "local_preferred_and_feasible"
        elif preferred is ElasticRoute.REMOTE and remote_feasible:
            route, reason = ElasticRoute.REMOTE, "remote_preferred_and_feasible"
        else:
            alternate = (
                ElasticRoute.REMOTE
                if preferred is ElasticRoute.LOCAL
                else ElasticRoute.LOCAL
            )
            preferred_score = local_score if preferred is ElasticRoute.LOCAL else remote_score
            alternate_score = remote_score if alternate is ElasticRoute.REMOTE else local_score
            alternate_feasible = (
                remote_feasible if alternate is ElasticRoute.REMOTE else local_feasible
            )
            if (
                alternate_feasible
                and alternate_score
                <= preferred_score + self.config.spill_regression_budget_ms
            ):
                route = alternate
                reason = "bounded_spill_to_alternate"
                remote_probe = (
                    route is ElasticRoute.REMOTE
                    and self._regime is ElasticRegime.RECOVERY_PROBE
                )
            elif preferred is ElasticRoute.LOCAL and local_intrinsic and not local_credit:
                reason = "local_credit_exhausted_remote_too_costly"
            elif preferred is ElasticRoute.REMOTE and remote_intrinsic and not remote_credit:
                reason = "remote_credit_exhausted_local_too_costly"
            elif not local_deadline and not remote_deadline:
                reason = "both_deadlines_infeasible"
            elif not estimate.local_tbt_safe and not remote_intrinsic:
                reason = "local_tbt_unsafe_remote_unavailable"

        phase = ElasticPhase.QUEUED
        if route is ElasticRoute.LOCAL:
            self._local_owned[request.request_id] = request.local_compute_cost_us
            phase = ElasticPhase.LOCAL_RESERVED
            remote_probe = False
        elif route is ElasticRoute.REMOTE:
            self._remote_owned[request.request_id] = request.remote_kv_bytes
            phase = ElasticPhase.REMOTE_RESERVED
            if remote_probe:
                if self._remote_probe_request is not None:
                    raise AssertionError("multiple recovery probes")
                self._remote_probe_request = request.request_id

        return ElasticDecision(
            request_id=request.request_id,
            route=route,
            reason=reason,
            phase=phase,
            regime=self._regime,
            attempt=attempt,
            median_gap_ns=self._median_gap_ns,
            cache_residency=request.cache_residency,
            local_score_ms=local_score,
            remote_score_ms=remote_score,
            local_compute_used_before_us=local_used,
            local_compute_budget_us=self.config.local_compute_budget_us,
            remote_kv_used_before_bytes=remote_used,
            remote_kv_budget_bytes=self.config.remote_kv_budget_bytes,
            local_credit_available=local_credit,
            remote_credit_available=remote_credit,
            local_deadline_feasible=local_deadline,
            remote_deadline_feasible=remote_deadline,
            remote_probe=remote_probe,
        )

    def _release_credit(self, entry: _Entry) -> None:
        request_id = entry.request.request_id
        if entry.decision.route is ElasticRoute.LOCAL:
            if self._local_owned.pop(request_id, None) is None:
                raise AssertionError("missing local credit")
        elif entry.decision.route is ElasticRoute.REMOTE:
            if self._remote_owned.pop(request_id, None) is None:
                raise AssertionError("missing remote credit")

    def _release_remote(self, request_id: str) -> None:
        if self._remote_owned.pop(request_id, None) is None:
            raise AssertionError("missing remote credit")

    def _get(self, request_id: str) -> _Entry:
        entry = self._entries.get(request_id)
        if entry is None:
            raise ValueError("unknown request_id")
        return entry
