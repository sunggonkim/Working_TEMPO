"""Endpoint-completion controller for TEMPO Elastic-PD.

This module owns admission only.  It does not read sysfs, fetch ``/metrics``,
or infer a physical switch bottleneck.  Callers provide deployment-profile
priors and endpoint-owned first-response completions.  The controller keeps
four independent in-flight resources:

* decoder-local prefill token-milliseconds;
* remote-P prefill token-milliseconds;
* remote KV bytes; and
* remote semantic transfer/install operations.

First-response feedback is normalized by the matching route's idle TTFT
prior.  The resulting service stretch adjusts the static E2E prior without
confusing full-stream decode time with prefill/handoff occupancy.  The same
normalization may observe route-pinned tenants passively, but passive samples
never grant or release admission credit.  Failed or SLO-violating routes
recover only through a bounded explicit probe.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import threading
from typing import Deque


SCHEMA = "tempo-pd-endpoint-controller-v1"


class EndpointRoute(str, Enum):
    LOCAL = "decoder_local_chunked_prefill"
    REMOTE = "official_lmcache_remote_prefill"
    QUEUE = "bounded_ingress_queue"


class RouteHealth(str, Enum):
    GOOD = "good"
    SKIP = "skip"
    DENIED = "denied"
    PROBE = "probe"


@dataclass(frozen=True)
class EndpointAdmissionConfig:
    local_token_ms_window: int
    remote_prefill_token_ms_window: int
    remote_kv_bytes_window: int
    remote_semantic_ops_window: int
    feedback_history: int
    feedback_quantile: float
    minimum_feedback: int
    route_margin_ms: float
    feedback_fresh_ns: int
    probe_after_ns: int
    denied_probe_after_ns: int

    def __post_init__(self) -> None:
        for name, value in (
            ("local_token_ms_window", self.local_token_ms_window),
            ("remote_prefill_token_ms_window", self.remote_prefill_token_ms_window),
            ("remote_kv_bytes_window", self.remote_kv_bytes_window),
            ("remote_semantic_ops_window", self.remote_semantic_ops_window),
            ("feedback_history", self.feedback_history),
            ("minimum_feedback", self.minimum_feedback),
            ("feedback_fresh_ns", self.feedback_fresh_ns),
            ("probe_after_ns", self.probe_after_ns),
            ("denied_probe_after_ns", self.denied_probe_after_ns),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        if self.minimum_feedback > self.feedback_history:
            raise ValueError("minimum_feedback cannot exceed feedback_history")
        if self.probe_after_ns > self.feedback_fresh_ns:
            raise ValueError(
                "probe_after_ns cannot exceed feedback_fresh_ns"
            )
        if (
            isinstance(self.feedback_quantile, bool)
            or not isinstance(self.feedback_quantile, (int, float))
            or not math.isfinite(float(self.feedback_quantile))
            or not 0.5 <= float(self.feedback_quantile) <= 1.0
        ):
            raise ValueError("feedback_quantile must be finite and in [0.5, 1]")
        if (
            isinstance(self.route_margin_ms, bool)
            or not isinstance(self.route_margin_ms, (int, float))
            or not math.isfinite(float(self.route_margin_ms))
            or float(self.route_margin_ms) < 0.0
        ):
            raise ValueError("route_margin_ms must be finite and non-negative")


@dataclass(frozen=True)
class EndpointWork:
    local_token_ms: int
    remote_prefill_token_ms: int
    remote_kv_bytes: int
    remote_semantic_ops: int

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")


@dataclass(frozen=True)
class EndpointRequest:
    request_id: str
    local_e2e_prior_ms: float
    remote_e2e_prior_ms: float
    local_ttft_prior_ms: float
    remote_ttft_prior_ms: float
    uncertainty_ms: float
    e2e_deadline_ms: float
    work: EndpointWork
    local_allowed: bool = True
    remote_allowed: bool = True

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id.strip():
            raise ValueError("request_id must be nonempty")
        for name, value in (
            ("local_e2e_prior_ms", self.local_e2e_prior_ms),
            ("remote_e2e_prior_ms", self.remote_e2e_prior_ms),
            ("local_ttft_prior_ms", self.local_ttft_prior_ms),
            ("remote_ttft_prior_ms", self.remote_ttft_prior_ms),
            ("e2e_deadline_ms", self.e2e_deadline_ms),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.uncertainty_ms, bool)
            or not isinstance(self.uncertainty_ms, (int, float))
            or not math.isfinite(float(self.uncertainty_ms))
            or float(self.uncertainty_ms) < 0.0
        ):
            raise ValueError("uncertainty_ms must be finite and non-negative")
        if not isinstance(self.work, EndpointWork):
            raise TypeError("work must be EndpointWork")
        if type(self.local_allowed) is not bool or type(self.remote_allowed) is not bool:
            raise TypeError("route-allowed flags must be bool")
        if not self.local_allowed and not self.remote_allowed:
            raise ValueError("at least one route must be allowed")

    def e2e_prior_ms(self, route: EndpointRoute) -> float:
        if route is EndpointRoute.LOCAL:
            return float(self.local_e2e_prior_ms)
        if route is EndpointRoute.REMOTE:
            return float(self.remote_e2e_prior_ms)
        raise ValueError("queue has no E2E prior")

    def ttft_prior_ms(self, route: EndpointRoute) -> float:
        if route is EndpointRoute.LOCAL:
            return float(self.local_ttft_prior_ms)
        if route is EndpointRoute.REMOTE:
            return float(self.remote_ttft_prior_ms)
        raise ValueError("queue has no TTFT prior")

    def allowed(self, route: EndpointRoute) -> bool:
        if route is EndpointRoute.LOCAL:
            return self.local_allowed
        if route is EndpointRoute.REMOTE:
            return self.remote_allowed
        return False


@dataclass(frozen=True)
class EndpointDecision:
    request_id: str
    route: EndpointRoute
    reason: str
    decided_ns: int
    local_score_ms: float
    remote_score_ms: float
    local_multiplier: float
    remote_multiplier: float
    local_state: RouteHealth
    remote_state: RouteHealth
    probe: bool
    resource_used_before: dict[str, int]
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("endpoint decision schema mismatch")
        if not isinstance(self.route, EndpointRoute):
            raise TypeError("route must be EndpointRoute")
        if not isinstance(self.local_state, RouteHealth) or not isinstance(
            self.remote_state, RouteHealth
        ):
            raise TypeError("route states must be RouteHealth")


@dataclass
class _RouteFeedback:
    stretches: Deque[float]
    last_feedback_ns: int | None = None
    denied_until_ns: int = 0
    probe_request_id: str | None = None
    failures: int = 0
    active_samples: int = 0
    passive_samples: int = 0
    passive_failures: int = 0
    active_ignored_while_unhealthy: int = 0
    passive_ignored_while_unhealthy: int = 0


@dataclass(frozen=True)
class _Reservation:
    request: EndpointRequest
    decision: EndpointDecision


@dataclass(frozen=True)
class _ExternalReservation:
    route: EndpointRoute
    work: EndpointWork
    prior_ttft_ms: float
    e2e_deadline_ms: float
    started_ns: int


def _nearest_rank(values: Deque[float], quantile: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    rank = max(1, math.ceil(float(quantile) * len(ordered)))
    return float(ordered[rank - 1])


class EndpointFeedbackController:
    """Atomic dual-route admission with exact first-response release."""

    def __init__(self, config: EndpointAdmissionConfig) -> None:
        if not isinstance(config, EndpointAdmissionConfig):
            raise TypeError("config must be EndpointAdmissionConfig")
        self.config = config
        self._feedback = {
            route: _RouteFeedback(deque(maxlen=config.feedback_history))
            for route in (EndpointRoute.LOCAL, EndpointRoute.REMOTE)
        }
        self._inflight: dict[str, _Reservation] = {}
        self._external_inflight: dict[str, _ExternalReservation] = {}
        self._completed: set[str] = set()
        self._passive_completed: set[str] = set()
        self._local_token_ms_used = 0
        self._remote_prefill_token_ms_used = 0
        self._remote_kv_bytes_used = 0
        self._remote_semantic_ops_used = 0
        self._external_local_token_ms_used = 0
        self._external_remote_prefill_token_ms_used = 0
        self._external_remote_kv_bytes_used = 0
        self._external_remote_semantic_ops_used = 0
        self._lock = threading.Lock()

    def _multiplier(self, route: EndpointRoute) -> float:
        feedback = self._feedback[route]
        if len(feedback.stretches) < self.config.minimum_feedback:
            return 1.0
        return max(
            1.0,
            _nearest_rank(feedback.stretches, self.config.feedback_quantile),
        )

    def _score(self, request: EndpointRequest, route: EndpointRoute) -> float:
        multiplier = self._multiplier(route)
        dynamic_ttft = request.ttft_prior_ms(route) * multiplier
        service_inflation = max(0.0, dynamic_ttft - request.ttft_prior_ms(route))
        return (
            request.e2e_prior_ms(route)
            + service_inflation
            + float(request.uncertainty_ms)
        )

    def _owned_resource_snapshot(self) -> dict[str, int]:
        return {
            "local_token_ms": self._local_token_ms_used,
            "remote_prefill_token_ms": self._remote_prefill_token_ms_used,
            "remote_kv_bytes": self._remote_kv_bytes_used,
            "remote_semantic_ops": self._remote_semantic_ops_used,
        }

    def _external_resource_snapshot(self) -> dict[str, int]:
        return {
            "local_token_ms": self._external_local_token_ms_used,
            "remote_prefill_token_ms": (
                self._external_remote_prefill_token_ms_used),
            "remote_kv_bytes": self._external_remote_kv_bytes_used,
            "remote_semantic_ops": self._external_remote_semantic_ops_used,
        }

    def _resource_snapshot(self) -> dict[str, int]:
        owned = self._owned_resource_snapshot()
        external = self._external_resource_snapshot()
        return {name: owned[name] + external[name] for name in owned}

    def _fits(self, request: EndpointRequest, route: EndpointRoute) -> bool:
        work = request.work
        if route is EndpointRoute.LOCAL:
            return (
                self._local_token_ms_used
                + self._external_local_token_ms_used
                + work.local_token_ms
                <= self.config.local_token_ms_window
            )
        if route is EndpointRoute.REMOTE:
            return (
                self._remote_prefill_token_ms_used
                + self._external_remote_prefill_token_ms_used
                + work.remote_prefill_token_ms
                <= self.config.remote_prefill_token_ms_window
                and self._remote_kv_bytes_used
                + self._external_remote_kv_bytes_used
                + work.remote_kv_bytes
                <= self.config.remote_kv_bytes_window
                and self._remote_semantic_ops_used
                + self._external_remote_semantic_ops_used
                + work.remote_semantic_ops
                <= self.config.remote_semantic_ops_window
            )
        return False

    def _reserve(self, request: EndpointRequest, route: EndpointRoute) -> None:
        work = request.work
        if route is EndpointRoute.LOCAL:
            self._local_token_ms_used += work.local_token_ms
        elif route is EndpointRoute.REMOTE:
            self._remote_prefill_token_ms_used += work.remote_prefill_token_ms
            self._remote_kv_bytes_used += work.remote_kv_bytes
            self._remote_semantic_ops_used += work.remote_semantic_ops
        else:
            raise ValueError("cannot reserve queue route")

    def _release(self, reservation: _Reservation) -> None:
        work = reservation.request.work
        route = reservation.decision.route
        if route is EndpointRoute.LOCAL:
            self._local_token_ms_used -= work.local_token_ms
        elif route is EndpointRoute.REMOTE:
            self._remote_prefill_token_ms_used -= work.remote_prefill_token_ms
            self._remote_kv_bytes_used -= work.remote_kv_bytes
            self._remote_semantic_ops_used -= work.remote_semantic_ops
        else:
            raise ValueError("queue route cannot own resources")
        if any(value < 0 for value in self._owned_resource_snapshot().values()):
            raise RuntimeError("endpoint resource ownership became negative")

    def _reserve_external(self, reservation: _ExternalReservation) -> None:
        work = reservation.work
        if reservation.route is EndpointRoute.LOCAL:
            self._external_local_token_ms_used += work.local_token_ms
        elif reservation.route is EndpointRoute.REMOTE:
            self._external_remote_prefill_token_ms_used += (
                work.remote_prefill_token_ms)
            self._external_remote_kv_bytes_used += work.remote_kv_bytes
            self._external_remote_semantic_ops_used += work.remote_semantic_ops
        else:
            raise ValueError("external queue route cannot own resources")

    def _release_external(self, reservation: _ExternalReservation) -> None:
        work = reservation.work
        if reservation.route is EndpointRoute.LOCAL:
            self._external_local_token_ms_used -= work.local_token_ms
        elif reservation.route is EndpointRoute.REMOTE:
            self._external_remote_prefill_token_ms_used -= (
                work.remote_prefill_token_ms)
            self._external_remote_kv_bytes_used -= work.remote_kv_bytes
            self._external_remote_semantic_ops_used -= work.remote_semantic_ops
        else:
            raise ValueError("external queue route cannot own resources")
        if any(
            value < 0 for value in self._external_resource_snapshot().values()
        ):
            raise RuntimeError("external endpoint ownership became negative")

    def _base_state(self, route: EndpointRoute, now_ns: int) -> RouteHealth:
        feedback = self._feedback[route]
        if feedback.probe_request_id is not None:
            return RouteHealth.PROBE
        if now_ns < feedback.denied_until_ns:
            return RouteHealth.DENIED
        if (
            feedback.last_feedback_ns is not None
            and now_ns - feedback.last_feedback_ns > self.config.feedback_fresh_ns
        ):
            return RouteHealth.SKIP
        return RouteHealth.GOOD

    def _probe_due(self, route: EndpointRoute, now_ns: int) -> bool:
        feedback = self._feedback[route]
        if feedback.probe_request_id is not None:
            return False
        if now_ns < feedback.denied_until_ns:
            return False
        if feedback.last_feedback_ns is None:
            return False
        age = now_ns - feedback.last_feedback_ns
        delay = (
            self.config.denied_probe_after_ns
            if feedback.failures
            else self.config.probe_after_ns
        )
        return age >= delay

    def submit(self, request: EndpointRequest, *, now_ns: int) -> EndpointDecision:
        if not isinstance(request, EndpointRequest):
            raise TypeError("request must be EndpointRequest")
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative int")
        with self._lock:
            if (
                request.request_id in self._inflight
                or request.request_id in self._external_inflight
                or request.request_id in self._completed
                or request.request_id in self._passive_completed
            ):
                raise ValueError("request_id already owned")
            before = self._resource_snapshot()
            scores = {
                EndpointRoute.LOCAL: self._score(request, EndpointRoute.LOCAL),
                EndpointRoute.REMOTE: self._score(request, EndpointRoute.REMOTE),
            }
            states = {
                route: self._base_state(route, now_ns)
                for route in (EndpointRoute.LOCAL, EndpointRoute.REMOTE)
            }

            healthy = [
                route
                for route in (EndpointRoute.LOCAL, EndpointRoute.REMOTE)
                if request.allowed(route)
                and states[route] is RouteHealth.GOOD
                and self._fits(request, route)
                and scores[route] <= request.e2e_deadline_ms
            ]
            healthy.sort(key=lambda route: (scores[route], route.value))

            # A route with enough feedback can be skipped when it is
            # decisively dominated.  This is request-local; no physical
            # bottleneck label is assigned.
            if len(healthy) == 2:
                best, other = healthy
                if (
                    len(self._feedback[other].stretches)
                    >= self.config.minimum_feedback
                    and scores[other]
                    > scores[best] + float(self.config.route_margin_ms)
                ):
                    states[other] = RouteHealth.SKIP
                    healthy = [best]

            probe_candidates = [
                route
                for route in (EndpointRoute.LOCAL, EndpointRoute.REMOTE)
                if request.allowed(route)
                and states[route] in {RouteHealth.SKIP, RouteHealth.DENIED}
                and self._probe_due(route, now_ns)
                and self._fits(request, route)
                and scores[route] <= request.e2e_deadline_ms
            ]
            probe_candidates.sort(key=lambda route: (
                self._feedback[route].last_feedback_ns or now_ns,
                scores[route],
                route.value,
            ))

            probe = False
            if probe_candidates:
                route = probe_candidates[0]
                probe = True
                states[route] = RouteHealth.PROBE
                reason = f"endpoint_{route.name.lower()}_recovery_probe"
            elif healthy:
                route = healthy[0]
                reason = f"endpoint_{route.name.lower()}_lower_service_cost"
            else:
                route = EndpointRoute.QUEUE
                reason = "endpoint_no_fresh_deadline_safe_window"

            decision = EndpointDecision(
                request_id=request.request_id,
                route=route,
                reason=reason,
                decided_ns=now_ns,
                local_score_ms=scores[EndpointRoute.LOCAL],
                remote_score_ms=scores[EndpointRoute.REMOTE],
                local_multiplier=self._multiplier(EndpointRoute.LOCAL),
                remote_multiplier=self._multiplier(EndpointRoute.REMOTE),
                local_state=states[EndpointRoute.LOCAL],
                remote_state=states[EndpointRoute.REMOTE],
                probe=probe,
                resource_used_before=before,
            )
            if route is not EndpointRoute.QUEUE:
                self._reserve(request, route)
                self._inflight[request.request_id] = _Reservation(request, decision)
                if probe:
                    self._feedback[route].probe_request_id = request.request_id
            return decision

    def observe_first_response(
        self,
        request_id: str,
        *,
        observed_ttft_ms: float,
        now_ns: int,
    ) -> bool:
        if type(request_id) is not str or not request_id:
            raise ValueError("request_id must be nonempty")
        if (
            isinstance(observed_ttft_ms, bool)
            or not isinstance(observed_ttft_ms, (int, float))
            or not math.isfinite(float(observed_ttft_ms))
            or float(observed_ttft_ms) <= 0.0
        ):
            raise ValueError("observed_ttft_ms must be finite and positive")
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative int")
        with self._lock:
            reservation = self._inflight.get(request_id)
            if reservation is None:
                raise ValueError("request has no in-flight endpoint reservation")
            if now_ns < reservation.decision.decided_ns:
                raise ValueError("feedback time precedes decision")
            route = reservation.decision.route
            feedback = self._feedback[route]
            probe_matches = feedback.probe_request_id == request_id
            if reservation.decision.probe != probe_matches:
                raise RuntimeError("endpoint probe ownership differs")
            self._release(reservation)
            self._inflight.pop(request_id)
            if probe_matches:
                feedback.probe_request_id = None
            prior = reservation.request.ttft_prior_ms(route)
            stretch = float(observed_ttft_ms) / prior
            slo_violation = (
                float(observed_ttft_ms) > reservation.request.e2e_deadline_ms)
            accepted = False
            if slo_violation:
                feedback.stretches.append(stretch)
                feedback.last_feedback_ns = now_ns
                feedback.active_samples += 1
                feedback.failures += 1
                feedback.denied_until_ns = now_ns + self.config.denied_probe_after_ns
                accepted = True
            elif reservation.decision.probe:
                # Recovery evidence must not remain dominated by stale
                # pre-failure samples.
                feedback.stretches.clear()
                feedback.stretches.append(stretch)
                feedback.last_feedback_ns = now_ns
                feedback.active_samples += 1
                feedback.failures = 0
                feedback.denied_until_ns = 0
                accepted = True
            elif (
                feedback.failures > 0
                or feedback.probe_request_id is not None
                or now_ns < feedback.denied_until_ns
            ):
                # This request was admitted before newer failure/probe state.
                # It releases its own credit but cannot recover the route.
                feedback.active_ignored_while_unhealthy += 1
            else:
                feedback.stretches.append(stretch)
                feedback.last_feedback_ns = now_ns
                feedback.active_samples += 1
                feedback.failures = 0
                feedback.denied_until_ns = 0
                accepted = True
            self._completed.add(request_id)
            return accepted

    def observe_external_start(
        self,
        sample_id: str,
        *,
        route: EndpointRoute,
        work: EndpointWork,
        prior_ttft_ms: float,
        e2e_deadline_ms: float,
        now_ns: int,
    ) -> None:
        """Account route-pinned endpoint work without controlling its route."""
        if type(sample_id) is not str or not sample_id:
            raise ValueError("external sample_id must be nonempty")
        if route not in {EndpointRoute.LOCAL, EndpointRoute.REMOTE}:
            raise ValueError("external route must be local or remote")
        if not isinstance(work, EndpointWork):
            raise TypeError("external work must be EndpointWork")
        for name, value in (
            ("prior_ttft_ms", prior_ttft_ms),
            ("e2e_deadline_ms", e2e_deadline_ms),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"external {name} must be finite and positive")
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative int")
        reservation = _ExternalReservation(
            route=route,
            work=work,
            prior_ttft_ms=float(prior_ttft_ms),
            e2e_deadline_ms=float(e2e_deadline_ms),
            started_ns=now_ns,
        )
        with self._lock:
            if (
                sample_id in self._inflight
                or sample_id in self._external_inflight
                or sample_id in self._completed
                or sample_id in self._passive_completed
            ):
                raise ValueError("external sample_id already observed or owned")
            self._reserve_external(reservation)
            self._external_inflight[sample_id] = reservation

    def observe_external_first_response(
        self, sample_id: str, *, observed_ttft_ms: float, now_ns: int,
    ) -> bool:
        """Release external credit and retain completion-derived route health."""
        if type(sample_id) is not str or not sample_id:
            raise ValueError("external sample_id must be nonempty")
        if (
            isinstance(observed_ttft_ms, bool)
            or not isinstance(observed_ttft_ms, (int, float))
            or not math.isfinite(float(observed_ttft_ms))
            or float(observed_ttft_ms) <= 0.0
        ):
            raise ValueError("external observed_ttft_ms must be finite and positive")
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative int")
        with self._lock:
            reservation = self._external_inflight.get(sample_id)
            if reservation is None:
                raise ValueError("external sample has no in-flight reservation")
            if now_ns < reservation.started_ns:
                raise ValueError("external feedback time precedes admission")
            self._release_external(reservation)
            self._external_inflight.pop(sample_id)
            feedback = self._feedback[reservation.route]
            stretch = (
                float(observed_ttft_ms) / reservation.prior_ttft_ms)
            slo_violation = (
                float(observed_ttft_ms) > reservation.e2e_deadline_ms)
            accepted = False
            if slo_violation:
                feedback.stretches.append(stretch)
                feedback.last_feedback_ns = now_ns
                feedback.failures += 1
                feedback.passive_samples += 1
                feedback.passive_failures += 1
                feedback.denied_until_ns = (
                    now_ns + self.config.denied_probe_after_ns)
                accepted = True
            elif (
                feedback.failures == 0
                and feedback.probe_request_id is None
                and now_ns >= feedback.denied_until_ns
            ):
                feedback.stretches.append(stretch)
                feedback.last_feedback_ns = max(
                    now_ns, feedback.last_feedback_ns or now_ns)
                feedback.passive_samples += 1
                accepted = True
            else:
                feedback.passive_ignored_while_unhealthy += 1
            self._passive_completed.add(sample_id)
            return accepted

    def fail_external(self, sample_id: str, *, now_ns: int) -> None:
        """Release failed external work and deny its observed endpoint."""
        if type(sample_id) is not str or not sample_id:
            raise ValueError("external sample_id must be nonempty")
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative int")
        with self._lock:
            reservation = self._external_inflight.get(sample_id)
            if reservation is None:
                raise ValueError("external sample has no in-flight reservation")
            if now_ns < reservation.started_ns:
                raise ValueError("external failure time precedes admission")
            self._release_external(reservation)
            self._external_inflight.pop(sample_id)
            feedback = self._feedback[reservation.route]
            feedback.last_feedback_ns = now_ns
            feedback.failures += 1
            feedback.passive_failures += 1
            feedback.denied_until_ns = (
                now_ns + self.config.denied_probe_after_ns)
            self._passive_completed.add(sample_id)

    def observe_passive_first_response(
        self,
        sample_id: str,
        *,
        route: EndpointRoute,
        observed_ttft_ms: float,
        prior_ttft_ms: float,
        now_ns: int,
    ) -> bool:
        """Observe route service without granting or releasing admission credit.

        Route-pinned tenants are useful endpoint evidence but are not owned by
        TEMPO admission.  A passive success cannot recover a denied route or
        replace the controller's single explicit probe.
        """

        if type(sample_id) is not str or not sample_id:
            raise ValueError("passive sample_id must be nonempty")
        if route not in {EndpointRoute.LOCAL, EndpointRoute.REMOTE}:
            raise ValueError("passive feedback route must be local or remote")
        for name, value in (
            ("observed_ttft_ms", observed_ttft_ms),
            ("prior_ttft_ms", prior_ttft_ms),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"passive {name} must be finite and positive")
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative int")
        stretch = float(observed_ttft_ms) / float(prior_ttft_ms)
        with self._lock:
            if (
                sample_id in self._inflight
                or sample_id in self._external_inflight
                or sample_id in self._completed
                or sample_id in self._passive_completed
            ):
                raise ValueError("passive sample_id already observed or owned")
            feedback = self._feedback[route]
            accepted = (
                feedback.failures == 0
                and feedback.probe_request_id is None
                and now_ns >= feedback.denied_until_ns
            )
            if accepted:
                feedback.stretches.append(stretch)
                feedback.last_feedback_ns = max(
                    now_ns, feedback.last_feedback_ns or now_ns)
                feedback.passive_samples += 1
            else:
                feedback.passive_ignored_while_unhealthy += 1
            self._passive_completed.add(sample_id)
            return accepted

    def fail_passive(
        self, sample_id: str, *, route: EndpointRoute, now_ns: int,
    ) -> None:
        """Deny a route after an observed failure without touching credits."""

        if type(sample_id) is not str or not sample_id:
            raise ValueError("passive sample_id must be nonempty")
        if route not in {EndpointRoute.LOCAL, EndpointRoute.REMOTE}:
            raise ValueError("passive failure route must be local or remote")
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative int")
        with self._lock:
            if (
                sample_id in self._inflight
                or sample_id in self._external_inflight
                or sample_id in self._completed
                or sample_id in self._passive_completed
            ):
                raise ValueError("passive sample_id already observed or owned")
            feedback = self._feedback[route]
            feedback.last_feedback_ns = max(
                now_ns, feedback.last_feedback_ns or now_ns)
            feedback.failures += 1
            feedback.passive_failures += 1
            feedback.denied_until_ns = (
                now_ns + self.config.denied_probe_after_ns)
            self._passive_completed.add(sample_id)

    def fail(self, request_id: str, *, now_ns: int) -> None:
        if type(request_id) is not str or not request_id:
            raise ValueError("request_id must be nonempty")
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative int")
        with self._lock:
            reservation = self._inflight.get(request_id)
            if reservation is None:
                raise ValueError("request has no in-flight endpoint reservation")
            if now_ns < reservation.decision.decided_ns:
                raise ValueError("failure time precedes decision")
            route = reservation.decision.route
            feedback = self._feedback[route]
            probe_matches = feedback.probe_request_id == request_id
            if reservation.decision.probe != probe_matches:
                raise RuntimeError("endpoint probe ownership differs")
            self._release(reservation)
            self._inflight.pop(request_id)
            if probe_matches:
                feedback.probe_request_id = None
            feedback.last_feedback_ns = now_ns
            feedback.failures += 1
            feedback.denied_until_ns = now_ns + self.config.denied_probe_after_ns
            self._completed.add(request_id)

    def snapshot(self, *, now_ns: int) -> dict[str, object]:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative int")
        with self._lock:
            routes = {}
            for route in (EndpointRoute.LOCAL, EndpointRoute.REMOTE):
                feedback = self._feedback[route]
                routes[route.value] = {
                    "state": self._base_state(route, now_ns).value,
                    "service_multiplier": self._multiplier(route),
                    "feedback_count": len(feedback.stretches),
                    "last_feedback_ns": feedback.last_feedback_ns,
                    "denied_until_ns": feedback.denied_until_ns,
                    "probe_request_id": feedback.probe_request_id,
                    "failures": feedback.failures,
                    "active_samples": feedback.active_samples,
                    "passive_samples": feedback.passive_samples,
                    "passive_failures": feedback.passive_failures,
                    "active_ignored_while_unhealthy": (
                        feedback.active_ignored_while_unhealthy),
                    "passive_ignored_while_unhealthy": (
                        feedback.passive_ignored_while_unhealthy),
                }
            return {
                "schema": SCHEMA,
                "resources": self._resource_snapshot(),
                "owned_resources": self._owned_resource_snapshot(),
                "external_resources": self._external_resource_snapshot(),
                "inflight": len(self._inflight),
                "external_inflight": len(self._external_inflight),
                "completed": len(self._completed),
                "passive_completed": len(self._passive_completed),
                "routes": routes,
            }


__all__ = [
    "EndpointAdmissionConfig",
    "EndpointDecision",
    "EndpointFeedbackController",
    "EndpointRequest",
    "EndpointRoute",
    "EndpointWork",
    "RouteHealth",
    "SCHEMA",
]
