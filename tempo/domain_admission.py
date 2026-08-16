"""Policy-independent resource-domain admission for TEMPO-RD.

The controller is intentionally coarse: one request is admitted against an
explicit route, per-domain service envelopes, and a deadline.  It does not
predict per-step collective gaps and it never infers a domain merely from
topology.  Training and inference adapters can share this state machine while
retaining different request producers and correctness endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Mapping

from tempo.resource_domain import ResourceDomain


@dataclass(frozen=True)
class DomainBudget:
    domain: ResourceDomain
    service_rate_bytes_per_second: int
    max_inflight_bytes: int
    minimum_service_bytes_per_second: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.domain, ResourceDomain):
            raise TypeError("domain must be a ResourceDomain")
        for name, value in (
            ("service_rate_bytes_per_second", self.service_rate_bytes_per_second),
            ("max_inflight_bytes", self.max_inflight_bytes),
            ("minimum_service_bytes_per_second", self.minimum_service_bytes_per_second),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if self.service_rate_bytes_per_second <= 0 or self.max_inflight_bytes <= 0:
            raise ValueError("service rate and max inflight must be positive")
        if self.minimum_service_bytes_per_second > self.service_rate_bytes_per_second:
            raise ValueError("minimum service cannot exceed service rate")


@dataclass(frozen=True)
class DomainRequest:
    """One auxiliary reservation plus an optional foreground footprint.

    ``foreground_domains`` is deliberately separate from ``route``: the
    former describes the latency-critical operation already in flight, while
    the latter describes the bytes this request will move.  Their explicit
    intersection is carried into :class:`DomainDecision` so adapters can
    reproduce why a tail/SLO budget was applied.
    """

    request_id: str
    flow_id: str
    bytes: int
    route: tuple[ResourceDomain, ...]
    now_ns: int
    deadline_ns: int
    nonpreemptible_residual_bytes: int = 0
    # A local foreground-tail/SLO budget measured from ``now_ns``.  This is
    # distinct from the auxiliary flow's absolute completion deadline: a
    # request may be feasible for durability but still be forbidden while a
    # latency-critical collective/token SLO is active.
    tail_budget_ns: int = 0
    # CPU/control-plane time already spent (or reserved) to create the
    # admission decision: packet formation, group rendezvous, planner solve,
    # and stream-transition preparation.  It is deliberately separate from
    # domain service time; otherwise a fast PFS/NIC envelope can be reported
    # feasible while the controller itself has consumed the deadline.
    control_overhead_ns: int = 0
    # ``None`` means the foreground footprint is unknown and admission stays
    # conservative.  An explicit empty tuple proves that no foreground
    # resource domain overlaps this auxiliary request.
    foreground_domains: tuple[ResourceDomain, ...] | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.flow_id:
            raise ValueError("request_id and flow_id must be non-empty")
        if type(self.bytes) is not int or self.bytes <= 0:
            raise ValueError("bytes must be a positive int")
        if not self.route or len(set(self.route)) != len(self.route):
            raise ValueError("route must contain unique ResourceDomain values")
        if any(not isinstance(domain, ResourceDomain) for domain in self.route):
            raise TypeError("route must contain ResourceDomain values")
        if type(self.now_ns) is not int or self.now_ns < 0:
            raise ValueError("now_ns must be a non-negative int")
        if type(self.deadline_ns) is not int or self.deadline_ns <= self.now_ns:
            raise ValueError("deadline_ns must be greater than now_ns")
        if type(self.nonpreemptible_residual_bytes) is not int or not 0 <= self.nonpreemptible_residual_bytes <= self.bytes:
            raise ValueError("residual bytes must be within the request")
        if type(self.tail_budget_ns) is not int or self.tail_budget_ns < 0:
            raise ValueError("tail_budget_ns must be a non-negative int")
        if type(self.control_overhead_ns) is not int or self.control_overhead_ns < 0:
            raise ValueError("control_overhead_ns must be a non-negative int")
        if self.foreground_domains is not None:
            if type(self.foreground_domains) is not tuple:
                raise TypeError("foreground_domains must be a tuple or None")
            if len(set(self.foreground_domains)) != len(self.foreground_domains):
                raise ValueError("foreground_domains must contain unique domains")
            if any(not isinstance(domain, ResourceDomain) for domain in self.foreground_domains):
                raise TypeError("foreground_domains must contain ResourceDomain values")

    @property
    def shared_domains(self) -> tuple[ResourceDomain, ...] | None:
        """Return the explicit foreground/auxiliary domain intersection."""

        if self.foreground_domains is None:
            return None
        return tuple(domain for domain in self.route if domain in self.foreground_domains)


@dataclass(frozen=True)
class DomainDecision:
    """Admission result with explicit foreground/auxiliary overlap."""

    request_id: str
    admitted: bool
    admitted_bytes: int
    estimated_completion_ns: int
    catch_up: bool
    reason: str
    per_domain_service_ns: Mapping[ResourceDomain, int]
    # The residual is part of the admission contract, not an advisory field.
    # A caller must be able to carry it to the completion/trace record without
    # reconstructing it from the original request after ownership changes.
    nonpreemptible_residual_bytes: int = 0
    shared_domains: tuple[ResourceDomain, ...] | None = None


class DomainAdmissionController:
    """A deterministic multi-domain envelope with bounded outstanding work."""

    def __init__(self, budgets: Mapping[ResourceDomain, DomainBudget], *, catch_up_slack_ns: int) -> None:
        if not budgets:
            raise ValueError("at least one domain budget is required")
        if type(catch_up_slack_ns) is not int or catch_up_slack_ns < 0:
            raise ValueError("catch_up_slack_ns must be a non-negative int")
        if set(budgets) != {budget.domain for budget in budgets.values()}:
            raise ValueError("budget keys must match budget.domain")
        self.budgets = dict(budgets)
        self.catch_up_slack_ns = catch_up_slack_ns
        self._inflight: dict[ResourceDomain, int] = {domain: 0 for domain in budgets}
        self._requests: dict[str, DomainRequest] = {}
        self._residuals: dict[str, int] = {}

    @property
    def inflight_bytes(self) -> Mapping[ResourceDomain, int]:
        return dict(self._inflight)

    @property
    def active_residual_bytes(self) -> Mapping[str, int]:
        """Return admitted request residuals for exact telemetry/accounting."""

        return dict(self._residuals)

    def admit(self, request: DomainRequest) -> DomainDecision:
        if request.request_id in self._requests:
            raise ValueError("duplicate request_id")
        shared_domains = request.shared_domains
        missing = [domain for domain in request.route if domain not in self.budgets]
        if missing:
            return DomainDecision(
                request.request_id, False, 0, request.now_ns, False,
                f"unsupported domains: {','.join(domain.value for domain in missing)}", {},
                0,
                shared_domains,
            )
        if request.nonpreemptible_residual_bytes > min(
            self.budgets[domain].max_inflight_bytes for domain in request.route
        ):
            return DomainDecision(
                request.request_id,
                False,
                0,
                request.now_ns,
                False,
                "residual",
                {},
                0,
                shared_domains,
            )
        # A positive minimum service envelope is the conservative rate used
        # for deadline admission.  The measured rate remains available to the
        # caller as provenance, but optimistic nominal bandwidth must not turn
        # an infeasible request into a false guarantee.
        service_ns = {
            domain: ceil(
                request.bytes
                * 1_000_000_000
                / (
                    self.budgets[domain].minimum_service_bytes_per_second
                    or self.budgets[domain].service_rate_bytes_per_second
                )
            )
            for domain in request.route
        }
        # Deadline admission must account for bytes already reserved on each
        # route domain.  Capacity and service rate are separate constraints:
        # accepting a request because it fits the in-flight cap must not turn
        # an already queued prefix into an optimistic zero-queue estimate.
        queued_completion_ns = {
            domain: ceil(
                (self._inflight[domain] + request.bytes)
                * 1_000_000_000
                / (
                    self.budgets[domain].minimum_service_bytes_per_second
                    or self.budgets[domain].service_rate_bytes_per_second
                )
            )
            for domain in request.route
        }
        completion = (
            request.now_ns
            + request.control_overhead_ns
            + max(queued_completion_ns.values())
        )
        capacity_ok = all(
            self._inflight[domain] + request.bytes <= self.budgets[domain].max_inflight_bytes
            for domain in request.route
        )
        deadline_ok = completion <= request.deadline_ns
        catch_up = request.deadline_ns - completion <= self.catch_up_slack_ns
        if not capacity_ok:
            return DomainDecision(
                request.request_id, False, 0, completion, catch_up, "capacity", service_ns,
                0, shared_domains,
            )
        if not deadline_ok:
            return DomainDecision(
                request.request_id, False, 0, completion, catch_up, "deadline", service_ns,
                0, shared_domains,
            )
        tail_budget_applies = (
            request.tail_budget_ns
            and (request.foreground_domains is None or bool(shared_domains))
        )
        if tail_budget_applies and completion - request.now_ns > request.tail_budget_ns:
            return DomainDecision(
                request.request_id,
                False,
                0,
                completion,
                catch_up,
                "tail_budget",
                service_ns,
                request.nonpreemptible_residual_bytes,
                shared_domains,
            )
        for domain in request.route:
            self._inflight[domain] += request.bytes
        self._requests[request.request_id] = request
        self._residuals[request.request_id] = request.nonpreemptible_residual_bytes
        reason = "catch_up" if catch_up else "open"
        return DomainDecision(
            request.request_id,
            True,
            request.bytes,
            completion,
            catch_up,
            reason,
            service_ns,
            request.nonpreemptible_residual_bytes,
            shared_domains,
        )

    def complete(self, request_id: str, completed_bytes: int) -> None:
        request = self._requests.pop(request_id, None)
        if request is None:
            raise ValueError("unknown request_id")
        if type(completed_bytes) is not int or completed_bytes != request.bytes:
            self._requests[request_id] = request
            raise ValueError("completion must equal admitted bytes")
        for domain in request.route:
            self._inflight[domain] -= completed_bytes
        del self._residuals[request_id]

    def cancel(self, request_id: str) -> None:
        """Release an admitted reservation without claiming completion."""

        request = self._requests.pop(request_id, None)
        if request is None:
            raise ValueError("unknown request_id")
        for domain in request.route:
            self._inflight[domain] -= request.bytes
            if self._inflight[domain] < 0:
                # Preserve ownership if an internal accounting invariant is
                # already broken; callers must not continue with corruption.
                for rollback_domain in request.route:
                    self._inflight[rollback_domain] += request.bytes
                self._requests[request_id] = request
                raise RuntimeError("domain inflight reservation underflow")
        del self._residuals[request_id]


class FlowAdmissionLedger:
    """Adapter-neutral request ownership shared by training and inference.

    The ledger intentionally knows nothing about checkpoints, KV versions, or
    CUDA.  It owns only the controller reservation and exact completion
    lifecycle.  Training and KV adapters layer their own correctness/version
    checks around this same object instead of reimplementing release logic.
    """

    def __init__(self, controller: DomainAdmissionController) -> None:
        if not isinstance(controller, DomainAdmissionController):
            raise TypeError("controller must be a DomainAdmissionController")
        self.controller = controller
        self._admitted: dict[str, DomainRequest] = {}

    def admit(self, request: DomainRequest) -> DomainDecision:
        if request.request_id in self._admitted:
            raise ValueError("duplicate request_id")
        decision = self.controller.admit(request)
        if decision.admitted:
            self._admitted[request.request_id] = request
        return decision

    def complete(self, request_id: str, completed_bytes: int) -> None:
        request = self._admitted.get(request_id)
        if request is None:
            raise ValueError("unknown request_id")
        self.controller.complete(request_id, completed_bytes)
        del self._admitted[request_id]

    def cancel(self, request_id: str) -> None:
        """Release an active request after a failed/aborted transport."""

        if request_id not in self._admitted:
            raise ValueError("unknown request_id")
        self.controller.cancel(request_id)
        del self._admitted[request_id]

    def is_active(self, request_id: str) -> bool:
        return request_id in self._admitted
