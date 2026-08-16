"""Adapter-neutral conversion into the TEMPO-RD state-flow DAG.

The controller consumes ordered, exact-byte stages rather than a backend-specific
checkpoint or KV object.  This module is deliberately small: it only translates
the two current producers into :class:`tempo.resource_domain.StateFlow` and
does not infer a route from topology or perform I/O.
"""

from __future__ import annotations

from typing import Iterable

from tempo.kv_flow import KVTransferRequest
from tempo.domain_admission import (
    DomainAdmissionController,
    DomainDecision,
    DomainRequest,
    FlowAdmissionLedger,
)
from tempo.resource_domain import FlowStage, ResourceDomain, StateFlow


_DEFAULT_D2H_DOMAINS = (
    ResourceDomain.GPU_LOCAL,
    ResourceDomain.PCIE_HOST,
    ResourceDomain.HOST_NUMA,
)
_DEFAULT_PERSIST_DOMAINS = (
    ResourceDomain.NIC_FABRIC,
    ResourceDomain.SLINGSHOT_FABRIC,
    ResourceDomain.PERSISTENT_ENDPOINT,
)


def _domains(value: Iterable[ResourceDomain], name: str) -> tuple[ResourceDomain, ...]:
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique domains")
    if any(not isinstance(domain, ResourceDomain) for domain in result):
        raise TypeError(f"{name} must contain ResourceDomain values")
    return result


def checkpoint_state_flow(
    *,
    flow_id: str,
    state_bytes: int,
    deadline_ns: int,
    d2h_deadline_ns: int,
    persist_deadline_ns: int,
    version: str = "",
    d2h_domains: Iterable[ResourceDomain] = _DEFAULT_D2H_DOMAINS,
    persist_domains: Iterable[ResourceDomain] = _DEFAULT_PERSIST_DOMAINS,
    d2h_residual_bytes: int = 0,
    persist_residual_bytes: int = 0,
    d2h_tail_budget_ns: int = 0,
    persist_tail_budget_ns: int = 0,
    d2h_control_overhead_ns: int = 0,
    persist_control_overhead_ns: int = 0,
) -> StateFlow:
    """Build the two-stage checkpoint flow used by the training adapter.

    The two stages intentionally each carry ``state_bytes``: they are service
    work at different resource domains, not a claim that bytes are duplicated
    in storage.  The caller supplies the observed route domains explicitly.
    """

    if type(state_bytes) is not int or state_bytes <= 0:
        raise ValueError("state_bytes must be a positive int")
    for name, value in (
        ("deadline_ns", deadline_ns),
        ("d2h_deadline_ns", d2h_deadline_ns),
        ("persist_deadline_ns", persist_deadline_ns),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive int")
    if d2h_deadline_ns > deadline_ns or persist_deadline_ns > deadline_ns:
        raise ValueError("stage deadlines cannot exceed flow deadline")
    for name, value in (
        ("d2h_residual_bytes", d2h_residual_bytes),
        ("persist_residual_bytes", persist_residual_bytes),
        ("d2h_tail_budget_ns", d2h_tail_budget_ns),
        ("persist_tail_budget_ns", persist_tail_budget_ns),
        ("d2h_control_overhead_ns", d2h_control_overhead_ns),
        ("persist_control_overhead_ns", persist_control_overhead_ns),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative int")
        if name.endswith("residual_bytes") and value > state_bytes:
            raise ValueError(f"{name} must be within state_bytes")
    if type(flow_id) is not str or not flow_id:
        raise ValueError("flow_id must be non-empty")
    if type(version) is not str:
        raise TypeError("version must be a string")
    return StateFlow(
        flow_id=flow_id,
        version=version,
        deadline_ns=deadline_ns,
        stages=(
            FlowStage(
                stage_id="d2h",
                bytes=state_bytes,
                domains=_domains(d2h_domains, "d2h_domains"),
                deadline_ns=d2h_deadline_ns,
                max_residual_bytes=d2h_residual_bytes,
                tail_budget_ns=d2h_tail_budget_ns,
                control_overhead_ns=d2h_control_overhead_ns,
            ),
            FlowStage(
                stage_id="persist",
                bytes=state_bytes,
                domains=_domains(persist_domains, "persist_domains"),
                deadline_ns=persist_deadline_ns,
                max_residual_bytes=persist_residual_bytes,
                tail_budget_ns=persist_tail_budget_ns,
                control_overhead_ns=persist_control_overhead_ns,
            ),
        ),
    )


def kv_state_flow(request: KVTransferRequest) -> StateFlow:
    """Translate a versioned KV request into the same stage-DAG contract."""

    if not isinstance(request, KVTransferRequest):
        raise TypeError("request must be a KVTransferRequest")
    return StateFlow(
        flow_id=f"kv:{request.request_id}",
        version=(
            f"{request.version.session_id}:{request.version.sequence}:"
            f"{request.version.content_digest}"
        ),
        deadline_ns=request.deadline_ns,
        stages=(
            FlowStage(
                stage_id=f"kv:{request.operation.value}",
                bytes=request.bytes,
                domains=tuple(request.route),
                deadline_ns=request.deadline_ns,
                max_residual_bytes=request.max_residual_bytes,
                tail_budget_ns=request.tail_budget_ns,
                control_overhead_ns=0,
            ),
        ),
    )


def flow_route_signature(flow: StateFlow) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return a stable stage/route signature for cross-adapter comparisons."""

    if not isinstance(flow, StateFlow):
        raise TypeError("flow must be a StateFlow")
    return tuple(
        (stage.stage_id, tuple(domain.value for domain in stage.domains))
        for stage in flow.stages
    )


class StateFlowAdmission:
    """Sequential stage lifecycle shared by checkpoint and KV adapters.

    The flow itself supplies exact bytes, route, residual, absolute deadline,
    and optional tail budget.  This wrapper adds only stage ordering and exact
    completion ownership; all capacity/rate decisions remain in the shared
    :class:`DomainAdmissionController`.  The optional foreground-domain tuple
    is passed unchanged to every stage so training and KV adapters use the same
    overlap semantics.
    """

    def __init__(
        self,
        flow: StateFlow,
        controller: DomainAdmissionController,
        *,
        foreground_domains: tuple[ResourceDomain, ...] | None = None,
    ) -> None:
        if not isinstance(flow, StateFlow):
            raise TypeError("flow must be a StateFlow")
        if not isinstance(controller, DomainAdmissionController):
            raise TypeError("controller must be a DomainAdmissionController")
        self.flow = flow
        self.ledger = FlowAdmissionLedger(controller)
        if foreground_domains is not None and type(foreground_domains) is not tuple:
            raise TypeError("foreground_domains must be a tuple or None")
        if foreground_domains is not None and (
            len(set(foreground_domains)) != len(foreground_domains)
            or any(not isinstance(domain, ResourceDomain) for domain in foreground_domains)
        ):
            raise ValueError("foreground_domains must contain unique ResourceDomain values")
        self.foreground_domains = foreground_domains
        self._next_stage = 0
        self._active: tuple[int, str] | None = None

    @property
    def completed_stages(self) -> int:
        return self._next_stage

    def admit_next(self, *, now_ns: int) -> DomainDecision:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative int")
        if self._active is not None:
            raise ValueError("previous stage must complete before the next admission")
        if self._next_stage >= len(self.flow.stages):
            raise ValueError("all flow stages are already complete")
        stage = self.flow.stages[self._next_stage]
        request_id = f"{self.flow.flow_id}:{stage.stage_id}:{self._next_stage}"
        decision = self.ledger.admit(
            DomainRequest(
                request_id=request_id,
                flow_id=self.flow.flow_id,
                bytes=stage.bytes,
                route=stage.domains,
                now_ns=now_ns,
                deadline_ns=stage.deadline_ns,
                nonpreemptible_residual_bytes=stage.max_residual_bytes,
                tail_budget_ns=stage.tail_budget_ns,
                control_overhead_ns=stage.control_overhead_ns,
                foreground_domains=self.foreground_domains,
            )
        )
        if decision.admitted:
            self._active = (self._next_stage, request_id)
        return decision

    def complete_active(self, completed_bytes: int) -> None:
        if self._active is None:
            raise ValueError("no stage is currently admitted")
        index, request_id = self._active
        self.ledger.complete(request_id, completed_bytes)
        self._active = None
        self._next_stage = index + 1

    def abort_active(self) -> None:
        """Release the active stage without advancing completion ownership."""

        if self._active is None:
            raise ValueError("no stage is currently admitted")
        _index, request_id = self._active
        self.ledger.cancel(request_id)
        self._active = None
