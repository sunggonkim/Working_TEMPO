"""Versioned inference KV movement contracts for TEMPO-RD.

This module is an endpoint-neutral correctness/admission adapter.  It does
not implement a vLLM, SGLang, LMCache, or storage backend.  The adapter makes
the parts that must be identical across those backends explicit: session and
KV version identity, an observed resource-domain route, exact bytes, a
prefetch/eviction deadline, and stale-version rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Mapping

from tempo.resource_domain import ResourceDomain


class KVOperation(str, Enum):
    EVICT = "evict"
    PREFETCH = "prefetch"
    MIGRATE = "migrate"


@dataclass(frozen=True)
class KVVersion:
    session_id: str
    sequence: int
    content_digest: str

    def __post_init__(self) -> None:
        if type(self.session_id) is not str or not self.session_id:
            raise ValueError("session_id must be non-empty")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sequence must be a non-negative int")
        if type(self.content_digest) is not str or len(self.content_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.content_digest
        ):
            raise ValueError("content_digest must be lowercase SHA-256 hex")

    @classmethod
    def from_bytes(cls, session_id: str, sequence: int, payload: bytes) -> "KVVersion":
        return cls(session_id, sequence, sha256(payload).hexdigest())


@dataclass(frozen=True)
class KVTransferRequest:
    request_id: str
    version: KVVersion
    operation: KVOperation
    bytes: int
    source: ResourceDomain
    destination: ResourceDomain
    route: tuple[ResourceDomain, ...]
    deadline_ns: int
    max_residual_bytes: int = 0
    # Shared tail/SLO budget for the request's admission interval.  A value
    # of zero means that only the absolute request deadline is enforced.
    tail_budget_ns: int = 0

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not isinstance(self.version, KVVersion):
            raise TypeError("version must be a KVVersion")
        if not isinstance(self.operation, KVOperation):
            raise TypeError("operation must be a KVOperation")
        if type(self.bytes) is not int or self.bytes <= 0:
            raise ValueError("bytes must be a positive int")
        if not isinstance(self.source, ResourceDomain) or not isinstance(
            self.destination, ResourceDomain
        ):
            raise TypeError("source and destination must be ResourceDomain values")
        if type(self.route) is not tuple or not self.route:
            raise ValueError("route must be a non-empty tuple")
        if any(not isinstance(domain, ResourceDomain) for domain in self.route):
            raise TypeError("route must contain ResourceDomain values")
        if self.route[0] is not self.source or self.route[-1] is not self.destination:
            raise ValueError("route must begin at source and end at destination")
        if len(set(self.route)) != len(self.route):
            raise ValueError("route domains must be unique")
        if type(self.deadline_ns) is not int or self.deadline_ns <= 0:
            raise ValueError("deadline_ns must be a positive int")
        if type(self.max_residual_bytes) is not int or not 0 <= self.max_residual_bytes <= self.bytes:
            raise ValueError("max_residual_bytes must be within the request")
        if type(self.tail_budget_ns) is not int or self.tail_budget_ns < 0:
            raise ValueError("tail_budget_ns must be a non-negative int")


@dataclass(frozen=True)
class KVAdmissionDecision:
    request_id: str
    session_id: str
    sequence: int
    admitted_bytes: int
    estimated_service_ns: int
    deadline_ns: int
    status: str
    reason: str
    shared_domains: tuple[ResourceDomain, ...] | None = None


class KVFlowLedger:
    """Small deterministic ledger shared by inference adapters and tests.

    A backend calls ``publish`` when a new KV version is atomically produced,
    ``admit_via_domain_controller`` before moving bytes on a live multi-domain
    path, and ``complete`` after an exact transfer.  The scalar ``admit``
    helper is retained for version/deadline unit tests; it has no per-domain
    queue state and must not be used to claim resource-domain orchestration.
    A stale request is never silently upgraded to the newest version.
    """

    def __init__(self) -> None:
        self._published: dict[str, KVVersion] = {}
        self._admitted: dict[str, KVTransferRequest] = {}
        self._completed: dict[str, int] = {}
        # Requests admitted through the shared flow ledger retain their owner
        # so the public ``complete`` lifecycle releases the same per-domain
        # inflight reservation.  A KV adapter must not need a second,
        # out-of-band controller.complete() call: doing so would make
        # training and inference use different ownership semantics.
        self._domain_ledgers: dict[str, object] = {}

    @property
    def published(self) -> Mapping[str, KVVersion]:
        return dict(self._published)

    def publish(self, version: KVVersion) -> None:
        if not isinstance(version, KVVersion):
            raise TypeError("version must be a KVVersion")
        previous = self._published.get(version.session_id)
        if previous is not None and version.sequence <= previous.sequence:
            raise ValueError("KV version sequence must increase monotonically")
        self._published[version.session_id] = version

    def _check_current(self, request: KVTransferRequest) -> None:
        current = self._published.get(request.version.session_id)
        if current != request.version:
            raise ValueError("stale or unpublished KV version")

    def admit(
        self,
        request: KVTransferRequest,
        *,
        now_ns: int,
        service_rate_bytes_per_second: int,
        available_bytes: int,
    ) -> KVAdmissionDecision:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative int")
        if type(service_rate_bytes_per_second) is not int or service_rate_bytes_per_second <= 0:
            raise ValueError("service rate must be a positive int")
        if type(available_bytes) is not int or available_bytes < 0:
            raise ValueError("available_bytes must be a non-negative int")
        self._check_current(request)
        if request.request_id in self._admitted:
            raise ValueError("duplicate KV request id")
        service_ns = (request.bytes * 1_000_000_000 + service_rate_bytes_per_second - 1) // service_rate_bytes_per_second
        if request.bytes > available_bytes:
            return KVAdmissionDecision(
                request.request_id,
                request.version.session_id,
                request.version.sequence,
                0,
                service_ns,
                request.deadline_ns,
                "rejected",
                "capacity",
            )
        if now_ns + service_ns > request.deadline_ns:
            return KVAdmissionDecision(
                request.request_id,
                request.version.session_id,
                request.version.sequence,
                0,
                service_ns,
                request.deadline_ns,
                "rejected",
                "deadline",
            )
        self._admitted[request.request_id] = request
        self._completed[request.request_id] = 0
        return KVAdmissionDecision(
            request.request_id,
            request.version.session_id,
            request.version.sequence,
            request.bytes,
            service_ns,
            request.deadline_ns,
            "admitted",
            "ok",
        )

    def complete(self, request_id: str, completed_bytes: int) -> None:
        request = self._admitted.get(request_id)
        if request is None:
            raise ValueError("unknown KV request")
        if type(completed_bytes) is not int or completed_bytes != request.bytes:
            raise ValueError("completion must exactly match admitted KV bytes")
        if self._completed.get(request_id, 0) == request.bytes:
            raise ValueError("KV request is already complete")
        ledger = self._domain_ledgers.get(request_id)
        # A newer version may have been published while an older prefetch was
        # in flight.  The transport may finish, but that payload must never be
        # accepted as the current session state.  Release the exact controller
        # reservation before rejecting it so stale traffic cannot leak domain
        # capacity and block the newer version.
        current = self._published.get(request.version.session_id)
        if current != request.version:
            if ledger is not None:
                from tempo.domain_admission import FlowAdmissionLedger

                if not isinstance(ledger, FlowAdmissionLedger):
                    raise TypeError("stored domain ledger has an invalid type")
                ledger.complete(request_id, completed_bytes)
                del self._domain_ledgers[request_id]
            del self._admitted[request_id]
            self._completed.pop(request_id, None)
            raise ValueError("stale or superseded KV version")
        if ledger is not None:
            # The shared ledger validates the exact request and releases every
            # domain atomically before the ledger marks the KV version done.
            # If it raises, both ledger completion and ownership remain
            # unchanged so the caller can fail closed.
            from tempo.domain_admission import FlowAdmissionLedger

            if not isinstance(ledger, FlowAdmissionLedger):
                raise TypeError("stored domain ledger has an invalid type")
            ledger.complete(request_id, completed_bytes)
            del self._domain_ledgers[request_id]
        self._completed[request_id] = completed_bytes

    def cancel(self, request_id: str) -> None:
        """Abort a request and release any shared-domain reservation."""

        request = self._admitted.get(request_id)
        if request is None:
            raise ValueError("unknown KV request")
        ledger = self._domain_ledgers.pop(request_id, None)
        if ledger is not None:
            from tempo.domain_admission import FlowAdmissionLedger

            if not isinstance(ledger, FlowAdmissionLedger):
                raise TypeError("stored domain ledger has an invalid type")
            ledger.cancel(request_id)
        del self._admitted[request_id]
        self._completed.pop(request_id, None)

    def admit_via_domain_controller(
        self,
        request: KVTransferRequest,
        controller: object,
        *,
        now_ns: int,
        foreground_domains: tuple[ResourceDomain, ...] | None = None,
        control_overhead_ns: int = 0,
    ) -> KVAdmissionDecision:
        """Use the shared envelope while preserving KV identity checks.

        ``foreground_domains`` is the observed token/foreground footprint;
        the returned decision exposes its intersection with the KV route.
        """

        from tempo.domain_admission import DomainAdmissionController, DomainRequest, FlowAdmissionLedger

        if not isinstance(controller, DomainAdmissionController):
            raise TypeError("controller must be a DomainAdmissionController")
        if type(control_overhead_ns) is not int or control_overhead_ns < 0:
            raise ValueError("control_overhead_ns must be a non-negative int")
        self._check_current(request)
        if request.request_id in self._admitted:
            raise ValueError("duplicate KV request id")
        domain_ledger = FlowAdmissionLedger(controller)
        decision = domain_ledger.admit(
            DomainRequest(
                request_id=request.request_id,
                flow_id=request.version.session_id,
                bytes=request.bytes,
                route=request.route,
                now_ns=now_ns,
                deadline_ns=request.deadline_ns,
                nonpreemptible_residual_bytes=request.max_residual_bytes,
                tail_budget_ns=request.tail_budget_ns,
                control_overhead_ns=control_overhead_ns,
                foreground_domains=foreground_domains,
            )
        )
        estimated = decision.estimated_completion_ns - now_ns
        if not decision.admitted:
            return KVAdmissionDecision(
                request.request_id,
                request.version.session_id,
                request.version.sequence,
                0,
                estimated,
                request.deadline_ns,
                "rejected",
                decision.reason,
                decision.shared_domains,
            )
        self._admitted[request.request_id] = request
        self._completed[request.request_id] = 0
        self._domain_ledgers[request.request_id] = domain_ledger
        return KVAdmissionDecision(
            request.request_id,
            request.version.session_id,
            request.version.sequence,
            request.bytes,
            estimated,
            request.deadline_ns,
            "admitted",
            decision.reason,
            decision.shared_domains,
        )

    def is_complete(self, request_id: str) -> bool:
        request = self._admitted.get(request_id)
        return request is not None and self._completed.get(request_id, 0) == request.bytes
