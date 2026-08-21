"""Fail-closed endpoint evidence contracts for TEMPO Elastic-PD.

The router must not subtract monotonic timestamps generated on different
hosts.  Each endpoint therefore publishes only endpoint-local queue/service
state.  The router records its own receive timestamp and uses that timestamp
only for freshness.  Service feedback carries endpoint-local durations with
explicit endpoint ownership.

This module is policy-free.  It neither computes a local/remote score nor
infers a physical fabric bottleneck from endpoint counters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Mapping

from tempo.domain_evidence import CounterSupport


SCHEMA = "tempo-pd-endpoint-evidence-v1"
TRANSFER_SCHEMA = "tempo-pd-transfer-feedback-v1"


class PDEndpointRole(str, Enum):
    PREFILL = "prefill"
    DECODER = "decoder"


class TransferStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


_COMMON_METRICS = frozenset({
    "running_requests",
    "waiting_requests",
    "kv_cache_usage_fraction",
    "kv_transfer_bytes_inflight",
    "kv_transfer_ops_inflight",
})
_PREFILL_METRICS = _COMMON_METRICS | frozenset({
    "active_prefill_tokens",
    "prefill_token_ms_inflight",
    "prefill_service_p50_ns",
    "prefill_service_p90_ns",
})
_DECODER_METRICS = _COMMON_METRICS | frozenset({
    "active_decode_tokens",
    "active_local_prefill_tokens",
    "local_prefill_token_ms_inflight",
    "decode_step_p90_ns",
    "kv_install_p90_ns",
})

_INTEGER_METRICS = (
    _PREFILL_METRICS | _DECODER_METRICS
) - {"kv_cache_usage_fraction"}


def endpoint_metric_names(role: PDEndpointRole) -> frozenset[str]:
    if not isinstance(role, PDEndpointRole):
        raise TypeError("role must be a PDEndpointRole")
    return _PREFILL_METRICS if role is PDEndpointRole.PREFILL else _DECODER_METRICS


@dataclass(frozen=True)
class EndpointMetric:
    """One explicitly supported or unavailable endpoint metric."""

    name: str
    support: CounterSupport
    value: int | float | None

    def __post_init__(self) -> None:
        if self.name not in _PREFILL_METRICS | _DECODER_METRICS:
            raise ValueError(f"unknown endpoint metric: {self.name}")
        if not isinstance(self.support, CounterSupport):
            raise TypeError("support must be CounterSupport")
        if self.support is not CounterSupport.SUPPORTED:
            if self.value is not None:
                raise ValueError("unsupported endpoint metrics must have value=None")
            return
        if self.value is None or isinstance(self.value, bool):
            raise ValueError("supported endpoint metrics require a value")
        if self.name in _INTEGER_METRICS:
            if type(self.value) is not int or self.value < 0:
                raise ValueError(f"{self.name} must be a non-negative int")
            return
        if not isinstance(self.value, (int, float)) or not math.isfinite(
            float(self.value)
        ):
            raise ValueError(f"{self.name} must be finite")
        if not 0.0 <= float(self.value) <= 1.0:
            raise ValueError("kv_cache_usage_fraction must be in [0, 1]")


@dataclass(frozen=True)
class PDEndpointIdentity:
    endpoint_id: str
    role: PDEndpointRole
    pair_index: int

    def __post_init__(self) -> None:
        if type(self.endpoint_id) is not str or not self.endpoint_id.strip():
            raise ValueError("endpoint_id must be nonempty")
        if not isinstance(self.role, PDEndpointRole):
            raise TypeError("role must be a PDEndpointRole")
        if type(self.pair_index) is not int or self.pair_index < 0:
            raise ValueError("pair_index must be a non-negative int")


@dataclass(frozen=True)
class PDEndpointSnapshot:
    identity: PDEndpointIdentity
    sequence: int
    endpoint_monotonic_ns: int
    source: str
    metrics: tuple[EndpointMetric, ...]
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PDEndpointIdentity):
            raise TypeError("identity must be PDEndpointIdentity")
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("sequence must be a positive int")
        if type(self.endpoint_monotonic_ns) is not int or self.endpoint_monotonic_ns < 0:
            raise ValueError("endpoint_monotonic_ns must be a non-negative int")
        if type(self.source) is not str or not self.source.strip():
            raise ValueError("source must be nonempty")
        if self.schema != SCHEMA:
            raise ValueError("endpoint evidence schema is not canonical")
        if type(self.metrics) is not tuple:
            raise TypeError("metrics must be a tuple")
        if any(not isinstance(metric, EndpointMetric) for metric in self.metrics):
            raise TypeError("metrics must contain EndpointMetric values")
        names = tuple(metric.name for metric in self.metrics)
        if len(names) != len(set(names)):
            raise ValueError("endpoint metric names must be unique")
        if names != tuple(sorted(names)):
            raise ValueError("endpoint metrics must be sorted by name")
        expected = endpoint_metric_names(self.identity.role)
        if frozenset(names) != expected:
            missing = sorted(expected - frozenset(names))
            extra = sorted(frozenset(names) - expected)
            raise ValueError(
                f"endpoint metric inventory is not exact: missing={missing}, extra={extra}"
            )

    def metric(self, name: str) -> EndpointMetric:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "endpoint_id": self.identity.endpoint_id,
            "role": self.identity.role.value,
            "pair_index": self.identity.pair_index,
            "sequence": self.sequence,
            "endpoint_monotonic_ns": self.endpoint_monotonic_ns,
            "source": self.source,
            "metrics": {
                metric.name: {
                    "support": metric.support.value,
                    "value": metric.value,
                }
                for metric in self.metrics
            },
        }


def endpoint_metrics(
    role: PDEndpointRole,
    *,
    supported: Mapping[str, int | float] | None = None,
    unavailable: Mapping[str, CounterSupport] | None = None,
) -> tuple[EndpointMetric, ...]:
    """Build an exact role-specific metric inventory without inventing zeroes."""

    expected = endpoint_metric_names(role)
    supported_values = dict(supported or {})
    unavailable_values = dict(unavailable or {})
    supplied = set(supported_values) | set(unavailable_values)
    if set(supported_values) & set(unavailable_values):
        raise ValueError("a metric cannot be both supported and unavailable")
    if supplied != set(expected):
        raise ValueError(
            "supported and unavailable mappings must cover the exact role inventory"
        )
    metrics: list[EndpointMetric] = []
    for name in sorted(expected):
        if name in supported_values:
            metrics.append(EndpointMetric(
                name, CounterSupport.SUPPORTED, supported_values[name]
            ))
        else:
            support = unavailable_values[name]
            if support is CounterSupport.SUPPORTED:
                raise ValueError("unavailable metrics cannot be marked supported")
            metrics.append(EndpointMetric(name, support, None))
    return tuple(metrics)


@dataclass(frozen=True)
class ReceivedEndpointSnapshot:
    snapshot: PDEndpointSnapshot
    router_received_monotonic_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, PDEndpointSnapshot):
            raise TypeError("snapshot must be PDEndpointSnapshot")
        if (
            type(self.router_received_monotonic_ns) is not int
            or self.router_received_monotonic_ns < 0
        ):
            raise ValueError("router_received_monotonic_ns must be non-negative")


@dataclass(frozen=True)
class PDEndpointPairView:
    pair_index: int
    prefill: ReceivedEndpointSnapshot
    decoder: ReceivedEndpointSnapshot
    viewed_monotonic_ns: int

    def __post_init__(self) -> None:
        if type(self.pair_index) is not int or self.pair_index < 0:
            raise ValueError("pair_index must be non-negative")
        for name, value, role in (
            ("prefill", self.prefill, PDEndpointRole.PREFILL),
            ("decoder", self.decoder, PDEndpointRole.DECODER),
        ):
            if not isinstance(value, ReceivedEndpointSnapshot):
                raise TypeError(f"{name} must be ReceivedEndpointSnapshot")
            if value.snapshot.identity.pair_index != self.pair_index:
                raise ValueError(f"{name} pair index does not match")
            if value.snapshot.identity.role is not role:
                raise ValueError(f"{name} role does not match")
        if type(self.viewed_monotonic_ns) is not int or self.viewed_monotonic_ns < 0:
            raise ValueError("viewed_monotonic_ns must be non-negative")

    def age_ns(self, role: PDEndpointRole) -> int:
        if role is PDEndpointRole.PREFILL:
            received = self.prefill.router_received_monotonic_ns
        elif role is PDEndpointRole.DECODER:
            received = self.decoder.router_received_monotonic_ns
        else:
            raise TypeError("role must be a PDEndpointRole")
        return self.viewed_monotonic_ns - received


class PDEndpointEvidenceStore:
    """Latest-value store for endpoint-pushed, asynchronously received state."""

    def __init__(self, identities: Iterable[PDEndpointIdentity]) -> None:
        values = tuple(identities)
        if not values or any(not isinstance(item, PDEndpointIdentity) for item in values):
            raise ValueError("identities must contain PDEndpointIdentity values")
        by_id = {item.endpoint_id: item for item in values}
        if len(by_id) != len(values):
            raise ValueError("endpoint_id values must be unique")
        by_pair_role: dict[tuple[int, PDEndpointRole], PDEndpointIdentity] = {}
        for identity in values:
            key = (identity.pair_index, identity.role)
            if key in by_pair_role:
                raise ValueError("each pair must have one endpoint per role")
            by_pair_role[key] = identity
        pairs = {identity.pair_index for identity in values}
        for pair_index in pairs:
            for role in PDEndpointRole:
                if (pair_index, role) not in by_pair_role:
                    raise ValueError("each pair requires prefill and decoder endpoints")
        self._identities = by_id
        self._by_pair_role = by_pair_role
        self._latest: dict[str, ReceivedEndpointSnapshot] = {}

    def observe(
        self, snapshot: PDEndpointSnapshot, *, router_received_monotonic_ns: int,
    ) -> None:
        if not isinstance(snapshot, PDEndpointSnapshot):
            raise TypeError("snapshot must be PDEndpointSnapshot")
        expected = self._identities.get(snapshot.identity.endpoint_id)
        if expected is None or expected != snapshot.identity:
            raise ValueError("snapshot endpoint identity is not registered")
        received = ReceivedEndpointSnapshot(
            snapshot, router_received_monotonic_ns
        )
        previous = self._latest.get(snapshot.identity.endpoint_id)
        if previous is not None:
            if snapshot.sequence <= previous.snapshot.sequence:
                raise ValueError("endpoint sequence must increase")
            if (
                snapshot.endpoint_monotonic_ns
                <= previous.snapshot.endpoint_monotonic_ns
            ):
                raise ValueError("endpoint-local monotonic timestamp must increase")
            if (
                router_received_monotonic_ns
                < previous.router_received_monotonic_ns
            ):
                raise ValueError("router receive timestamp must not regress")
        self._latest[snapshot.identity.endpoint_id] = received

    def pair_view(
        self, pair_index: int, *, now_monotonic_ns: int, max_age_ns: int,
    ) -> PDEndpointPairView:
        if type(pair_index) is not int or pair_index < 0:
            raise ValueError("pair_index must be non-negative")
        if type(now_monotonic_ns) is not int or now_monotonic_ns < 0:
            raise ValueError("now_monotonic_ns must be non-negative")
        if type(max_age_ns) is not int or max_age_ns <= 0:
            raise ValueError("max_age_ns must be positive")
        received: dict[PDEndpointRole, ReceivedEndpointSnapshot] = {}
        for role in PDEndpointRole:
            identity = self._by_pair_role.get((pair_index, role))
            if identity is None:
                raise KeyError(f"unknown pair index {pair_index}")
            value = self._latest.get(identity.endpoint_id)
            if value is None:
                raise ValueError(f"missing {role.value} endpoint snapshot")
            age_ns = now_monotonic_ns - value.router_received_monotonic_ns
            if age_ns < 0:
                raise ValueError("router view time precedes endpoint receipt")
            if age_ns > max_age_ns:
                raise ValueError(f"stale {role.value} endpoint snapshot")
            received[role] = value
        return PDEndpointPairView(
            pair_index=pair_index,
            prefill=received[PDEndpointRole.PREFILL],
            decoder=received[PDEndpointRole.DECODER],
            viewed_monotonic_ns=now_monotonic_ns,
        )


@dataclass(frozen=True)
class EndpointDuration:
    name: str
    endpoint_id: str
    duration_ns: int
    source: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ValueError("duration name must be nonempty")
        if type(self.endpoint_id) is not str or not self.endpoint_id.strip():
            raise ValueError("duration endpoint_id must be nonempty")
        if type(self.duration_ns) is not int or self.duration_ns < 0:
            raise ValueError("duration_ns must be a non-negative int")
        if type(self.source) is not str or not self.source.strip():
            raise ValueError("duration source must be nonempty")


@dataclass(frozen=True)
class KVTransferFeedback:
    """Exact transfer outcome with endpoint-owned, non-joinable durations."""

    request_id: str
    pair_index: int
    source_endpoint_id: str
    destination_endpoint_id: str
    potential_kv_bytes: int
    completed_kv_bytes: int | None
    semantic_operations: int
    status: TransferStatus
    durations: tuple[EndpointDuration, ...]
    error: str | None = None
    schema: str = TRANSFER_SCHEMA

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("source_endpoint_id", self.source_endpoint_id),
            ("destination_endpoint_id", self.destination_endpoint_id),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        if self.source_endpoint_id == self.destination_endpoint_id:
            raise ValueError("transfer endpoints must differ")
        if type(self.pair_index) is not int or self.pair_index < 0:
            raise ValueError("pair_index must be non-negative")
        if type(self.potential_kv_bytes) is not int or self.potential_kv_bytes <= 0:
            raise ValueError("potential_kv_bytes must be positive")
        if (
            self.completed_kv_bytes is not None
            and (
                type(self.completed_kv_bytes) is not int
                or not 0 <= self.completed_kv_bytes <= self.potential_kv_bytes
            )
        ):
            raise ValueError("completed_kv_bytes must be within potential bytes")
        if type(self.semantic_operations) is not int or self.semantic_operations < 1:
            raise ValueError("semantic_operations must be positive")
        if not isinstance(self.status, TransferStatus):
            raise TypeError("status must be TransferStatus")
        if type(self.durations) is not tuple or any(
            not isinstance(item, EndpointDuration) for item in self.durations
        ):
            raise TypeError("durations must be a tuple of EndpointDuration")
        duration_names = [item.name for item in self.durations]
        if len(duration_names) != len(set(duration_names)):
            raise ValueError("duration names must be unique")
        allowed_endpoints = {self.source_endpoint_id, self.destination_endpoint_id}
        if any(item.endpoint_id not in allowed_endpoints for item in self.durations):
            raise ValueError("duration endpoint is outside the transfer pair")
        if self.status is TransferStatus.SUCCESS:
            if self.completed_kv_bytes is None:
                raise ValueError("successful transfer needs completed byte evidence")
            if self.error is not None:
                raise ValueError("successful transfer cannot contain an error")
        elif type(self.error) is not str or not self.error.strip():
            raise ValueError("failed transfer needs an error")
        if self.schema != TRANSFER_SCHEMA:
            raise ValueError("transfer feedback schema is not canonical")

    def total_duration_ns(self) -> int:
        """Disallow an invalid cross-endpoint duration sum by construction."""

        endpoints = {item.endpoint_id for item in self.durations}
        if len(endpoints) > 1:
            raise ValueError("durations from different endpoint clocks cannot be summed")
        return sum(item.duration_ns for item in self.durations)


class KVTransferFeedbackLedger:
    """Exact-once transfer feedback ownership for controller observations."""

    def __init__(self, identities: Iterable[PDEndpointIdentity]) -> None:
        values = tuple(identities)
        if not values or any(
            not isinstance(item, PDEndpointIdentity) for item in values
        ):
            raise ValueError(
                "identities must be unique PDEndpointIdentity values")
        self._identities = {item.endpoint_id: item for item in values}
        if len(self._identities) != len(values):
            raise ValueError("identities must be unique PDEndpointIdentity values")
        self._feedback: dict[str, KVTransferFeedback] = {}

    def observe(self, feedback: KVTransferFeedback) -> None:
        if not isinstance(feedback, KVTransferFeedback):
            raise TypeError("feedback must be KVTransferFeedback")
        if feedback.request_id in self._feedback:
            raise ValueError("transfer feedback already observed")
        source = self._identities.get(feedback.source_endpoint_id)
        destination = self._identities.get(feedback.destination_endpoint_id)
        if source is None or destination is None:
            raise ValueError("transfer endpoint is not registered")
        if (
            source.role is not PDEndpointRole.PREFILL
            or destination.role is not PDEndpointRole.DECODER
            or source.pair_index != feedback.pair_index
            or destination.pair_index != feedback.pair_index
        ):
            raise ValueError("transfer endpoints do not match the P/D pair")
        self._feedback[feedback.request_id] = feedback

    def get(self, request_id: str) -> KVTransferFeedback:
        try:
            return self._feedback[request_id]
        except KeyError as exc:
            raise KeyError(request_id) from exc


__all__ = [
    "EndpointDuration",
    "EndpointMetric",
    "KVTransferFeedback",
    "KVTransferFeedbackLedger",
    "PDEndpointEvidenceStore",
    "PDEndpointIdentity",
    "PDEndpointPairView",
    "PDEndpointRole",
    "PDEndpointSnapshot",
    "ReceivedEndpointSnapshot",
    "SCHEMA",
    "TRANSFER_SCHEMA",
    "TransferStatus",
    "endpoint_metric_names",
    "endpoint_metrics",
]
