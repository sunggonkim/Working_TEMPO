"""Allocation-scoped multi-pair orchestration for TEMPO Elastic-PD.

The existing endpoint controller owns local-prefill or remote-handoff work
until first response.  The existing frontend owns decoder work until HTTP
EOF.  This module joins those lifetimes without pretending they are one
resource.  It is intentionally transport-neutral: vLLM/LMCache adapters feed
application-visible endpoint telemetry and execute the returned immutable
pair/route decision.

No physical-switch label, privileged NIC counter, future arrival, or benchmark
phase is accepted by this policy.  A telemetry adapter may only report totals
visible to the endpoint.  Effective use is ``max(controller_owned,
endpoint_observed_total)`` so controller traffic is never double counted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
import threading
from typing import Iterable


SCHEMA = "tempo-go-global-orchestrator-v1"
TELEMETRY_SCHEMA = "tempo-go-endpoint-telemetry-v1"
FAILURE_SCHEMA = "tempo-go-global-failure-v1"
CROSS_LAYER_SCHEMA = "tempo-go-cross-layer-envelope-v1"
JOINT_ACTUATION_SCHEMA = "tempo-go-joint-actuation-v1"
JOINT_ACTUATION_SCHEMA_V2 = "tempo-go-joint-actuation-v2"
JOINT_ACTUATION_SCHEMA_V3 = "tempo-go-joint-actuation-v3"
SERVICE_LANE_RESERVATION_SCHEMA = "tempo-go-service-lane-reservation-v1"
SERVICE_LANE_QUEUE_PROMOTION_SCHEMA = (
    "tempo-go-service-lane-queue-promotion-v1")
PRIORITY_SERVICE_LANE_BINDING = "vllm_priority_remote_cache_service_lane"
BUSINESS_PRIORITY_SERVICE_LANE_BINDING = (
    "vllm_priority_business_dual_route_service_lane")
REMOTE_CACHE_PRIORITY_SERVICE_LANE_MODE = "vllm_priority_remote_cache_v1"
BUSINESS_DUAL_ROUTE_PRIORITY_SERVICE_LANE_MODE = (
    "vllm_priority_business_dual_route_v2")
MESH_NEAR_TIE_SOURCE_BALANCE_BINDING = (
    "mesh_telemetry_uncertainty_source_virtual_service")
PROTECTED_SERVICE_LANE_BINDING = (
    "global_protected_service_lane_reservation_v1")
PROTECTED_SERVICE_LANE_MODE = "tenant_pair_edge_reservation_v1"
PROTECTED_SERVICE_LANE_RESERVE_MODE = "tenant_pair_edge_reservation_v2"


def _positive_int(name: str, value: int, *, zero: bool = False) -> None:
    if type(value) is not int or value < (0 if zero else 1):
        qualifier = "non-negative" if zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} int")


def _finite(name: str, value: float, *, minimum: float = 0.0) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"{name} must be finite and >= {minimum}")


def _sha256(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


class GlobalRoute(str, Enum):
    LOCAL = "decoder_local_chunked_prefill"
    REMOTE = "official_lmcache_remote_prefill"


class PathHealth(str, Enum):
    GOOD = "good"
    SKIP = "skip"
    DENIED = "denied"
    PROBE = "probe"


class GlobalDecisionKind(str, Enum):
    ADMIT = "admit"
    QUEUE = "queue"
    REJECT = "reject"


class GlobalRequestPhase(str, Enum):
    QUEUED = "queued"
    ROUTE_COMMITTED = "route_committed"
    FIRST_RESPONSE = "first_response"
    COMPLETE = "complete"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ResourceVector:
    """Independent semantic resources, never converted into fake bytes."""

    decode_tokens: int = 0
    active_sequences: int = 0
    endpoint_requests: int = 0
    local_prefill_token_ms: int = 0
    remote_prefill_token_ms: int = 0
    remote_kv_bytes: int = 0
    remote_semantic_ops: int = 0

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            _positive_int(name, value, zero=True)

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return (
            "decode_tokens",
            "active_sequences",
            "endpoint_requests",
            "local_prefill_token_ms",
            "remote_prefill_token_ms",
            "remote_kv_bytes",
            "remote_semantic_ops",
        )

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.names()}

    def __add__(self, other: "ResourceVector") -> "ResourceVector":
        if not isinstance(other, ResourceVector):
            return NotImplemented
        return ResourceVector(**{
            name: getattr(self, name) + getattr(other, name)
            for name in self.names()
        })

    def subtract(self, other: "ResourceVector") -> "ResourceVector":
        if not isinstance(other, ResourceVector):
            raise TypeError("resource subtraction requires ResourceVector")
        values = {
            name: getattr(self, name) - getattr(other, name)
            for name in self.names()
        }
        if any(value < 0 for value in values.values()):
            raise RuntimeError("global resource ownership underflow")
        return ResourceVector(**values)

    def stage_release(self, route: GlobalRoute) -> "ResourceVector":
        """Work released by the route's first response."""

        if route is GlobalRoute.LOCAL:
            return ResourceVector(
                endpoint_requests=self.endpoint_requests,
                local_prefill_token_ms=self.local_prefill_token_ms)
        if route is GlobalRoute.REMOTE:
            return ResourceVector(
                endpoint_requests=self.endpoint_requests,
                remote_prefill_token_ms=self.remote_prefill_token_ms,
                remote_kv_bytes=self.remote_kv_bytes,
                remote_semantic_ops=self.remote_semantic_ops,
            )
        raise TypeError("unsupported route")

    def dominant_ratio(self, capacity: "ResourceVector") -> float:
        ratios = [
            getattr(self, name) / getattr(capacity, name)
            for name in self.names()
            if getattr(capacity, name) > 0
        ]
        return max(ratios, default=0.0)


@dataclass(frozen=True)
class PairCapacity:
    pair_index: int
    resources: ResourceVector

    def __post_init__(self) -> None:
        _positive_int("pair_index", self.pair_index, zero=True)
        if not isinstance(self.resources, ResourceVector):
            raise TypeError("resources must be ResourceVector")
        if any(value <= 0 for value in self.resources.as_dict().values()):
            raise ValueError("every pair capacity must be positive")


@dataclass(frozen=True)
class TenantPolicy:
    tenant_id: str
    weight: float = 1.0
    ttft_slo_ms: float = 3_000.0
    tpot_slo_ms: float = 250.0
    e2e_slo_ms: float = 16_000.0
    maximum_queue_wait_ns: int = 5_000_000_000
    minimum_service_fraction: float = 0.0
    # Optional business reservation inside the bounded global ingress queue.
    # These slots protect a tenant from a burst of another tenant while the
    # request remains queued; they do not reserve GPU/endpoint capacity.
    queue_reservation_slots: int = 0
    # When the global reservation window expires, this tenant may still be
    # forwarded to the native vLLM waiting queue under an explicit lease.
    # The lease is still globally routed, fair, and provenance-bound; it is
    # not an untracked bypass of TEMPO.  Keep the default fail-closed so old
    # profiles retain their exact overload semantics.
    queue_lease_on_timeout: bool = False
    # A business tenant may opt into a bounded reuse window when a
    # request-triggered refresh fails.  This is not a telemetry bypass:
    # candidate capacity, path health, endpoint queue, cache affinity, and
    # deadline guards still run against the last atomic snapshot.
    telemetry_stale_grace_ns: int = 0
    # Fraction of each pair's resource vector kept available for tenants with
    # a strictly higher admission priority.  This is a physical admission
    # reserve, distinct from queue reservation slots and service fairness.
    # Lower-priority work remains work-conserving only up to the unreserved
    # capacity; it cannot fill the protected lane before an urgent request
    # arrives.
    admission_priority: int = 0
    protected_capacity_fraction: float = 0.0
    # Optional decoder-pair spread bound for a business class.  A bounded
    # class is packed onto at most this many decoder pairs for one busy epoch;
    # higher-priority tenants may then activate/use a pair outside that scope.
    # ``None`` preserves the historical fully work-conserving placement.
    pair_spread_limit: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("tenant_id must be nonempty")
        _finite("weight", self.weight, minimum=1e-12)
        for name in ("ttft_slo_ms", "tpot_slo_ms", "e2e_slo_ms"):
            _finite(name, getattr(self, name), minimum=1e-12)
        _positive_int("maximum_queue_wait_ns", self.maximum_queue_wait_ns)
        _positive_int(
            "queue_reservation_slots", self.queue_reservation_slots, zero=True)
        _positive_int(
            "telemetry_stale_grace_ns", self.telemetry_stale_grace_ns,
            zero=True)
        _positive_int(
            "admission_priority", self.admission_priority, zero=True)
        if type(self.queue_lease_on_timeout) is not bool:
            raise TypeError("queue_lease_on_timeout must be bool")
        if (
            isinstance(self.minimum_service_fraction, bool)
            or not isinstance(self.minimum_service_fraction, (int, float))
            or not math.isfinite(float(self.minimum_service_fraction))
            or not 0.0 <= float(self.minimum_service_fraction) <= 1.0
        ):
            raise ValueError("minimum_service_fraction must be in [0, 1]")
        if (
            isinstance(self.protected_capacity_fraction, bool)
            or not isinstance(self.protected_capacity_fraction, (int, float))
            or not math.isfinite(float(self.protected_capacity_fraction))
            or not 0.0 <= float(self.protected_capacity_fraction) < 1.0
        ):
            raise ValueError(
                "protected_capacity_fraction must be in [0, 1)"
            )
        if self.pair_spread_limit is not None:
            _positive_int("pair_spread_limit", self.pair_spread_limit)


@dataclass(frozen=True)
class CrossLayerSignal:
    """One scoped signal retained without collapsing the state into a label.

    ``support`` is deliberately part of the value.  ``not_collected`` and
    ``ambiguous`` are not equivalent to zero and therefore never contribute to
    the controller's derived shadow price.
    """

    name: str
    value: float | int | None
    unit: str
    support: str
    source: str
    uncertainty: float = 0.0
    scope: str = "pair"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("cross-layer signal name must be nonempty")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("cross-layer signal unit must be nonempty")
        if self.support not in {
            "supported", "not_supported", "not_collected", "ambiguous",
        }:
            raise ValueError("cross-layer signal support is invalid")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("cross-layer signal source must be nonempty")
        if self.scope not in {"node", "pair", "endpoint", "communicator"}:
            raise ValueError("cross-layer signal scope is invalid")
        if self.support == "supported":
            if (
                self.value is None
                or isinstance(self.value, bool)
                or not isinstance(self.value, (int, float))
                or not math.isfinite(float(self.value))
                or float(self.value) < 0.0
            ):
                raise ValueError("supported cross-layer signal needs a value")
        elif self.value is not None:
            raise ValueError("unsupported cross-layer signal must have no value")
        _finite("cross-layer uncertainty", self.uncertainty)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "support": self.support,
            "source": self.source,
            "uncertainty": self.uncertainty,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class CrossLayerTelemetry:
    """Immutable node/pair/endpoint evidence used by global orchestration."""

    pair_index: int
    node_id: str
    endpoint_id: str
    communicator_id: str
    source_epoch: str
    topology_fingerprint_sha256: str
    sequence: int
    sampled_ns: int
    window_ms: float
    signals: tuple[CrossLayerSignal, ...]
    cassini_by_nic: tuple[
        tuple[tuple[int, float | None, float | None], ...], ...
    ] = ()
    schema: str = CROSS_LAYER_SCHEMA

    def __post_init__(self) -> None:
        _positive_int("cross-layer pair_index", self.pair_index, zero=True)
        for name in ("node_id", "endpoint_id", "communicator_id", "source_epoch"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"cross-layer {name} must be nonempty")
        _sha256(
            "cross-layer topology_fingerprint_sha256",
            self.topology_fingerprint_sha256,
        )
        _positive_int("cross-layer sequence", self.sequence)
        _positive_int("cross-layer sampled_ns", self.sampled_ns, zero=True)
        _finite("cross-layer window_ms", self.window_ms, minimum=1e-9)
        if not self.signals or any(
            not isinstance(item, CrossLayerSignal) for item in self.signals
        ):
            raise TypeError("cross-layer telemetry requires signal values")
        names = [item.name for item in self.signals]
        if len(names) != len(set(names)):
            raise ValueError("cross-layer signal names must be unique")
        for nic in self.cassini_by_nic:
            if not isinstance(nic, tuple):
                raise TypeError("cross-layer Cassini NIC vector is invalid")
            traffic_classes = [item[0] for item in nic]
            if traffic_classes != list(range(len(traffic_classes))):
                raise ValueError("cross-layer Cassini traffic classes are not ordered")
            for traffic_class, rx_pause, tx_pause in nic:
                _positive_int(
                    "cross-layer Cassini traffic class", traffic_class, zero=True)
                for name, value in (
                    ("rx_pause_fraction", rx_pause),
                    ("tx_pause_fraction", tx_pause),
                ):
                    if value is not None:
                        _finite(
                            f"cross-layer Cassini {name}", value,
                            minimum=0.0)
                        if float(value) > 1.0:
                            raise ValueError(
                                f"cross-layer Cassini {name} exceeds one")
        if self.schema != CROSS_LAYER_SCHEMA:
            raise ValueError("cross-layer schema mismatch")

    def signal(self, name: str) -> CrossLayerSignal | None:
        return next((item for item in self.signals if item.name == name), None)

    def _supported_value(self, name: str) -> float | None:
        item = self.signal(name)
        if item is None or item.support != "supported" or item.value is None:
            return None
        return float(item.value)

    def cassini_nic_pause_maxima(self) -> tuple[float, ...]:
        """Return one pause maximum per NIC from the raw Cassini vector.

        A missing value is not congestion.  NICs with no supported RX/TX
        class value are omitted, while a supported zero remains a real zero.
        The result is intentionally derived from the vector at decision time;
        it is not exported as a synthetic policy label.
        """

        maxima: list[float] = []
        for nic in self.cassini_by_nic:
            values = [
                float(value)
                for _traffic_class, rx_pause, tx_pause in nic
                for value in (rx_pause, tx_pause)
                if value is not None
            ]
            if values:
                maxima.append(max(values))
        return tuple(maxima)

    def cassini_nic_pause_max(self) -> float | None:
        maxima = self.cassini_nic_pause_maxima()
        return max(maxima) if maxima else None

    def route_externality(self, route: "GlobalRoute") -> tuple[float, dict[str, float], float]:
        """Return derived route cost, contributions, and evidence coverage.

        The vector and support states remain authoritative.  This method is a
        small action-specific shadow-price projection, not a replacement for
        the vector and not a ``fabric_pressure`` label.
        """

        if not isinstance(route, GlobalRoute):
            raise TypeError("route must be GlobalRoute")
        prices: dict[str, float] = {
            "nccl_collective_p99_ms": 0.20 if route is GlobalRoute.REMOTE else 0.15,
            "nccl_arrival_spread_ms": 0.10,
            "lmcache_transfer_p99_ms": 0.75 if route is GlobalRoute.REMOTE else 0.0,
            "lmcache_remote_semantic_ops_inflight": (
                8.0 if route is GlobalRoute.REMOTE else 0.0),
            "lmcache_remote_kv_bytes_inflight": 1.0 / (256 * 1024 * 1024)
            if route is GlobalRoute.REMOTE else 0.0,
            "cassini_rx_pause_fraction_max": (
                250.0 if route is GlobalRoute.REMOTE else 0.0),
            "cassini_tx_pause_fraction_max": (
                250.0 if route is GlobalRoute.REMOTE else 0.0),
            "cassini_host_posted_cycles_per_packet_max": (
                0.05 if route is GlobalRoute.REMOTE else 0.0),
            "cassini_ecn_fraction_max": (
                100.0 if route is GlobalRoute.REMOTE else 0.0),
            "cassini_retries": (
                0.02 if route is GlobalRoute.REMOTE else 0.0),
            "cassini_timeouts": (
                0.50 if route is GlobalRoute.REMOTE else 0.0),
        }
        contributions: dict[str, float] = {}
        supported = 0
        considered = 0
        nic_pause = self.cassini_nic_pause_max()
        vector_replaces = {
            "cassini_rx_pause_fraction_max": nic_pause is not None,
            "cassini_tx_pause_fraction_max": nic_pause is not None,
        }
        for name, price in prices.items():
            if price == 0.0:
                continue
            if vector_replaces.get(name, False):
                # The raw per-NIC vector is more specific than a scalar
                # endpoint maximum.  Do not double-charge the same pause
                # observation when both are present in the envelope.
                continue
            considered += 1
            value = self._supported_value(name)
            # Early observer envelopes used the shorter name.  Accept it as
            # a source-compatible alias while the canonical producer emits
            # the explicit semantic-op name.
            if value is None and name == "lmcache_remote_semantic_ops_inflight":
                value = self._supported_value("lmcache_remote_ops_inflight")
            if value is None:
                continue
            supported += 1
            contributions[name] = value * price
        if nic_pause is not None and route is GlobalRoute.REMOTE:
            # Cassini pause is charged to the REMOTE action that injects KV
            # traffic.  Pair pressure still projects REMOTE externality for
            # global scaling, while LOCAL remains an explicit network-avoidance
            # fallback governed by decoder/GPU/completion state.
            considered += 1
            supported += 1
            contributions["cassini_by_nic_pause_fraction_max"] = (
                nic_pause * 250.0)
        total = sum(contributions.values())
        confidence = supported / considered if considered else 0.0
        return total, contributions, confidence

    def as_dict(self) -> dict[str, object]:
        externality = {
            route.value: {
                "cost_ms": self.route_externality(route)[0],
                "contributions_ms": self.route_externality(route)[1],
                "confidence": self.route_externality(route)[2],
            }
            for route in GlobalRoute
        }
        return {
            "schema": self.schema,
            "pair_index": self.pair_index,
            "node_id": self.node_id,
            "endpoint_id": self.endpoint_id,
            "communicator_id": self.communicator_id,
            "source_epoch": self.source_epoch,
            "topology_fingerprint_sha256": self.topology_fingerprint_sha256,
            "sequence": self.sequence,
            "sampled_ns": self.sampled_ns,
            "window_ms": self.window_ms,
            "signals": [item.as_dict() for item in self.signals],
            "cassini_by_nic": [
                [
                    {
                        "traffic_class": traffic_class,
                        "rx_pause_fraction": rx_pause,
                        "tx_pause_fraction": tx_pause,
                    }
                    for traffic_class, rx_pause, tx_pause in nic
                ]
                for nic in self.cassini_by_nic
            ],
            "derived_route_externality": externality,
        }


@dataclass(frozen=True)
class JointActuationPlan:
    """One immutable cross-layer action committed with a route.

    The plan deliberately carries independent limits for local prefill,
    remote prefill, KV bytes, and semantic transfer operations.  A caller
    may derive a dominant action pressure internally, but the decision
    receipt retains the contributing vector and never exposes a fabricated
    scalar ``fabric_pressure`` input.  ``dispatch_stagger_us`` is a bounded
    request-start delay used to spread work already admitted by the global
    controller; it is not a future-arrival or phase oracle.
    """

    pair_index: int
    route: GlobalRoute
    local_prefill_token_ms_limit: int
    remote_prefill_token_ms_limit: int
    remote_kv_bytes_limit: int
    remote_semantic_ops_limit: int
    dispatch_stagger_us: int
    telemetry_sequence: int
    confidence: float
    signal_contributions: tuple[tuple[str, float], ...] = ()
    schema: str = JOINT_ACTUATION_SCHEMA
    # v2 separates a cross-layer action target from the endpoint-enforced
    # lease.  The target is the shadow-price boundary; the enforced lease may
    # temporarily cover the atomically committed request so a transient
    # externality does not turn a work-conserving global decision into a local
    # 503.  v1 leaves these fields unset and retains its hard-window meaning.
    action_mode: str = "hard_window_v1"
    critical_guard: bool = False
    enforced_local_prefill_token_ms_limit: int | None = None
    enforced_remote_prefill_token_ms_limit: int | None = None
    enforced_remote_kv_bytes_limit: int | None = None
    enforced_remote_semantic_ops_limit: int | None = None
    overage_fraction: float = 0.0
    overage_penalty_ms: float = 0.0
    soft_overage_resources: tuple[str, ...] = ()
    # v3 adds a resource-specific budget for remote work shared by compatible
    # P/D pairs.  These are receipt fields as well as policy evidence: the
    # global admission transaction enforces the same limits before commit.
    shared_fabric_group: str | None = None
    shared_remote_requests_limit: int | None = None
    shared_remote_kv_bytes_limit: int | None = None
    shared_remote_semantic_ops_limit: int | None = None
    shared_remote_requests_used_before: int | None = None
    shared_remote_kv_bytes_used_before: int | None = None
    shared_remote_semantic_ops_used_before: int | None = None
    shared_budget_action: str = "none"
    shared_budget_contributions: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _positive_int("joint pair_index", self.pair_index, zero=True)
        if not isinstance(self.route, GlobalRoute):
            raise TypeError("joint route must be GlobalRoute")
        for name in (
            "local_prefill_token_ms_limit",
            "remote_prefill_token_ms_limit",
            "remote_kv_bytes_limit",
            "remote_semantic_ops_limit",
            "telemetry_sequence",
        ):
            _positive_int(f"joint {name}", getattr(self, name))
        _positive_int(
            "joint dispatch_stagger_us", self.dispatch_stagger_us, zero=True)
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("joint confidence must be in [0, 1]")
        if not isinstance(self.signal_contributions, tuple):
            raise TypeError("joint signal_contributions must be a tuple")
        names = []
        for item in self.signal_contributions:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0].strip()
            ):
                raise ValueError("joint signal contribution is invalid")
            _finite(f"joint contribution {item[0]}", item[1])
            if float(item[1]) > 1.0:
                raise ValueError("joint signal contribution exceeds one")
            names.append(item[0])
        if len(names) != len(set(names)):
            raise ValueError("joint signal contributions must be unique")
        if self.schema not in {
            JOINT_ACTUATION_SCHEMA,
            JOINT_ACTUATION_SCHEMA_V2,
            JOINT_ACTUATION_SCHEMA_V3,
        }:
            raise ValueError("joint actuation schema mismatch")
        if type(self.critical_guard) is not bool:
            raise TypeError("joint critical_guard must be bool")
        _finite("joint overage_fraction", self.overage_fraction)
        if float(self.overage_fraction) > 1.0:
            raise ValueError("joint overage_fraction exceeds one")
        _finite("joint overage_penalty_ms", self.overage_penalty_ms)
        if not isinstance(self.soft_overage_resources, tuple):
            raise TypeError("joint soft_overage_resources must be a tuple")
        if any(
            not isinstance(name, str) or name not in {
                "local_prefill_token_ms",
                "remote_prefill_token_ms",
                "remote_kv_bytes",
                "remote_semantic_ops",
            }
            for name in self.soft_overage_resources
        ) or len(set(self.soft_overage_resources)) != len(
            self.soft_overage_resources
        ):
            raise ValueError("joint soft overage resources are invalid")
        if self.schema == JOINT_ACTUATION_SCHEMA:
            if self.action_mode != "hard_window_v1":
                raise ValueError("v1 joint actuation must use hard_window_v1")
        elif self.schema in {JOINT_ACTUATION_SCHEMA_V2, JOINT_ACTUATION_SCHEMA_V3}:
            expected_action_mode = (
                "shared_budget_v3"
                if self.schema == JOINT_ACTUATION_SCHEMA_V3
                else "soft_shadow_price_v2"
            )
            if self.action_mode != expected_action_mode:
                raise ValueError("unsupported joint action mode")
            for name in (
                "enforced_local_prefill_token_ms_limit",
                "enforced_remote_prefill_token_ms_limit",
                "enforced_remote_kv_bytes_limit",
                "enforced_remote_semantic_ops_limit",
            ):
                _positive_int(
                    f"joint {name}", getattr(self, name))
        if self.schema == JOINT_ACTUATION_SCHEMA_V3:
            if not isinstance(self.shared_fabric_group, str) or not self.shared_fabric_group.strip():
                raise ValueError("v3 joint actuation group is invalid")
            for name in (
                "shared_remote_requests_limit",
                "shared_remote_kv_bytes_limit",
                "shared_remote_semantic_ops_limit",
                "shared_remote_requests_used_before",
                "shared_remote_kv_bytes_used_before",
                "shared_remote_semantic_ops_used_before",
            ):
                value = getattr(self, name)
                _positive_int(f"joint {name}", value, zero=True)
                if value <= 0 and name.endswith("_limit"):
                    raise ValueError(f"joint {name} must be positive")
            if self.shared_budget_action not in {
                "none", "global_remote_budget", "global_remote_stagger",
            }:
                raise ValueError("v3 joint shared budget action is invalid")
            if not isinstance(self.shared_budget_contributions, tuple):
                raise TypeError("v3 shared budget contributions must be a tuple")
            shared_names = []
            for item in self.shared_budget_contributions:
                if (
                    not isinstance(item, tuple)
                    or len(item) != 2
                    or not isinstance(item[0], str)
                    or not item[0].strip()
                ):
                    raise ValueError("v3 shared budget contribution is invalid")
                _finite(f"v3 shared contribution {item[0]}", item[1])
                if float(item[1]) > 1.0:
                    raise ValueError("v3 shared contribution exceeds one")
                shared_names.append(item[0])
            if len(shared_names) != len(set(shared_names)):
                raise ValueError("v3 shared budget contributions must be unique")

    def as_dict(self) -> dict[str, object]:
        value = {
            "schema": self.schema,
            "pair_index": self.pair_index,
            "route": self.route.value,
            "local_prefill_token_ms_limit": self.local_prefill_token_ms_limit,
            "remote_prefill_token_ms_limit": self.remote_prefill_token_ms_limit,
            "remote_kv_bytes_limit": self.remote_kv_bytes_limit,
            "remote_semantic_ops_limit": self.remote_semantic_ops_limit,
            "dispatch_stagger_us": self.dispatch_stagger_us,
            "telemetry_sequence": self.telemetry_sequence,
            "confidence": self.confidence,
            "signal_contributions": [
                {"name": name, "pressure": pressure}
                for name, pressure in self.signal_contributions
            ],
        }
        if self.schema == JOINT_ACTUATION_SCHEMA_V2:
            value.update({
                "action_mode": self.action_mode,
                "critical_guard": self.critical_guard,
                "enforced_local_prefill_token_ms_limit": (
                    self.enforced_local_prefill_token_ms_limit),
                "enforced_remote_prefill_token_ms_limit": (
                    self.enforced_remote_prefill_token_ms_limit),
                "enforced_remote_kv_bytes_limit": (
                    self.enforced_remote_kv_bytes_limit),
                "enforced_remote_semantic_ops_limit": (
                    self.enforced_remote_semantic_ops_limit),
                "overage_fraction": self.overage_fraction,
                "overage_penalty_ms": self.overage_penalty_ms,
                "soft_overage_resources": list(self.soft_overage_resources),
            })
        elif self.schema == JOINT_ACTUATION_SCHEMA_V3:
            value.update({
                "action_mode": self.action_mode,
                "critical_guard": self.critical_guard,
                "enforced_local_prefill_token_ms_limit": (
                    self.enforced_local_prefill_token_ms_limit),
                "enforced_remote_prefill_token_ms_limit": (
                    self.enforced_remote_prefill_token_ms_limit),
                "enforced_remote_kv_bytes_limit": (
                    self.enforced_remote_kv_bytes_limit),
                "enforced_remote_semantic_ops_limit": (
                    self.enforced_remote_semantic_ops_limit),
                "overage_fraction": self.overage_fraction,
                "overage_penalty_ms": self.overage_penalty_ms,
                "soft_overage_resources": list(self.soft_overage_resources),
                "shared_fabric_group": self.shared_fabric_group,
                "shared_remote_requests_limit": (
                    self.shared_remote_requests_limit),
                "shared_remote_kv_bytes_limit": (
                    self.shared_remote_kv_bytes_limit),
                "shared_remote_semantic_ops_limit": (
                    self.shared_remote_semantic_ops_limit),
                "shared_remote_requests_used_before": (
                    self.shared_remote_requests_used_before),
                "shared_remote_kv_bytes_used_before": (
                    self.shared_remote_kv_bytes_used_before),
                "shared_remote_semantic_ops_used_before": (
                    self.shared_remote_semantic_ops_used_before),
                "shared_budget_action": self.shared_budget_action,
                "shared_budget_contributions": [
                    {"name": name, "pressure": pressure}
                    for name, pressure in self.shared_budget_contributions
                ],
            })
        return value


@dataclass(frozen=True)
class PairTelemetry:
    """Causal endpoint totals sampled from an unprivileged agent."""

    pair_index: int
    sequence: int
    sampled_ns: int
    collected_ns: int
    agent_epoch: str
    profile_fingerprint_sha256: str
    controller_generation: int
    observed_total: ResourceVector
    local_health: PathHealth = PathHealth.GOOD
    remote_health: PathHealth = PathHealth.GOOD
    local_service_multiplier: float = 1.0
    remote_service_multiplier: float = 1.0
    local_failure_count: int = 0
    remote_failure_count: int = 0
    local_last_failure_kind: str | None = None
    remote_last_failure_kind: str | None = None
    scheduler_running_requests: int | None = None
    scheduler_waiting_requests: int | None = None
    scheduler_kv_cache_usage_fraction: float | None = None
    scheduler_schema: str | None = None
    scheduler_source: str | None = None
    endpoint_completed_first_responses: int | None = None
    endpoint_residual_inflight: int | None = None
    completion_schema: str | None = None
    service_lane_queue_requests: int | None = None
    service_lane_queue_offers: int | None = None
    service_lane_pending_global_commits: int | None = None
    service_lane_active_reservations: int | None = None
    service_lane_active_queue_leases: int | None = None
    quarantine_reason: str | None = None
    source: str = "application_endpoint_agent"
    schema: str = TELEMETRY_SCHEMA
    cross_layer: CrossLayerTelemetry | None = None

    def __post_init__(self) -> None:
        _positive_int("pair_index", self.pair_index, zero=True)
        _positive_int("sequence", self.sequence)
        _positive_int("sampled_ns", self.sampled_ns, zero=True)
        _positive_int("collected_ns", self.collected_ns, zero=True)
        if self.collected_ns < self.sampled_ns:
            raise ValueError("telemetry collection finishes before it starts")
        if not isinstance(self.agent_epoch, str) or not self.agent_epoch.strip():
            raise ValueError("agent_epoch must be nonempty")
        _sha256("profile_fingerprint_sha256", self.profile_fingerprint_sha256)
        _positive_int(
            "controller_generation", self.controller_generation, zero=True)
        if not isinstance(self.observed_total, ResourceVector):
            raise TypeError("observed_total must be ResourceVector")
        if not isinstance(self.local_health, PathHealth) or not isinstance(
            self.remote_health, PathHealth
        ):
            raise TypeError("health values must be PathHealth")
        _finite("local_service_multiplier", self.local_service_multiplier, minimum=1.0)
        _finite("remote_service_multiplier", self.remote_service_multiplier, minimum=1.0)
        for name in ("local_failure_count", "remote_failure_count"):
            _positive_int(name, getattr(self, name), zero=True)
        for name in ("local_last_failure_kind", "remote_last_failure_kind"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be nonempty when present")
        scheduler_values = (
            self.scheduler_running_requests,
            self.scheduler_waiting_requests,
            self.scheduler_kv_cache_usage_fraction,
        )
        if any(value is not None for value in scheduler_values):
            if any(value is None for value in scheduler_values):
                raise ValueError("scheduler telemetry fields must be complete")
            for name in ("scheduler_running_requests", "scheduler_waiting_requests"):
                _positive_int(name, getattr(self, name), zero=True)
            usage = float(self.scheduler_kv_cache_usage_fraction)
            if not math.isfinite(usage) or not 0.0 <= usage <= 1.0:
                raise ValueError("scheduler KV-cache usage must be in [0, 1]")
            if not isinstance(self.scheduler_schema, str) or not self.scheduler_schema.strip():
                raise ValueError("scheduler_schema is required with scheduler telemetry")
            if not isinstance(self.scheduler_source, str) or not self.scheduler_source.strip():
                raise ValueError("scheduler_source is required with scheduler telemetry")
        elif self.scheduler_schema is not None or self.scheduler_source is not None:
            raise ValueError("scheduler identity requires scheduler telemetry")
        completion_values = (
            self.endpoint_completed_first_responses,
            self.endpoint_residual_inflight,
        )
        if any(value is not None for value in completion_values):
            if any(value is None for value in completion_values):
                raise ValueError("completion telemetry fields must be complete")
            _positive_int(
                "endpoint_completed_first_responses",
                self.endpoint_completed_first_responses,
                zero=True,
            )
            _positive_int(
                "endpoint_residual_inflight",
                self.endpoint_residual_inflight,
                zero=True,
            )
            if not isinstance(self.completion_schema, str) or not self.completion_schema.strip():
                raise ValueError("completion_schema is required with completion telemetry")
        elif self.completion_schema is not None:
            raise ValueError("completion identity requires completion telemetry")
        service_lane_values = (
            self.service_lane_queue_requests,
            self.service_lane_queue_offers,
            self.service_lane_pending_global_commits,
            self.service_lane_active_reservations,
            self.service_lane_active_queue_leases,
        )
        if any(value is not None for value in service_lane_values):
            if any(value is None for value in service_lane_values):
                raise ValueError("service-lane telemetry fields must be complete")
            for name in (
                "service_lane_queue_requests",
                "service_lane_queue_offers",
                "service_lane_pending_global_commits",
                "service_lane_active_reservations",
                "service_lane_active_queue_leases",
            ):
                _positive_int(name, getattr(self, name), zero=True)
        if self.quarantine_reason is not None and (
            not isinstance(self.quarantine_reason, str)
            or not self.quarantine_reason.strip()
        ):
            raise ValueError("quarantine_reason must be nonempty when present")
        if self.schema != TELEMETRY_SCHEMA:
            raise ValueError("telemetry schema mismatch")
        if self.source != "application_endpoint_agent":
            raise ValueError("telemetry source is not policy-eligible")
        if self.cross_layer is not None:
            if not isinstance(self.cross_layer, CrossLayerTelemetry):
                raise TypeError("cross_layer must be CrossLayerTelemetry")
            if self.cross_layer.pair_index != self.pair_index:
                raise ValueError("cross-layer pair identity differs")

    def health(self, route: GlobalRoute) -> PathHealth:
        return self.local_health if route is GlobalRoute.LOCAL else self.remote_health

    def multiplier(self, route: GlobalRoute) -> float:
        return (
            float(self.local_service_multiplier)
            if route is GlobalRoute.LOCAL
            else float(self.remote_service_multiplier)
        )


@dataclass(frozen=True)
class RouteCandidate:
    """One immutable local placement or remote P->D edge candidate.

    ``pair_index`` remains the decoder-resource index for compatibility with
    the v1 profiles and ledgers.  C6 makes the formerly implicit placement
    explicit: local work is ``D_i`` and remote work is ``P_i -> D_j``.
    Legacy callers that omit the new fields are normalized to the historical
    paired edge ``P_i -> D_i``.
    """

    pair_index: int
    route: GlobalRoute
    work: ResourceVector
    predicted_e2e_ms: float
    predicted_ttft_ms: float
    uncertainty_ms: float = 0.0
    cache_affinity: bool = False
    prefill_index: int | None = None
    decoder_index: int | None = None
    edge_id: str | None = None

    def __post_init__(self) -> None:
        _positive_int("pair_index", self.pair_index, zero=True)
        if not isinstance(self.route, GlobalRoute):
            raise TypeError("route must be GlobalRoute")
        if not isinstance(self.work, ResourceVector):
            raise TypeError("work must be ResourceVector")
        _finite("predicted_e2e_ms", self.predicted_e2e_ms, minimum=1e-12)
        _finite("predicted_ttft_ms", self.predicted_ttft_ms, minimum=1e-12)
        _finite("uncertainty_ms", self.uncertainty_ms)
        if type(self.cache_affinity) is not bool:
            raise TypeError("cache_affinity must be bool")
        prefill_index = (
            self.pair_index if self.prefill_index is None
            else self.prefill_index
        )
        decoder_index = (
            self.pair_index if self.decoder_index is None
            else self.decoder_index
        )
        _positive_int("prefill_index", prefill_index, zero=True)
        _positive_int("decoder_index", decoder_index, zero=True)
        if decoder_index != self.pair_index:
            raise ValueError(
                "pair_index must remain the decoder_index compatibility alias")
        if self.route is GlobalRoute.LOCAL and prefill_index != decoder_index:
            raise ValueError("local candidate cannot cross P/D endpoints")
        canonical_edge_id = (
            f"local:d{decoder_index}"
            if self.route is GlobalRoute.LOCAL
            else f"remote:p{prefill_index}->d{decoder_index}"
        )
        if self.edge_id is not None and self.edge_id != canonical_edge_id:
            raise ValueError("candidate edge_id is not canonical")
        object.__setattr__(self, "prefill_index", prefill_index)
        object.__setattr__(self, "decoder_index", decoder_index)
        object.__setattr__(self, "edge_id", canonical_edge_id)
        if self.work.decode_tokens <= 0 or self.work.active_sequences <= 0:
            raise ValueError("every candidate must reserve decoder work")
        if self.work.endpoint_requests != 1:
            raise ValueError(
                "every candidate must reserve one first-response endpoint slot")
        if self.route is GlobalRoute.LOCAL:
            if self.work.local_prefill_token_ms <= 0 or any((
                self.work.remote_prefill_token_ms,
                self.work.remote_kv_bytes,
                self.work.remote_semantic_ops,
            )):
                raise ValueError("local candidate has invalid route work")
        elif (
            self.work.local_prefill_token_ms
            or self.work.remote_prefill_token_ms <= 0
            or self.work.remote_kv_bytes <= 0
            or self.work.remote_semantic_ops <= 0
        ):
            raise ValueError("remote candidate has invalid route work")

    @property
    def identity_key(self) -> tuple[int, int, GlobalRoute]:
        assert self.prefill_index is not None
        assert self.decoder_index is not None
        return self.prefill_index, self.decoder_index, self.route


@dataclass(frozen=True)
class GlobalRequest:
    request_id: str
    tenant_id: str
    arrival_ns: int
    deadline_ns: int
    candidates: tuple[RouteCandidate, ...]
    # Deterministic token-chunk group derived by the native tokenizer.  It is
    # optional for legacy/replay requests, but when present the global
    # controller serializes concurrent remote transfers for the same
    # pair-local cache chunk.  This prevents push-based LMCache receiver
    # ownership from being raced by duplicate shared-prefix transfers.
    cache_group_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be nonempty")
        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("tenant_id must be nonempty")
        _positive_int("arrival_ns", self.arrival_ns, zero=True)
        if type(self.deadline_ns) is not int or self.deadline_ns <= self.arrival_ns:
            raise ValueError("deadline_ns must be greater than arrival_ns")
        if not self.candidates:
            raise ValueError("request requires route candidates")
        keys = [item.identity_key for item in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "request candidates must be unique by prefill, decoder, and route")
        if self.cache_group_key is not None:
            if (
                not isinstance(self.cache_group_key, str)
                or len(self.cache_group_key) != 64
                or any(
                    char not in "0123456789abcdef"
                    for char in self.cache_group_key
                )
            ):
                raise ValueError("cache_group_key must be a SHA-256 hex digest")


@dataclass(frozen=True)
class GlobalOrchestratorConfig:
    capacities: tuple[PairCapacity, ...]
    tenants: tuple[TenantPolicy, ...]
    telemetry_fresh_ns: int
    queue_capacity: int
    # A fresh batch is preferred, but a bounded allocation-scoped grace can
    # keep admission work-conserving when a request-triggered refresh times
    # out.  This never creates capacity: the last installed snapshot, local
    # resource ledger, route health, and transport hard guards still apply.
    telemetry_stale_grace_ns: int = 0
    minimum_active_pairs: int = 1
    maximum_active_pairs: int | None = None
    scale_up_utilization: float = 0.8
    scale_down_idle_ns: int = 5_000_000_000
    utilization_penalty_ms: float = 100.0
    activation_penalty_ms: float = 1.0
    probe_penalty_ms: float = 10.0
    maximum_queue_wait_ns: int = 5_000_000_000
    overload_action: str = "reject_new_request"
    # An explicit endpoint queue is a real downstream service phase.  The
    # default keeps historical profiles fail-closed; the work-conserving
    # v1 mode lets an opted-in queue lease carry only endpoint-window debt to
    # vLLM's bounded retry queue.  v2 additionally permits a failure-free
    # stale-completion route to receive one bounded liveness probe.  Shared
    # remote-fabric budgets, transport hard guards, explicit route failures,
    # cache affinity, and tenant deadlines remain hard limits.
    endpoint_queue_debt_mode: str = "disabled"
    # A headroom-first profile may hand a queued request to the native
    # endpoint queue immediately when the current scheduler/completion
    # snapshot proves bounded service-lane headroom.  The default preserves
    # the historical after-global-timeout behavior.
    endpoint_queue_admission_mode: str = "after_timeout"
    # Maximum number of TEMPO-owned queue leases that may wait at one
    # endpoint. ``None`` inherits the frozen global ingress queue capacity
    # for backwards-compatible profiles.
    endpoint_queue_capacity: int | None = None
    # A stale-feedback recovery probe is normally one-shot per route.  The
    # opt-in shared mode lets later waiters reuse that already-running proof
    # while explicit native queue headroom remains available.  It does not
    # create another probe and does not bypass completion/fabric guards.
    completion_liveness_shared_probe_mode: str = "disabled"
    # An opt-in queue lease may use fresh, failure-free completion progress as
    # the initial credit when the native endpoint queue still has headroom.
    # This is what lets a burst enter the bounded vLLM queue before the global
    # reservation window expires; it never waives endpoint headroom or route
    # and fabric guards.
    endpoint_queue_headroom_admission_mode: str = "disabled"
    # A production decoder can expose a bounded, business-owned priority lane
    # independently of its ordinary FCFS backlog.  The v1 lane is deliberately
    # narrow: only a proven remote cache-affinity candidate for an opted-in
    # business tenant may use it.  The global controller still enforces mesh,
    # cache, fabric, route-health, and deadline guards; it waives only the
    # ordinary scheduler-waiting headroom and prices the request as priority
    # service instead of placing it behind the observed FCFS queue.
    priority_service_lane_mode: str = "disabled"
    priority_service_lane_capacity: int = 0
    priority_service_lane_min_admission_priority: int = 0
    # vLLM orders lower integer priorities first.  Zero is required while the
    # lane is disabled; an enabled lane must bind a negative priority so the
    # control-plane lease corresponds to a real downstream scheduling action.
    priority_service_lane_priority: int = 0
    # Preserve decoder service capacity before vLLM's non-preemptive running
    # set and waiting queue are saturated by lower-business-priority work.
    # The frontend derives each decoder's background concurrency from
    # ``active_sequences - priority_service_lane_capacity`` and pauses new
    # background upstream starts while a globally committed protected tenant
    # is active.  Every offered request remains queued and a bounded escape
    # prevents indefinite starvation.
    decoder_business_admission_mode: str = "disabled"
    decoder_business_background_max_wait_ns: int = 0
    # Candidate B can open a prewarmed spare when the currently observed
    # ingress queue or tenant queue age indicates approaching SLO risk.  The
    # default of 1.0 preserves the original pressure-only behavior for legacy
    # profiles: the trigger fires only at a full queue or at the full wait
    # budget.  These are current-state business signals, never future arrivals.
    proactive_scale_up_queue_fraction: float = 1.0
    proactive_scale_up_wait_fraction: float = 1.0
    proactive_scale_up_active_pair_penalty_ms: float = 0.0
    # A prewarmed decoder may be inactive while an active decoder still has
    # an admissible route.  In a P-by-D mesh that is not sufficient reason to
    # discard the inactive frontier: a cool remote edge can be the globally
    # cheaper route once live scheduler/receiver pressure is included.  Open
    # the spare only when its fully-priced live score beats the best active
    # score by this margin.  Zero means strict score improvement and keeps
    # legacy profiles' economics explicit rather than using a phase label.
    proactive_scale_up_route_benefit_margin_ms: float = 0.0
    # Business pair packing protects a higher-priority tenant from a lower
    # priority tenant's packed pair, but it must not turn into permanent
    # single-pair pinning.  Once every clean candidate is at this observed
    # utilization, re-admit dirty candidates and let the complete live score
    # (decoder, endpoint, mesh, and fabric) choose a spill/scale route.
    business_clean_pair_pressure_fraction: float = 1.0
    # The endpoint controller's semantic-op window is a transport/data-plane
    # safety boundary, not a promise that the final slot is safe to use under
    # contention.  A frozen non-zero reserve lets discovery protect that
    # boundary before route commit while keeping the decision auditable.
    remote_semantic_ops_safety_reserve: int = 0
    # Candidate C converts an endpoint failure receipt into an immediate
    # fail-closed route/pair circuit breaker.  Recovery requires a later
    # explicit PROBE telemetry observation.
    route_failure_quarantine_mode: str = "disabled"
    # Candidate I promotes a cumulative endpoint failure observation into a
    # pair-health circuit before a new request can reuse the pair.  This is
    # deliberately separate from the explicit per-request failure receipt
    # mode above: the adapter may observe a dead EngineCore/cache-key path
    # before a request-level HTTP failure is delivered to the frontend.
    telemetry_failure_quarantine_mode: str = "disabled"
    telemetry_failure_quarantine_scope: str = "pair"
    # When a pair is quarantined, preserve this fraction of every surviving
    # pair's capacity for urgent/minimum-service tenants.  This is a service
    # capacity reservation, not an ingress queue reservation.
    survivor_capacity_reserve_fraction: float = 0.0
    survivor_reserve_bypass_min_weight: float = 0.0
    # Cross-layer action bounds.  These are frozen policy parameters, not
    # privileged fabric settings.  They leave a minimum useful service
    # window while the live vector asks the global controller to shed or
    # stagger transfer work under coupled NCCL/LMCache/Cassini pressure.
    cross_layer_remote_limit_floor_fraction: float = 0.25
    cross_layer_local_limit_floor_fraction: float = 0.50
    cross_layer_stagger_max_us: int = 2_000
    # v1 treats the action target as a hard endpoint window.  v2 makes the
    # target a resource-specific shadow price and only hard-guards explicitly
    # critical transport signals; the committed request receives an enforced
    # lease so global admission remains work-conserving.
    cross_layer_control_mode: str = "hard_window_v1"
    cross_layer_shadow_price_ms: float = 0.0
    # Price receiver-side LMCache transfer latency for LOCAL candidates when
    # the observer is pair-scoped.  A local route avoids issuing a new KV
    # transfer, but it still competes with the observed receiver/collective
    # work on that decoder pair; treating that externality as zero was the
    # measured cause of the C9 normal-control D0 tail.
    cross_layer_local_receiver_price_ms: float = 0.0
    # A pair-scoped LMCache transfer tail above this explicit service ceiling
    # is a receiver admission guard, not merely a score penalty.  It denies
    # new REMOTE work on that pair while still allowing LOCAL/spare-pair
    # candidates to be evaluated.  Disabled by default so legacy profiles
    # preserve their exact semantics.
    cross_layer_remote_receiver_guard_mode: str = "disabled"
    # ``pair`` protects only the receiver identified by the candidate's
    # decoder pair. ``shared_group`` is the allocation-wide safety mode:
    # when one compatible pair is hot, a new remote edge is not admitted to
    # another member merely to move the incast victim.
    cross_layer_remote_receiver_guard_scope: str = "pair"
    # A deployment may bind several P/D pairs to one allocation-wide fabric
    # group even when one member cannot publish a cross-layer sample (for
    # example, a co-job observer covers only P0-D0).  This explicit identity
    # lets the guard fail closed for the whole declared group; an empty value
    # retains telemetry-derived group membership for legacy profiles.
    cross_layer_remote_receiver_guard_group_id: str = ""
    cross_layer_remote_receiver_guard_p99_ms: float = 0.0
    cross_layer_critical_pressure_fraction: float = 2.0
    # v3 enables an allocation-scoped shared remote budget.  Zero capacities
    # mean "derive the sum of configured pair capacities"; explicit values
    # are used by a frozen Perlmutter profile.  The default remains disabled
    # so v1/v2 historical profiles retain their exact contract.
    shared_fabric_control_mode: str = "disabled"
    shared_remote_requests_capacity: int = 0
    shared_remote_kv_bytes_capacity: int = 0
    shared_remote_semantic_ops_capacity: int = 0
    shared_remote_limit_floor_fraction: float = 0.25
    shared_remote_stagger_max_us: int = 2_000
    # C6 separates the producer P, destination D, and remote edge while
    # retaining the v1 pair capacity rows as endpoint capacity priors.  The
    # receiver owns KV/semantic credits; source-prefill and edge residuals are
    # tracked independently and returned at first response.
    mesh_control_mode: str = "disabled"
    mesh_receiver_stagger_max_us: int = 2_000
    mesh_edge_service_ewma_alpha: float = 0.25
    # A live P-by-D mesh can expose two healthy sources whose telemetry-priced
    # route scores differ by less than the predictor's own uncertainty.  A
    # deterministic source-index tie break then concentrates all cache reads
    # on one Slingshot edge even though the controller cannot statistically
    # distinguish the alternatives.  The opt-in v1 mode uses controller-owned
    # admitted source/edge service as a virtual-finish tie break *only* inside
    # that uncertainty envelope and only for the same decoder.  It never
    # overrides a materially cheaper route and it does not target a route
    # ratio or inspect a workload/request ordinal.
    mesh_near_tie_source_balance_mode: str = "disabled"
    mesh_near_tie_source_balance_uncertainty_fraction: float = 0.0
    # In a live hot/cool crossover, a remote edge's lower measured TTFT prior
    # can be hidden by a conservative remote E2E upper bound.  Credit that
    # prior only when an active decoder is hot while both the remote source
    # and destination are cool; normal phases retain ordinary E2E scoring.
    mesh_cool_remote_route_pressure_fraction: float = 0.5
    # Candidate J: forecast currently observed service waves across the
    # destination decoder/endpoint and, for remote work, source/edge/receiver
    # before committing a request. Disabled preserves legacy profiles; the
    # enabled mode is a hard deadline feasibility lease, not a future-arrival
    # predictor or a second route-ratio policy.
    service_feasibility_mode: str = "disabled"
    service_forecast_safety_factor: float = 1.0
    # Candidate K: reserve a small, explicit service lane per business tenant
    # and physical P->D edge.  Lower-priority work is admitted only from the
    # residual endpoint/decoder/edge budget; protected work consumes the
    # reservation atomically with the normal route lease.
    protected_service_lane_mode: str = "disabled"
    protected_service_lane_capacity: int = 0
    protected_service_lane_min_admission_priority: int = 0

    def __post_init__(self) -> None:
        if not self.capacities:
            raise ValueError("at least one pair capacity is required")
        indices = [item.pair_index for item in self.capacities]
        if indices != list(range(len(indices))):
            raise ValueError("pair capacities must be contiguous and ordered")
        if not self.tenants:
            raise ValueError("at least one tenant policy is required")
        tenant_ids = [item.tenant_id for item in self.tenants]
        if len(tenant_ids) != len(set(tenant_ids)):
            raise ValueError("tenant policies must be unique")
        for name in (
            "telemetry_fresh_ns",
            "queue_capacity",
            "minimum_active_pairs",
            "scale_down_idle_ns",
            "maximum_queue_wait_ns",
        ):
            _positive_int(name, getattr(self, name))
        _positive_int(
            "telemetry_stale_grace_ns",
            self.telemetry_stale_grace_ns,
            zero=True,
        )
        reserved_slots = sum(
            item.queue_reservation_slots for item in self.tenants)
        if reserved_slots > self.queue_capacity:
            raise ValueError(
                "tenant queue reservations exceed global queue capacity")
        _positive_int(
            "remote_semantic_ops_safety_reserve",
            self.remote_semantic_ops_safety_reserve,
            zero=True,
        )
        if any(
            self.remote_semantic_ops_safety_reserve
            >= capacity.resources.remote_semantic_ops
            for capacity in self.capacities
        ):
            raise ValueError(
                "remote semantic-op safety reserve leaves no admission slot")
        maximum = self.maximum_active_pairs
        if maximum is None:
            maximum = len(self.capacities)
            object.__setattr__(self, "maximum_active_pairs", maximum)
        if not self.minimum_active_pairs <= maximum <= len(self.capacities):
            raise ValueError("active pair bounds are invalid")
        for tenant in self.tenants:
            if (
                tenant.pair_spread_limit is not None
                and tenant.pair_spread_limit > len(self.capacities)
            ):
                raise ValueError(
                    "tenant pair spread limit exceeds configured pair count"
                )
        if self.overload_action not in {
            "reject_new_request", "endpoint_queue_lease",
        }:
            raise ValueError("unsupported overload_action")
        if self.endpoint_queue_debt_mode not in {
            "disabled",
            "work_conserving_endpoint_queue_v1",
            "completion_liveness_endpoint_queue_v2",
            "completion_credit_endpoint_queue_v3",
            "completion_credit_mesh_endpoint_queue_v1",
        }:
            raise ValueError("unsupported endpoint_queue_debt_mode")
        if self.endpoint_queue_admission_mode not in {
            "after_timeout", "headroom_first_v1",
        }:
            raise ValueError("unsupported endpoint_queue_admission_mode")
        if self.completion_liveness_shared_probe_mode not in {
            "disabled", "headroom_shared_v1",
        }:
            raise ValueError("unsupported completion_liveness_shared_probe_mode")
        if self.endpoint_queue_headroom_admission_mode not in {
            "disabled", "completion_progress_v1",
        }:
            raise ValueError("unsupported endpoint_queue_headroom_admission_mode")
        if (
            self.completion_liveness_shared_probe_mode != "disabled"
            and self.endpoint_queue_debt_mode
            != "completion_credit_mesh_endpoint_queue_v1"
        ):
            raise ValueError(
                "shared liveness probes require receiver-credit mesh queue mode"
            )
        endpoint_queue_capacity = self.endpoint_queue_capacity
        if endpoint_queue_capacity is None:
            endpoint_queue_capacity = self.queue_capacity
            object.__setattr__(
                self, "endpoint_queue_capacity", endpoint_queue_capacity)
        _positive_int("endpoint_queue_capacity", endpoint_queue_capacity)
        if self.priority_service_lane_mode not in {
            "disabled",
            REMOTE_CACHE_PRIORITY_SERVICE_LANE_MODE,
            BUSINESS_DUAL_ROUTE_PRIORITY_SERVICE_LANE_MODE,
        }:
            raise ValueError("unsupported priority_service_lane_mode")
        _positive_int(
            "priority_service_lane_capacity",
            self.priority_service_lane_capacity,
            zero=True,
        )
        _positive_int(
            "priority_service_lane_min_admission_priority",
            self.priority_service_lane_min_admission_priority,
            zero=True,
        )
        if type(self.priority_service_lane_priority) is not int:
            raise TypeError("priority_service_lane_priority must be an int")
        if self.priority_service_lane_mode == "disabled":
            if (
                self.priority_service_lane_capacity != 0
                or self.priority_service_lane_priority != 0
            ):
                raise ValueError(
                    "disabled priority service lane cannot reserve capacity "
                    "or a downstream priority"
                )
        elif (
            self.priority_service_lane_capacity <= 0
            or self.priority_service_lane_min_admission_priority <= 0
            or self.priority_service_lane_priority not in {-2, -1}
            or self.overload_action != "endpoint_queue_lease"
        ):
            raise ValueError(
                "vLLM priority service lane requires endpoint queue leasing, "
                "positive capacity/business priority, and priority -1 or -2"
            )
        if self.decoder_business_admission_mode not in {
            "disabled", "priority_drain_v1",
        }:
            raise ValueError("unsupported decoder_business_admission_mode")
        _positive_int(
            "decoder_business_background_max_wait_ns",
            self.decoder_business_background_max_wait_ns,
            zero=True,
        )
        if self.decoder_business_admission_mode == "disabled":
            if self.decoder_business_background_max_wait_ns != 0:
                raise ValueError(
                    "disabled decoder business admission cannot retain wait")
        elif (
            self.priority_service_lane_mode not in {
                REMOTE_CACHE_PRIORITY_SERVICE_LANE_MODE,
                BUSINESS_DUAL_ROUTE_PRIORITY_SERVICE_LANE_MODE,
            }
            or self.decoder_business_background_max_wait_ns <= 0
            or any(
                capacity.resources.active_sequences
                <= self.priority_service_lane_capacity
                for capacity in self.capacities
            )
        ):
            raise ValueError(
                "priority decoder drain requires the priority lane, bounded "
                "background wait, and residual active-sequence capacity"
            )
        if self.route_failure_quarantine_mode not in {
            "disabled", "deny_until_probe",
        }:
            raise ValueError("unsupported route_failure_quarantine_mode")
        if self.telemetry_failure_quarantine_mode not in {
            "disabled", "deny_until_probe",
        }:
            raise ValueError("unsupported telemetry_failure_quarantine_mode")
        if self.telemetry_failure_quarantine_scope not in {"route", "pair"}:
            raise ValueError("unsupported telemetry_failure_quarantine_scope")
        if not 0.0 <= float(self.survivor_capacity_reserve_fraction) < 1.0:
            raise ValueError(
                "survivor_capacity_reserve_fraction must be in [0, 1)")
        _finite(
            "survivor_reserve_bypass_min_weight",
            self.survivor_reserve_bypass_min_weight,
        )
        for name in (
            "cross_layer_remote_limit_floor_fraction",
            "cross_layer_local_limit_floor_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        _positive_int(
            "cross_layer_stagger_max_us",
            self.cross_layer_stagger_max_us,
            zero=True,
        )
        if self.cross_layer_control_mode not in {
            "hard_window_v1", "soft_shadow_price_v2",
        }:
            raise ValueError("unsupported cross_layer_control_mode")
        _finite(
            "cross_layer_shadow_price_ms",
            self.cross_layer_shadow_price_ms,
        )
        _finite(
            "cross_layer_local_receiver_price_ms",
            self.cross_layer_local_receiver_price_ms,
        )
        if self.cross_layer_remote_receiver_guard_mode not in {
            "disabled", "deny_while_hot",
        }:
            raise ValueError(
                "unsupported cross_layer_remote_receiver_guard_mode")
        if self.cross_layer_remote_receiver_guard_scope not in {
            "pair", "shared_group",
        }:
            raise ValueError(
                "unsupported cross_layer_remote_receiver_guard_scope")
        if (
            self.cross_layer_remote_receiver_guard_scope == "shared_group"
            and self.mesh_control_mode == "disabled"
        ):
            raise ValueError(
                "shared_group receiver guard requires P/D mesh control")
        if (
            self.cross_layer_remote_receiver_guard_scope == "shared_group"
            and self.cross_layer_remote_receiver_guard_group_id
            and not self.cross_layer_remote_receiver_guard_group_id.strip()
        ):
            raise ValueError(
                "receiver guard group id must be non-whitespace when set")
        _finite(
            "cross_layer_remote_receiver_guard_p99_ms",
            self.cross_layer_remote_receiver_guard_p99_ms,
            minimum=0.0,
        )
        if (
            self.cross_layer_remote_receiver_guard_mode == "deny_while_hot"
            and self.cross_layer_remote_receiver_guard_p99_ms <= 0.0
        ):
            raise ValueError(
                "receiver guard mode requires a positive p99 ceiling")
        _finite(
            "cross_layer_critical_pressure_fraction",
            self.cross_layer_critical_pressure_fraction,
            minimum=1.0,
        )
        if (
            self.cross_layer_control_mode == "soft_shadow_price_v2"
            and self.cross_layer_shadow_price_ms <= 0.0
        ):
            raise ValueError(
                "soft_shadow_price_v2 requires a positive shadow price")
        if self.shared_fabric_control_mode not in {
            "disabled", "global_budget_v3",
        }:
            raise ValueError("unsupported shared_fabric_control_mode")
        for name in (
            "shared_remote_requests_capacity",
            "shared_remote_kv_bytes_capacity",
            "shared_remote_semantic_ops_capacity",
        ):
            _positive_int(name, getattr(self, name), zero=True)
        value = float(self.shared_remote_limit_floor_fraction)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("shared_remote_limit_floor_fraction is invalid")
        _positive_int(
            "shared_remote_stagger_max_us",
            self.shared_remote_stagger_max_us,
            zero=True,
        )
        if self.mesh_control_mode not in {
            "disabled", "receiver_credit_pxd_v1",
        }:
            raise ValueError("unsupported mesh_control_mode")
        if (
            self.endpoint_queue_debt_mode
            == "completion_credit_mesh_endpoint_queue_v1"
            and self.mesh_control_mode != "receiver_credit_pxd_v1"
        ):
            raise ValueError(
                "mesh endpoint queue credit requires receiver-credit mesh mode"
            )
        _positive_int(
            "mesh_receiver_stagger_max_us",
            self.mesh_receiver_stagger_max_us,
            zero=True,
        )
        if not 0.0 < float(self.mesh_edge_service_ewma_alpha) <= 1.0:
            raise ValueError("mesh_edge_service_ewma_alpha must be in (0, 1]")
        if self.mesh_near_tie_source_balance_mode not in {
            "disabled", "telemetry_uncertainty_virtual_service_v1",
        }:
            raise ValueError("unsupported mesh_near_tie_source_balance_mode")
        _finite(
            "mesh_near_tie_source_balance_uncertainty_fraction",
            self.mesh_near_tie_source_balance_uncertainty_fraction,
        )
        if self.mesh_near_tie_source_balance_mode == "disabled":
            if self.mesh_near_tie_source_balance_uncertainty_fraction != 0.0:
                raise ValueError(
                    "disabled mesh near-tie source balance cannot retain "
                    "an uncertainty fraction"
                )
        elif (
            self.mesh_control_mode != "receiver_credit_pxd_v1"
            or not 0.0
            < self.mesh_near_tie_source_balance_uncertainty_fraction
            <= 1.0
        ):
            raise ValueError(
                "mesh near-tie source balance requires receiver-credit mesh "
                "mode and an uncertainty fraction in (0, 1]"
            )
        if not 0.0 < float(self.mesh_cool_remote_route_pressure_fraction) <= 1.0:
            raise ValueError(
                "mesh_cool_remote_route_pressure_fraction must be in (0, 1]"
            )
        if self.service_feasibility_mode not in {
            "disabled", "deadline_residual_v1",
        }:
            raise ValueError("unsupported service_feasibility_mode")
        _finite(
            "service_forecast_safety_factor",
            self.service_forecast_safety_factor,
            minimum=1.0,
        )
        if self.protected_service_lane_mode not in {
            "disabled", PROTECTED_SERVICE_LANE_MODE,
            PROTECTED_SERVICE_LANE_RESERVE_MODE,
        }:
            raise ValueError("unsupported protected_service_lane_mode")
        _positive_int(
            "protected_service_lane_capacity",
            self.protected_service_lane_capacity,
            zero=True,
        )
        _positive_int(
            "protected_service_lane_min_admission_priority",
            self.protected_service_lane_min_admission_priority,
            zero=True,
        )
        if self.protected_service_lane_mode == "disabled":
            if (
                self.protected_service_lane_capacity != 0
                or self.protected_service_lane_min_admission_priority != 0
            ):
                raise ValueError(
                    "disabled protected service lane cannot retain capacity "
                    "or a business priority"
                )
        elif (
            self.mesh_control_mode != "receiver_credit_pxd_v1"
            or self.protected_service_lane_capacity <= 0
            or self.protected_service_lane_min_admission_priority <= 0
            or any(
                capacity.resources.endpoint_requests
                <= self.protected_service_lane_capacity
                for capacity in self.capacities
            )
        ):
            raise ValueError(
                "protected service lane requires mesh control, positive "
                "capacity/priority, and residual endpoint capacity"
            )
        if not 0.0 < float(self.scale_up_utilization) <= 1.0:
            raise ValueError("scale_up_utilization must be in (0, 1]")
        if not 0.0 < float(self.business_clean_pair_pressure_fraction) <= 1.0:
            raise ValueError(
                "business_clean_pair_pressure_fraction must be in (0, 1]"
            )
        for name in (
            "proactive_scale_up_queue_fraction",
            "proactive_scale_up_wait_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        _finite(
            "proactive_scale_up_active_pair_penalty_ms",
            self.proactive_scale_up_active_pair_penalty_ms,
        )
        _finite(
            "proactive_scale_up_route_benefit_margin_ms",
            self.proactive_scale_up_route_benefit_margin_ms,
        )
        if self.proactive_scale_up_route_benefit_margin_ms < 0.0:
            raise ValueError(
                "proactive_scale_up_route_benefit_margin_ms must be non-negative"
            )
        for name in (
            "utilization_penalty_ms",
            "activation_penalty_ms",
            "probe_penalty_ms",
        ):
            _finite(name, getattr(self, name))


@dataclass(frozen=True)
class RejectedCandidate:
    pair_index: int
    route: GlobalRoute
    reason: str
    binding_resources: tuple[str, ...] = ()
    prefill_index: int | None = None
    decoder_index: int | None = None
    edge_id: str | None = None
    evaluated_score_ms: float | None = None
    score_delta_ms: float | None = None
    uncertainty_ms: float | None = None
    mesh_near_tie_eligible: bool | None = None
    mesh_source_virtual_service_before: float | None = None
    mesh_edge_virtual_service_before: float | None = None

    def __post_init__(self) -> None:
        _positive_int("rejected pair_index", self.pair_index, zero=True)
        if not isinstance(self.route, GlobalRoute):
            raise TypeError("rejected route must be GlobalRoute")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("rejected reason must be nonempty")
        if any(not isinstance(item, str) or not item for item in self.binding_resources):
            raise ValueError("rejected binding resources must be nonempty strings")
        prefill_index = (
            self.pair_index if self.prefill_index is None
            else self.prefill_index
        )
        decoder_index = (
            self.pair_index if self.decoder_index is None
            else self.decoder_index
        )
        _positive_int("rejected prefill_index", prefill_index, zero=True)
        _positive_int("rejected decoder_index", decoder_index, zero=True)
        if decoder_index != self.pair_index:
            raise ValueError("rejected pair_index must alias decoder_index")
        if self.route is GlobalRoute.LOCAL and prefill_index != decoder_index:
            raise ValueError("rejected local candidate cannot cross endpoints")
        canonical_edge_id = (
            f"local:d{decoder_index}"
            if self.route is GlobalRoute.LOCAL
            else f"remote:p{prefill_index}->d{decoder_index}"
        )
        if self.edge_id is not None and self.edge_id != canonical_edge_id:
            raise ValueError("rejected edge_id is not canonical")
        for name in (
            "evaluated_score_ms",
            "uncertainty_ms",
            "mesh_source_virtual_service_before",
            "mesh_edge_virtual_service_before",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite(name, value)
        if self.score_delta_ms is not None and (
            isinstance(self.score_delta_ms, bool)
            or not isinstance(self.score_delta_ms, (int, float))
            or not math.isfinite(float(self.score_delta_ms))
        ):
            raise ValueError("score_delta_ms must be finite")
        if (
            self.mesh_near_tie_eligible is not None
            and type(self.mesh_near_tie_eligible) is not bool
        ):
            raise TypeError("mesh_near_tie_eligible must be bool or None")
        object.__setattr__(self, "prefill_index", prefill_index)
        object.__setattr__(self, "decoder_index", decoder_index)
        object.__setattr__(self, "edge_id", canonical_edge_id)

    @classmethod
    def from_candidate(
        cls,
        candidate: RouteCandidate,
        reason: str,
        binding_resources: tuple[str, ...] = (),
    ) -> "RejectedCandidate":
        if not isinstance(candidate, RouteCandidate):
            raise TypeError("candidate must be RouteCandidate")
        return cls(
            pair_index=candidate.pair_index,
            route=candidate.route,
            reason=reason,
            binding_resources=binding_resources,
            prefill_index=candidate.prefill_index,
            decoder_index=candidate.decoder_index,
            edge_id=candidate.edge_id,
        )


@dataclass(frozen=True)
class GlobalDecision:
    request_id: str
    tenant_id: str
    kind: GlobalDecisionKind
    decided_ns: int
    reason: str
    pair_index: int | None
    route: GlobalRoute | None
    score_ms: float | None
    deadline_slack_ms: float | None
    selected_work: dict[str, int]
    predicted_e2e_ms: float | None
    predicted_ttft_ms: float | None
    uncertainty_ms: float | None
    cache_affinity: bool | None
    binding_resources: tuple[str, ...]
    rejected_candidates: tuple[RejectedCandidate, ...]
    resource_used_before: dict[str, int]
    active_pairs_before: tuple[int, ...]
    active_pairs_after: tuple[int, ...]
    pair_activated: bool
    tenant_virtual_service_before: float
    tenant_virtual_service_after: float
    telemetry_sequences: dict[int, int]
    telemetry_provenance: dict[int, dict[str, object]]
    fairness_basis: str = "weighted_dominant_resource_service"
    schema: str = SCHEMA
    joint_actuation: JointActuationPlan | None = None
    queue_lease: bool = False
    cache_group_key: str | None = None
    prefill_index: int | None = None
    decoder_index: int | None = None
    edge_id: str | None = None
    receiver_stagger_us: int = 0
    mesh_near_tie_source_balanced: bool = False
    mesh_near_tie_score_window_ms: float | None = None
    mesh_near_tie_score_delta_ms: float | None = None
    mesh_source_virtual_service_before: float | None = None
    mesh_edge_virtual_service_before: float | None = None
    service_queue_delay_ms: float | None = None
    service_forecast_ms: float | None = None
    protected_service_lane: bool = False
    protected_service_lane_key: str | None = None
    protected_service_lane_before: int | None = None
    protected_service_lane_after: int | None = None

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("global decision schema mismatch")
        if self.fairness_basis != "weighted_dominant_resource_service":
            raise ValueError("global decision fairness basis mismatch")
        if self.joint_actuation is not None:
            if not isinstance(self.joint_actuation, JointActuationPlan):
                raise TypeError("global decision joint actuation is invalid")
            if self.kind is not GlobalDecisionKind.ADMIT:
                raise ValueError("non-admission decision cannot carry actuation")
            if (
                self.pair_index != self.joint_actuation.pair_index
                or self.route is not self.joint_actuation.route
            ):
                raise ValueError("joint actuation route identity differs")
        if type(self.queue_lease) is not bool:
            raise TypeError("queue_lease must be bool")
        if self.queue_lease and self.kind is not GlobalDecisionKind.ADMIT:
            raise ValueError("queue lease requires an admission decision")
        if self.cache_group_key is not None:
            if (
                not isinstance(self.cache_group_key, str)
                or len(self.cache_group_key) != 64
                or any(
                    char not in "0123456789abcdef"
                    for char in self.cache_group_key
                )
            ):
                raise ValueError("decision cache_group_key is invalid")
        if self.kind is GlobalDecisionKind.ADMIT:
            if self.pair_index is None or self.route is None or self.score_ms is None:
                raise ValueError("admission decision lacks a committed route")
            prefill_index = (
                self.pair_index if self.prefill_index is None
                else self.prefill_index
            )
            decoder_index = (
                self.pair_index if self.decoder_index is None
                else self.decoder_index
            )
            _positive_int("decision prefill_index", prefill_index, zero=True)
            _positive_int("decision decoder_index", decoder_index, zero=True)
            if decoder_index != self.pair_index:
                raise ValueError("decision pair_index must alias decoder_index")
            if self.route is GlobalRoute.LOCAL and prefill_index != decoder_index:
                raise ValueError("local decision cannot cross P/D endpoints")
            canonical_edge_id = (
                f"local:d{decoder_index}"
                if self.route is GlobalRoute.LOCAL
                else f"remote:p{prefill_index}->d{decoder_index}"
            )
            if self.edge_id is not None and self.edge_id != canonical_edge_id:
                raise ValueError("decision edge_id is not canonical")
            object.__setattr__(self, "prefill_index", prefill_index)
            object.__setattr__(self, "decoder_index", decoder_index)
            object.__setattr__(self, "edge_id", canonical_edge_id)
            if set(self.selected_work) != set(ResourceVector.names()):
                raise ValueError("admission decision lacks selected work")
            if any(value is None for value in (
                self.predicted_e2e_ms,
                self.predicted_ttft_ms,
                self.uncertainty_ms,
                self.cache_affinity,
            )):
                raise ValueError("admission decision lacks selected prediction")
        elif (
            self.pair_index is not None
            or self.route is not None
            or self.prefill_index is not None
            or self.decoder_index is not None
            or self.edge_id is not None
        ):
            raise ValueError("queue decision cannot commit a route")
        elif self.selected_work or any(value is not None for value in (
            self.predicted_e2e_ms,
            self.predicted_ttft_ms,
            self.uncertainty_ms,
            self.cache_affinity,
        )):
            raise ValueError("queue decision cannot carry selected work")
        _positive_int(
            "decision receiver_stagger_us", self.receiver_stagger_us, zero=True)
        if self.kind is not GlobalDecisionKind.ADMIT and self.receiver_stagger_us:
            raise ValueError("non-admission decision cannot stagger a receiver")
        if type(self.protected_service_lane) is not bool:
            raise TypeError("protected_service_lane must be bool")
        protected_values = (
            self.protected_service_lane_key,
            self.protected_service_lane_before,
            self.protected_service_lane_after,
        )
        if self.protected_service_lane:
            if (
                self.kind is not GlobalDecisionKind.ADMIT
                or not isinstance(self.protected_service_lane_key, str)
                or not self.protected_service_lane_key
                or self.protected_service_lane_before is None
                or self.protected_service_lane_after is None
                or self.protected_service_lane_before < 0
                or self.protected_service_lane_after
                <= self.protected_service_lane_before
            ):
                raise ValueError(
                    "protected service lane lacks an admission receipt")
        elif any(value is not None for value in protected_values):
            raise ValueError(
                "unbound decision cannot carry protected service lane state")
        if type(self.mesh_near_tie_source_balanced) is not bool:
            raise TypeError("mesh_near_tie_source_balanced must be bool")
        for name in (
            "mesh_near_tie_score_window_ms",
            "mesh_near_tie_score_delta_ms",
            "mesh_source_virtual_service_before",
            "mesh_edge_virtual_service_before",
            "service_queue_delay_ms",
            "service_forecast_ms",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite(name, value)
        if self.mesh_near_tie_source_balanced and (
            self.kind is not GlobalDecisionKind.ADMIT
            or self.route is not GlobalRoute.REMOTE
            or self.mesh_near_tie_score_window_ms is None
            or self.mesh_near_tie_score_delta_ms is None
            or self.mesh_source_virtual_service_before is None
            or self.mesh_edge_virtual_service_before is None
        ):
            raise ValueError(
                "mesh near-tie source balance lacks a remote admission receipt"
            )


def global_decision_dict(decision: GlobalDecision) -> dict[str, object]:
    """Return the canonical provenance committed across the HTTP boundary."""

    if not isinstance(decision, GlobalDecision):
        raise TypeError("decision must be GlobalDecision")
    return {
        "schema": decision.schema,
        "request_id": decision.request_id,
        "tenant_id": decision.tenant_id,
        "kind": decision.kind.value,
        "decided_ns": decision.decided_ns,
        "reason": decision.reason,
        "pair_index": decision.pair_index,
        "prefill_index": decision.prefill_index,
        "decoder_index": decision.decoder_index,
        "edge_id": decision.edge_id,
        "route": decision.route.value if decision.route is not None else None,
        "score_ms": decision.score_ms,
        "deadline_slack_ms": decision.deadline_slack_ms,
        "selected_work": dict(decision.selected_work),
        "predicted_e2e_ms": decision.predicted_e2e_ms,
        "predicted_ttft_ms": decision.predicted_ttft_ms,
        "uncertainty_ms": decision.uncertainty_ms,
        "cache_affinity": decision.cache_affinity,
        "binding_resources": list(decision.binding_resources),
        "rejected_candidates": [
            {
                "pair_index": item.pair_index,
                "prefill_index": item.prefill_index,
                "decoder_index": item.decoder_index,
                "edge_id": item.edge_id,
                "route": item.route.value,
                "reason": item.reason,
                "binding_resources": list(item.binding_resources),
                "evaluated_score_ms": item.evaluated_score_ms,
                "score_delta_ms": item.score_delta_ms,
                "uncertainty_ms": item.uncertainty_ms,
                "mesh_near_tie_eligible": item.mesh_near_tie_eligible,
                "mesh_source_virtual_service_before": (
                    item.mesh_source_virtual_service_before),
                "mesh_edge_virtual_service_before": (
                    item.mesh_edge_virtual_service_before),
            }
            for item in decision.rejected_candidates
        ],
        "resource_used_before": dict(decision.resource_used_before),
        "active_pairs_before": list(decision.active_pairs_before),
        "active_pairs_after": list(decision.active_pairs_after),
        "pair_activated": decision.pair_activated,
        "tenant_virtual_service_before": (
            decision.tenant_virtual_service_before),
        "tenant_virtual_service_after": (
            decision.tenant_virtual_service_after),
        "fairness_basis": decision.fairness_basis,
        "queue_lease": decision.queue_lease,
        "receiver_stagger_us": decision.receiver_stagger_us,
        "mesh_near_tie_source_balanced": (
            decision.mesh_near_tie_source_balanced),
        "mesh_near_tie_score_window_ms": (
            decision.mesh_near_tie_score_window_ms),
        "mesh_near_tie_score_delta_ms": (
            decision.mesh_near_tie_score_delta_ms),
        "mesh_source_virtual_service_before": (
            decision.mesh_source_virtual_service_before),
        "mesh_edge_virtual_service_before": (
            decision.mesh_edge_virtual_service_before),
        "service_queue_delay_ms": decision.service_queue_delay_ms,
        "service_forecast_ms": decision.service_forecast_ms,
        "protected_service_lane": decision.protected_service_lane,
        "protected_service_lane_key": decision.protected_service_lane_key,
        "protected_service_lane_before": decision.protected_service_lane_before,
        "protected_service_lane_after": decision.protected_service_lane_after,
        "cache_group_key": decision.cache_group_key,
        "joint_actuation": (
            decision.joint_actuation.as_dict()
            if decision.joint_actuation is not None else None
        ),
        "telemetry_sequences": {
            str(index): value
            for index, value in sorted(decision.telemetry_sequences.items())
        },
        "telemetry_provenance": {
            str(index): dict(value)
            for index, value in sorted(decision.telemetry_provenance.items())
        },
    }


def global_decision_fingerprint(decision: GlobalDecision) -> str:
    payload = json.dumps(
        global_decision_dict(decision),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GlobalFailureReceipt:
    """Auditable terminal receipt for an endpoint route failure.

    The failed request is never migrated under the same request ID.  A client
    retry must use a new request ID; only already queued *other* requests may
    be dispatched after the failed route is quarantined.
    """

    request_id: str
    tenant_id: str
    decided_ns: int
    failure_kind: str
    reason: str
    quarantine_scope: str
    pair_index: int
    route: GlobalRoute
    phase_before: GlobalRequestPhase
    terminal_phase: GlobalRequestPhase
    released_work: dict[str, int]
    quarantined_routes: tuple[tuple[int, GlobalRoute], ...]
    telemetry_sequences: dict[int, int]
    reassignment_policy: str = "new_request_id_required"
    schema: str = FAILURE_SCHEMA
    prefill_index: int | None = None
    decoder_index: int | None = None
    edge_id: str | None = None
    quarantined_edges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.schema != FAILURE_SCHEMA:
            raise ValueError("global failure schema mismatch")
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("failure request_id must be nonempty")
        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("failure tenant_id must be nonempty")
        _positive_int("failure decided_ns", self.decided_ns, zero=True)
        for name in ("failure_kind", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        if self.quarantine_scope not in {
            "route", "pair", "edge", "prefill", "decoder",
        }:
            raise ValueError("failure quarantine scope is invalid")
        _positive_int("failure pair_index", self.pair_index, zero=True)
        if not isinstance(self.route, GlobalRoute):
            raise TypeError("failure route must be GlobalRoute")
        if not isinstance(self.phase_before, GlobalRequestPhase):
            raise TypeError("failure phase_before is invalid")
        if self.phase_before not in {
            GlobalRequestPhase.ROUTE_COMMITTED,
            GlobalRequestPhase.FIRST_RESPONSE,
        }:
            raise ValueError("route failure must start from an in-flight phase")
        if self.terminal_phase is not GlobalRequestPhase.FAILED:
            raise ValueError("route failure terminal phase must be failed")
        if set(self.released_work) != set(ResourceVector.names()):
            raise ValueError("failure receipt lacks released work")
        for name, value in self.released_work.items():
            _positive_int(f"released_work.{name}", value, zero=True)
        prefill_index = (
            self.pair_index if self.prefill_index is None
            else self.prefill_index
        )
        decoder_index = (
            self.pair_index if self.decoder_index is None
            else self.decoder_index
        )
        _positive_int("failure prefill_index", prefill_index, zero=True)
        _positive_int("failure decoder_index", decoder_index, zero=True)
        if decoder_index != self.pair_index:
            raise ValueError("failure pair_index must alias decoder_index")
        canonical_edge_id = (
            f"local:d{decoder_index}"
            if self.route is GlobalRoute.LOCAL
            else f"remote:p{prefill_index}->d{decoder_index}"
        )
        if self.edge_id is not None and self.edge_id != canonical_edge_id:
            raise ValueError("failure edge_id is not canonical")
        object.__setattr__(self, "prefill_index", prefill_index)
        object.__setattr__(self, "decoder_index", decoder_index)
        object.__setattr__(self, "edge_id", canonical_edge_id)
        if not self.quarantined_routes and not self.quarantined_edges:
            raise ValueError("failure receipt lacks a quarantine target")
        for pair_index, route in self.quarantined_routes:
            _positive_int("quarantined pair_index", pair_index, zero=True)
            if not isinstance(route, GlobalRoute):
                raise TypeError("quarantined route must be GlobalRoute")
        for edge_prefill, edge_decoder in self.quarantined_edges:
            _positive_int(
                "quarantined edge prefill_index", edge_prefill, zero=True)
            _positive_int(
                "quarantined edge decoder_index", edge_decoder, zero=True)
        for pair_index, sequence in self.telemetry_sequences.items():
            _positive_int("failure telemetry pair_index", pair_index, zero=True)
            _positive_int("failure telemetry sequence", sequence)
        if self.reassignment_policy != "new_request_id_required":
            raise ValueError("failure reassignment policy is invalid")


def global_failure_dict(receipt: GlobalFailureReceipt) -> dict[str, object]:
    """Serialize one terminal failure receipt for the native ledger."""

    if not isinstance(receipt, GlobalFailureReceipt):
        raise TypeError("receipt must be GlobalFailureReceipt")
    return {
        "schema": receipt.schema,
        "request_id": receipt.request_id,
        "tenant_id": receipt.tenant_id,
        "decided_ns": receipt.decided_ns,
        "failure_kind": receipt.failure_kind,
        "reason": receipt.reason,
        "quarantine_scope": receipt.quarantine_scope,
        "pair_index": receipt.pair_index,
        "prefill_index": receipt.prefill_index,
        "decoder_index": receipt.decoder_index,
        "edge_id": receipt.edge_id,
        "route": receipt.route.value,
        "phase_before": receipt.phase_before.value,
        "terminal_phase": receipt.terminal_phase.value,
        "released_work": dict(receipt.released_work),
        "quarantined_routes": [
            {"pair_index": pair_index, "route": route.value}
            for pair_index, route in receipt.quarantined_routes
        ],
        "quarantined_edges": [
            {
                "prefill_index": prefill_index,
                "decoder_index": decoder_index,
                "edge_id": f"remote:p{prefill_index}->d{decoder_index}",
            }
            for prefill_index, decoder_index in receipt.quarantined_edges
        ],
        "telemetry_sequences": {
            str(pair_index): sequence
            for pair_index, sequence in sorted(receipt.telemetry_sequences.items())
        },
        "reassignment_policy": receipt.reassignment_policy,
    }


def global_failure_fingerprint(receipt: GlobalFailureReceipt) -> str:
    payload = json.dumps(
        global_failure_dict(receipt),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GlobalFailureReport:
    receipt: GlobalFailureReceipt
    dispatched: tuple[GlobalDecision, ...]


@dataclass(frozen=True)
class GlobalServiceLaneReservationFailureReceipt:
    """Terminal receipt for a failed endpoint service-lane handshake.

    This is deliberately separate from route failure quarantine.  The
    endpoint did not start the request and therefore the route is not
    unhealthy; only the provisional global queue lease failed to acquire a
    physical endpoint service credit.  The global reservation is released
    exactly once and the request must not be forwarded under the same lease.
    """

    request_id: str
    tenant_id: str
    decided_ns: int
    failure_kind: str
    reason: str
    pair_index: int
    route: GlobalRoute
    phase_before: GlobalRequestPhase
    terminal_phase: GlobalRequestPhase
    released_work: dict[str, int]
    telemetry_sequences: dict[int, int]
    reassignment_policy: str = "new_request_id_required"
    schema: str = SERVICE_LANE_RESERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SERVICE_LANE_RESERVATION_SCHEMA:
            raise ValueError("service-lane reservation schema mismatch")
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("reservation failure request_id must be nonempty")
        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("reservation failure tenant_id must be nonempty")
        _positive_int("reservation failure decided_ns", self.decided_ns, zero=True)
        for name in ("failure_kind", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        _positive_int("reservation failure pair_index", self.pair_index, zero=True)
        if not isinstance(self.route, GlobalRoute):
            raise TypeError("reservation failure route is invalid")
        if self.phase_before is not GlobalRequestPhase.ROUTE_COMMITTED:
            raise ValueError("reservation failure must start after global commit")
        if self.terminal_phase is not GlobalRequestPhase.FAILED:
            raise ValueError("reservation failure terminal phase must be failed")
        if set(self.released_work) != set(ResourceVector.names()):
            raise ValueError("reservation failure lacks released work")
        for name, value in self.released_work.items():
            _positive_int(f"released_work.{name}", value, zero=True)
        for pair_index, sequence in self.telemetry_sequences.items():
            _positive_int("reservation telemetry pair_index", pair_index, zero=True)
            _positive_int("reservation telemetry sequence", sequence)
        if self.reassignment_policy != "new_request_id_required":
            raise ValueError("reservation reassignment policy is invalid")


def global_service_lane_reservation_failure_dict(
    receipt: GlobalServiceLaneReservationFailureReceipt,
) -> dict[str, object]:
    if not isinstance(receipt, GlobalServiceLaneReservationFailureReceipt):
        raise TypeError("receipt must be a service-lane reservation receipt")
    return {
        "schema": receipt.schema,
        "request_id": receipt.request_id,
        "tenant_id": receipt.tenant_id,
        "decided_ns": receipt.decided_ns,
        "failure_kind": receipt.failure_kind,
        "reason": receipt.reason,
        "pair_index": receipt.pair_index,
        "route": receipt.route.value,
        "phase_before": receipt.phase_before.value,
        "terminal_phase": receipt.terminal_phase.value,
        "released_work": dict(receipt.released_work),
        "telemetry_sequences": {
            str(pair_index): sequence
            for pair_index, sequence in sorted(receipt.telemetry_sequences.items())
        },
        "reassignment_policy": receipt.reassignment_policy,
    }


def global_service_lane_reservation_failure_fingerprint(
    receipt: GlobalServiceLaneReservationFailureReceipt,
) -> str:
    payload = json.dumps(
        global_service_lane_reservation_failure_dict(receipt),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GlobalServiceLaneReservationFailureReport:
    receipt: GlobalServiceLaneReservationFailureReceipt
    dispatched: tuple[GlobalDecision, ...]


@dataclass(frozen=True)
class GlobalServiceLaneQueuePromotionReceipt:
    """Auditable result of reconciling global and endpoint admission.

    The global controller has already committed and owns ``selected_work``
    when the endpoint reports that the same route needs its bounded native
    queue.  Promotion does not add work or change route/placement.  It only
    makes that downstream queue debt explicit after rechecking current
    scheduler, completion, deadline, route-health, mesh, and fabric guards.
    """

    request_id: str
    tenant_id: str
    decided_ns: int
    status: str
    reason: str
    pair_index: int
    route: GlobalRoute
    phase_before: GlobalRequestPhase
    queue_lease_before: bool
    queue_lease_after: bool
    completion_credit_consumed: bool
    completion_liveness_probe: bool
    endpoint_queue_debt_before: int
    endpoint_queue_debt_after: int
    scheduler_waiting_requests: int
    endpoint_residual_inflight: int
    endpoint_queue_capacity: int
    binding_resources: tuple[str, ...]
    telemetry_sequences: dict[int, int]
    schema: str = SERVICE_LANE_QUEUE_PROMOTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SERVICE_LANE_QUEUE_PROMOTION_SCHEMA:
            raise ValueError("service-lane queue promotion schema mismatch")
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("queue promotion request_id must be nonempty")
        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("queue promotion tenant_id must be nonempty")
        _positive_int("queue promotion decided_ns", self.decided_ns, zero=True)
        if self.status not in {"promoted", "rejected"}:
            raise ValueError("queue promotion status is invalid")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("queue promotion reason must be nonempty")
        _positive_int("queue promotion pair_index", self.pair_index, zero=True)
        if not isinstance(self.route, GlobalRoute):
            raise TypeError("queue promotion route is invalid")
        if self.phase_before is not GlobalRequestPhase.ROUTE_COMMITTED:
            raise ValueError("queue promotion must follow a route commit")
        for name in (
            "queue_lease_before", "queue_lease_after",
            "completion_credit_consumed", "completion_liveness_probe",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.queue_lease_before:
            raise ValueError("queue promotion cannot begin with a queue lease")
        if self.queue_lease_after != (self.status == "promoted"):
            raise ValueError("queue promotion lease state differs from status")
        if self.completion_credit_consumed and self.completion_liveness_probe:
            raise ValueError("queue promotion cannot consume two progress proofs")
        for name in (
            "endpoint_queue_debt_before", "endpoint_queue_debt_after",
            "scheduler_waiting_requests", "endpoint_residual_inflight",
            "endpoint_queue_capacity",
        ):
            _positive_int(name, getattr(self, name), zero=True)
        debt_transition_valid = (
            self.endpoint_queue_debt_after > self.endpoint_queue_debt_before
            if self.status == "promoted" else
            self.endpoint_queue_debt_after == self.endpoint_queue_debt_before
        )
        if not debt_transition_valid:
            raise ValueError("queue promotion debt transition is invalid")
        if not isinstance(self.binding_resources, tuple) or any(
            not isinstance(value, str) or not value
            for value in self.binding_resources
        ):
            raise ValueError("queue promotion bindings are invalid")
        for pair_index, sequence in self.telemetry_sequences.items():
            _positive_int("queue promotion telemetry pair", pair_index, zero=True)
            _positive_int("queue promotion telemetry sequence", sequence)


def global_service_lane_queue_promotion_dict(
    receipt: GlobalServiceLaneQueuePromotionReceipt,
) -> dict[str, object]:
    if not isinstance(receipt, GlobalServiceLaneQueuePromotionReceipt):
        raise TypeError("receipt must be a service-lane queue promotion receipt")
    return {
        "schema": receipt.schema,
        "request_id": receipt.request_id,
        "tenant_id": receipt.tenant_id,
        "decided_ns": receipt.decided_ns,
        "status": receipt.status,
        "reason": receipt.reason,
        "pair_index": receipt.pair_index,
        "route": receipt.route.value,
        "phase_before": receipt.phase_before.value,
        "queue_lease_before": receipt.queue_lease_before,
        "queue_lease_after": receipt.queue_lease_after,
        "completion_credit_consumed": receipt.completion_credit_consumed,
        "completion_liveness_probe": receipt.completion_liveness_probe,
        "endpoint_queue_debt_before": receipt.endpoint_queue_debt_before,
        "endpoint_queue_debt_after": receipt.endpoint_queue_debt_after,
        "scheduler_waiting_requests": receipt.scheduler_waiting_requests,
        "endpoint_residual_inflight": receipt.endpoint_residual_inflight,
        "endpoint_queue_capacity": receipt.endpoint_queue_capacity,
        "binding_resources": list(receipt.binding_resources),
        "telemetry_sequences": {
            str(pair_index): sequence
            for pair_index, sequence in sorted(receipt.telemetry_sequences.items())
        },
    }


def global_service_lane_queue_promotion_fingerprint(
    receipt: GlobalServiceLaneQueuePromotionReceipt,
) -> str:
    payload = json.dumps(
        global_service_lane_queue_promotion_dict(receipt),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GlobalServiceLaneQueuePromotionReport:
    receipt: GlobalServiceLaneQueuePromotionReceipt
    decision: GlobalDecision | None


@dataclass(frozen=True)
class _SharedRemoteBudget:
    group: str
    members: tuple[int, ...]
    requests_limit: int
    kv_bytes_limit: int
    semantic_ops_limit: int
    requests_used: int
    kv_bytes_used: int
    semantic_ops_used: int
    dispatch_stagger_us: int
    contributions: tuple[tuple[str, float], ...]
    limited: bool
    suppress_pair_activation: bool


@dataclass(frozen=True)
class _CandidateEvaluation:
    candidate: RouteCandidate
    score_ms: float
    slack_ms: float
    effective_used: ResourceVector
    utilization: float
    activate: bool
    activation_basis: str = "resource_pressure_or_no_active_fit"
    joint_actuation: JointActuationPlan | None = None
    # A soft action target can remain work-conserving for the request that
    # triggered it, but it still represents a resource-specific capacity
    # deficit.  Keep that fact explicit so the global selector can open a
    # prewarmed spare pair in the same atomic decision.
    cross_layer_scale_required: bool = False
    shared_scale_suppressed: bool = False
    endpoint_queue_debt_resources: tuple[str, ...] = ()
    completion_liveness_probe: bool = False
    completion_liveness_shared_probe: bool = False
    endpoint_queue_headroom_admission: bool = False
    endpoint_queue_deadline_grace: bool = False
    # Endpoint service feedback can age out while the request-triggered
    # scheduler/completion envelope remains fresh.  In the global P-by-D mesh
    # that is uncertainty, not evidence of a failed path.  Keep the fallback
    # explicit so its use is visible in the global decision receipt.
    stale_feedback_fallback: bool = False
    receiver_stagger_us: int = 0
    # True only when the frozen global profile has bound this remote,
    # cache-affine business request to vLLM's bounded priority scheduler lane.
    priority_service_lane: bool = False
    mesh_near_tie_source_balanced: bool = False
    mesh_near_tie_score_window_ms: float | None = None
    mesh_near_tie_score_delta_ms: float | None = None
    mesh_source_virtual_service_before: float | None = None
    mesh_edge_virtual_service_before: float | None = None
    service_queue_delay_ms: float | None = None
    service_forecast_ms: float | None = None
    protected_service_lane: bool = False
    protected_service_lane_key: str | None = None
    protected_service_lane_before: int | None = None
    protected_service_lane_after: int | None = None


@dataclass
class _Reservation:
    request: GlobalRequest
    candidate: RouteCandidate
    decision: GlobalDecision
    held: ResourceVector
    committed_ns: int
    mesh_stage_held: bool = False
    phase: GlobalRequestPhase = GlobalRequestPhase.ROUTE_COMMITTED


@dataclass
class _MeshEdgeState:
    """Controller-owned residual and measured completion state for P_i->D_j."""

    held_remote_prefill_token_ms: int = 0
    held_remote_kv_bytes: int = 0
    held_remote_semantic_ops: int = 0
    inflight_transfers: int = 0
    completed_first_responses: int = 0
    first_response_ewma_ms: float | None = None
    last_completion_ns: int | None = None


@dataclass(frozen=True)
class _RouteQuarantine:
    pair_index: int
    route: GlobalRoute
    failure_kind: str
    count: int
    first_failed_ns: int
    last_failed_ns: int
    telemetry_sequence: int
    scope: str
    trigger: str = "explicit_failure_receipt"


@dataclass(frozen=True)
class _MeshEdgeQuarantine:
    prefill_index: int
    decoder_index: int
    failure_kind: str
    count: int
    first_failed_ns: int
    last_failed_ns: int
    prefill_telemetry_sequence: int
    decoder_telemetry_sequence: int
    scope: str


class GlobalOrchestrator:
    """Thread-safe global admission, fairness, route, and pair controller."""

    def __init__(self, config: GlobalOrchestratorConfig) -> None:
        if not isinstance(config, GlobalOrchestratorConfig):
            raise TypeError("config must be GlobalOrchestratorConfig")
        self.config = config
        self._capacities = {
            item.pair_index: item.resources for item in config.capacities
        }
        self._tenants = {item.tenant_id: item for item in config.tenants}
        self._telemetry: dict[int, PairTelemetry] = {}
        # Shared-fabric limits depend only on the installed telemetry epoch;
        # current remote usage is overlaid at admission time.  This avoids
        # rescanning every compatible pair for every candidate while keeping
        # usage exact under the global lock.
        self._shared_budget_static: dict[str, _SharedRemoteBudget] = {}
        self._shared_group_members: dict[str, tuple[int, ...]] = {}
        self._owned = {
            item.pair_index: ResourceVector() for item in config.capacities
        }
        # C6 source-P credits and per-edge receiver residuals are independent
        # of the destination decoder ledger above.  They remain allocated for
        # route-commit -> first-response only.
        self._mesh_source_prefill_owned = {
            item.pair_index: 0 for item in config.capacities
        }
        # Long-lived, controller-owned virtual service prevents a healthy
        # source/edge from being selected forever solely because its integer
        # index wins an otherwise indistinguishable score tie.  Unlike held
        # credits, this ledger survives first response; unlike a quota, it is
        # consulted only within the predictor uncertainty envelope.
        self._mesh_source_virtual_service = {
            item.pair_index: 0.0 for item in config.capacities
        }
        self._mesh_edge_virtual_service = {
            (prefill, decoder): 0.0
            for prefill in self._capacities
            for decoder in self._capacities
        }
        self._mesh_edges = {
            (prefill, decoder): _MeshEdgeState()
            for prefill in self._capacities
            for decoder in self._capacities
        }
        self._mesh_edge_quarantines: dict[
            tuple[int, int], _MeshEdgeQuarantine
        ] = {}
        self._active_pairs = set(range(config.minimum_active_pairs))
        self._last_busy_ns = {
            item.pair_index: 0 for item in config.capacities
        }
        self._queued: dict[str, GlobalRequest] = {}
        # Preserve candidate-level causes when a queued request reaches the
        # endpoint-queue lease boundary but no safe route can be committed.
        # Without this handoff, the terminal timeout receipt hides the actual
        # global overload evidence from native Perlmutter runs.
        self._queue_lease_rejections: dict[
            str, tuple[RejectedCandidate, ...]
        ] = {}
        self._inflight: dict[str, _Reservation] = {}
        self._terminal: dict[str, GlobalRequestPhase] = {}
        self._decision_history: dict[str, list[GlobalDecision]] = {}
        self._route_quarantines: dict[tuple[int, GlobalRoute], _RouteQuarantine] = {}
        # A bounded endpoint queue timeout is feedback for future queue
        # leases on that pair, not a route-health failure.  The lease circuit
        # remains closed until a newer scheduler sample proves the queue has
        # drained.
        self._endpoint_queue_lease_cooldowns: dict[int, int] = {}
        # v3 turns observed first-response completions into one-shot queue
        # lease credits.  Credits are allocation/pair local, capped by the
        # physical endpoint window, and consumed at global route commit.
        self._completion_credit_balance = {
            item.pair_index: 0 for item in config.capacities
        }
        # One failure-free bootstrap promotion is allowed per pair/route and
        # telemetry sequence when the endpoint itself has returned a bounded
        # queue offer but no completion delta has yet produced a consumable
        # credit.  This closes the startup race without turning a stale
        # completion counter into an unlimited queue bypass.
        self._completion_liveness_bootstrap_sequences: dict[
            tuple[int, GlobalRoute], int
        ] = {}
        self._service_lane_queue_promotions: dict[
            str, GlobalServiceLaneQueuePromotionReceipt
        ] = {}
        # Push-based LMCache receivers have pair-local ownership for a cache
        # chunk.  A repeated shared-prefix request must not start a second
        # transfer for the same pair/chunk until the first transfer reaches
        # first response.  This is global admission state, not an endpoint
        # retry: it lets another pair or the local path remain eligible.
        self._cache_group_holds: dict[tuple[int, str], str] = {}
        self._failure_history: dict[str, list[GlobalFailureReceipt]] = {}
        self._tenant_virtual_service = {
            tenant_id: 0.0 for tenant_id in self._tenants
        }
        # ``_tenant_virtual_service`` is weighted service debt (dominant
        # resource service divided by the tenant weight).  Keep raw service
        # units separately: minimum-service guarantees are fractions of work
        # actually admitted, not fractions of weighted debt.
        self._tenant_service_units = {
            tenant_id: 0.0 for tenant_id in self._tenants
        }
        self._tenant_admitted_decode_tokens = {
            tenant_id: 0 for tenant_id in self._tenants
        }
        self._tenant_completed_decode_tokens = {
            tenant_id: 0 for tenant_id in self._tenants
        }
        # A low-priority tenant may deliberately consolidate work onto a
        # bounded decoder set.  The assignment is global-authority state: it
        # is updated only at route commit, retained across first response/EOF,
        # and expires only after a pair-scale idle epoch.  This lets an urgent
        # tenant use a clean prewarmed pair without relying on a workload phase
        # label or a future-arrival oracle.
        self._tenant_pair_assignments: dict[str, set[int]] = {
            tenant_id: set() for tenant_id in self._tenants
        }
        self._tenant_pair_last_busy_ns: dict[str, dict[int, int]] = {
            tenant_id: {} for tenant_id in self._tenants
        }
        self._lock = threading.Lock()

    def _observe_completion_credit_locked(
        self,
        prior: PairTelemetry | None,
        current: PairTelemetry,
    ) -> None:
        """Accrue one-shot v3 lease credit from causal completion deltas."""

        pair = current.pair_index
        if self.config.endpoint_queue_debt_mode not in {
            "completion_credit_endpoint_queue_v3",
            "completion_credit_mesh_endpoint_queue_v1",
        }:
            return
        if (
            prior is None
            or prior.agent_epoch != current.agent_epoch
            or prior.controller_generation != current.controller_generation
            or prior.endpoint_completed_first_responses is None
            or current.endpoint_completed_first_responses is None
            or current.endpoint_completed_first_responses
            < prior.endpoint_completed_first_responses
        ):
            self._completion_credit_balance[pair] = 0
            return
        completed_delta = (
            current.endpoint_completed_first_responses
            - prior.endpoint_completed_first_responses
        )
        self._completion_credit_balance[pair] = min(
            self._capacities[pair].endpoint_requests,
            self._completion_credit_balance[pair] + completed_delta,
        )

    def update_telemetry(self, telemetry: PairTelemetry) -> None:
        if not isinstance(telemetry, PairTelemetry):
            raise TypeError("telemetry must be PairTelemetry")
        if telemetry.pair_index not in self._capacities:
            raise ValueError("telemetry pair is not configured")
        with self._lock:
            prior = self._validate_telemetry_update_locked(telemetry)
            self._observe_completion_credit_locked(prior, telemetry)
            self._telemetry[telemetry.pair_index] = telemetry
            self._shared_budget_static.clear()
            self._shared_group_members.clear()
            self._observe_telemetry_failure_delta_locked(prior, telemetry)
            self._recover_route_quarantines_locked((telemetry,))

    def update_telemetry_batch(
        self, telemetry: Iterable[PairTelemetry]
    ) -> None:
        """Install one complete all-pair sample atomically.

        The frontend agent assigns one sequence and conservative collection
        interval to every pair.  Requiring a complete batch prevents a route
        decision from comparing one pair's new state with another pair's old
        state while an asynchronous poll is being installed.
        """

        values = tuple(telemetry)
        if not values or any(not isinstance(item, PairTelemetry) for item in values):
            raise TypeError("telemetry batch must contain PairTelemetry values")
        indices = [item.pair_index for item in values]
        if len(indices) != len(set(indices)):
            raise ValueError("telemetry batch contains a duplicate pair")
        if set(indices) != set(self._capacities):
            raise ValueError("telemetry batch must contain every configured pair")
        for name, projected in (
            ("sequence", {item.sequence for item in values}),
            ("sampled_ns", {item.sampled_ns for item in values}),
            ("collected_ns", {item.collected_ns for item in values}),
            ("agent_epoch", {item.agent_epoch for item in values}),
        ):
            if len(projected) != 1:
                raise ValueError(f"telemetry batch has mixed {name}")
        with self._lock:
            prior = {
                item.pair_index: self._telemetry.get(item.pair_index)
                for item in values
            }
            for item in values:
                self._validate_telemetry_update_locked(item)
            for item in values:
                self._observe_completion_credit_locked(
                    prior[item.pair_index], item)
                self._telemetry[item.pair_index] = item
            self._shared_budget_static.clear()
            self._shared_group_members.clear()
            for item in values:
                self._observe_telemetry_failure_delta_locked(
                    prior[item.pair_index], item)
            self._recover_route_quarantines_locked(values)

    def admission_wait_budget_ns(self, tenant_id: str) -> int:
        """Return the effective queue wait contract for one tenant.

        The global limit bounds every tenant, while a tenant policy may impose
        a stricter SLO.  Keeping this lookup in the orchestrator prevents an
        async frontend from accidentally replacing tenant business policy
        with one process-wide timeout.
        """

        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be nonempty")
        try:
            policy = self._tenants[tenant_id]
        except KeyError as exc:
            raise ValueError("request tenant is not configured") from exc
        return min(self.config.maximum_queue_wait_ns,
                   policy.maximum_queue_wait_ns)

    def _effective_deadline_ns(self, request: GlobalRequest) -> int:
        """Cap an external deadline by the tenant's frozen E2E SLO.

        The client may provide a stricter remaining deadline, but a generic
        endpoint deadline must not let a low-priority request outrank a
        tenant whose business contract expires earlier.  This is a global
        queue-ordering rule; it is independent of route, phase, and future
        arrivals.
        """

        policy = self._tenants[request.tenant_id]
        tenant_deadline = request.arrival_ns + int(
            policy.e2e_slo_ms * 1_000_000)
        return min(request.deadline_ns, tenant_deadline)

    def _queue_reservation_rejection_reason_locked(
        self, request: GlobalRequest
    ) -> str | None:
        """Protect tenant queue reservations during ingress overload.

        Reservations apply only to requests that remain queued.  The projected
        queue is allowed to consume all non-reserved slots, while a request
        that fills its tenant's missing reservation may consume that reserved
        slot.  This is a business/fairness admission guard, not GPU capacity
        and not a route selector.
        """

        queued_by_tenant = Counter(
            item.tenant_id for item in self._queued.values())
        queued_by_tenant[request.tenant_id] += 1
        missing_reserved = sum(
            max(
                0,
                policy.queue_reservation_slots
                - queued_by_tenant.get(policy.tenant_id, 0),
            )
            for policy in self._tenants.values()
        )
        projected_occupancy = len(self._queued) + 1
        available_capacity = self.config.queue_capacity - missing_reserved
        if projected_occupancy > available_capacity:
            return "global_tenant_queue_reservation"
        return None

    def submit(self, request: GlobalRequest, *, now_ns: int) -> GlobalDecision:
        if not isinstance(request, GlobalRequest):
            raise TypeError("request must be GlobalRequest")
        _positive_int("now_ns", now_ns, zero=True)
        if now_ns < request.arrival_ns:
            raise ValueError("request arrives in the future")
        if request.tenant_id not in self._tenants:
            raise ValueError("request tenant is not configured")
        for candidate in request.candidates:
            self._validate_candidate_topology(candidate)
        with self._lock:
            self._expire_tenant_pair_assignments_locked(now_ns)
            if (
                request.request_id in self._queued
                or request.request_id in self._inflight
                or request.request_id in self._terminal
            ):
                raise ValueError("duplicate request_id")
            if len(self._queued) >= self.config.queue_capacity:
                return self._reject_locked(
                    request,
                    now_ns,
                    reason="global_ingress_overload_reject",
                )
            reservation_reason = (
                self._queue_reservation_rejection_reason_locked(request))
            if reservation_reason is not None:
                return self._reject_locked(
                    request,
                    now_ns,
                    reason=reservation_reason,
                )
            self._queued[request.request_id] = request
            dispatched = self._dispatch_locked(now_ns)
            for decision in dispatched:
                if decision.request_id == request.request_id:
                    return decision
            decision = self._queue_decision_locked(request, now_ns)
            self._decision_history.setdefault(request.request_id, []).append(decision)
            return decision

    def mark_first_response(
        self, request_id: str, *, now_ns: int
    ) -> tuple[GlobalDecision, ...]:
        _positive_int("now_ns", now_ns, zero=True)
        with self._lock:
            reservation = self._reservation(request_id)
            if reservation.phase is not GlobalRequestPhase.ROUTE_COMMITTED:
                raise ValueError("first response is not the next request phase")
            released = reservation.held.stage_release(
                reservation.candidate.route)
            pair = reservation.candidate.pair_index
            self._owned[pair] = self._owned[pair].subtract(released)
            reservation.held = reservation.held.subtract(released)
            self._release_mesh_stage_locked(
                reservation,
                now_ns=now_ns,
                completed_first_response=True,
            )
            reservation.phase = GlobalRequestPhase.FIRST_RESPONSE
            self._release_cache_group_locked(
                reservation.request, reservation.candidate)
            self._touch_tenant_pair_locked(
                reservation.request.tenant_id, pair, now_ns)
            self._last_busy_ns[pair] = now_ns
            return tuple(self._dispatch_locked(now_ns))

    def complete(
        self, request_id: str, *, now_ns: int
    ) -> tuple[GlobalDecision, ...]:
        _positive_int("now_ns", now_ns, zero=True)
        with self._lock:
            reservation = self._reservation(request_id)
            if reservation.phase is not GlobalRequestPhase.FIRST_RESPONSE:
                raise ValueError("EOF requires an observed first response")
            pair = reservation.candidate.pair_index
            self._owned[pair] = self._owned[pair].subtract(reservation.held)
            self._release_mesh_stage_locked(
                reservation,
                now_ns=now_ns,
                completed_first_response=False,
            )
            self._release_cache_group_locked(
                reservation.request, reservation.candidate)
            reservation.phase = GlobalRequestPhase.COMPLETE
            tenant = reservation.request.tenant_id
            self._tenant_completed_decode_tokens[tenant] += (
                reservation.candidate.work.decode_tokens)
            del self._inflight[request_id]
            self._terminal[request_id] = GlobalRequestPhase.COMPLETE
            self._touch_tenant_pair_locked(tenant, pair, now_ns)
            self._last_busy_ns[pair] = now_ns
            self._reconcile_pairs_locked(now_ns)
            return tuple(self._dispatch_locked(now_ns))

    def fail(
        self, request_id: str, *, now_ns: int
    ) -> tuple[GlobalDecision, ...]:
        _positive_int("now_ns", now_ns, zero=True)
        with self._lock:
            reservation = self._reservation(request_id)
            pair = reservation.candidate.pair_index
            self._owned[pair] = self._owned[pair].subtract(reservation.held)
            self._release_mesh_stage_locked(
                reservation,
                now_ns=now_ns,
                completed_first_response=False,
            )
            self._release_cache_group_locked(
                reservation.request, reservation.candidate)
            reservation.phase = GlobalRequestPhase.FAILED
            del self._inflight[request_id]
            self._terminal[request_id] = GlobalRequestPhase.FAILED
            self._touch_tenant_pair_locked(
                reservation.request.tenant_id, pair, now_ns)
            self._last_busy_ns[pair] = now_ns
            self._reconcile_pairs_locked(now_ns)
            return tuple(self._dispatch_locked(now_ns))

    def promote_service_lane_queue_lease(
        self, request_id: str, *, now_ns: int
    ) -> GlobalServiceLaneQueuePromotionReport:
        """Atomically convert an owned route into bounded endpoint queue debt.

        This is the reconciliation half of two-level admission.  It is valid
        only after the endpoint has returned ``queue_required`` for the exact
        globally committed route.  No work is added, no route is changed, and
        no fairness service is charged twice.  A rejected promotion leaves
        the original reservation intact so the caller can close it through
        ``fail_service_lane_reservation`` exactly once.
        """

        _positive_int("now_ns", now_ns, zero=True)
        with self._lock:
            reservation = self._reservation(request_id)
            if reservation.phase is not GlobalRequestPhase.ROUTE_COMMITTED:
                raise ValueError(
                    "service-lane queue promotion is not the next phase")
            initial = reservation.decision
            if initial.queue_lease:
                raise ValueError("service-lane queue lease was already promoted")
            request = reservation.request
            candidate = reservation.candidate
            pair = candidate.pair_index
            telemetry = self._telemetry.get(pair)
            if telemetry is None:
                raise RuntimeError("queue promotion lacks endpoint telemetry")
            debt_before = self._endpoint_queue_lease_debt(pair)
            scheduler_waiting = telemetry.scheduler_waiting_requests
            endpoint_residual = telemetry.endpoint_residual_inflight
            scheduler_waiting_value = (
                scheduler_waiting if scheduler_waiting is not None else 0)
            endpoint_residual_value = (
                endpoint_residual if endpoint_residual is not None else 0)

            def finish_rejected(
                reason: str,
                bindings: tuple[str, ...] = (),
            ) -> GlobalServiceLaneQueuePromotionReport:
                receipt = GlobalServiceLaneQueuePromotionReceipt(
                    request_id=request_id,
                    tenant_id=request.tenant_id,
                    decided_ns=now_ns,
                    status="rejected",
                    reason=reason,
                    pair_index=pair,
                    route=candidate.route,
                    phase_before=reservation.phase,
                    queue_lease_before=False,
                    queue_lease_after=False,
                    completion_credit_consumed=False,
                    completion_liveness_probe=False,
                    endpoint_queue_debt_before=debt_before,
                    endpoint_queue_debt_after=debt_before,
                    scheduler_waiting_requests=scheduler_waiting_value,
                    endpoint_residual_inflight=endpoint_residual_value,
                    endpoint_queue_capacity=self.config.endpoint_queue_capacity,
                    binding_resources=bindings,
                    telemetry_sequences={
                        index: value.sequence
                        for index, value in self._telemetry.items()
                    },
                )
                self._service_lane_queue_promotions[request_id] = receipt
                return GlobalServiceLaneQueuePromotionReport(
                    receipt=receipt, decision=None)

            policy = self._tenants[request.tenant_id]
            if (
                self.config.overload_action != "endpoint_queue_lease"
                or not policy.queue_lease_on_timeout
            ):
                return finish_rejected("queue_lease_policy_disabled")
            priority_service_lane = self._priority_service_lane_headroom(
                request, candidate)
            if not (
                self._endpoint_scheduler_queue_headroom(pair)
                or priority_service_lane
            ):
                return finish_rejected(
                    "endpoint_queue_capacity_full",
                    ("endpoint_queue_capacity",),
                )

            evaluated = self._evaluate_queue_lease_candidate(
                candidate,
                request=request,
                now_ns=now_ns,
                already_owned=True,
                allow_completion_liveness_bootstrap=True,
            )
            if isinstance(evaluated, RejectedCandidate):
                return finish_rejected(
                    evaluated.reason, evaluated.binding_resources)

            completion_credit_mode = self.config.endpoint_queue_debt_mode in {
                "completion_credit_endpoint_queue_v3",
                "completion_credit_mesh_endpoint_queue_v1",
            }
            credit_consumed = bool(
                completion_credit_mode
                and not evaluated.priority_service_lane
                and not evaluated.endpoint_queue_headroom_admission
                and self._completion_credit_balance[pair] > 0
            )
            liveness_probe = bool(
                evaluated.completion_liveness_probe and not credit_consumed)
            shared_probe = bool(evaluated.completion_liveness_shared_probe)
            headroom_admission = bool(
                evaluated.endpoint_queue_headroom_admission)
            if (
                completion_credit_mode
                and not evaluated.priority_service_lane
                and not (
                    credit_consumed
                    or liveness_probe
                    or shared_probe
                    or headroom_admission
                )
            ):
                raise RuntimeError(
                    "queue promotion lost its completion progress proof")

            binding = tuple(dict.fromkeys(
                initial.binding_resources
                + evaluated.endpoint_queue_debt_resources
                + ("endpoint_service_lane_queue",)
                + (
                    (self._priority_service_lane_binding(),)
                    if evaluated.priority_service_lane else ()
                )
                + (
                    ("completion_first_response_credit",)
                    if credit_consumed else
                    ("completion_liveness_bootstrap",)
                    if liveness_probe else
                    ("completion_liveness_shared_probe",)
                    if shared_probe else
                    ("completion_progress_headroom",)
                    if headroom_admission else
                    ()
                )
            ))
            reason = (
                self._priority_service_lane_reason(promoted=True)
                if evaluated.priority_service_lane else
                "global_endpoint_service_lane_completion_credit_promoted"
                if credit_consumed else
                "global_endpoint_service_lane_completion_liveness_promoted"
                if liveness_probe else
                "global_endpoint_service_lane_shared_liveness_route_committed"
                if shared_probe else
                "global_endpoint_service_lane_completion_progress_headroom_route_committed"
                if headroom_admission else
                "global_endpoint_service_lane_queue_promoted"
            )
            promoted = replace(
                initial,
                decided_ns=now_ns,
                reason=reason,
                score_ms=evaluated.score_ms,
                deadline_slack_ms=evaluated.slack_ms,
                binding_resources=binding,
                queue_lease=True,
            )
            if credit_consumed:
                self._completion_credit_balance[pair] -= 1
            if liveness_probe:
                self._completion_liveness_bootstrap_sequences[
                    (pair, candidate.route)
                ] = telemetry.sequence
            reservation.decision = promoted
            self._decision_history.setdefault(request_id, []).append(promoted)
            debt_after = self._endpoint_queue_lease_debt(pair)
            receipt = GlobalServiceLaneQueuePromotionReceipt(
                request_id=request_id,
                tenant_id=request.tenant_id,
                decided_ns=now_ns,
                status="promoted",
                reason=reason,
                pair_index=pair,
                route=candidate.route,
                phase_before=reservation.phase,
                queue_lease_before=False,
                queue_lease_after=True,
                completion_credit_consumed=credit_consumed,
                completion_liveness_probe=liveness_probe,
                endpoint_queue_debt_before=debt_before,
                endpoint_queue_debt_after=debt_after,
                scheduler_waiting_requests=scheduler_waiting_value,
                endpoint_residual_inflight=endpoint_residual_value,
                endpoint_queue_capacity=self.config.endpoint_queue_capacity,
                binding_resources=binding,
                telemetry_sequences={
                    index: value.sequence
                    for index, value in self._telemetry.items()
                },
            )
            self._service_lane_queue_promotions[request_id] = receipt
            return GlobalServiceLaneQueuePromotionReport(
                receipt=receipt, decision=promoted)

    def fail_service_lane_reservation(
        self,
        request_id: str,
        *,
        failure_kind: str,
        reason: str,
        now_ns: int,
    ) -> GlobalServiceLaneReservationFailureReport:
        """Release a global lease that failed endpoint admission.

        The endpoint service lane is the second phase of a queue lease.  A
        failed handshake is not a route failure: no upstream request started,
        so quarantining the route would turn transient capacity pressure into
        a false health signal.  The global held vector is nevertheless
        released with the same exactly-once ownership rules as ``fail``.
        """

        _positive_int("now_ns", now_ns, zero=True)
        if not isinstance(failure_kind, str) or not failure_kind.strip():
            raise ValueError("reservation failure_kind must be nonempty")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reservation failure reason must be nonempty")
        with self._lock:
            reservation = self._reservation(request_id)
            if reservation.phase is not GlobalRequestPhase.ROUTE_COMMITTED:
                raise ValueError(
                    "service-lane reservation failure is not the next phase")
            phase_before = reservation.phase
            pair = reservation.candidate.pair_index
            route = reservation.candidate.route
            released = reservation.held
            self._owned[pair] = self._owned[pair].subtract(released)
            self._release_mesh_stage_locked(
                reservation,
                now_ns=now_ns,
                completed_first_response=False,
            )
            self._release_cache_group_locked(
                reservation.request, reservation.candidate)
            if (
                "queue_lease" in failure_kind
                or "queue_lease" in reason
                or "bounded_queue" in failure_kind
                or "bounded_queue" in reason
            ):
                self._endpoint_queue_lease_cooldowns[pair] = (
                    self._telemetry[pair].sequence
                    if pair in self._telemetry else 0
                )
            reservation.phase = GlobalRequestPhase.FAILED
            del self._inflight[request_id]
            self._terminal[request_id] = GlobalRequestPhase.FAILED
            self._touch_tenant_pair_locked(
                reservation.request.tenant_id, pair, now_ns)
            self._last_busy_ns[pair] = now_ns
            self._reconcile_pairs_locked(now_ns)
            dispatched = tuple(self._dispatch_locked(now_ns))
            receipt = GlobalServiceLaneReservationFailureReceipt(
                request_id=request_id,
                tenant_id=reservation.request.tenant_id,
                decided_ns=now_ns,
                failure_kind=failure_kind,
                reason=reason,
                pair_index=pair,
                route=route,
                phase_before=phase_before,
                terminal_phase=GlobalRequestPhase.FAILED,
                released_work=released.as_dict(),
                telemetry_sequences={
                    index: value.sequence
                    for index, value in self._telemetry.items()
                },
            )
            return GlobalServiceLaneReservationFailureReport(
                receipt=receipt,
                dispatched=dispatched,
            )

    def report_route_failure(
        self,
        request_id: str,
        *,
        failure_kind: str,
        now_ns: int,
        scope: str = "route",
        route: GlobalRoute | None = None,
    ) -> "GlobalFailureReport":
        """Fail one committed request and quarantine its failed path.

        This is intentionally separate from generic lifecycle failure.  It is
        called only when the endpoint/router supplies an explicit failure
        receipt.  The current request is terminally failed and is never
        migrated; queued requests are reconsidered against surviving paths.
        """

        _positive_int("now_ns", now_ns, zero=True)
        if self.config.route_failure_quarantine_mode != "deny_until_probe":
            raise ValueError(
                "route failure quarantine is disabled by the frozen profile")
        if not isinstance(failure_kind, str) or not failure_kind.strip():
            raise ValueError("failure_kind must be nonempty")
        if scope not in {"route", "pair", "edge", "prefill", "decoder"}:
            raise ValueError(
                "failure quarantine scope must be route, pair, edge, "
                "prefill, or decoder")
        if route is not None and not isinstance(route, GlobalRoute):
            raise TypeError("failure route must be GlobalRoute")
        with self._lock:
            reservation = self._reservation(request_id)
            committed_route = reservation.candidate.route
            if route is not None and route is not committed_route:
                raise ValueError("failure route differs from committed route")
            if scope in {"edge", "prefill"} and (
                not self._mesh_enabled()
                or committed_route is not GlobalRoute.REMOTE
            ):
                raise ValueError(
                    "edge/prefill quarantine requires a remote mesh commitment")
            pair = reservation.candidate.pair_index
            prefill = int(reservation.candidate.prefill_index)
            decoder = int(reservation.candidate.decoder_index)
            phase_before = reservation.phase
            if scope in {"pair", "decoder"}:
                quarantined_routes = (
                    (decoder, GlobalRoute.LOCAL),
                    (decoder, GlobalRoute.REMOTE),
                )
            elif scope == "route":
                quarantined_routes = ((decoder, committed_route),)
            else:
                quarantined_routes = ()
            if scope == "edge":
                quarantined_edges = ((prefill, decoder),)
            elif scope == "prefill":
                quarantined_edges = tuple(
                    (prefill, destination)
                    for destination in sorted(self._capacities)
                )
            elif scope == "decoder":
                quarantined_edges = tuple(
                    (source, decoder)
                    for source in sorted(self._capacities)
                )
            else:
                quarantined_edges = ()
            telemetry_sequence = {
                index: item.sequence
                for index, item in self._telemetry.items()
            }
            sequence_for_recovery = telemetry_sequence.get(pair, 0)
            for quarantine_pair, quarantine_route in quarantined_routes:
                self._quarantine_route_locked(
                    quarantine_pair,
                    quarantine_route,
                    failure_kind=failure_kind.strip(),
                    now_ns=now_ns,
                    telemetry_sequence=sequence_for_recovery,
                    scope=scope,
                )
            for edge_prefill, edge_decoder in quarantined_edges:
                self._quarantine_mesh_edge_locked(
                    edge_prefill,
                    edge_decoder,
                    failure_kind=failure_kind.strip(),
                    now_ns=now_ns,
                    scope=scope,
                )
            released = reservation.held
            self._owned[pair] = self._owned[pair].subtract(released)
            self._release_mesh_stage_locked(
                reservation,
                now_ns=now_ns,
                completed_first_response=False,
            )
            self._release_cache_group_locked(
                reservation.request, reservation.candidate)
            reservation.phase = GlobalRequestPhase.FAILED
            del self._inflight[request_id]
            self._terminal[request_id] = GlobalRequestPhase.FAILED
            self._touch_tenant_pair_locked(
                reservation.request.tenant_id, pair, now_ns)
            self._last_busy_ns[pair] = now_ns
            self._reconcile_pairs_locked(now_ns)
            receipt = GlobalFailureReceipt(
                request_id=request_id,
                tenant_id=reservation.request.tenant_id,
                decided_ns=now_ns,
                failure_kind=failure_kind.strip(),
                reason=(
                    "global_mesh_failure_quarantine"
                    if quarantined_edges
                    else "global_route_failure_quarantine"
                ),
                quarantine_scope=scope,
                pair_index=pair,
                route=committed_route,
                phase_before=phase_before,
                terminal_phase=GlobalRequestPhase.FAILED,
                released_work=released.as_dict(),
                quarantined_routes=quarantined_routes,
                telemetry_sequences=telemetry_sequence,
                prefill_index=prefill,
                decoder_index=decoder,
                edge_id=reservation.candidate.edge_id,
                quarantined_edges=quarantined_edges,
            )
            self._failure_history.setdefault(request_id, []).append(receipt)
            dispatched = tuple(self._dispatch_locked(now_ns))
            return GlobalFailureReport(receipt=receipt, dispatched=dispatched)

    def reconcile_pairs(self, *, now_ns: int) -> tuple[int, ...]:
        _positive_int("now_ns", now_ns, zero=True)
        with self._lock:
            self._reconcile_pairs_locked(now_ns)
            return tuple(sorted(self._active_pairs))

    def dispatch(self, *, now_ns: int) -> tuple[GlobalDecision, ...]:
        """Re-evaluate the bounded queue after an atomic telemetry refresh."""

        _positive_int("now_ns", now_ns, zero=True)
        with self._lock:
            self._expire_tenant_pair_assignments_locked(now_ns)
            return tuple(self._dispatch_locked(now_ns))

    def cancel_queued(
        self, request_id: str, *, now_ns: int, reason: str
    ) -> GlobalDecision:
        """Remove one request that never received a route commitment."""

        _positive_int("now_ns", now_ns, zero=True)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("queue cancellation reason must be nonempty")
        with self._lock:
            request = self._queued.pop(request_id, None)
            if request is None:
                raise ValueError("request is not globally queued")
            self._queue_lease_rejections.pop(request_id, None)
            self._terminal[request_id] = GlobalRequestPhase.FAILED
            virtual = self._tenant_virtual_service[request.tenant_id]
            decision = GlobalDecision(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                kind=GlobalDecisionKind.QUEUE,
                decided_ns=now_ns,
                reason=reason,
                pair_index=None,
                route=None,
                score_ms=None,
                deadline_slack_ms=None,
                selected_work={},
                predicted_e2e_ms=None,
                predicted_ttft_ms=None,
                uncertainty_ms=None,
                cache_affinity=None,
                binding_resources=(),
                rejected_candidates=(),
                resource_used_before={},
                active_pairs_before=tuple(sorted(self._active_pairs)),
                active_pairs_after=tuple(sorted(self._active_pairs)),
                pair_activated=False,
                tenant_virtual_service_before=virtual,
                tenant_virtual_service_after=virtual,
                telemetry_sequences={
                    index: value.sequence
                    for index, value in self._telemetry.items()
                },
                telemetry_provenance=self._decision_telemetry_provenance(),
                cache_group_key=request.cache_group_key,
            )
            self._decision_history.setdefault(request_id, []).append(decision)
            return decision

    def reject_queued(
        self, request_id: str, *, now_ns: int, reason: str
    ) -> GlobalDecision:
        """Terminally reject a request that exceeded its queue budget."""

        _positive_int("now_ns", now_ns, zero=True)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("queue rejection reason must be nonempty")
        with self._lock:
            request = self._queued.pop(request_id, None)
            if request is None:
                raise ValueError("request is not globally queued")
            rejected = self._queue_lease_rejections.pop(request_id, ())
            return self._reject_locked(
                request,
                now_ns,
                reason=reason,
                rejected_candidates=rejected,
            )

    def lease_queued_to_endpoint(
        self, request_id: str, *, now_ns: int
    ) -> GlobalDecision | None:
        """Commit one expired queue waiter to a native endpoint queue.

        This is an explicit business-policy action, not an implicit timeout
        bypass.  The tenant must opt in and the profile must select
        ``endpoint_queue_lease``.  The selected work remains owned by TEMPO,
        so the over-capacity debt is visible to later global decisions and is
        released through the normal first-response/EOF lifecycle.
        """

        _positive_int("now_ns", now_ns, zero=True)
        with self._lock:
            self._expire_tenant_pair_assignments_locked(now_ns)
            request = self._queued.get(request_id)
            if request is None:
                raise ValueError("request is not globally queued")
            policy = self._tenants[request.tenant_id]
            if (
                self.config.overload_action != "endpoint_queue_lease"
                or not policy.queue_lease_on_timeout
            ):
                self._queue_lease_rejections[request_id] = tuple(
                    RejectedCandidate.from_candidate(
                        candidate, "queue_lease_policy_disabled")
                    for candidate in request.candidates
                )
                return None
            evaluations: list[_CandidateEvaluation] = []
            rejected: list[RejectedCandidate] = []
            can_scale = len(self._active_pairs) < int(
                self.config.maximum_active_pairs)
            for candidate in request.candidates:
                if candidate.pair_index not in self._active_pairs and not can_scale:
                    rejected.append(RejectedCandidate.from_candidate(
                        candidate, "pair_inactive_at_maximum"))
                    continue
                value = self._evaluate_queue_lease_candidate(
                    candidate, request=request, now_ns=now_ns)
                if isinstance(value, RejectedCandidate):
                    rejected.append(value)
                else:
                    evaluations.append(value)
            if not evaluations:
                self._queue_lease_rejections[request_id] = tuple(sorted(
                    rejected,
                    key=lambda item: (
                        item.pair_index, item.route.value, item.reason),
                ))
                return None
            evaluations = self._prefer_business_clean_evaluations(
                request, evaluations, rejected)
            self._queue_lease_rejections.pop(request_id, None)
            # A queue lease is useful only when the selected endpoint can
            # make progress.  If one candidate already carries endpoint
            # service-window debt while another candidate is physically
            # serviceable, choosing the lower static score can pin the
            # request to a downstream queue for its entire E2E budget.  That
            # is exactly the failure mode seen in native P/D contention:
            # local prefill was over the endpoint window, while the remote
            # candidate was admissible but lost by a small prior-score gap.
            # Proven cache placement is a semantic constraint, not a soft
            # service score.  A P_ONLY request cannot be redirected to a
            # decoder-local route merely because that route has less endpoint
            # debt: the downstream vLLM/LMCache request can then ask the
            # decoder for a key that was only produced on the prefill side.
            # Preserve cache affinity first, then prefer an already healthy
            # and serviceable endpoint, then use score and deterministic
            # ordering.  A failure-free completion-liveness probe is useful
            # only when no healthy candidate is available; it must not steal
            # work from a route already making measured progress.
            evaluations.sort(key=lambda item: (
                not item.candidate.cache_affinity,
                not item.priority_service_lane,
                item.completion_liveness_probe,
                bool(item.endpoint_queue_debt_resources),
                item.score_ms,
                item.candidate.pair_index,
                item.candidate.route.value,
            ))
            static_best = evaluations[0]
            evaluations = self._mesh_near_tie_source_order(evaluations)
            selected = evaluations[0]
            rejected.extend(
                self._higher_score_rejection(
                    item,
                    selected=selected,
                    static_best=static_best,
                )
                for item in evaluations[1:]
            )
            candidate = selected.candidate
            active_before = tuple(sorted(self._active_pairs))
            if selected.activate:
                self._active_pairs.add(candidate.pair_index)
            active_after = tuple(sorted(self._active_pairs))
            used_before = selected.effective_used.as_dict()
            self._assign_tenant_pair_locked(
                request.tenant_id, candidate.pair_index, now_ns)
            virtual_before = self._tenant_virtual_service[request.tenant_id]
            weight = policy.weight
            service_units = candidate.work.dominant_ratio(
                self._capacities[candidate.pair_index])
            virtual_after = virtual_before + service_units / weight
            self._tenant_virtual_service[request.tenant_id] = virtual_after
            self._tenant_service_units[request.tenant_id] += service_units
            self._tenant_admitted_decode_tokens[request.tenant_id] += (
                candidate.work.decode_tokens)
            destination_work = self._destination_work(candidate)
            self._owned[candidate.pair_index] = (
                self._owned[candidate.pair_index] + destination_work)
            self._last_busy_ns[candidate.pair_index] = now_ns
            capacity = self._capacities[candidate.pair_index]
            binding = tuple(
                name for name in ResourceVector.names()
                if getattr(selected.effective_used + destination_work, name)
                > getattr(capacity, name)
            )
            binding = tuple(dict.fromkeys(
                binding + selected.endpoint_queue_debt_resources))
            if selected.protected_service_lane:
                binding = tuple(dict.fromkeys(
                    binding + (PROTECTED_SERVICE_LANE_BINDING,)))
            if selected.priority_service_lane:
                binding = tuple(dict.fromkeys(
                    binding + (self._priority_service_lane_binding(),)))
            if selected.mesh_near_tie_source_balanced:
                binding = tuple(dict.fromkeys(
                    binding + (MESH_NEAR_TIE_SOURCE_BALANCE_BINDING,)))
            if selected.completion_liveness_shared_probe:
                binding = tuple(dict.fromkeys(
                    binding + ("completion_liveness_shared_probe",)))
            if selected.endpoint_queue_headroom_admission:
                binding = tuple(dict.fromkeys(
                    binding + ("completion_progress_headroom",)))
            completion_credit_used = bool(
                not selected.priority_service_lane
                and not selected.completion_liveness_shared_probe
                and not selected.endpoint_queue_headroom_admission
                and self.config.endpoint_queue_debt_mode in {
                    "completion_credit_endpoint_queue_v3",
                    "completion_credit_mesh_endpoint_queue_v1",
                }
            )
            if completion_credit_used:
                binding = tuple(dict.fromkeys(
                    binding + ("completion_first_response_credit",)))
            decision = GlobalDecision(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                kind=GlobalDecisionKind.ADMIT,
                decided_ns=now_ns,
                reason=(
                    "global_protected_service_lane_queue_lease_committed"
                    if selected.protected_service_lane else
                    self._priority_service_lane_reason(promoted=False)
                    if selected.priority_service_lane else
                    "global_endpoint_completion_progress_headroom_route_committed"
                    if selected.endpoint_queue_headroom_admission else
                    "global_endpoint_completion_liveness_shared_probe_route_committed"
                    if selected.completion_liveness_shared_probe else
                    "global_endpoint_completion_liveness_probe_route_committed"
                    if selected.completion_liveness_probe else
                    "global_endpoint_completion_credit_route_committed"
                    if completion_credit_used else
                    "global_endpoint_queue_lease_route_committed"
                ),
                pair_index=candidate.pair_index,
                route=candidate.route,
                score_ms=selected.score_ms,
                deadline_slack_ms=selected.slack_ms,
                selected_work=candidate.work.as_dict(),
                predicted_e2e_ms=candidate.predicted_e2e_ms,
                predicted_ttft_ms=candidate.predicted_ttft_ms,
                uncertainty_ms=candidate.uncertainty_ms,
                cache_affinity=candidate.cache_affinity,
                binding_resources=binding,
                rejected_candidates=tuple(sorted(
                    rejected,
                    key=lambda item: (
                        item.pair_index, item.route.value, item.reason),
                )),
                resource_used_before=used_before,
                active_pairs_before=active_before,
                active_pairs_after=active_after,
                pair_activated=selected.activate,
                tenant_virtual_service_before=virtual_before,
                tenant_virtual_service_after=virtual_after,
                telemetry_sequences={
                    index: value.sequence
                    for index, value in self._telemetry.items()
                },
                telemetry_provenance=self._decision_telemetry_provenance(
                    candidate.pair_index, selected.joint_actuation),
                joint_actuation=selected.joint_actuation,
                cache_group_key=request.cache_group_key,
                queue_lease=True,
                prefill_index=candidate.prefill_index,
                decoder_index=candidate.decoder_index,
                edge_id=candidate.edge_id,
                receiver_stagger_us=selected.receiver_stagger_us,
                mesh_near_tie_source_balanced=(
                    selected.mesh_near_tie_source_balanced),
                mesh_near_tie_score_window_ms=(
                    selected.mesh_near_tie_score_window_ms),
                mesh_near_tie_score_delta_ms=(
                    selected.mesh_near_tie_score_delta_ms),
                mesh_source_virtual_service_before=(
                    selected.mesh_source_virtual_service_before),
                mesh_edge_virtual_service_before=(
                    selected.mesh_edge_virtual_service_before),
                service_queue_delay_ms=selected.service_queue_delay_ms,
                service_forecast_ms=selected.service_forecast_ms,
                protected_service_lane=selected.protected_service_lane,
                protected_service_lane_key=selected.protected_service_lane_key,
                protected_service_lane_before=(
                    selected.protected_service_lane_before),
                protected_service_lane_after=(
                    selected.protected_service_lane_after),
            )
            if completion_credit_used:
                if self._completion_credit_balance[candidate.pair_index] <= 0:
                    raise RuntimeError("completion credit vanished before commit")
                self._completion_credit_balance[candidate.pair_index] -= 1
            self._hold_cache_group_locked(request, candidate)
            self._inflight[request.request_id] = _Reservation(
                request=request,
                candidate=candidate,
                decision=decision,
                held=destination_work,
                committed_ns=now_ns,
                mesh_stage_held=self._reserve_mesh_stage_locked(candidate),
            )
            del self._queued[request.request_id]
            self._decision_history.setdefault(
                request.request_id, []).append(decision)
            return decision

    def reject_unadmitted(
        self, request: GlobalRequest, *, now_ns: int, reason: str,
    ) -> GlobalDecision:
        """Record a terminal reject before admission when telemetry is unavailable.

        A request-triggered telemetry failure happens before ``submit`` can
        place a request in the global queue.  It must still have the same
        explicit business/terminal receipt as an overload reject; otherwise
        the HTTP 503 is indistinguishable from a missing decision in the
        native ledger.
        """

        if not isinstance(request, GlobalRequest):
            raise TypeError("request must be GlobalRequest")
        _positive_int("now_ns", now_ns, zero=True)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("unadmitted rejection reason must be nonempty")
        with self._lock:
            request_id = request.request_id
            if (
                request_id in self._queued
                or request_id in self._inflight
                or request_id in self._terminal
            ):
                raise ValueError("request already has a global lifecycle")
            return self._reject_locked(request, now_ns, reason=reason)

    def decision_history(self, request_id: str) -> tuple[GlobalDecision, ...]:
        with self._lock:
            return tuple(self._decision_history.get(request_id, ()))

    def failure_history(self, request_id: str) -> tuple[GlobalFailureReceipt, ...]:
        with self._lock:
            return tuple(self._failure_history.get(request_id, ()))

    @staticmethod
    def _shared_budget_evidence(
        budget: _SharedRemoteBudget,
    ) -> dict[str, object]:
        return {
            "group": budget.group,
            "members": list(budget.members),
            "limits": {
                "remote_requests": budget.requests_limit,
                "remote_kv_bytes": budget.kv_bytes_limit,
                "remote_semantic_ops": budget.semantic_ops_limit,
            },
            "used_before": {
                "remote_requests": budget.requests_used,
                "remote_kv_bytes": budget.kv_bytes_used,
                "remote_semantic_ops": budget.semantic_ops_used,
            },
            "dispatch_stagger_us": budget.dispatch_stagger_us,
            "limited": budget.limited,
            "suppress_pair_activation": budget.suppress_pair_activation,
            "contributions": [
                {"name": name, "pressure": pressure}
                for name, pressure in budget.contributions
            ],
        }

    def snapshot(self, *, now_ns: int) -> dict[str, object]:
        _positive_int("now_ns", now_ns, zero=True)
        with self._lock:
            shared_groups: dict[str, dict[str, object]] = {}
            for pair in sorted(self._capacities):
                budget = self._shared_remote_budget_for_pair(pair)
                if budget is not None:
                    shared_groups[budget.group] = (
                        self._shared_budget_evidence(budget))
            protected_debt: dict[str, int] = {}
            for reservation in self._inflight.values():
                if not reservation.decision.protected_service_lane:
                    continue
                candidate = reservation.candidate
                key = self._protected_service_lane_key(candidate)
                key_text = (
                    f"local:d{key[1]}"
                    if key[2] is GlobalRoute.LOCAL
                    else f"remote:p{key[0]}->d{key[1]}"
                )
                protected_debt[key_text] = (
                    protected_debt.get(key_text, 0)
                    + reservation.held.active_sequences
                )
            return {
                "schema": SCHEMA,
                "now_ns": now_ns,
                "active_pairs": sorted(self._active_pairs),
                "shared_fabric_control_mode": (
                    self.config.shared_fabric_control_mode),
                "telemetry_fresh_ns": self.config.telemetry_fresh_ns,
                "telemetry_stale_grace_ns": (
                    self.config.telemetry_stale_grace_ns),
                "shared_remote_budgets": shared_groups,
                "cross_layer_remote_receiver_guard": {
                    "mode": (
                        self.config.cross_layer_remote_receiver_guard_mode),
                    "scope": (
                        self.config.cross_layer_remote_receiver_guard_scope),
                    "p99_ceiling_ms": (
                        self.config.cross_layer_remote_receiver_guard_p99_ms),
                },
                "queued": len(self._queued),
                "inflight": len(self._inflight),
                "terminal": len(self._terminal),
                "overload_action": self.config.overload_action,
                "endpoint_queue_debt_mode": (
                    self.config.endpoint_queue_debt_mode),
                "endpoint_queue_capacity": self.config.endpoint_queue_capacity,
                "priority_service_lane_mode": (
                    self.config.priority_service_lane_mode),
                "priority_service_lane_capacity": (
                    self.config.priority_service_lane_capacity),
                "priority_service_lane_min_admission_priority": (
                    self.config.priority_service_lane_min_admission_priority),
                "priority_service_lane_priority": (
                    self.config.priority_service_lane_priority),
                "decoder_business_admission_mode": (
                    self.config.decoder_business_admission_mode),
                "decoder_business_background_max_wait_ns": (
                    self.config.decoder_business_background_max_wait_ns),
                "mesh_near_tie_source_balance_mode": (
                    self.config.mesh_near_tie_source_balance_mode),
                "mesh_near_tie_source_balance_uncertainty_fraction": (
                    self.config.
                    mesh_near_tie_source_balance_uncertainty_fraction),
                "service_feasibility_mode": (
                    self.config.service_feasibility_mode),
                "service_forecast_safety_factor": (
                    self.config.service_forecast_safety_factor),
                "protected_service_lane_mode": (
                    self.config.protected_service_lane_mode),
                "protected_service_lane_capacity": (
                    self.config.protected_service_lane_capacity),
                "protected_service_lane_min_admission_priority": (
                    self.config.protected_service_lane_min_admission_priority),
                "protected_service_lane_debt": protected_debt,
                "priority_service_lane_debt": {
                    str(pair): self._priority_service_lane_debt(pair)
                    for pair in sorted(self._capacities)
                },
                "completion_credit_balance": {
                    str(pair): value
                    for pair, value in sorted(
                        self._completion_credit_balance.items())
                },
                "completion_liveness_bootstrap_sequences": {
                    f"{pair}:{route.value}": sequence
                    for (pair, route), sequence in sorted(
                        self._completion_liveness_bootstrap_sequences.items(),
                        key=lambda item: (item[0][0], item[0][1].value),
                    )
                },
                "service_lane_queue_promotions": {
                    request_id: global_service_lane_queue_promotion_dict(
                        receipt)
                    for request_id, receipt in sorted(
                        self._service_lane_queue_promotions.items())
                },
                "route_failure_quarantine_mode": (
                    self.config.route_failure_quarantine_mode),
                "telemetry_failure_quarantine_mode": (
                    self.config.telemetry_failure_quarantine_mode),
                "telemetry_failure_quarantine_scope": (
                    self.config.telemetry_failure_quarantine_scope),
                "endpoint_queue_lease_cooldowns": {
                    str(pair): sequence
                    for pair, sequence in sorted(
                        self._endpoint_queue_lease_cooldowns.items())
                },
                "cache_group_holds": [
                    {
                        "pair_index": pair,
                        "cache_group_key": cache_group_key,
                        "request_id": request_id,
                    }
                    for (pair, cache_group_key), request_id in sorted(
                        self._cache_group_holds.items())
                ],
                "route_failure_quarantines": [
                    {
                        "pair_index": value.pair_index,
                        "route": value.route.value,
                        "failure_kind": value.failure_kind,
                        "count": value.count,
                        "first_failed_ns": value.first_failed_ns,
                        "last_failed_ns": value.last_failed_ns,
                        "telemetry_sequence": value.telemetry_sequence,
                        "scope": value.scope,
                        "trigger": value.trigger,
                    }
                    for value in sorted(
                        self._route_quarantines.values(),
                        key=lambda item: (item.pair_index, item.route.value),
                    )
                ],
                "mesh_edge_failure_quarantines": [
                    {
                        "prefill_index": value.prefill_index,
                        "decoder_index": value.decoder_index,
                        "edge_id": (
                            f"remote:p{value.prefill_index}"
                            f"->d{value.decoder_index}"
                        ),
                        "failure_kind": value.failure_kind,
                        "count": value.count,
                        "first_failed_ns": value.first_failed_ns,
                        "last_failed_ns": value.last_failed_ns,
                        "prefill_telemetry_sequence": (
                            value.prefill_telemetry_sequence),
                        "decoder_telemetry_sequence": (
                            value.decoder_telemetry_sequence),
                        "scope": value.scope,
                    }
                    for value in sorted(
                        self._mesh_edge_quarantines.values(),
                        key=lambda item: (
                            item.prefill_index, item.decoder_index),
                    )
                ],
                "route_failure_events": [
                    global_failure_dict(receipt)
                    for receipts in self._failure_history.values()
                    for receipt in receipts
                ],
                "admission_guards": {
                    "remote_semantic_ops_safety_reserve": (
                        self.config.remote_semantic_ops_safety_reserve),
                    "remote_semantic_ops_limit_by_pair": {
                        str(pair): (
                            self._capacities[pair].remote_semantic_ops
                            - self.config.remote_semantic_ops_safety_reserve)
                        for pair in sorted(self._capacities)
                    },
                    "tenant_queue_reservation_slots": {
                        tenant_id: policy.queue_reservation_slots
                        for tenant_id, policy in self._tenants.items()
                    },
                    "survivor_capacity_reserve_fraction": (
                        self.config.survivor_capacity_reserve_fraction),
                    "survivor_reserve_bypass_min_weight": (
                        self.config.survivor_reserve_bypass_min_weight),
                    "fully_quarantined_pairs": sorted(
                        self._fully_quarantined_pairs()),
                    "survivor_reserve_active_by_pair": {
                        str(pair): self._survivor_reserve_active(pair)
                        for pair in sorted(self._capacities)
                    },
                },
                "owned_by_pair": {
                    str(pair): value.as_dict()
                    for pair, value in self._owned.items()
                },
                "mesh_control_mode": self.config.mesh_control_mode,
                "mesh_source_prefill_owned": {
                    str(prefill): value
                    for prefill, value in sorted(
                        self._mesh_source_prefill_owned.items())
                },
                "mesh_source_virtual_service": {
                    str(prefill): value
                    for prefill, value in sorted(
                        self._mesh_source_virtual_service.items())
                },
                "mesh_edge_virtual_service": {
                    f"p{prefill}->d{decoder}": value
                    for (prefill, decoder), value in sorted(
                        self._mesh_edge_virtual_service.items())
                },
                "mesh_edges": {
                    f"p{prefill}->d{decoder}": {
                        "held_remote_prefill_token_ms": (
                            state.held_remote_prefill_token_ms),
                        "held_remote_kv_bytes": state.held_remote_kv_bytes,
                        "held_remote_semantic_ops": (
                            state.held_remote_semantic_ops),
                        "inflight_transfers": state.inflight_transfers,
                        "completed_first_responses": (
                            state.completed_first_responses),
                        "first_response_ewma_ms": (
                            state.first_response_ewma_ms),
                        "last_completion_ns": state.last_completion_ns,
                    }
                    for (prefill, decoder), state in sorted(
                        self._mesh_edges.items())
                },
                "telemetry_sequences": {
                    str(pair): value.sequence
                    for pair, value in self._telemetry.items()
                },
                "telemetry_provenance": {
                    str(pair): self._telemetry_evidence(value)
                    for pair, value in self._telemetry.items()
                },
                "tenant_virtual_service": dict(self._tenant_virtual_service),
                "tenant_service_units": dict(self._tenant_service_units),
                "tenant_pair_assignments": {
                    tenant_id: sorted(pairs)
                    for tenant_id, pairs in self._tenant_pair_assignments.items()
                },
                "tenant_pair_last_busy_ns": {
                    tenant_id: {
                        str(pair): value
                        for pair, value in sorted(last_busy.items())
                    }
                    for tenant_id, last_busy in (
                        self._tenant_pair_last_busy_ns.items())
                },
                "tenant_policies": {
                    tenant_id: {
                        "weight": policy.weight,
                        "ttft_slo_ms": policy.ttft_slo_ms,
                        "tpot_slo_ms": policy.tpot_slo_ms,
                        "e2e_slo_ms": policy.e2e_slo_ms,
                        "maximum_queue_wait_ns": policy.maximum_queue_wait_ns,
                        "minimum_service_fraction": (
                            policy.minimum_service_fraction),
                        "queue_reservation_slots": (
                            policy.queue_reservation_slots),
                        "queue_lease_on_timeout": (
                            policy.queue_lease_on_timeout),
                        "telemetry_stale_grace_ns": (
                            policy.telemetry_stale_grace_ns),
                        "admission_priority": policy.admission_priority,
                        "protected_capacity_fraction": (
                            policy.protected_capacity_fraction),
                        "pair_spread_limit": policy.pair_spread_limit,
                    }
                    for tenant_id, policy in self._tenants.items()
                },
                "tenant_admitted_decode_tokens": dict(
                    self._tenant_admitted_decode_tokens),
                "tenant_completed_decode_tokens": dict(
                    self._tenant_completed_decode_tokens),
                "fairness_basis": "weighted_dominant_resource_service",
                "phases": {
                    **{
                        request_id: GlobalRequestPhase.QUEUED.value
                        for request_id in self._queued
                    },
                    **{
                        request_id: reservation.phase.value
                        for request_id, reservation in self._inflight.items()
                    },
                    **{
                        request_id: phase.value
                        for request_id, phase in self._terminal.items()
                    },
                },
            }

    def telemetry_snapshot(self) -> tuple[PairTelemetry, ...]:
        """Return the latest immutable endpoint observations for hierarchy fan-in.

        A hierarchy reducer uses this bounded snapshot to prove that its node
        and pair envelopes belong to one allocation-scoped identity before it
        forwards candidates to this global authority.  The returned values are
        immutable dataclasses; callers cannot mutate controller state.
        """

        with self._lock:
            return tuple(
                self._telemetry[pair]
                for pair in sorted(self._telemetry)
            )

    def submit_hierarchical(
        self,
        request: GlobalRequest,
        *,
        reducer: object,
        now_ns: int,
    ) -> tuple[GlobalDecision, object]:
        """Submit through node/pair/shard bounded fan-in.

        The import is lazy to keep the core controller usable by the legacy
        two-pair path without an import cycle.  The reducer returns an
        immutable fan-in receipt alongside the reduced request; this method
        then invokes the same atomic global lifecycle as ``submit``.
        """

        from tempo.pd_global_hierarchy import submit_hierarchical

        return submit_hierarchical(
            self,
            reducer,
            request,
            now_ns=now_ns,
        )

    def submit_hierarchical_frontiers(
        self,
        header: object,
        *,
        frontiers: object,
        reducer: object,
        now_ns: int,
    ) -> tuple[GlobalDecision, object]:
        """Submit pair-agent frontiers without central raw-candidate scanning."""

        from tempo.pd_global_hierarchy import submit_hierarchical_frontiers

        return submit_hierarchical_frontiers(
            self,
            reducer,
            header,
            frontiers,
            now_ns=now_ns,
        )

    def _reservation(self, request_id: str) -> _Reservation:
        reservation = self._inflight.get(request_id)
        if reservation is None:
            raise ValueError("request is not globally in flight")
        return reservation

    @staticmethod
    def _cache_group_hold_key(
        request: GlobalRequest, candidate: RouteCandidate
    ) -> tuple[int, str] | None:
        """Return the pair-local cache transfer identity, when applicable.

        The key intentionally includes the pair.  LMCache receiver ownership
        is local to the selected P/D pair, while the same token chunk may be
        transferred concurrently to a different pair without sharing that
        receiver state.  Local execution never enters this hold table.
        """

        if (
            candidate.route is not GlobalRoute.REMOTE
            or request.cache_group_key is None
        ):
            return None
        return candidate.pair_index, request.cache_group_key

    def _cache_group_rejection_locked(
        self, request: GlobalRequest, candidate: RouteCandidate
    ) -> RejectedCandidate | None:
        key = self._cache_group_hold_key(request, candidate)
        if key is None:
            return None
        owner = self._cache_group_holds.get(key)
        if owner is None or owner == request.request_id:
            return None
        return RejectedCandidate.from_candidate(
            candidate,
            "cache_chunk_transfer_serialization",
            ("cache_group_hold",),
        )

    def _hold_cache_group_locked(
        self, request: GlobalRequest, candidate: RouteCandidate
    ) -> None:
        key = self._cache_group_hold_key(request, candidate)
        if key is None:
            return
        owner = self._cache_group_holds.get(key)
        if owner is not None and owner != request.request_id:
            raise RuntimeError(
                "cache group was admitted concurrently under the global lock")
        self._cache_group_holds[key] = request.request_id

    def _release_cache_group_locked(
        self, request: GlobalRequest, candidate: RouteCandidate
    ) -> None:
        key = self._cache_group_hold_key(request, candidate)
        if key is not None and self._cache_group_holds.get(key) == request.request_id:
            del self._cache_group_holds[key]

    def _quarantine_route_locked(
        self,
        pair_index: int,
        route: GlobalRoute,
        *,
        failure_kind: str,
        now_ns: int,
        telemetry_sequence: int,
        scope: str,
        trigger: str = "explicit_failure_receipt",
    ) -> None:
        key = (pair_index, route)
        prior = self._route_quarantines.get(key)
        if prior is None:
            self._route_quarantines[key] = _RouteQuarantine(
                pair_index=pair_index,
                route=route,
                failure_kind=failure_kind,
                count=1,
                first_failed_ns=now_ns,
                last_failed_ns=now_ns,
                telemetry_sequence=telemetry_sequence,
                scope=scope,
                trigger=trigger,
            )
            return
        self._route_quarantines[key] = _RouteQuarantine(
            pair_index=pair_index,
            route=route,
            failure_kind=failure_kind,
            count=prior.count + 1,
            first_failed_ns=prior.first_failed_ns,
            last_failed_ns=now_ns,
            telemetry_sequence=telemetry_sequence,
            scope=scope,
            trigger=trigger,
        )

    def _quarantine_mesh_edge_locked(
        self,
        prefill_index: int,
        decoder_index: int,
        *,
        failure_kind: str,
        now_ns: int,
        scope: str,
    ) -> None:
        key = (prefill_index, decoder_index)
        prior = self._mesh_edge_quarantines.get(key)
        prefill_sequence = (
            self._telemetry[prefill_index].sequence
            if prefill_index in self._telemetry else 0
        )
        decoder_sequence = (
            self._telemetry[decoder_index].sequence
            if decoder_index in self._telemetry else 0
        )
        self._mesh_edge_quarantines[key] = _MeshEdgeQuarantine(
            prefill_index=prefill_index,
            decoder_index=decoder_index,
            failure_kind=failure_kind,
            count=1 if prior is None else prior.count + 1,
            first_failed_ns=(
                now_ns if prior is None else prior.first_failed_ns),
            last_failed_ns=now_ns,
            prefill_telemetry_sequence=prefill_sequence,
            decoder_telemetry_sequence=decoder_sequence,
            scope=scope,
        )

    def _quarantine_enabled(self) -> bool:
        return (
            self.config.route_failure_quarantine_mode == "deny_until_probe"
            or self.config.telemetry_failure_quarantine_mode == "deny_until_probe"
        )

    def _observe_telemetry_failure_delta_locked(
        self, prior: PairTelemetry | None, current: PairTelemetry
    ) -> None:
        """Turn a new endpoint failure count into a pre-admission circuit.

        Endpoint failure counters are cumulative within an allocation epoch.
        A delta is therefore a causal event even when the endpoint health field
        has not yet changed from GOOD.  Candidate I uses pair scope by default
        because an EngineCore/cache-key failure can invalidate both routes
        sharing that decoder.  Recovery remains explicit PROBE-only.
        """

        if self.config.telemetry_failure_quarantine_mode != "deny_until_probe":
            return
        previous_counts = {
            GlobalRoute.LOCAL: (
                prior.local_failure_count if prior is not None else 0),
            GlobalRoute.REMOTE: (
                prior.remote_failure_count if prior is not None else 0),
        }
        current_counts = {
            GlobalRoute.LOCAL: current.local_failure_count,
            GlobalRoute.REMOTE: current.remote_failure_count,
        }
        last_kinds = {
            GlobalRoute.LOCAL: current.local_last_failure_kind,
            GlobalRoute.REMOTE: current.remote_last_failure_kind,
        }
        for route in (GlobalRoute.LOCAL, GlobalRoute.REMOTE):
            if current_counts[route] <= previous_counts[route]:
                continue
            failure_kind = last_kinds[route] or "telemetry_failure_delta"
            scope = self.config.telemetry_failure_quarantine_scope
            routes = (
                ((current.pair_index, GlobalRoute.LOCAL),
                 (current.pair_index, GlobalRoute.REMOTE))
                if scope == "pair"
                else ((current.pair_index, route),)
            )
            for pair_index, quarantined_route in routes:
                self._quarantine_route_locked(
                    pair_index,
                    quarantined_route,
                    failure_kind=failure_kind,
                    now_ns=current.collected_ns,
                    telemetry_sequence=current.sequence,
                    scope=scope,
                    trigger="telemetry_failure_delta",
                )

    def _recover_route_quarantines_locked(
        self, telemetry: Iterable[PairTelemetry]
    ) -> None:
        if not self._quarantine_enabled():
            return
        for item in telemetry:
            for route in (GlobalRoute.LOCAL, GlobalRoute.REMOTE):
                key = (item.pair_index, route)
                quarantine = self._route_quarantines.get(key)
                if quarantine is None:
                    continue
                if item.sequence <= quarantine.telemetry_sequence:
                    continue
                # PROBE is an explicit application-endpoint recovery signal.
                # GOOD without a probe does not erase a failure receipt.
                if item.health(route) is PathHealth.PROBE:
                    del self._route_quarantines[key]
        for key, quarantine in tuple(self._mesh_edge_quarantines.items()):
            prefill = self._telemetry.get(quarantine.prefill_index)
            decoder = self._telemetry.get(quarantine.decoder_index)
            if prefill is None or decoder is None:
                continue
            if (
                prefill.sequence <= quarantine.prefill_telemetry_sequence
                or decoder.sequence <= quarantine.decoder_telemetry_sequence
            ):
                continue
            # An edge spans two endpoint agents.  Recovery therefore requires
            # a newer explicit remote-path PROBE from both ends; one GOOD
            # sample cannot erase an edge failure receipt.
            if (
                prefill.remote_health is PathHealth.PROBE
                and decoder.remote_health is PathHealth.PROBE
            ):
                del self._mesh_edge_quarantines[key]

    def _validate_telemetry_update_locked(
        self, telemetry: PairTelemetry
    ) -> PairTelemetry | None:
        prior = self._telemetry.get(telemetry.pair_index)
        if prior is not None and (
            telemetry.sequence <= prior.sequence
            or telemetry.sampled_ns < prior.sampled_ns
            or telemetry.collected_ns < prior.collected_ns
            or telemetry.agent_epoch != prior.agent_epoch
            or telemetry.profile_fingerprint_sha256
            != prior.profile_fingerprint_sha256
            or telemetry.controller_generation != prior.controller_generation
        ):
            raise ValueError(
                "telemetry sequence/time/epoch/profile is not monotonic")
        return prior

    @staticmethod
    def _telemetry_evidence(telemetry: PairTelemetry) -> dict[str, object]:
        return {
            "sequence": telemetry.sequence,
            "sampled_ns": telemetry.sampled_ns,
            "collected_ns": telemetry.collected_ns,
            "agent_epoch": telemetry.agent_epoch,
            "profile_fingerprint_sha256": (
                telemetry.profile_fingerprint_sha256),
            "controller_generation": telemetry.controller_generation,
            "source": telemetry.source,
            "schema": telemetry.schema,
            "scheduler": {
                "schema": telemetry.scheduler_schema,
                "source": telemetry.scheduler_source,
                "running_requests": telemetry.scheduler_running_requests,
                "waiting_requests": telemetry.scheduler_waiting_requests,
                "kv_cache_usage_fraction": (
                    telemetry.scheduler_kv_cache_usage_fraction),
            },
            "service_multipliers": {
                "local": telemetry.local_service_multiplier,
                "remote": telemetry.remote_service_multiplier,
            },
            "completion": {
                "schema": telemetry.completion_schema,
                "completed_first_responses": (
                    telemetry.endpoint_completed_first_responses),
                "residual_inflight": telemetry.endpoint_residual_inflight,
            },
            "route_failures": {
                "local_count": telemetry.local_failure_count,
                "remote_count": telemetry.remote_failure_count,
                "local_last_kind": telemetry.local_last_failure_kind,
                "remote_last_kind": telemetry.remote_last_failure_kind,
            },
            "quarantine_reason": telemetry.quarantine_reason,
            "cross_layer": (
                telemetry.cross_layer.as_dict()
                if telemetry.cross_layer is not None else None
            ),
        }

    def _telemetry_provenance(self) -> dict[int, dict[str, object]]:
        return {
            index: self._telemetry_evidence(value)
            for index, value in self._telemetry.items()
        }

    def _decision_telemetry_provenance(
        self,
        selected_pair: int | None = None,
        joint_actuation: JointActuationPlan | None = None,
    ) -> dict[int, dict[str, object]]:
        """Keep v3 receipts bounded while retaining shared-group identity.

        Legacy decisions keep the historical all-pair evidence map.  The v3
        shared budget is already computed from an atomic batch and carries the
        complete sequence/group/contribution receipt; duplicating every raw
        endpoint vector into every request decision creates a central fan-in
        bottleneck at large pair counts.
        """

        if self.config.shared_fabric_control_mode != "global_budget_v3":
            return self._telemetry_provenance()
        value: dict[int, dict[str, object]] = {}
        if selected_pair is not None and selected_pair in self._telemetry:
            value[selected_pair] = self._telemetry_evidence(
                self._telemetry[selected_pair])
        groups: dict[str, dict[str, object]] = {}
        group_first_pair: dict[str, int] = {}
        for pair in sorted(self._telemetry):
            group = self._cross_layer_group_key(pair)
            if group is not None:
                group_first_pair.setdefault(group, pair)
        for group, pair in group_first_pair.items():
            self._shared_remote_budget_for_pair(pair)
        for group, static_budget in self._shared_budget_static.items():
            used = self._shared_remote_usage(group)
            budget = replace(
                static_budget,
                requests_used=used[0],
                kv_bytes_used=used[1],
                semantic_ops_used=used[2],
            )
            groups[group] = self._shared_budget_evidence(budget)
        if groups:
            # Negative keys cannot collide with a configured pair index and
            # are serialized as strings by global_decision_dict.
            value[-1] = {
                "schema": "tempo-go-shared-fabric-provenance-v1",
                "selected_pair": selected_pair,
                "joint_actuation_schema": (
                    joint_actuation.schema
                    if joint_actuation is not None else None),
                "groups": groups,
                "telemetry_sequences": {
                    str(index): item.sequence
                    for index, item in sorted(self._telemetry.items())
                },
            }
        return value

    def _effective_used(self, pair: int) -> ResourceVector:
        observed = self._telemetry[pair].observed_total
        owned = self._owned[pair]
        return ResourceVector(**{
            name: max(getattr(observed, name), getattr(owned, name))
            for name in ResourceVector.names()
        })

    def _mesh_enabled(self) -> bool:
        return self.config.mesh_control_mode == "receiver_credit_pxd_v1"

    def _validate_candidate_topology(self, candidate: RouteCandidate) -> None:
        assert candidate.prefill_index is not None
        assert candidate.decoder_index is not None
        if candidate.decoder_index not in self._capacities:
            raise ValueError("request candidate decoder is not configured")
        if candidate.prefill_index not in self._capacities:
            raise ValueError("request candidate prefill is not configured")
        if (
            not self._mesh_enabled()
            and candidate.prefill_index != candidate.decoder_index
        ):
            raise ValueError("cross P/D candidate requires mesh control mode")

    def _destination_work(self, candidate: RouteCandidate) -> ResourceVector:
        """Project one candidate onto the destination D/receiver ledger."""

        if not self._mesh_enabled() or candidate.route is GlobalRoute.LOCAL:
            return candidate.work
        return ResourceVector(
            decode_tokens=candidate.work.decode_tokens,
            active_sequences=candidate.work.active_sequences,
            endpoint_requests=candidate.work.endpoint_requests,
            remote_kv_bytes=candidate.work.remote_kv_bytes,
            remote_semantic_ops=candidate.work.remote_semantic_ops,
        )

    def _effective_destination_used(
        self, candidate: RouteCandidate,
    ) -> ResourceVector:
        decoder = int(candidate.decoder_index)
        if not self._mesh_enabled():
            return self._effective_used(decoder)
        observed = self._telemetry[decoder].observed_total
        # Producer-prefill token-ms belongs to P, not to the destination D.
        # All receiver/decode dimensions remain visible on D.
        observed_destination = ResourceVector(
            decode_tokens=observed.decode_tokens,
            active_sequences=observed.active_sequences,
            endpoint_requests=observed.endpoint_requests,
            local_prefill_token_ms=observed.local_prefill_token_ms,
            remote_kv_bytes=observed.remote_kv_bytes,
            remote_semantic_ops=observed.remote_semantic_ops,
        )
        owned = self._owned[decoder]
        return ResourceVector(**{
            name: max(
                getattr(observed_destination, name),
                getattr(owned, name),
            )
            for name in ResourceVector.names()
        })

    def _mesh_source_used(self, prefill: int) -> int:
        observed = self._telemetry[prefill].observed_total
        return max(
            observed.remote_prefill_token_ms,
            self._mesh_source_prefill_owned[prefill],
        )

    def _mesh_receiver_inflight(self, decoder: int) -> int:
        return sum(
            state.inflight_transfers
            for (prefill, destination), state in self._mesh_edges.items()
            if destination == decoder
        )

    def _mesh_candidate_rejection(
        self, candidate: RouteCandidate, *, already_owned: bool = False,
    ) -> RejectedCandidate | None:
        """Enforce source, edge, and receiver-advertised semantic credits."""

        if not self._mesh_enabled() or candidate.route is GlobalRoute.LOCAL:
            return None
        prefill = int(candidate.prefill_index)
        decoder = int(candidate.decoder_index)
        additional_prefill = (
            0 if already_owned else candidate.work.remote_prefill_token_ms)
        additional_kv = 0 if already_owned else candidate.work.remote_kv_bytes
        additional_semantic = (
            0 if already_owned else candidate.work.remote_semantic_ops)
        additional_transfer = 0 if already_owned else 1
        source_after = (
            self._mesh_source_used(prefill)
            + additional_prefill
        )
        source_limit = self._capacities[prefill].remote_prefill_token_ms
        if source_after > source_limit:
            return RejectedCandidate.from_candidate(
                candidate,
                "mesh_source_prefill_credit",
                ("source_remote_prefill_token_ms",),
            )
        edge = self._mesh_edges[(prefill, decoder)]
        receiver_capacity = self._capacities[decoder]
        semantic_limit = (
            receiver_capacity.remote_semantic_ops
            - self.config.remote_semantic_ops_safety_reserve
        )
        bindings = []
        if (
            edge.held_remote_kv_bytes + additional_kv
            > receiver_capacity.remote_kv_bytes
        ):
            bindings.append("edge_remote_kv_bytes")
        if (
            edge.held_remote_semantic_ops + additional_semantic
            > semantic_limit
        ):
            bindings.append("edge_remote_semantic_ops")
        if edge.inflight_transfers + additional_transfer > semantic_limit:
            bindings.append("edge_inflight_transfers")
        if bindings:
            return RejectedCandidate.from_candidate(
                candidate,
                "mesh_edge_receiver_credit",
                tuple(bindings),
            )
        return None

    def _mesh_receiver_stagger_us(self, candidate: RouteCandidate) -> int:
        if (
            not self._mesh_enabled()
            or candidate.route is GlobalRoute.LOCAL
            or self.config.mesh_receiver_stagger_max_us == 0
        ):
            return 0
        decoder = int(candidate.decoder_index)
        inflight = self._mesh_receiver_inflight(decoder)
        if inflight <= 0:
            return 0
        limit = max(
            1,
            self._capacities[decoder].remote_semantic_ops
            - self.config.remote_semantic_ops_safety_reserve,
        )
        return int(round(
            self.config.mesh_receiver_stagger_max_us
            * min(1.0, inflight / limit)
        ))

    def _mesh_edge_feedback_penalty_ms(
        self, candidate: RouteCandidate,
    ) -> float:
        if not self._mesh_enabled() or candidate.route is GlobalRoute.LOCAL:
            return 0.0
        state = self._mesh_edges[
            (int(candidate.prefill_index), int(candidate.decoder_index))]
        if state.first_response_ewma_ms is None or state.inflight_transfers <= 0:
            return 0.0
        limit = max(1, self._capacities[
            int(candidate.decoder_index)].remote_semantic_ops)
        residual = max(
            0.0,
            state.first_response_ewma_ms - candidate.predicted_ttft_ms,
        )
        return residual * min(1.0, state.inflight_transfers / limit)

    def _mesh_virtual_finish(
        self, candidate: RouteCandidate,
    ) -> tuple[float, float, float]:
        """Return dominant, source, and edge virtual finish for remote work."""

        prefill = int(candidate.prefill_index)
        decoder = int(candidate.decoder_index)
        source_before = self._mesh_source_virtual_service[prefill]
        edge_before = self._mesh_edge_virtual_service[(prefill, decoder)]
        source_units = (
            candidate.work.remote_prefill_token_ms
            / max(1, self._capacities[prefill].remote_prefill_token_ms)
        )
        receiver_capacity = self._capacities[decoder]
        edge_units = max(
            candidate.work.remote_kv_bytes
            / max(1, receiver_capacity.remote_kv_bytes),
            candidate.work.remote_semantic_ops
            / max(1, receiver_capacity.remote_semantic_ops),
        )
        return (
            max(source_before + source_units, edge_before + edge_units),
            source_before,
            edge_before,
        )

    def _mesh_near_tie_peer(
        self,
        best: _CandidateEvaluation,
        item: _CandidateEvaluation,
    ) -> bool:
        """Whether ``item`` is semantically equivalent inside model error."""

        if (
            self.config.mesh_near_tie_source_balance_mode
            != "telemetry_uncertainty_virtual_service_v1"
            or best.candidate.route is not GlobalRoute.REMOTE
            or item.candidate.route is not GlobalRoute.REMOTE
            or best.candidate.decoder_index != item.candidate.decoder_index
            or best.candidate.cache_affinity != item.candidate.cache_affinity
            or best.priority_service_lane != item.priority_service_lane
            or best.completion_liveness_probe
            != item.completion_liveness_probe
            or best.endpoint_queue_debt_resources
            != item.endpoint_queue_debt_resources
            or best.candidate.work != item.candidate.work
        ):
            return False
        fraction = (
            self.config.mesh_near_tie_source_balance_uncertainty_fraction)
        window_ms = min(
            best.candidate.uncertainty_ms,
            item.candidate.uncertainty_ms,
        ) * fraction
        return item.score_ms - best.score_ms <= window_ms

    def _mesh_near_tie_source_order(
        self,
        ordered: list[_CandidateEvaluation],
    ) -> list[_CandidateEvaluation]:
        """Use virtual service only when telemetry scores are indistinguishable.

        ``ordered`` must already follow the normal semantic/service/score
        ordering.  The first item therefore remains the static score anchor;
        source balancing cannot recursively widen its own eligibility window.
        """

        if len(ordered) < 2:
            return ordered
        best = ordered[0]
        peers = [
            item for item in ordered
            if self._mesh_near_tie_peer(best, item)
        ]
        if len({int(item.candidate.prefill_index) for item in peers}) < 2:
            return ordered
        selected = min(
            peers,
            key=lambda item: (
                self._mesh_virtual_finish(item.candidate)[0],
                item.score_ms,
                int(item.candidate.prefill_index),
            ),
        )
        _, source_before, edge_before = self._mesh_virtual_finish(
            selected.candidate)
        window_ms = min(
            best.candidate.uncertainty_ms,
            selected.candidate.uncertainty_ms,
        ) * self.config.mesh_near_tie_source_balance_uncertainty_fraction
        selected = replace(
            selected,
            mesh_near_tie_source_balanced=True,
            mesh_near_tie_score_window_ms=window_ms,
            mesh_near_tie_score_delta_ms=(
                selected.score_ms - best.score_ms),
            mesh_source_virtual_service_before=source_before,
            mesh_edge_virtual_service_before=edge_before,
        )
        return [selected] + [
            item for item in ordered if item is not selected
            and item.candidate != selected.candidate
        ]

    def _higher_score_rejection(
        self,
        item: _CandidateEvaluation,
        *,
        selected: _CandidateEvaluation,
        static_best: _CandidateEvaluation,
    ) -> RejectedCandidate:
        near_tie = self._mesh_near_tie_peer(static_best, item)
        _, source_before, edge_before = (
            self._mesh_virtual_finish(item.candidate)
            if item.candidate.route is GlobalRoute.REMOTE
            else (0.0, 0.0, 0.0)
        )
        return replace(
            RejectedCandidate.from_candidate(
                item.candidate,
                "mesh_near_tie_source_virtual_service"
                if selected.mesh_near_tie_source_balanced and near_tie
                else "higher_global_score",
            ),
            evaluated_score_ms=item.score_ms,
            score_delta_ms=item.score_ms - static_best.score_ms,
            uncertainty_ms=item.candidate.uncertainty_ms,
            mesh_near_tie_eligible=near_tie,
            mesh_source_virtual_service_before=(
                source_before
                if item.candidate.route is GlobalRoute.REMOTE else None),
            mesh_edge_virtual_service_before=(
                edge_before
                if item.candidate.route is GlobalRoute.REMOTE else None),
        )

    def _reserve_mesh_stage_locked(self, candidate: RouteCandidate) -> bool:
        if not self._mesh_enabled() or candidate.route is GlobalRoute.LOCAL:
            return False
        prefill = int(candidate.prefill_index)
        decoder = int(candidate.decoder_index)
        self._mesh_source_prefill_owned[prefill] += (
            candidate.work.remote_prefill_token_ms)
        edge = self._mesh_edges[(prefill, decoder)]
        edge.held_remote_prefill_token_ms += (
            candidate.work.remote_prefill_token_ms)
        edge.held_remote_kv_bytes += candidate.work.remote_kv_bytes
        edge.held_remote_semantic_ops += candidate.work.remote_semantic_ops
        edge.inflight_transfers += 1
        _, source_before, edge_before = self._mesh_virtual_finish(candidate)
        source_capacity = self._capacities[prefill]
        receiver_capacity = self._capacities[decoder]
        self._mesh_source_virtual_service[prefill] = (
            source_before
            + candidate.work.remote_prefill_token_ms
            / max(1, source_capacity.remote_prefill_token_ms)
        )
        self._mesh_edge_virtual_service[(prefill, decoder)] = (
            edge_before
            + max(
                candidate.work.remote_kv_bytes
                / max(1, receiver_capacity.remote_kv_bytes),
                candidate.work.remote_semantic_ops
                / max(1, receiver_capacity.remote_semantic_ops),
            )
        )
        return True

    def _release_mesh_stage_locked(
        self,
        reservation: _Reservation,
        *,
        now_ns: int,
        completed_first_response: bool,
    ) -> None:
        if not reservation.mesh_stage_held:
            return
        candidate = reservation.candidate
        prefill = int(candidate.prefill_index)
        decoder = int(candidate.decoder_index)
        edge = self._mesh_edges[(prefill, decoder)]
        values = {
            "source_prefill": (
                self._mesh_source_prefill_owned[prefill]
                - candidate.work.remote_prefill_token_ms),
            "edge_prefill": (
                edge.held_remote_prefill_token_ms
                - candidate.work.remote_prefill_token_ms),
            "edge_kv": (
                edge.held_remote_kv_bytes - candidate.work.remote_kv_bytes),
            "edge_semantic": (
                edge.held_remote_semantic_ops
                - candidate.work.remote_semantic_ops),
            "edge_inflight": edge.inflight_transfers - 1,
        }
        if any(value < 0 for value in values.values()):
            raise RuntimeError("mesh receiver-credit ownership underflow")
        self._mesh_source_prefill_owned[prefill] = values["source_prefill"]
        edge.held_remote_prefill_token_ms = values["edge_prefill"]
        edge.held_remote_kv_bytes = values["edge_kv"]
        edge.held_remote_semantic_ops = values["edge_semantic"]
        edge.inflight_transfers = values["edge_inflight"]
        if completed_first_response:
            latency_ms = (now_ns - reservation.committed_ns) / 1_000_000.0
            if latency_ms < 0.0:
                raise RuntimeError("mesh first response precedes route commit")
            alpha = self.config.mesh_edge_service_ewma_alpha
            edge.first_response_ewma_ms = (
                latency_ms
                if edge.first_response_ewma_ms is None
                else alpha * latency_ms
                + (1.0 - alpha) * edge.first_response_ewma_ms
            )
            edge.completed_first_responses += 1
            edge.last_completion_ns = now_ns
        reservation.mesh_stage_held = False

    def _scheduler_pressure(self, pair: int) -> float:
        telemetry = self._telemetry.get(pair)
        if telemetry is None or telemetry.scheduler_running_requests is None:
            return 0.0
        requests = (
            telemetry.scheduler_running_requests
            + telemetry.scheduler_waiting_requests
        )
        return requests / self._capacities[pair].active_sequences

    def _scheduler_queue_penalty_ms(
        self, pair: int, candidate: RouteCandidate,
    ) -> float:
        """Project measured decoder occupancy into one queue-wave cost.

        The fixed utilization penalty is intentionally small and symmetric; it
        is useful for tie-breaking but cannot represent a decoder whose
        scheduler is already full.  A candidate consumes one first-response
        service window on its destination decoder, so the current
        running+waiting occupancy is converted into that candidate's TTFT
        prior.  This remains an online scheduler observation, not a phase or
        future-arrival hint.  Missing scheduler telemetry contributes zero and
        is still handled by the existing fail-closed admission checks.
        """

        telemetry = self._telemetry.get(pair)
        if (
            telemetry is None
            or telemetry.scheduler_running_requests is None
            or telemetry.scheduler_waiting_requests is None
        ):
            return 0.0
        capacity = self._capacities[pair].active_sequences
        occupancy = min(
            1.0,
            max(
                0.0,
                (
                    telemetry.scheduler_running_requests
                    + telemetry.scheduler_waiting_requests
                ) / capacity,
            ),
        )
        return occupancy * candidate.predicted_ttft_ms

    def _mesh_cool_remote_ttft_credit_ms(
        self, candidate: RouteCandidate, request: GlobalRequest,
    ) -> float:
        """Credit remote TTFT only in a measured live hot/cool crossover."""

        if (
            not self._mesh_enabled()
            or candidate.route is not GlobalRoute.REMOTE
            or candidate.prefill_index is None
            or candidate.decoder_index is None
        ):
            return 0.0
        source_pressure = self._scheduler_pressure(candidate.prefill_index)
        destination_pressure = self._scheduler_pressure(candidate.decoder_index)
        threshold = self.config.mesh_cool_remote_route_pressure_fraction
        if source_pressure > threshold or destination_pressure > threshold:
            return 0.0
        active_hot = any(
            self._scheduler_pressure(pair) >= threshold
            for pair in self._active_pairs
        )
        if not active_hot:
            return 0.0
        local_reference = next(
            (
                item for item in request.candidates
                if item.route is GlobalRoute.LOCAL
                and item.decoder_index == candidate.decoder_index
            ),
            None,
        )
        if local_reference is None:
            return 0.0
        return max(
            0.0,
            local_reference.predicted_ttft_ms - candidate.predicted_ttft_ms,
        )

    def _completion_pressure(self, pair: int) -> float:
        """Return observed endpoint residual pressure for this pair.

        vLLM's waiting gauge is not a complete service signal: an endpoint can
        have requests in first-response/transfer completion while its visible
        queue is empty.  The completion snapshot is therefore an independent
        pressure term, not folded into ``observed_total`` and not treated as a
        physical fabric counter.
        """

        telemetry = self._telemetry.get(pair)
        if telemetry is None or telemetry.endpoint_residual_inflight is None:
            return 0.0
        capacity = self._capacities[pair].endpoint_requests
        return telemetry.endpoint_residual_inflight / capacity

    def _service_feasibility_forecast(
        self,
        candidate: RouteCandidate,
        *,
        effective_used: ResourceVector,
    ) -> tuple[float, float, tuple[str, ...]] | None:
        """Forecast completion from currently observed service waves.

        The forecast intentionally uses only state already installed in the
        controller: decoder running/waiting work, endpoint residual/queue
        work, and (for a mesh remote candidate) source/edge/receiver work.
        ``max`` is used across overlapping observations so the same request
        is not counted once as a scheduler request and again as completion
        debt.  It is a conservative admission guard, not a claimed physical
        service-rate model.
        """

        if self.config.service_feasibility_mode == "disabled":
            return None
        pair = int(candidate.decoder_index)
        telemetry = self._telemetry[pair]
        capacity = self._capacities[pair]
        candidate_ttft = max(1e-9, candidate.predicted_ttft_ms)
        candidate_service = max(1e-9, candidate.predicted_e2e_ms)
        waves: list[tuple[str, float]] = []

        decoder_ahead = float(effective_used.active_sequences)
        if telemetry.scheduler_running_requests is not None:
            decoder_ahead = max(
                decoder_ahead,
                float(
                    telemetry.scheduler_running_requests
                    + int(telemetry.scheduler_waiting_requests or 0)
                ),
            )
        waves.append((
            "decode_tokens",
            decoder_ahead / max(1, capacity.active_sequences),
        ))

        endpoint_ahead = float(effective_used.endpoint_requests)
        if telemetry.endpoint_residual_inflight is not None:
            endpoint_ahead = max(
                endpoint_ahead,
                float(telemetry.endpoint_residual_inflight),
            )
        if telemetry.service_lane_queue_requests is not None:
            endpoint_ahead = max(
                endpoint_ahead,
                float(telemetry.service_lane_queue_requests),
            )
        if telemetry.service_lane_pending_global_commits is not None:
            endpoint_ahead = max(
                endpoint_ahead,
                float(
                    telemetry.service_lane_pending_global_commits
                    + telemetry.service_lane_active_reservations
                    + telemetry.service_lane_active_queue_leases
                ),
            )
        waves.append((
            "endpoint_requests",
            endpoint_ahead / max(1, capacity.endpoint_requests),
        ))

        if candidate.route is GlobalRoute.LOCAL:
            waves.append((
                "local_prefill_token_ms",
                effective_used.local_prefill_token_ms
                / max(1, capacity.local_prefill_token_ms),
            ))
        elif self._mesh_enabled():
            prefill = int(candidate.prefill_index)
            source_capacity = self._capacities[prefill]
            waves.append((
                "remote_prefill_token_ms",
                self._mesh_source_used(prefill)
                / max(1, source_capacity.remote_prefill_token_ms),
            ))
            edge = self._mesh_edges[(prefill, pair)]
            waves.append((
                "remote_semantic_ops",
                max(
                    edge.inflight_transfers,
                    edge.held_remote_semantic_ops,
                    effective_used.remote_semantic_ops,
                ) / max(
                    1,
                    capacity.remote_semantic_ops
                    - self.config.remote_semantic_ops_safety_reserve,
                ),
            ))
            waves.append((
                "remote_kv_bytes",
                max(
                    edge.held_remote_kv_bytes,
                    effective_used.remote_kv_bytes,
                ) / max(1, capacity.remote_kv_bytes),
            ))

        wave = max((value for _, value in waves), default=0.0)
        queue_delay_ms = (
            wave
            * candidate_ttft
            * self.config.service_forecast_safety_factor
        )
        multiplier = telemetry.multiplier(candidate.route)
        if (
            self._mesh_enabled()
            and candidate.route is GlobalRoute.REMOTE
        ):
            multiplier = max(
                multiplier,
                self._telemetry[int(candidate.prefill_index)].remote_service_multiplier,
            )
        forecast_ms = (
            queue_delay_ms
            + candidate_service * multiplier
            + candidate.uncertainty_ms
        )
        bindings = tuple(sorted({
            name for name, value in waves if value >= wave and value > 0.0
        }))
        return queue_delay_ms, forecast_ms, bindings

    def _pair_pressure(self, pair: int) -> float:
        values = [
            self._effective_used(pair).dominant_ratio(
                self._capacities[pair]),
            self._scheduler_pressure(pair),
            self._completion_pressure(pair),
        ]
        telemetry = self._telemetry.get(pair)
        if telemetry is not None and telemetry.cross_layer is not None:
            externality, _contributions, _confidence = (
                telemetry.cross_layer.route_externality(GlobalRoute.REMOTE))
            values.append(min(1.0, externality / 1000.0))
        return max(values)

    def _fully_quarantined_pairs(self) -> set[int]:
        return {
            pair
            for pair in self._capacities
            if all(
                (pair, route) in self._route_quarantines
                for route in (GlobalRoute.LOCAL, GlobalRoute.REMOTE)
            )
        }

    def _survivor_reserve_active(self, pair: int) -> bool:
        if self.config.survivor_capacity_reserve_fraction <= 0.0:
            return False
        failed_pairs = self._fully_quarantined_pairs()
        return bool(failed_pairs) and pair not in failed_pairs

    def _tenant_can_bypass_survivor_reserve(
        self, request: GlobalRequest, now_ns: int
    ) -> bool:
        policy = self._tenants[request.tenant_id]
        waited = max(0, now_ns - request.arrival_ns)
        wait_budget = self.admission_wait_budget_ns(request.tenant_id)
        if waited >= wait_budget:
            return True
        total_service = sum(self._tenant_service_units.values())
        service_fraction = (
            self._tenant_service_units[request.tenant_id] / total_service
            if total_service > 0.0 else 0.0
        )
        if service_fraction < policy.minimum_service_fraction:
            return True
        return policy.weight >= self.config.survivor_reserve_bypass_min_weight

    def _survivor_capacity_limit(
        self, pair: int
    ) -> ResourceVector:
        fraction = self.config.survivor_capacity_reserve_fraction
        capacity = self._capacities[pair]
        return ResourceVector(**{
            name: max(
                1,
                math.floor(getattr(capacity, name) * (1.0 - fraction)),
            )
            for name in ResourceVector.names()
        })

    def _expire_tenant_pair_assignments_locked(self, now_ns: int) -> None:
        """Expire idle business-pair scopes without observing future work."""

        busy_tenants = {
            request.tenant_id for request in self._queued.values()
        } | {
            reservation.request.tenant_id
            for reservation in self._inflight.values()
        }
        for tenant_id, assigned in self._tenant_pair_assignments.items():
            if not assigned or tenant_id in busy_tenants:
                continue
            last_busy = self._tenant_pair_last_busy_ns[tenant_id]
            expired = {
                pair for pair in assigned
                if (
                    pair not in last_busy
                    or now_ns - last_busy[pair]
                    >= self.config.scale_down_idle_ns
                )
            }
            assigned.difference_update(expired)
            for pair in expired:
                last_busy.pop(pair, None)

    def _tenant_pair_spread_rejection(
        self, request: GlobalRequest, candidate: RouteCandidate,
    ) -> RejectedCandidate | None:
        policy = self._tenants[request.tenant_id]
        limit = policy.pair_spread_limit
        assigned = self._tenant_pair_assignments[request.tenant_id]
        if (
            limit is not None
            and len(assigned) >= limit
            and candidate.pair_index not in assigned
        ):
            return RejectedCandidate.from_candidate(
                candidate,
                "tenant_pair_spread_limit",
                ("tenant_pair_scope",),
            )
        return None

    def _assign_tenant_pair_locked(
        self, tenant_id: str, pair: int, now_ns: int,
    ) -> bool:
        policy = self._tenants[tenant_id]
        limit = policy.pair_spread_limit
        if limit is None:
            return False
        assigned = self._tenant_pair_assignments[tenant_id]
        newly_assigned = pair not in assigned
        if newly_assigned:
            if len(assigned) >= limit:
                raise RuntimeError(
                    "route commit exceeds tenant pair spread limit")
            assigned.add(pair)
        self._tenant_pair_last_busy_ns[tenant_id][pair] = now_ns
        return newly_assigned

    def _touch_tenant_pair_locked(
        self, tenant_id: str, pair: int, now_ns: int,
    ) -> None:
        if pair in self._tenant_pair_assignments[tenant_id]:
            self._tenant_pair_last_busy_ns[tenant_id][pair] = now_ns

    def _lower_priority_packed_tenants(
        self, tenant_id: str, pair: int,
    ) -> tuple[str, ...]:
        priority = self._tenants[tenant_id].admission_priority
        return tuple(sorted(
            other_id
            for other_id, other in self._tenants.items()
            if (
                other.admission_priority < priority
                and other.pair_spread_limit is not None
                and pair in self._tenant_pair_assignments[other_id]
            )
        ))

    def _business_clean_candidate(
        self, request: GlobalRequest, evaluation: _CandidateEvaluation,
    ) -> bool:
        return not self._lower_priority_packed_tenants(
            request.tenant_id, evaluation.candidate.pair_index)

    def _prefer_business_clean_evaluations(
        self,
        request: GlobalRequest,
        evaluations: list[_CandidateEvaluation],
        rejected: list[RejectedCandidate],
    ) -> list[_CandidateEvaluation]:
        clean = [
            item for item in evaluations
            if self._business_clean_candidate(request, item)
        ]
        if not clean:
            return evaluations
        # Pair isolation is a business preference, not a capacity guarantee.
        # The old unconditional filter kept a protected MISS workload on one
        # decoder even after that decoder's live service pressure was high;
        # the v9 Perlmutter receipt shows the resulting 8.7 s miss-hot tail.
        # At the configured current-state threshold, retain both clean and
        # packed candidates so the normal global score can spread work.  This
        # does not alter cache-affinity semantics: a candidate whose proven
        # cache placement forbids another route is still rejected by the
        # ordinary cache/edge guards below.
        if min(item.utilization for item in clean) >= (
            self.config.business_clean_pair_pressure_fraction
        ):
            return evaluations
        for item in evaluations:
            if item not in clean:
                rejected.append(RejectedCandidate.from_candidate(
                    item.candidate,
                    "higher_priority_clean_pair_available",
                    ("tenant_pair_isolation",),
                ))
        return clean

    def _protected_capacity_fraction(self, tenant_id: str) -> float:
        """Return the strongest higher-priority tenant's reserve.

        Reserves are intentionally static and pair-local.  This prevents a
        burst of background work from consuming the capacity needed by an
        interactive/latency request, while still allowing each lower-priority
        class to use its unreserved share.  The reservation is not a queue
        hint and is therefore applied to both immediate admission and an
        explicit endpoint queue lease.
        """

        policy = self._tenants[tenant_id]
        return max(
            (
                float(other.protected_capacity_fraction)
                for other in self._tenants.values()
                if other.admission_priority > policy.admission_priority
            ),
            default=0.0,
        )

    def _tenant_capacity_limit(
        self, tenant_id: str, capacity: ResourceVector
    ) -> ResourceVector:
        fraction = self._protected_capacity_fraction(tenant_id)
        if fraction <= 0.0:
            return capacity
        return ResourceVector(**{
            name: max(
                1,
                math.floor(getattr(capacity, name) * (1.0 - fraction)),
            )
            for name in ResourceVector.names()
        })

    def _telemetry_age_ns(self, pair: int, now_ns: int) -> int | None:
        telemetry = self._telemetry.get(pair)
        if telemetry is None or telemetry.sampled_ns > now_ns:
            return None
        return now_ns - telemetry.sampled_ns

    def _telemetry_fresh(self, pair: int, now_ns: int) -> bool:
        age = self._telemetry_age_ns(pair, now_ns)
        return age is not None and age <= self.config.telemetry_fresh_ns

    def _telemetry_admissible(
        self, pair: int, now_ns: int, *, tenant_id: str | None = None,
    ) -> bool:
        """Return whether the installed snapshot is safe to use for admission.

        The normal path requires fresh telemetry.  The optional grace is a
        control-plane continuity mechanism for a transient refresh timeout;
        it is bounded by allocation policy and does not bypass any physical
        capacity, route health, shared remote budget, or critical transport
        guard checked later in candidate evaluation.
        """

        age = self._telemetry_age_ns(pair, now_ns)
        if age is None:
            return False
        grace_ns = self.config.telemetry_stale_grace_ns
        if tenant_id is not None:
            try:
                grace_ns = max(
                    grace_ns, self._tenants[tenant_id].telemetry_stale_grace_ns)
            except KeyError as exc:
                raise ValueError("request tenant is not configured") from exc
        return age <= self.config.telemetry_fresh_ns + grace_ns

    def telemetry_admission_available(
        self, *, now_ns: int, tenant_id: str | None = None,
    ) -> bool:
        """Whether every pair has a usable snapshot for one business tenant."""

        _positive_int("now_ns", now_ns, zero=True)
        with self._lock:
            return all(
                self._telemetry_admissible(
                    pair, now_ns, tenant_id=tenant_id)
                for pair in self._capacities
            )

    def _endpoint_queue_lease_cooldown_reason(
        self, pair: int
    ) -> str | None:
        sequence = self._endpoint_queue_lease_cooldowns.get(pair)
        if sequence is None:
            return None
        telemetry = self._telemetry.get(pair)
        if telemetry is None or telemetry.sequence <= sequence:
            return "endpoint_queue_lease_cooldown"
        waiting = telemetry.scheduler_waiting_requests
        if waiting is None or waiting > 0:
            return "endpoint_queue_lease_cooldown"
        del self._endpoint_queue_lease_cooldowns[pair]
        return None

    def _endpoint_scheduler_queue_headroom(self, pair: int) -> bool:
        """Return whether one bounded endpoint queue lease is supportable.

        The endpoint service window is an SLO-safe first-response credit, not
        the same thing as the vLLM scheduler's waiting queue.  A queue lease
        may cross the former only when the latter reports a bounded slot and
        previously leased debt has not consumed it.  Missing scheduler
        telemetry remains fail-closed; fabric hard guards are checked by the
        caller before this method is used.
        """

        telemetry = self._telemetry.get(pair)
        if telemetry is None or telemetry.scheduler_waiting_requests is None:
            return False
        queue_occupancy = self._endpoint_queue_occupancy(pair)
        # ``endpoint_requests`` is the active route reservation and must stay
        # a hard global resource.  It is not the size of vLLM's waiting
        # queue: under real contention vLLM can report dozens of waiting
        # requests while its active sequence window remains 16.  Queue debt
        # is therefore bounded by the explicit, business-approved
        # ``endpoint_queue_capacity``.  Keeping these two windows separate
        # lets the global controller remain work-conserving without silently
        # converting scheduler backlog into an unbounded route reservation.
        queue_capacity = int(self.config.endpoint_queue_capacity)
        return queue_occupancy < queue_capacity

    def _endpoint_queue_occupancy(self, pair: int) -> int:
        """Return one de-duplicated downstream queue occupancy estimate.

        ``scheduler_waiting_requests`` and TEMPO's queue-lease debt describe
        the same downstream work from different observers.  Newer endpoint
        agents additionally report their own queue records and two-phase
        offers.  Use the maximum of the overlapping steady-state counters,
        then add only the transient offer/commit states and active first-
        response reservations.  This keeps global admission conservative
        without counting one request three times.
        """

        telemetry = self._telemetry.get(pair)
        if telemetry is None:
            return 0
        scheduler_waiting = telemetry.scheduler_waiting_requests
        queue_lease_debt = self._endpoint_queue_lease_debt(pair)
        queue_requests = telemetry.service_lane_queue_requests
        if any(value is None for value in (
            scheduler_waiting, queue_requests,
            telemetry.service_lane_queue_offers,
            telemetry.service_lane_pending_global_commits,
        )):
            # Preserve the legacy contract for old endpoint agents.
            return (
                (scheduler_waiting or 0)
                + (telemetry.endpoint_residual_inflight or 0)
                + queue_lease_debt
            )
        steady = max(scheduler_waiting, queue_requests, queue_lease_debt)
        transient = (
            telemetry.service_lane_queue_offers
            + telemetry.service_lane_pending_global_commits
        )
        return steady + (telemetry.endpoint_residual_inflight or 0) + transient

    def _priority_service_lane_binding(self) -> str:
        if (
            self.config.priority_service_lane_mode
            == BUSINESS_DUAL_ROUTE_PRIORITY_SERVICE_LANE_MODE
        ):
            return BUSINESS_PRIORITY_SERVICE_LANE_BINDING
        return PRIORITY_SERVICE_LANE_BINDING

    def _priority_service_lane_reason(self, *, promoted: bool) -> str:
        suffix = "promoted" if promoted else "route_committed"
        if (
            self.config.priority_service_lane_mode
            == BUSINESS_DUAL_ROUTE_PRIORITY_SERVICE_LANE_MODE
        ):
            return f"global_priority_business_dual_route_service_lane_{suffix}"
        return f"global_priority_remote_cache_service_lane_{suffix}"

    def _priority_service_lane_candidate(
        self, request: GlobalRequest, candidate: RouteCandidate,
    ) -> bool:
        """Whether one candidate is authorized to ask for priority service.

        This is intentionally based only on the frozen business policy and
        the candidate's proven cache/route semantics.  Workload phase names,
        benchmark labels, and future arrivals are not policy inputs.
        """

        policy = self._tenants[request.tenant_id]
        mode = self.config.priority_service_lane_mode
        if not (
            policy.queue_lease_on_timeout
            and policy.admission_priority
            >= self.config.priority_service_lane_min_admission_priority
        ):
            return False
        if mode == REMOTE_CACHE_PRIORITY_SERVICE_LANE_MODE:
            return bool(
                candidate.route is GlobalRoute.REMOTE
                and candidate.cache_affinity
            )
        if mode == BUSINESS_DUAL_ROUTE_PRIORITY_SERVICE_LANE_MODE:
            return bool(
                candidate.route is GlobalRoute.LOCAL
                or (
                    candidate.route is GlobalRoute.REMOTE
                    and candidate.cache_affinity
                )
            )
        return False

    def _priority_service_lane_debt(self, pair: int) -> int:
        """Return live globally-owned priority slots for one decoder."""

        return sum(
            reservation.held.endpoint_requests
            for reservation in self._inflight.values()
            if (
                reservation.candidate.pair_index == pair
                and reservation.decision.queue_lease
                and self._priority_service_lane_binding()
                in reservation.decision.binding_resources
                and reservation.phase is GlobalRequestPhase.ROUTE_COMMITTED
            )
        )

    def _priority_service_lane_headroom(
        self, request: GlobalRequest, candidate: RouteCandidate,
    ) -> bool:
        """Whether a fresh decoder snapshot supports one priority lease."""

        if not self._priority_service_lane_candidate(request, candidate):
            return False
        telemetry = self._telemetry.get(candidate.pair_index)
        if telemetry is None or any(value is None for value in (
            telemetry.scheduler_running_requests,
            telemetry.scheduler_waiting_requests,
            telemetry.endpoint_completed_first_responses,
            telemetry.endpoint_residual_inflight,
        )):
            return False
        return self._priority_service_lane_debt(candidate.pair_index) < (
            self.config.priority_service_lane_capacity)

    def _protected_service_lane_key(
        self, candidate: RouteCandidate,
    ) -> tuple[int, int, GlobalRoute]:
        """Return the physical service lane identity for one P/D edge."""

        return (
            int(candidate.prefill_index),
            int(candidate.decoder_index),
            candidate.route,
        )

    def _protected_service_lane_debt(
        self, candidate: RouteCandidate,
    ) -> int:
        """Count active protected service slots on this exact P/D edge."""

        key = self._protected_service_lane_key(candidate)
        return sum(
            reservation.held.active_sequences
            for reservation in self._inflight.values()
            if (
                reservation.candidate.identity_key == key
                and reservation.decision.protected_service_lane
            )
        )

    def _protected_service_lane_guard(
        self,
        request: GlobalRequest,
        candidate: RouteCandidate,
        *,
        effective_used: ResourceVector,
        already_owned: bool = False,
    ) -> tuple[bool, str, int, int] | RejectedCandidate:
        """Apply an atomic tenant/edge service-lane reservation.

        A protected request consumes a slot on its exact local or P->D edge.
        Other tenants may use only the residual endpoint and decoder slots, so
        the reservation exists before a victim arrives.  For remote work the
        same residual rule is applied to the edge semantic-operation ledger;
        this prevents a receiver reservation from being bypassed by a
        different source feeding the same destination.
        """

        if self.config.protected_service_lane_mode == "disabled":
            return False, "", 0, 0
        key = self._protected_service_lane_key(candidate)
        key_text = (
            f"local:d{key[1]}"
            if key[2] is GlobalRoute.LOCAL
            else f"remote:p{key[0]}->d{key[1]}"
        )
        before = self._protected_service_lane_debt(candidate)
        capacity = self.config.protected_service_lane_capacity
        policy = self._tenants[request.tenant_id]
        protected = (
            policy.admission_priority
            >= self.config.protected_service_lane_min_admission_priority
        )
        destination_work = self._destination_work(candidate)
        after = effective_used if already_owned else effective_used + destination_work
        if protected:
            projected = (
                before
                if already_owned
                else before + max(
                    destination_work.active_sequences,
                    destination_work.endpoint_requests,
                )
            )
            # v1 treated the protected reserve as a second hard concurrency
            # ceiling.  That makes a lane with capacity=2 reject the third
            # protected request even when the physical endpoint still has
            # room; under a real service wave this converts protection into
            # throughput loss.  v2 makes the field a lower-priority reserve:
            # protected work may consume physical capacity, while the normal
            # capacity/endpoint/deadline guards below remain authoritative.
            if (
                self.config.protected_service_lane_mode
                == PROTECTED_SERVICE_LANE_MODE
                and projected > capacity
            ):
                return RejectedCandidate.from_candidate(
                    candidate,
                    "protected_service_lane_exhausted",
                    (PROTECTED_SERVICE_LANE_BINDING, key_text),
                )
            return True, key_text, before, projected

        # Keep both the running decoder slot and the endpoint admission slot
        # available for a protected request.  This is a reservation, not a
        # second physical capacity vector, so ordinary capacity checks still
        # run after this guard.
        reserved = max(0, capacity - before)
        limits = {
            "active_sequences": self._capacities[int(candidate.decoder_index)]
            .active_sequences - reserved,
            "endpoint_requests": self._capacities[int(candidate.decoder_index)]
            .endpoint_requests - reserved,
        }
        binding = tuple(
            name for name, limit in limits.items()
            if getattr(after, name) > limit
        )
        if binding:
            return RejectedCandidate.from_candidate(
                candidate,
                "protected_service_lane_reserve",
                tuple(binding) + (PROTECTED_SERVICE_LANE_BINDING, key_text),
            )
        if candidate.route is GlobalRoute.REMOTE:
            edge = self._mesh_edges[key[:2]]
            edge_after = (
                edge.held_remote_semantic_ops
                if already_owned else edge.held_remote_semantic_ops
                + destination_work.remote_semantic_ops
            )
            edge_limit = max(
                1,
                self._capacities[int(candidate.decoder_index)]
                .remote_semantic_ops
                - self.config.remote_semantic_ops_safety_reserve
                - reserved,
            )
            if edge_after > edge_limit:
                return RejectedCandidate.from_candidate(
                    candidate,
                    "protected_service_lane_edge_reserve",
                    (
                        "remote_semantic_ops",
                        PROTECTED_SERVICE_LANE_BINDING,
                        key_text,
                    ),
                )
        return False, key_text, before, before

    def _endpoint_queue_lease_debt(self, pair: int) -> int:
        """Return live TEMPO-owned endpoint queue debt for one pair."""

        return sum(
            reservation.held.endpoint_requests
            for reservation in self._inflight.values()
            if (
                reservation.candidate.pair_index == pair
                and reservation.decision.queue_lease
                and reservation.phase is GlobalRequestPhase.ROUTE_COMMITTED
            )
        )

    def _completion_liveness_queue_delay_ms(
        self, pair: int, candidate: RouteCandidate,
    ) -> float:
        """Project bounded queue delay from observed completion headroom.

        Scheduler waiting and endpoint residual can overlap, so their maximum
        is the conservative non-double-counted work ahead.  TEMPO-owned queue
        leases are then added explicitly.  Dividing by the endpoint first-
        response window yields queue waves; the request's frozen TTFT prior is
        the service time per wave.  This is a causal capacity projection, not
        a phase label or a fitted fabric coefficient.
        """

        telemetry = self._telemetry[pair]
        if (
            telemetry.scheduler_waiting_requests is None
            or telemetry.endpoint_residual_inflight is None
        ):
            raise RuntimeError("completion liveness telemetry is incomplete")
        work_ahead = self._endpoint_queue_occupancy(pair)
        waves = work_ahead / max(
            1, self._capacities[pair].endpoint_requests)
        return waves * candidate.predicted_ttft_ms

    def _completion_liveness_probe_inflight(
        self, pair: int, route: GlobalRoute,
    ) -> bool:
        """Keep failure-free recovery to one first-response probe per route."""

        return any(
            reservation.candidate.pair_index == pair
            and reservation.candidate.route is route
            and reservation.decision.queue_lease
            and reservation.phase is GlobalRequestPhase.ROUTE_COMMITTED
            for reservation in self._inflight.values()
        )

    def _cross_layer_group_key(self, pair: int) -> str | None:
        telemetry = self._telemetry.get(pair)
        if telemetry is None or telemetry.cross_layer is None:
            return None
        cross_layer = telemetry.cross_layer
        return "|".join((
            cross_layer.source_epoch,
            cross_layer.topology_fingerprint_sha256,
            cross_layer.communicator_id,
        ))

    def _cross_layer_derived_value(
        self, cross_layer: CrossLayerTelemetry, name: str
    ) -> float | None:
        """Read a derived fabric value without inventing missing telemetry."""

        if name == "cassini_by_nic_pause_fraction_max":
            return cross_layer.cassini_nic_pause_max()
        return None

    def _local_receiver_externality_ms(
        self, cross_layer: CrossLayerTelemetry,
    ) -> float:
        """Price observed receiver pressure for a pair-local route.

        LOCAL avoids a new inter-pair KV transfer, but it still executes on
        the decoder pair whose receiver may be sharing the NCCL/LMCache
        service window.  This term is enabled only by an explicit profile
        value and only from a supported pair-scoped LMCache observer signal;
        missing or communicator-only evidence contributes zero.
        """

        price = self.config.cross_layer_local_receiver_price_ms
        if price <= 0.0:
            return 0.0
        signal = cross_layer.signal("lmcache_transfer_p99_ms")
        if (
            signal is None
            or signal.support != "supported"
            or signal.value is None
            or signal.scope != "pair"
        ):
            return 0.0
        return max(0.0, float(signal.value)) * price

    def _remote_receiver_guard_binding(
        self, cross_layer: CrossLayerTelemetry,
    ) -> tuple[str, ...]:
        """Return a hard receiver admission binding for a hot pair.

        A transfer-tail observation is actionable only when it is supported
        and pair-scoped.  Communicator-wide or missing observations cannot
        identify the destination receiver and therefore remain score-only or
        unsupported.  This guard is intentionally separate from the latency
        shadow price: once a pair's receiver service is beyond its configured
        ceiling, admitting more remote work there is not work-conserving; it
        increases the native queue/timeout risk for all later victims.
        """

        if (
            self.config.cross_layer_remote_receiver_guard_mode
            != "deny_while_hot"
        ):
            return ()
        signal = cross_layer.signal("lmcache_transfer_p99_ms")
        if (
            signal is None
            or signal.support != "supported"
            or signal.value is None
            or signal.scope != "pair"
        ):
            return ()
        if float(signal.value) >= self.config.cross_layer_remote_receiver_guard_p99_ms:
            return ("lmcache_transfer_p99_ms",)
        return ()

    def _remote_receiver_group_guard_binding(
        self, pair: int,
    ) -> tuple[str, ...]:
        """Guard every remote edge in a proven shared receiver group.

        Pair-local guarding is insufficient for an incast: after ``P0->D0``
        is denied, the selector can immediately choose ``P1->D1`` while the
        same allocation-wide receiver service is still overloaded.  This
        mode is fail-closed and only acts when every member has a usable
        pair-scoped transfer-tail sample in the same installed telemetry
        epoch.  Missing evidence never becomes congestion by assumption.
        """

        if (
            self.config.cross_layer_remote_receiver_guard_mode
            != "deny_while_hot"
            or self.config.cross_layer_remote_receiver_guard_scope
            != "shared_group"
        ):
            return ()
        group = self._cross_layer_group_key(pair)
        declared_group = self.config.cross_layer_remote_receiver_guard_group_id
        if declared_group:
            members = tuple(sorted(self._capacities))
        elif group is not None:
            members = tuple(
                other for other in sorted(self._capacities)
                if self._cross_layer_group_key(other) == group
            )
        else:
            return ()
        if len(members) < 2:
            return ()
        hot_members: list[int] = []
        for other in members:
            if other not in self._telemetry:
                continue
            cross_layer = self._telemetry[other].cross_layer
            if cross_layer is None:
                continue
            signal = cross_layer.signal("lmcache_transfer_p99_ms")
            if (
                signal is None
                or signal.support != "supported"
                or signal.value is None
                or signal.scope != "pair"
            ):
                continue
            if float(signal.value) >= self.config.cross_layer_remote_receiver_guard_p99_ms:
                hot_members.append(other)
        if not hot_members:
            return ()
        return (
            "lmcache_transfer_p99_ms",
            "shared_receiver_group",
            *(
                (f"receiver_guard_group:{declared_group}",)
                if declared_group else ()
            ),
            *(f"hot_receiver_pair:{other}" for other in hot_members),
        )

    def _shared_scale_suppressed_for_pair(
        self, pair: int, budget: _SharedRemoteBudget
    ) -> bool:
        """Keep shared-budget fail-closed semantics unless NICs disambiguate.

        A communicator-wide pressure signal alone cannot prove that a
        prewarmed spare pair has an independent service path.  When the
        producer supplies a per-NIC Cassini vector, however, a cool candidate
        pair is actionable evidence for global load balancing even if another
        NIC in the communicator is hot.  This is the only case in which
        shared pressure may open the spare pair.
        """

        if not budget.suppress_pair_activation:
            return False
        telemetry = self._telemetry.get(pair)
        if telemetry is None or telemetry.cross_layer is None:
            return True
        nic_pause = telemetry.cross_layer.cassini_nic_pause_max()
        if nic_pause is None:
            return True
        return nic_pause >= 0.20

    def _shared_remote_usage(self, group: str) -> tuple[int, int, int]:
        requests = 0
        kv_bytes = 0
        semantic_ops = 0
        for reservation in self._inflight.values():
            if reservation.candidate.route is not GlobalRoute.REMOTE:
                continue
            if self._cross_layer_group_key(reservation.candidate.pair_index) != group:
                continue
            requests += reservation.held.endpoint_requests
            kv_bytes += reservation.held.remote_kv_bytes
            semantic_ops += reservation.held.remote_semantic_ops
        return requests, kv_bytes, semantic_ops

    def _shared_remote_budget_binding(
        self,
        candidate: RouteCandidate,
        budget: _SharedRemoteBudget | None,
        *,
        already_owned: bool = False,
    ) -> tuple[str, ...]:
        """Return the shared resources that are hard-bound for this route.

        In the soft-shadow-price contract, transfer latency is a cost signal
        and not proof that the shared remote data plane has lost capacity.
        The old path nevertheless treated the configured KV/semantic-op
        targets as hard admission caps, so a transient LMCache p99 sample
        could reject work that the native vLLM waiting queue could still
        serve.  Keep the shared request count as a bounded business/admission
        guard, and make KV/semantic-op targets hard only when an explicit
        transport-pressure envelope is limited.  Hard-window v1 and a
        limited v3 envelope retain the fail-closed behavior.
        """

        if candidate.route is not GlobalRoute.REMOTE or budget is None:
            return ()
        request_delta = 0 if already_owned else candidate.work.endpoint_requests
        kv_delta = 0 if already_owned else candidate.work.remote_kv_bytes
        semantic_delta = (
            0 if already_owned else candidate.work.remote_semantic_ops)
        used = {
            "shared_remote_requests": budget.requests_used
            + request_delta
            > budget.requests_limit,
            "shared_remote_kv_bytes": budget.kv_bytes_used
            + kv_delta
            > budget.kv_bytes_limit,
            "shared_remote_semantic_ops": budget.semantic_ops_used
            + semantic_delta
            > budget.semantic_ops_limit,
        }
        if (
            self.config.cross_layer_control_mode == "soft_shadow_price_v2"
            and not budget.limited
            and self.config.endpoint_queue_debt_mode
            != "completion_credit_endpoint_queue_v3"
        ):
            # Requests remain a bounded global ingress/fairness guard.  The
            # byte/op targets are soft shadow-price targets until Cassini or
            # another explicitly supported hard signal crosses its limit.
            names = ("shared_remote_requests",)
        else:
            names = tuple(used)
        return tuple(name for name in names if used[name])

    def _shared_remote_budget_for_pair(
        self, pair: int
    ) -> _SharedRemoteBudget | None:
        if self.config.shared_fabric_control_mode != "global_budget_v3":
            return None
        group = self._cross_layer_group_key(pair)
        if group is None:
            return None
        members = self._shared_group_members.get(group)
        if members is None:
            members = tuple(sorted(
                other for other in self._capacities
                if self._cross_layer_group_key(other) == group
            ))
            self._shared_group_members[group] = members
        # A one-pair sample cannot identify shared externality.  Keep the
        # pair-local v1/v2 actuator in charge until a complete shared group is
        # present in the atomic telemetry batch.
        if len(members) < 2:
            return None

        cached = self._shared_budget_static.get(group)
        if cached is not None and cached.members == members:
            used = self._shared_remote_usage(group)
            return replace(
                cached,
                requests_used=used[0],
                kv_bytes_used=used[1],
                semantic_ops_used=used[2],
            )

        capacities = {
            "requests": sum(
                self._capacities[index].endpoint_requests
                for index in members),
            "kv_bytes": sum(
                self._capacities[index].remote_kv_bytes
                for index in members),
            "semantic_ops": sum(
                self._capacities[index].remote_semantic_ops
                for index in members),
        }
        configured = {
            "requests": self.config.shared_remote_requests_capacity,
            "kv_bytes": self.config.shared_remote_kv_bytes_capacity,
            "semantic_ops": self.config.shared_remote_semantic_ops_capacity,
        }
        for name in capacities:
            if configured[name] > 0:
                capacities[name] = configured[name]

        signal_specs = {
            "requests": (
                ("lmcache_transfer_p99_ms", 50.0),
                ("nccl_collective_p99_ms", 20.0),
                ("nccl_arrival_spread_ms", 5.0),
                ("cassini_rx_pause_fraction_max", 0.20),
                ("cassini_tx_pause_fraction_max", 0.20),
                ("cassini_ecn_fraction_max", 0.10),
                ("cassini_retries", 10.0),
                ("cassini_timeouts", 2.0),
                ("cassini_by_nic_pause_fraction_max", 0.20),
            ),
            "kv_bytes": (
                ("lmcache_transfer_p99_ms", 50.0),
                ("lmcache_remote_kv_bytes_inflight", float(
                    max(1, max(
                        self._capacities[index].remote_kv_bytes
                        for index in members)))),
            ),
            "semantic_ops": (
                ("lmcache_transfer_p99_ms", 50.0),
                ("lmcache_remote_semantic_ops_inflight", float(
                    max(1, max(
                        self._capacities[index].remote_semantic_ops
                        for index in members)))),
            ),
        }
        contributions: list[tuple[str, float]] = []
        pressures: dict[str, float] = {}
        for resource, specs in signal_specs.items():
            resource_values: list[float] = []
            for name, normalization in specs:
                values: list[float] = []
                for member in members:
                    cross_layer = self._telemetry[member].cross_layer
                    if cross_layer is None:
                        continue
                    derived = self._cross_layer_derived_value(cross_layer, name)
                    if derived is not None:
                        values.append(max(0.0, float(derived) / normalization))
                        continue
                    signal = cross_layer.signal(name)
                    if (
                        signal is None
                        and name == "lmcache_remote_semantic_ops_inflight"
                    ):
                        signal = cross_layer.signal("lmcache_remote_ops_inflight")
                    if (
                        signal is not None
                        and signal.support == "supported"
                        and signal.value is not None
                    ):
                        values.append(max(0.0, float(signal.value) / normalization))
                if not values:
                    continue
                value = min(1.0, max(values))
                contributions.append((f"shared.{resource}.{name}", value))
                resource_values.append(max(values))
            pressures[resource] = max(resource_values, default=0.0)

        floor = float(self.config.shared_remote_limit_floor_fraction)

        def bounded_budget(base: int, pressure: float) -> int:
            excess = min(1.0, max(0.0, pressure - 1.0))
            return max(1, math.ceil(base * max(floor, 1.0 - 0.75 * excess)))

        # In the soft-shadow-price contract, transfer/NCCL latency is a cost
        # signal, not proof that the shared remote data plane has lost hard
        # capacity.  The previous v3 path applied the latency pressure to the
        # shared byte/op limits and collapsed the configured 1.878 GB/8-op
        # budget to its 25% floor during a 1.8 s transfer tail.  That made the
        # global controller reject work even though Cassini reported no pause,
        # ECN, retry, or timeout backpressure.  Preserve the configured shared
        # capacity in soft mode; hard transport pressure may still shrink it.
        hard_signal_names = {
            "cassini_rx_pause_fraction_max",
            "cassini_tx_pause_fraction_max",
            "cassini_by_nic_pause_fraction_max",
            "cassini_ecn_fraction_max",
            "cassini_retries",
            "cassini_timeouts",
        }
        hard_pressure = 0.0
        for resource_specs in signal_specs.values():
            for name, normalization in resource_specs:
                if name not in hard_signal_names:
                    continue
                values = []
                for member in members:
                    cross_layer = self._telemetry[member].cross_layer
                    if cross_layer is None:
                        continue
                    derived = self._cross_layer_derived_value(
                        cross_layer, name)
                    if derived is not None:
                        values.append(max(0.0, float(derived) / normalization))
                        continue
                    signal = cross_layer.signal(name)
                    if (
                        signal is not None
                        and signal.support == "supported"
                        and signal.value is not None
                    ):
                        values.append(max(0.0, float(signal.value) / normalization))
                hard_pressure = max(hard_pressure, max(values, default=0.0))

        if self.config.cross_layer_control_mode == "soft_shadow_price_v2":
            limits = {
                resource: bounded_budget(capacities[resource], hard_pressure)
                if hard_pressure > 1.0 else capacities[resource]
                for resource in capacities
            }
        else:
            limits = {
                resource: bounded_budget(capacities[resource], pressures[resource])
                for resource in capacities
            }
        used = self._shared_remote_usage(group)
        limited = any(limits[name] < capacities[name] for name in limits)
        shared_pressure_present = max(pressures.values(), default=0.0) > 0.25
        request_pressure = pressures["requests"]
        stagger = 0
        if request_pressure > 0.25:
            stagger = min(
                self.config.shared_remote_stagger_max_us,
                max(
                    1,
                    round(
                        self.config.shared_remote_stagger_max_us
                        * (min(1.0, request_pressure) - 0.25) / 0.75
                    ),
                ),
            )
        budget = _SharedRemoteBudget(
            group=group,
            members=members,
            requests_limit=limits["requests"],
            kv_bytes_limit=limits["kv_bytes"],
            semantic_ops_limit=limits["semantic_ops"],
            requests_used=used[0],
            kv_bytes_used=used[1],
            semantic_ops_used=used[2],
            dispatch_stagger_us=stagger,
            contributions=tuple(sorted(contributions)),
            limited=limited,
            # Shared pressure is not evidence that a new pair has a new
            # service path.  Pair-local health/capacity rejection remains the
            # only route to spare-pair activation under this policy.
            # Shared latency pressure should not make an indistinguishable
            # spare pair look like independent fabric capacity.  A per-NIC
            # vector can still clear this suppression for a demonstrably cool
            # path in _shared_scale_suppressed_for_pair().
            suppress_pair_activation=limited or shared_pressure_present,
        )
        self._shared_budget_static[group] = replace(
            budget,
            requests_used=0,
            kv_bytes_used=0,
            semantic_ops_used=0,
        )
        return budget

    def _joint_actuation_plan(
        self,
        candidate: RouteCandidate,
        *,
        telemetry: PairTelemetry,
        capacity: ResourceVector,
        after: ResourceVector,
        shared_budget: _SharedRemoteBudget | None = None,
    ) -> JointActuationPlan | None:
        """Translate the live cross-layer vector into bounded action limits.

        This is intentionally action-specific.  NCCL collective values affect
        both routes because both execute tensor-parallel collectives.  Cassini
        endpoint pressure and LMCache transfer values affect only REMOTE,
        which injects cross-node KV traffic; LOCAL is the explicit fabric
        avoidance path.  The vector is retained in the
        plan receipt; the maximum is used only to choose a conservative limit
        and a bounded stagger.  Unsupported values contribute nothing and
        reduce confidence rather than becoming zero-valued congestion.
        """

        cross_layer = telemetry.cross_layer
        if cross_layer is None:
            return None

        common = (
            ("nccl_collective_p99_ms", 20.0),
            ("nccl_arrival_spread_ms", 5.0),
        )
        nic_pause = cross_layer.cassini_nic_pause_max()
        action_signals = list(common)
        # Pair-scoped LMCache receiver pressure is priced in the route score
        # by ``_local_receiver_externality_ms``.  It must not also become a
        # local dispatch sleep: the observation describes the already-running
        # receiver service window, while staggering a local GPU request would
        # delay the very fabric-avoiding escape route.  Local action
        # limits/stagger remain driven by NCCL and Cassini signals that local
        # work actually shares.
        if candidate.route is GlobalRoute.REMOTE:
            action_signals.extend((
                ("cassini_host_posted_cycles_per_packet_max", 24.0),
                ("cassini_ecn_fraction_max", 0.10),
                ("cassini_retries", 10.0),
                ("cassini_timeouts", 2.0),
                ("lmcache_transfer_p99_ms", 50.0),
                (
                    "lmcache_remote_semantic_ops_inflight",
                    float(max(1, capacity.remote_semantic_ops)),
                ),
                (
                    "lmcache_remote_kv_bytes_inflight",
                    float(max(1, capacity.remote_kv_bytes)),
                ),
            ))
            if nic_pause is not None:
                action_signals.append(
                    ("cassini_by_nic_pause_fraction_max", 0.20))
            else:
                action_signals.extend((
                    ("cassini_rx_pause_fraction_max", 0.20),
                    ("cassini_tx_pause_fraction_max", 0.20),
                ))

        contributions: list[tuple[str, float]] = []
        raw_contributions: list[float] = []
        raw_by_name: dict[str, float] = {}

        for name, normalization in action_signals:
            derived = self._cross_layer_derived_value(cross_layer, name)
            if derived is not None:
                value = float(derived)
                raw_contribution = max(0.0, value / normalization)
                contribution = min(1.0, raw_contribution)
                contributions.append((name, contribution))
                raw_contributions.append(raw_contribution)
                raw_by_name[name] = raw_contribution
                continue
            signal = cross_layer.signal(name)
            if (
                candidate.route is GlobalRoute.LOCAL
                and name == "lmcache_transfer_p99_ms"
                and signal is not None
                and signal.scope != "pair"
            ):
                # Communicator-wide transfer latency cannot identify which
                # decoder pair owns the receiver pressure.  It is preserved
                # in the raw envelope but cannot actuate LOCAL.
                continue
            if (
                signal is None
                and name == "lmcache_remote_semantic_ops_inflight"
            ):
                signal = cross_layer.signal("lmcache_remote_ops_inflight")
            if (
                signal is None
                or signal.support != "supported"
                or signal.value is None
            ):
                continue
            value = float(signal.value)
            raw_contribution = max(0.0, value / normalization)
            contribution = min(1.0, raw_contribution)
            contributions.append((name, contribution))
            raw_contributions.append(raw_contribution)
            raw_by_name[name] = raw_contribution

        considered = len(action_signals)
        confidence = len(contributions) / considered if considered else 0.0
        dominant = max((value for _, value in contributions), default=0.0)
        # A normalization is the safe-envelope boundary, not a target at
        # which useful capacity should already be discarded.  The previous
        # implementation applied ``1 - 0.75 * dominant`` even when every
        # observed value was below its boundary.  In a real co-job this made
        # an NCCL p99 of 11.8 ms (against a 20 ms envelope) remove roughly
        # 44% of local-prefill capacity and then turn ordinary queueing into
        # global 503 rejects.  Preserve the full window below the envelope;
        # only excess pressure can shrink the corresponding action window.
        excess_pressure = min(
            1.0,
            max((max(0.0, value - 1.0) for value in raw_contributions), default=0.0),
        )
        floor_fraction = (
            self.config.cross_layer_remote_limit_floor_fraction
            if candidate.route is GlobalRoute.REMOTE
            else self.config.cross_layer_local_limit_floor_fraction
        )
        # Keep a useful service window while making the live vector capable of
        # changing the actual admission boundary.  A pressure value is not
        # exported as policy state; only these independent limits are.
        limit_fraction = max(
            floor_fraction,
            1.0 - 0.75 * excess_pressure,
        )

        def bounded_limit(value: int) -> int:
            return max(1, min(value, math.ceil(value * limit_fraction)))

        remote_semantic_capacity = max(
            1,
            capacity.remote_semantic_ops
            - self.config.remote_semantic_ops_safety_reserve,
        )
        stagger = 0
        if self.config.cross_layer_stagger_max_us > 0 and dominant > 0.25:
            stagger = min(
                self.config.cross_layer_stagger_max_us,
                max(
                    1,
                    round(
                        self.config.cross_layer_stagger_max_us
                        * (dominant - 0.25) / 0.75
                    ),
                ),
            )
        action_targets = {
            "local_prefill_token_ms": bounded_limit(
                capacity.local_prefill_token_ms),
            "remote_prefill_token_ms": bounded_limit(
                capacity.remote_prefill_token_ms),
            "remote_kv_bytes": bounded_limit(capacity.remote_kv_bytes),
            "remote_semantic_ops": bounded_limit(remote_semantic_capacity),
        }
        # Transport failures and explicit fabric backpressure are safety
        # signals.  Latency inflation (NCCL/L缓存 transfer p99) is a cost,
        # not a reason to make the global admission loop non-work-conserving.
        critical_names = {
            "cassini_rx_pause_fraction_max",
            "cassini_tx_pause_fraction_max",
            "cassini_by_nic_pause_fraction_max",
            "cassini_ecn_fraction_max",
            "cassini_retries",
            "cassini_timeouts",
        }
        critical_guard = any(
            raw_by_name.get(name, 0.0)
            >= self.config.cross_layer_critical_pressure_fraction
            for name in critical_names
        )
        soft_overage_resources = tuple(
            name for name in action_targets
            if getattr(after, name) > action_targets[name]
        )
        overage_fraction = min(
            1.0,
            max(
                (
                    max(0.0, getattr(after, name) - action_targets[name])
                    / max(1, action_targets[name])
                    for name in action_targets
                ),
                default=0.0,
            ),
        )
        overage_penalty_ms = (
            self.config.cross_layer_shadow_price_ms * overage_fraction
            if self.config.cross_layer_control_mode == "soft_shadow_price_v2"
            else 0.0
        )

        def enforced_limit(name: str) -> int:
            # The global ledger may intentionally carry over-capacity debt for
            # an endpoint queue lease, but the endpoint controller rejects a
            # resource limit above its physical service window.  Keep the two
            # meanings separate: the global decision/receipt retains the
            # overage, while the downstream controller receives only a valid
            # per-endpoint window and can use its own queue until service
            # capacity is released.
            return min(
                getattr(capacity, name),
                max(action_targets[name], getattr(after, name)),
            )

        schema = (
            JOINT_ACTUATION_SCHEMA_V3
            if shared_budget is not None
            else JOINT_ACTUATION_SCHEMA_V2
            if self.config.cross_layer_control_mode == "soft_shadow_price_v2"
            else JOINT_ACTUATION_SCHEMA
        )
        action_mode = (
            "shared_budget_v3"
            if schema == JOINT_ACTUATION_SCHEMA_V3
            else self.config.cross_layer_control_mode
        )
        shared_action = "none"
        if shared_budget is not None:
            if shared_budget.limited:
                shared_action = "global_remote_budget"
            elif shared_budget.dispatch_stagger_us:
                shared_action = "global_remote_stagger"
        # The shared v3 budget is a remote-transfer actuator.  Applying its
        # stagger to a local-prefill request made a high LMCache p99 delay
        # local GPU work even when Cassini reported no hard backpressure.
        # Keep the pair-local vector stagger available for shared NCCL/Cassini
        # signals, but scope the shared remote stagger to remote work.  A
        # remote budget receipt remains attached to the decision for audit;
        # it must not silently become a global ingress sleep.
        shared_dispatch_stagger = (
            shared_budget.dispatch_stagger_us
            if shared_budget is not None
            and candidate.route is GlobalRoute.REMOTE
            else 0
        )
        return JointActuationPlan(
            pair_index=candidate.pair_index,
            route=candidate.route,
            local_prefill_token_ms_limit=action_targets[
                "local_prefill_token_ms"],
            remote_prefill_token_ms_limit=action_targets[
                "remote_prefill_token_ms"],
            remote_kv_bytes_limit=action_targets["remote_kv_bytes"],
            remote_semantic_ops_limit=action_targets["remote_semantic_ops"],
            dispatch_stagger_us=max(
                stagger,
                shared_dispatch_stagger,
            ),
            # The commit header carries the global, atomically installed
            # PairTelemetry batch sequence.  The cross-layer envelope has
            # its own producer sequence; retaining that producer identity in
            # telemetry provenance is correct, but mixing it into the joint
            # commit makes the router reject otherwise valid decisions.
            telemetry_sequence=telemetry.sequence,
            confidence=confidence,
            signal_contributions=tuple(sorted(contributions)),
            schema=schema,
            action_mode=action_mode,
            critical_guard=critical_guard,
            enforced_local_prefill_token_ms_limit=(
                enforced_limit("local_prefill_token_ms")
                if schema in {
                    JOINT_ACTUATION_SCHEMA_V2,
                    JOINT_ACTUATION_SCHEMA_V3,
                } else None),
            enforced_remote_prefill_token_ms_limit=(
                enforced_limit("remote_prefill_token_ms")
                if schema in {
                    JOINT_ACTUATION_SCHEMA_V2,
                    JOINT_ACTUATION_SCHEMA_V3,
                } else None),
            enforced_remote_kv_bytes_limit=(
                enforced_limit("remote_kv_bytes")
                if schema in {
                    JOINT_ACTUATION_SCHEMA_V2,
                    JOINT_ACTUATION_SCHEMA_V3,
                } else None),
            enforced_remote_semantic_ops_limit=(
                enforced_limit("remote_semantic_ops")
                if schema in {
                    JOINT_ACTUATION_SCHEMA_V2,
                    JOINT_ACTUATION_SCHEMA_V3,
                } else None),
            overage_fraction=overage_fraction,
            overage_penalty_ms=overage_penalty_ms,
            soft_overage_resources=soft_overage_resources,
            shared_fabric_group=(
                shared_budget.group if shared_budget is not None else None),
            shared_remote_requests_limit=(
                shared_budget.requests_limit
                if shared_budget is not None else None),
            shared_remote_kv_bytes_limit=(
                shared_budget.kv_bytes_limit
                if shared_budget is not None else None),
            shared_remote_semantic_ops_limit=(
                shared_budget.semantic_ops_limit
                if shared_budget is not None else None),
            shared_remote_requests_used_before=(
                shared_budget.requests_used
                if shared_budget is not None else None),
            shared_remote_kv_bytes_used_before=(
                shared_budget.kv_bytes_used
                if shared_budget is not None else None),
            shared_remote_semantic_ops_used_before=(
                shared_budget.semantic_ops_used
                if shared_budget is not None else None),
            shared_budget_action=shared_action,
            shared_budget_contributions=(
                shared_budget.contributions
                if shared_budget is not None else ()),
        )

    def _failure_free_stale_feedback_fallback(
        self, telemetry: PairTelemetry, route: GlobalRoute,
    ) -> bool:
        """Return whether fresh global evidence can cover stale route feedback.

        ``PathHealth.SKIP`` in endpoint telemetry means its last service
        sample aged past the endpoint feedback window.  It is not a transport
        or service failure.  A receiver-credit P-by-D controller already owns
        fresh scheduler, endpoint-completion, capacity, and deadline checks;
        when those causal signals are present and the route has no explicit
        failure, treating feedback age as a permanent path veto can strand an
        idle prewarmed decoder.  Non-mesh profiles retain the historical
        fail-closed behavior, and DENIED/failed routes never enter this path.
        """

        if (
            not self._mesh_enabled()
            or telemetry.health(route) is not PathHealth.SKIP
        ):
            return False
        failures = (
            telemetry.local_failure_count
            if route is GlobalRoute.LOCAL else telemetry.remote_failure_count
        )
        return bool(
            failures == 0
            and telemetry.scheduler_running_requests is not None
            and telemetry.scheduler_waiting_requests is not None
            and telemetry.endpoint_completed_first_responses is not None
            and telemetry.endpoint_completed_first_responses > 0
            and telemetry.endpoint_residual_inflight is not None
        )

    def _evaluate_candidate(
        self, candidate: RouteCandidate, *, request: GlobalRequest, now_ns: int
    ) -> _CandidateEvaluation | RejectedCandidate:
        pair = int(candidate.decoder_index)
        prefill = int(candidate.prefill_index)
        tenant_policy = self._tenants[request.tenant_id]
        spread_rejection = self._tenant_pair_spread_rejection(
            request, candidate)
        if spread_rejection is not None:
            return spread_rejection
        if candidate.predicted_ttft_ms > tenant_policy.ttft_slo_ms:
            return RejectedCandidate.from_candidate(
                candidate, "tenant_ttft_slo")
        if candidate.predicted_e2e_ms > tenant_policy.e2e_slo_ms:
            return RejectedCandidate.from_candidate(
                candidate, "tenant_e2e_slo")
        if not self._telemetry_admissible(
            pair, now_ns, tenant_id=request.tenant_id):
            return RejectedCandidate.from_candidate(
                candidate, "telemetry_missing_or_stale")
        if (
            self._mesh_enabled()
            and candidate.route is GlobalRoute.REMOTE
            and not self._telemetry_admissible(
                prefill, now_ns, tenant_id=request.tenant_id)
        ):
            return RejectedCandidate.from_candidate(
                candidate,
                "mesh_source_telemetry_missing_or_stale",
                ("source_telemetry",),
            )
        cache_group_rejection = self._cache_group_rejection_locked(
            request, candidate)
        if cache_group_rejection is not None:
            return cache_group_rejection
        telemetry = self._telemetry[pair]
        if candidate.route is GlobalRoute.REMOTE:
            receiver_guard_binding = self._remote_receiver_guard_binding(
                telemetry.cross_layer
            ) if telemetry.cross_layer is not None else ()
            if receiver_guard_binding:
                return RejectedCandidate.from_candidate(
                    candidate,
                    "cross_layer_remote_receiver_hot",
                    receiver_guard_binding,
                )
            receiver_group_guard_binding = (
                self._remote_receiver_group_guard_binding(pair))
            if receiver_group_guard_binding:
                return RejectedCandidate.from_candidate(
                    candidate,
                    "cross_layer_remote_receiver_group_hot",
                    receiver_group_guard_binding,
                )
        if (pair, candidate.route) in self._route_quarantines:
            return RejectedCandidate.from_candidate(
                candidate,
                "route_failure_quarantine",
                ("route_failure_quarantine",),
            )
        health = telemetry.health(candidate.route)
        stale_feedback_fallback = False
        if health in {PathHealth.SKIP, PathHealth.DENIED}:
            if self._failure_free_stale_feedback_fallback(
                telemetry, candidate.route
            ):
                stale_feedback_fallback = True
            else:
                guard_reason = (
                    "remote_pre_admission_guard"
                    if (
                        candidate.route is GlobalRoute.REMOTE
                        and telemetry.remote_failure_count > 0
                    )
                    else f"path_{health.value}"
                )
                return RejectedCandidate.from_candidate(candidate, guard_reason)
        source_stale_feedback_fallback = False
        if (
            self._mesh_enabled()
            and candidate.route is GlobalRoute.REMOTE
        ):
            if (prefill, pair) in self._mesh_edge_quarantines:
                return RejectedCandidate.from_candidate(
                    candidate,
                    "mesh_edge_failure_quarantine",
                    ("mesh_edge_failure_quarantine",),
                )
            source_health = self._telemetry[prefill].remote_health
            if source_health in {PathHealth.SKIP, PathHealth.DENIED}:
                source_telemetry = self._telemetry[prefill]
                if self._failure_free_stale_feedback_fallback(
                    source_telemetry, GlobalRoute.REMOTE
                ):
                    source_stale_feedback_fallback = True
                    stale_feedback_fallback = True
                else:
                    return RejectedCandidate.from_candidate(
                        candidate,
                        f"mesh_source_path_{source_health.value}",
                        ("source_remote_health",),
                    )
        mesh_rejection = self._mesh_candidate_rejection(candidate)
        if mesh_rejection is not None:
            return mesh_rejection
        effective = self._effective_destination_used(candidate)
        capacity = self._capacities[pair]
        destination_work = self._destination_work(candidate)
        protected_lane = self._protected_service_lane_guard(
            request,
            candidate,
            effective_used=effective,
        )
        if isinstance(protected_lane, RejectedCandidate):
            return protected_lane
        (
            protected_service_lane,
            protected_service_lane_key,
            protected_service_lane_before,
            protected_service_lane_after,
        ) = protected_lane
        service_forecast = self._service_feasibility_forecast(
            candidate, effective_used=effective)
        service_queue_delay_ms = None
        service_forecast_ms = None
        if service_forecast is not None:
            (
                service_queue_delay_ms,
                service_forecast_ms,
                service_bindings,
            ) = service_forecast
            remaining_ms = (
                self._effective_deadline_ns(request) - now_ns
            ) / 1_000_000.0
            if service_forecast_ms > remaining_ms:
                return RejectedCandidate.from_candidate(
                    candidate,
                    "global_service_lane_slo_infeasible",
                    tuple(service_bindings) or ("service_lane",),
                )
        after = effective + destination_work
        shared_budget = self._shared_remote_budget_for_pair(pair)
        if shared_budget is not None:
            shared_binding = self._shared_remote_budget_binding(
                candidate, shared_budget)
            if shared_binding:
                return RejectedCandidate.from_candidate(
                    candidate,
                    "shared_remote_budget",
                    shared_binding,
                )
        if candidate.route is GlobalRoute.REMOTE:
            semantic_limit = (
                capacity.remote_semantic_ops
                - self.config.remote_semantic_ops_safety_reserve)
            if after.remote_semantic_ops > semantic_limit:
                return RejectedCandidate.from_candidate(
                    candidate,
                    "remote_semantic_ops_admission_guard",
                    ("remote_semantic_ops_safety_reserve",),
                )
        binding = tuple(
            name for name in ResourceVector.names()
            if getattr(after, name) > getattr(capacity, name)
        )
        if binding:
            return RejectedCandidate.from_candidate(
                candidate, "capacity", binding)
        protected_capacity = self._tenant_capacity_limit(
            request.tenant_id, capacity)
        protected_binding = tuple(
            name for name in ResourceVector.names()
            if getattr(after, name) > getattr(protected_capacity, name)
        )
        if protected_binding:
            return RejectedCandidate.from_candidate(
                candidate,
                "tenant_protected_capacity_reserve",
                protected_binding,
            )
        if (
            self._survivor_reserve_active(pair)
            and not self._tenant_can_bypass_survivor_reserve(request, now_ns)
        ):
            survivor_limit = self._survivor_capacity_limit(pair)
            reserved_binding = tuple(
                name for name in ResourceVector.names()
                if getattr(after, name) > getattr(survivor_limit, name)
            )
            if reserved_binding:
                return RejectedCandidate.from_candidate(
                    candidate,
                    "survivor_capacity_reserve",
                    reserved_binding,
                )
        utilization = max(
            after.dominant_ratio(capacity),
            self._scheduler_pressure(pair),
            self._completion_pressure(pair),
        )
        if self._mesh_enabled() and candidate.route is GlobalRoute.REMOTE:
            source_capacity = self._capacities[prefill]
            source_after = (
                self._mesh_source_used(prefill)
                + candidate.work.remote_prefill_token_ms
            )
            source_utilization = (
                source_after / source_capacity.remote_prefill_token_ms)
            receiver_limit = max(
                1,
                capacity.remote_semantic_ops
                - self.config.remote_semantic_ops_safety_reserve,
            )
            receiver_utilization = (
                self._mesh_receiver_inflight(pair) + 1) / receiver_limit
            utilization = max(
                utilization,
                source_utilization,
                receiver_utilization,
            )
        cross_layer_externality_ms = 0.0
        cross_layer_confidence = 0.0
        if telemetry.cross_layer is not None:
            (
                cross_layer_externality_ms,
                _cross_layer_contributions,
                cross_layer_confidence,
            ) = telemetry.cross_layer.route_externality(candidate.route)
            if candidate.route is GlobalRoute.LOCAL:
                cross_layer_externality_ms += (
                    self._local_receiver_externality_ms(
                        telemetry.cross_layer))
            utilization = max(
                utilization,
                min(1.0, cross_layer_externality_ms / 1000.0),
            )
        if (
            self._mesh_enabled()
            and candidate.route is GlobalRoute.REMOTE
            and prefill != pair
            and self._telemetry[prefill].cross_layer is not None
        ):
            source_externality, _source_contributions, _source_confidence = (
                self._telemetry[prefill].cross_layer.route_externality(
                    GlobalRoute.REMOTE))
            cross_layer_externality_ms += source_externality
            utilization = max(
                utilization,
                min(1.0, source_externality / 1000.0),
            )
        joint_actuation = self._joint_actuation_plan(
            candidate,
            telemetry=telemetry,
            capacity=capacity,
            after=after,
            shared_budget=shared_budget,
        )
        if joint_actuation is not None:
            action_binding = []
            if (
                after.local_prefill_token_ms
                > joint_actuation.local_prefill_token_ms_limit
            ):
                action_binding.append("cross_layer_local_prefill_limit")
            if (
                after.remote_prefill_token_ms
                > joint_actuation.remote_prefill_token_ms_limit
            ):
                action_binding.append("cross_layer_remote_prefill_limit")
            if after.remote_kv_bytes > joint_actuation.remote_kv_bytes_limit:
                action_binding.append("cross_layer_remote_kv_limit")
            if (
                after.remote_semantic_ops
                > joint_actuation.remote_semantic_ops_limit
            ):
                action_binding.append("cross_layer_remote_semantic_limit")
            if action_binding and (
                joint_actuation.action_mode == "hard_window_v1"
                or joint_actuation.critical_guard
            ):
                return RejectedCandidate.from_candidate(
                    candidate,
                    "cross_layer_joint_actuation_limit",
                    tuple(action_binding),
                )
        cross_layer_scale_required = bool(
            joint_actuation is not None
            and joint_actuation.soft_overage_resources
            and not (
                shared_budget is not None
                and self._shared_scale_suppressed_for_pair(pair, shared_budget)
            )
        )
        multiplier = (
            1.0
            if stale_feedback_fallback and health is PathHealth.SKIP
            else telemetry.multiplier(candidate.route)
        )
        if self._mesh_enabled() and candidate.route is GlobalRoute.REMOTE:
            source_multiplier = (
                1.0
                if source_stale_feedback_fallback
                else self._telemetry[prefill].remote_service_multiplier
            )
            multiplier = max(
                multiplier,
                source_multiplier,
            )
        inflation = candidate.predicted_ttft_ms * (multiplier - 1.0)
        edge_feedback_penalty_ms = self._mesh_edge_feedback_penalty_ms(
            candidate)
        receiver_stagger_us = self._mesh_receiver_stagger_us(candidate)
        score = (
            candidate.predicted_e2e_ms
            + inflation
            + candidate.uncertainty_ms
            + utilization * self.config.utilization_penalty_ms
            + self._scheduler_queue_penalty_ms(pair, candidate)
            + (service_queue_delay_ms or 0.0)
            - self._mesh_cool_remote_ttft_credit_ms(candidate, request)
            + cross_layer_externality_ms
            + edge_feedback_penalty_ms
            + receiver_stagger_us / 1_000.0
            + (
                joint_actuation.dispatch_stagger_us / 1_000.0
                if joint_actuation is not None else 0.0
            )
            + (
                joint_actuation.overage_penalty_ms
                if joint_actuation is not None else 0.0
            )
        )
        activate = pair not in self._active_pairs
        if activate:
            score += self.config.activation_penalty_ms
        if health is PathHealth.PROBE or stale_feedback_fallback:
            score += self.config.probe_penalty_ms
        effective_deadline_ns = self._effective_deadline_ns(request)
        slack = (effective_deadline_ns - now_ns) / 1_000_000 - score
        if slack < 0.0:
            return RejectedCandidate.from_candidate(candidate, "deadline")
        return _CandidateEvaluation(
            candidate=candidate,
            score_ms=score,
            slack_ms=slack,
            effective_used=effective,
            utilization=utilization,
            activate=activate,
            joint_actuation=joint_actuation,
            cross_layer_scale_required=cross_layer_scale_required,
            shared_scale_suppressed=bool(
                shared_budget is not None
                and self._shared_scale_suppressed_for_pair(pair, shared_budget)
            ),
            stale_feedback_fallback=stale_feedback_fallback,
            receiver_stagger_us=receiver_stagger_us,
            service_queue_delay_ms=service_queue_delay_ms,
            service_forecast_ms=service_forecast_ms,
            protected_service_lane=protected_service_lane,
            protected_service_lane_key=protected_service_lane_key
            if protected_service_lane else None,
            protected_service_lane_before=protected_service_lane_before
            if protected_service_lane else None,
            protected_service_lane_after=protected_service_lane_after
            if protected_service_lane else None,
        )

    def _evaluate_queue_lease_candidate(
        self,
        candidate: RouteCandidate,
        *,
        request: GlobalRequest,
        now_ns: int,
        already_owned: bool = False,
        allow_completion_liveness_bootstrap: bool = False,
    ) -> _CandidateEvaluation | RejectedCandidate:
        """Evaluate a bounded native-endpoint queue lease.

        A queue lease is deliberately narrower than ordinary admission.  It
        is used only after the global reservation window expires, and it may
        not waive tenant service SLOs, explicit path failures, telemetry
        freshness, remote semantic execution limits, or critical cross-layer
        guards.  It only waives the *global reservation capacity* because the
        downstream vLLM scheduler already owns a waiting queue.  The v2 mode
        may turn failure-free stale completion feedback into one bounded
        recovery probe.  The resulting overage remains in global ownership
        and in the immutable decision receipt so subsequent work sees the debt
        instead of silently bypassing TEMPO.
        """

        def rejected(
            reason: str, binding_resources: tuple[str, ...] = (),
        ) -> RejectedCandidate:
            return RejectedCandidate.from_candidate(
                candidate, reason, binding_resources)

        mesh_queue_lease = (
            self._mesh_enabled()
            and self.config.endpoint_queue_debt_mode
            == "completion_credit_mesh_endpoint_queue_v1"
        )
        if self._mesh_enabled() and not mesh_queue_lease:
            # C6 normally uses explicit pre-prefill defer/reject.  Reusing a
            # downstream queue lease without the opt-in mesh mode would bypass
            # source/edge matching and turn receiver credit into hidden debt.
            return RejectedCandidate.from_candidate(
                candidate,
                "mesh_endpoint_queue_lease_disabled",
                ("mesh_receiver_credit",),
        )
        pair = candidate.pair_index
        prefill = int(candidate.prefill_index)
        tenant_policy = self._tenants[request.tenant_id]
        priority_service_lane = self._priority_service_lane_headroom(
            request, candidate)
        spread_rejection = self._tenant_pair_spread_rejection(
            request, candidate)
        if spread_rejection is not None:
            return spread_rejection
        if candidate.predicted_ttft_ms > tenant_policy.ttft_slo_ms:
            return rejected("tenant_ttft_slo")
        if candidate.predicted_e2e_ms > tenant_policy.e2e_slo_ms:
            return rejected("tenant_e2e_slo")
        if not self._telemetry_admissible(
            pair, now_ns, tenant_id=request.tenant_id):
            return rejected("telemetry_missing_or_stale")
        source_stale_feedback_fallback = False
        if mesh_queue_lease and candidate.route is GlobalRoute.REMOTE:
            if not self._telemetry_admissible(
                prefill, now_ns, tenant_id=request.tenant_id
            ):
                return rejected(
                    "mesh_source_telemetry_missing_or_stale",
                    ("source_telemetry",),
                )
            if (prefill, pair) in self._mesh_edge_quarantines:
                return rejected(
                    "mesh_edge_failure_quarantine",
                    ("mesh_edge_failure_quarantine",),
                )
            source_telemetry = self._telemetry[prefill]
            source_health = source_telemetry.remote_health
            if source_health in {PathHealth.SKIP, PathHealth.DENIED}:
                if self._failure_free_stale_feedback_fallback(
                    source_telemetry, GlobalRoute.REMOTE
                ):
                    source_stale_feedback_fallback = True
                else:
                    return rejected(
                        f"mesh_source_path_{source_health.value}",
                        ("source_remote_health",),
                    )
        cache_group_rejection = self._cache_group_rejection_locked(
            request, candidate)
        if cache_group_rejection is not None:
            return cache_group_rejection
        cooldown_reason = self._endpoint_queue_lease_cooldown_reason(pair)
        if cooldown_reason is not None:
            return rejected(cooldown_reason)
        if (pair, candidate.route) in self._route_quarantines:
            return rejected(
                "route_failure_quarantine", ("route_failure_quarantine",))
        telemetry = self._telemetry[pair]
        if candidate.route is GlobalRoute.REMOTE:
            receiver_guard_binding = self._remote_receiver_guard_binding(
                telemetry.cross_layer
            ) if telemetry.cross_layer is not None else ()
            if receiver_guard_binding:
                return rejected(
                    "cross_layer_remote_receiver_hot",
                    receiver_guard_binding,
                )
            receiver_group_guard_binding = (
                self._remote_receiver_group_guard_binding(pair))
            if receiver_group_guard_binding:
                return rejected(
                    "cross_layer_remote_receiver_group_hot",
                    receiver_group_guard_binding,
                )
        health = telemetry.health(candidate.route)
        completion_liveness_probe = False
        completion_liveness_shared_probe = False
        endpoint_queue_headroom_admission = False
        endpoint_queue_deadline_grace = False
        if health is PathHealth.DENIED:
            return rejected(f"path_{health.value}")
        failure_count = (
            telemetry.local_failure_count
            if candidate.route is GlobalRoute.LOCAL else
            telemetry.remote_failure_count
        )
        endpoint_queue_deadline_grace = bool(
            self.config.endpoint_queue_headroom_admission_mode
            == "completion_progress_v1"
            and self._endpoint_scheduler_queue_headroom(pair)
            and telemetry.endpoint_completed_first_responses is not None
            and telemetry.endpoint_completed_first_responses > 0
            and failure_count == 0
        )
        if health is PathHealth.SKIP:
            if (
                self.config.endpoint_queue_debt_mode not in {
                    "completion_liveness_endpoint_queue_v2",
                    "completion_credit_endpoint_queue_v3",
                    "completion_credit_mesh_endpoint_queue_v1",
                }
                or failure_count != 0
            ):
                return rejected("path_skip")
            if (
                telemetry.endpoint_completed_first_responses is None
                or telemetry.endpoint_completed_first_responses <= 0
                or telemetry.endpoint_residual_inflight is None
                or telemetry.scheduler_waiting_requests is None
            ):
                return rejected("completion_liveness_telemetry_missing")
            probe_inflight = self._completion_liveness_probe_inflight(
                pair, candidate.route
            )
            if probe_inflight:
                if (
                    self.config.completion_liveness_shared_probe_mode
                    != "headroom_shared_v1"
                ):
                    return rejected("completion_liveness_probe_inflight")
                # Reuse the already committed probe's failure-free evidence.
                # The native queue headroom check below remains mandatory, so
                # this mode cannot turn one probe into an unbounded bypass.
                completion_liveness_shared_probe = True
            if not (
                self._endpoint_scheduler_queue_headroom(pair)
                or priority_service_lane
            ):
                return rejected("endpoint_queue_capacity_full")
            if not completion_liveness_shared_probe:
                completion_liveness_probe = True

        completion_credit_mode = self.config.endpoint_queue_debt_mode in {
            "completion_credit_endpoint_queue_v3",
            "completion_credit_mesh_endpoint_queue_v1",
        }
        if completion_credit_mode:
            if (
                telemetry.endpoint_completed_first_responses is None
                or telemetry.endpoint_residual_inflight is None
                or telemetry.scheduler_waiting_requests is None
            ):
                return rejected("completion_credit_telemetry_missing")
            if (
                self._completion_credit_balance[pair] <= 0
                and not priority_service_lane
            ):
                failure_count = (
                    telemetry.local_failure_count
                    if candidate.route is GlobalRoute.LOCAL else
                    telemetry.remote_failure_count
                )
                bootstrap_key = (pair, candidate.route)
                bootstrap_available = bool(
                    allow_completion_liveness_bootstrap
                    and telemetry.endpoint_completed_first_responses > 0
                    and failure_count == 0
                    and not self._completion_liveness_probe_inflight(
                        pair, candidate.route)
                    and self._completion_liveness_bootstrap_sequences.get(
                        bootstrap_key) != telemetry.sequence
                )
                shared_probe_available = (
                    completion_liveness_shared_probe
                    and self._endpoint_scheduler_queue_headroom(pair)
                )
                endpoint_headroom_available = (
                    endpoint_queue_deadline_grace
                )
                if endpoint_headroom_available:
                    endpoint_queue_headroom_admission = True
                if not (
                    bootstrap_available
                    or shared_probe_available
                    or endpoint_headroom_available
                ):
                    return rejected("completion_credit_unavailable")
                if bootstrap_available:
                    completion_liveness_probe = True
            if not (
                self._endpoint_scheduler_queue_headroom(pair)
                or priority_service_lane
            ):
                return rejected("endpoint_queue_capacity_full")

        if mesh_queue_lease:
            # A mesh lease is allowed only when the destination has both
            # measured completion progress and a currently bounded endpoint
            # queue.  Source/edge/receiver credits are still hard: the lease
            # waives only destination reservation capacity.
            if (
                self._completion_credit_balance[pair] <= 0
                and not completion_liveness_probe
                and not completion_liveness_shared_probe
                and not endpoint_queue_headroom_admission
                and not priority_service_lane
            ):
                return rejected("completion_credit_unavailable")
            mesh_rejection = self._mesh_candidate_rejection(
                candidate, already_owned=already_owned)
            if mesh_rejection is not None:
                return mesh_rejection

        effective = self._effective_destination_used(candidate)
        capacity = self._capacities[pair]
        destination_work = self._destination_work(candidate)
        # A service-lane promotion begins after ordinary global admission;
        # the reservation is already present in ``_owned`` and (for remote
        # mesh work) in source/edge ledgers.  Re-adding it here would invent
        # a second request and reject the very debt transition being checked.
        after = effective if already_owned else effective + destination_work
        protected_lane = self._protected_service_lane_guard(
            request,
            candidate,
            effective_used=effective,
            already_owned=already_owned,
        )
        if isinstance(protected_lane, RejectedCandidate):
            return protected_lane
        (
            protected_service_lane,
            protected_service_lane_key,
            protected_service_lane_before,
            protected_service_lane_after,
        ) = protected_lane
        shared_budget = self._shared_remote_budget_for_pair(pair)
        if shared_budget is not None:
            shared_binding = self._shared_remote_budget_binding(
                candidate,
                shared_budget,
                already_owned=already_owned,
            )
            if shared_binding:
                return rejected("shared_remote_budget", shared_binding)
        if (
            candidate.route is GlobalRoute.REMOTE
            and self.config.endpoint_queue_debt_mode not in {
                "completion_liveness_endpoint_queue_v2",
                "completion_credit_endpoint_queue_v3",
            }
        ):
            semantic_limit = (
                capacity.remote_semantic_ops
                - self.config.remote_semantic_ops_safety_reserve)
            if after.remote_semantic_ops > semantic_limit:
                return rejected(
                    "remote_semantic_ops_admission_guard",
                    ("remote_semantic_ops_safety_reserve",),
                )
        protected_capacity = self._tenant_capacity_limit(
            request.tenant_id, capacity)
        # The ordinary reserve prevents controller-owned lower-priority work
        # from filling a higher-priority tenant's capacity.  Once unrelated
        # exogenous work has already filled the decoder snapshot, applying
        # that same max(observed, owned) test to a profile-bound priority lane
        # cannot restore the reserve; it only blocks the business request the
        # downstream priority scheduler is meant to rescue.  The lane has its
        # own per-decoder ownership cap and still retains every route/fabric/
        # mesh/deadline guard above and below this check.
        if protected_capacity != capacity and not priority_service_lane:
            protected_binding = tuple(
                name for name in ResourceVector.names()
                if getattr(after, name) > getattr(protected_capacity, name)
            )
            if protected_binding:
                return rejected(
                    "tenant_protected_capacity_reserve", protected_binding)
        # v3 does not waive the physical remote service lane blindly.  Its
        # one-shot completion credit, fresh scheduler headroom check, and the
        # endpoint_binding gate below jointly prove that one completed
        # first-response slot can be refilled.  Re-applying the stale
        # semantic occupancy window here double-counted that same slot and
        # caused v106 to reject useful remote work despite causal progress.
        if (
            self._survivor_reserve_active(pair)
            and not self._tenant_can_bypass_survivor_reserve(request, now_ns)
        ):
            survivor_limit = self._survivor_capacity_limit(pair)
            reserved_binding = tuple(
                name for name in ResourceVector.names()
                if getattr(after, name) > getattr(survivor_limit, name)
            )
            if reserved_binding:
                return rejected("survivor_capacity_reserve", reserved_binding)

        # A queue lease may carry decoder-window debt into vLLM's waiting
        # queue, but it must not carry endpoint service-lane debt into a
        # controller that has no physical route window left.  The previous
        # implementation treated every soft global overage as downstream
        # queueable.  In the native path that was false: the endpoint
        # controller rejected the request after global commit, producing an
        # HTTP 503 and requiring a compensating global release.  The global
        # telemetry and the immutable actuation plan already expose the
        # route-specific endpoint limits, so fail closed before ownership is
        # committed.  Decoder-only overage remains queue-lease eligible.

        utilization = max(
            after.dominant_ratio(capacity),
            self._scheduler_pressure(pair),
            self._completion_pressure(pair),
        )
        if mesh_queue_lease and candidate.route is GlobalRoute.REMOTE:
            source_capacity = self._capacities[prefill]
            source_after = self._mesh_source_used(prefill) + (
                0
                if already_owned
                else candidate.work.remote_prefill_token_ms
            )
            receiver_limit = max(
                1,
                capacity.remote_semantic_ops
                - self.config.remote_semantic_ops_safety_reserve,
            )
            utilization = max(
                utilization,
                source_after
                / max(1, source_capacity.remote_prefill_token_ms),
                (
                    self._mesh_receiver_inflight(pair)
                    + (0 if already_owned else 1)
                ) / receiver_limit,
            )
        cross_layer_externality_ms = 0.0
        if telemetry.cross_layer is not None:
            (
                cross_layer_externality_ms,
                _contributions,
                _confidence,
            ) = telemetry.cross_layer.route_externality(candidate.route)
            if candidate.route is GlobalRoute.LOCAL:
                cross_layer_externality_ms += (
                    self._local_receiver_externality_ms(
                        telemetry.cross_layer))
            utilization = max(
                utilization,
                min(1.0, cross_layer_externality_ms / 1000.0),
            )
        if (
            mesh_queue_lease
            and candidate.route is GlobalRoute.REMOTE
            and prefill != pair
            and self._telemetry[prefill].cross_layer is not None
        ):
            source_externality_ms, _source_contributions, _source_confidence = (
                self._telemetry[prefill].cross_layer.route_externality(
                    GlobalRoute.REMOTE)
            )
            cross_layer_externality_ms += source_externality_ms
            utilization = max(
                utilization,
                min(1.0, source_externality_ms / 1000.0),
            )
        joint_actuation = self._joint_actuation_plan(
            candidate,
            telemetry=telemetry,
            capacity=capacity,
            after=after,
            shared_budget=shared_budget,
        )
        endpoint_limits = {
            "local_prefill_token_ms": capacity.local_prefill_token_ms,
            "remote_prefill_token_ms": capacity.remote_prefill_token_ms,
            "remote_kv_bytes": capacity.remote_kv_bytes,
            "remote_semantic_ops": (
                capacity.remote_semantic_ops
                - self.config.remote_semantic_ops_safety_reserve
            ),
        }
        scheduler_queue_headroom = False
        if joint_actuation is not None:
            action_binding = tuple(
                name for name in (
                    "local_prefill_token_ms",
                    "remote_prefill_token_ms",
                    "remote_kv_bytes",
                    "remote_semantic_ops",
                )
                if getattr(after, name)
                > getattr(joint_actuation, f"{name}_limit")
            )
            for name, value in (
                (
                    "local_prefill_token_ms",
                    joint_actuation.enforced_local_prefill_token_ms_limit,
                ),
                (
                    "remote_prefill_token_ms",
                    joint_actuation.enforced_remote_prefill_token_ms_limit,
                ),
                (
                    "remote_kv_bytes",
                    joint_actuation.enforced_remote_kv_bytes_limit,
                ),
                (
                    "remote_semantic_ops",
                    joint_actuation.enforced_remote_semantic_ops_limit,
                ),
            ):
                if value is not None:
                    endpoint_limits[name] = min(endpoint_limits[name], value)
            # Critical transport pressure is never queue-leased.  A v1 hard
            # action target is also retained as a hard safety boundary.
            if joint_actuation.critical_guard or (
                joint_actuation.action_mode == "hard_window_v1"
                and action_binding
            ):
                return rejected(
                    "cross_layer_queue_lease_guard",
                    action_binding or ("critical_transport_pressure",),
                )

        endpoint_binding = tuple(
            name for name in (
                ("local_prefill_token_ms",)
                if candidate.route is GlobalRoute.LOCAL else
                (
                    "remote_prefill_token_ms",
                    "remote_kv_bytes",
                    "remote_semantic_ops",
                )
            )
            if getattr(after, name) > endpoint_limits[name]
        )
        if endpoint_binding:
            scheduler_queue_headroom = bool(
                self._endpoint_scheduler_queue_headroom(pair)
                or priority_service_lane
            )
            if self.config.endpoint_queue_debt_mode == "disabled":
                if not scheduler_queue_headroom:
                    return rejected(
                        "endpoint_service_window_full", endpoint_binding)
            elif not scheduler_queue_headroom:
                return rejected(
                    "endpoint_queue_capacity_full", endpoint_binding)

        over_capacity = tuple(
            name for name in ResourceVector.names()
            if getattr(after, name) > getattr(capacity, name)
        )
        overage_penalty_ms = (
            self.config.utilization_penalty_ms
            * sum(
                max(
                    0.0,
                    (getattr(after, name) - getattr(capacity, name))
                    / max(1, getattr(capacity, name)),
                )
                for name in over_capacity
            )
        )
        if joint_actuation is not None:
            overage_penalty_ms += joint_actuation.overage_penalty_ms
        # Queue leasing still chooses a route under the same live service
        # feedback as ordinary admission.  The lease only waives the global
        # reservation window; it must not discard a measured remote/local
        # service stretch and then send an expired waiter to the slower path.
        multiplier = (
            1.0
            if completion_liveness_probe else
            telemetry.multiplier(candidate.route)
        )
        if mesh_queue_lease and candidate.route is GlobalRoute.REMOTE:
            source_multiplier = (
                1.0
                if source_stale_feedback_fallback
                else self._telemetry[prefill].remote_service_multiplier
            )
            multiplier = max(multiplier, source_multiplier)
        inflation = candidate.predicted_ttft_ms * (multiplier - 1.0)
        edge_feedback_penalty_ms = self._mesh_edge_feedback_penalty_ms(
            candidate)
        receiver_stagger_us = self._mesh_receiver_stagger_us(candidate)
        completion_queue_delay_ms = (
            self._completion_liveness_queue_delay_ms(pair, candidate)
            if (
                not priority_service_lane
                and (
                    completion_liveness_probe
                    or self.config.endpoint_queue_debt_mode in {
                        "completion_credit_endpoint_queue_v3",
                        "completion_credit_mesh_endpoint_queue_v1",
                    }
                )
            )
            else 0.0
        )
        score = (
            candidate.predicted_e2e_ms
            + inflation
            + candidate.uncertainty_ms
            + completion_queue_delay_ms
            + utilization * self.config.utilization_penalty_ms
            + cross_layer_externality_ms
            + edge_feedback_penalty_ms
            + receiver_stagger_us / 1_000.0
            + overage_penalty_ms
            + (
                self.config.activation_penalty_ms
                if pair not in self._active_pairs else 0.0
            )
        )
        effective_deadline_ns = self._effective_deadline_ns(request)
        slack = (effective_deadline_ns - now_ns) / 1_000_000 - score
        if slack < 0.0 and not endpoint_queue_deadline_grace:
            return rejected("deadline")
        return _CandidateEvaluation(
            candidate=candidate,
            score_ms=score,
            slack_ms=slack,
            effective_used=effective,
            utilization=utilization,
            activate=pair not in self._active_pairs,
            activation_basis="endpoint_queue_lease",
            joint_actuation=joint_actuation,
            cross_layer_scale_required=bool(
                joint_actuation is not None
                and joint_actuation.soft_overage_resources
                and not (
                    shared_budget is not None
                    and self._shared_scale_suppressed_for_pair(pair, shared_budget)
                )
            ),
            shared_scale_suppressed=bool(
                shared_budget is not None
                and self._shared_scale_suppressed_for_pair(pair, shared_budget)
            ),
            endpoint_queue_debt_resources=tuple(endpoint_binding),
            completion_liveness_probe=completion_liveness_probe,
            completion_liveness_shared_probe=completion_liveness_shared_probe,
            endpoint_queue_headroom_admission=endpoint_queue_headroom_admission,
            endpoint_queue_deadline_grace=endpoint_queue_deadline_grace,
            priority_service_lane=priority_service_lane,
            stale_feedback_fallback=source_stale_feedback_fallback,
            receiver_stagger_us=receiver_stagger_us,
            protected_service_lane=protected_service_lane,
            protected_service_lane_key=protected_service_lane_key
            if protected_service_lane else None,
            protected_service_lane_before=protected_service_lane_before
            if protected_service_lane else None,
            protected_service_lane_after=protected_service_lane_after
            if protected_service_lane else None,
        )

    def _active_pressure(self) -> float:
        values = []
        for pair in self._active_pairs:
            if pair in self._telemetry:
                values.append(self._pair_pressure(pair))
        return max(values, default=0.0)

    def _proactive_scale_basis(self, now_ns: int) -> str | None:
        """Return a current-state reason to consider an inactive spare pair.

        Queue occupancy and queue age are deliberately evaluated from the
        requests already owned by the controller.  No future arrival,
        workload phase, or oracle route is involved.  The tenant-specific
        wait budget makes this a business/SLO signal rather than a second
        scalar fabric-pressure threshold.
        """

        if not self._queued:
            return None
        occupancy_limit = max(
            1,
            math.ceil(
                self.config.queue_capacity
                * self.config.proactive_scale_up_queue_fraction),
        )
        if len(self._queued) >= occupancy_limit:
            return "queue_occupancy"
        for request in self._queued.values():
            policy = self._tenants[request.tenant_id]
            wait_budget = min(
                self.config.maximum_queue_wait_ns,
                policy.maximum_queue_wait_ns,
            )
            risk_wait = max(
                1,
                math.ceil(
                    wait_budget
                    * self.config.proactive_scale_up_wait_fraction),
            )
            if now_ns - request.arrival_ns >= risk_wait:
                return "tenant_queue_slo_risk"
        return None

    def _options(
        self, request: GlobalRequest, now_ns: int
    ) -> tuple[list[_CandidateEvaluation], list[RejectedCandidate]]:
        accepted: list[_CandidateEvaluation] = []
        rejected: list[RejectedCandidate] = []
        for candidate in request.candidates:
            value = self._evaluate_candidate(
                candidate, request=request, now_ns=now_ns)
            if isinstance(value, RejectedCandidate):
                rejected.append(value)
            else:
                accepted.append(value)
        active = [item for item in accepted if not item.activate]
        inactive = [item for item in accepted if item.activate]
        can_scale = len(self._active_pairs) < int(
            self.config.maximum_active_pairs)
        business_isolation_scale_basis = (
            "tenant_protected_pair_isolation"
            if (
                can_scale
                and any(
                    not self._business_clean_candidate(request, item)
                    for item in active
                )
                and any(
                    self._business_clean_candidate(request, item)
                    for item in inactive
                )
            )
            else None
        )
        pressure = self._active_pressure()
        cross_layer_scale_basis = (
            "cross_layer_resource_envelope"
            if any(item.cross_layer_scale_required for item in active)
            else None
        )
        fabric_imbalanced_inactive = any(
            item.activate
            and not item.shared_scale_suppressed
            and item.joint_actuation is not None
            and item.joint_actuation.shared_budget_action
            == "global_remote_budget"
            for item in accepted
        )
        shared_fabric_scale_basis = (
            "cross_layer_fabric_imbalance"
            if fabric_imbalanced_inactive else None
        )
        shared_scale_suppressed = bool(active) and all(
            item.shared_scale_suppressed for item in active
        ) and not fabric_imbalanced_inactive
        queue_scale_basis = self._proactive_scale_basis(now_ns)
        route_benefit_basis = None
        if active and inactive and can_scale:
            best_active_score = min(item.score_ms for item in active)
            best_inactive_score = min(item.score_ms for item in inactive)
            if (
                best_inactive_score
                + self.config.proactive_scale_up_route_benefit_margin_ms
                < best_active_score
            ):
                route_benefit_basis = "route_benefit"
        proactive_scale = (
            not shared_scale_suppressed
            and (
                pressure >= self.config.scale_up_utilization
                or any(
                    item.utilization >= self.config.scale_up_utilization
                    for item in active
                )
                or cross_layer_scale_basis is not None
                or shared_fabric_scale_basis is not None
                or queue_scale_basis is not None
                or route_benefit_basis is not None
                or business_isolation_scale_basis is not None
            )
        )
        if active and (not proactive_scale or not can_scale):
            # Keep spare pairs cold below the global threshold.  Once actual
            # scheduler/endpoint pressure crosses it, inactive candidates are
            # eligible in this same atomic decision.
            accepted = active
        elif inactive and can_scale:
            accepted = active + inactive
            if queue_scale_basis is not None:
                accepted = [
                    replace(
                        item,
                        score_ms=(
                            item.score_ms
                            + self.config.proactive_scale_up_active_pair_penalty_ms
                        ),
                        slack_ms=(
                            item.slack_ms
                            - self.config.proactive_scale_up_active_pair_penalty_ms
                        ),
                        activation_basis=queue_scale_basis,
                    )
                    if not item.activate
                    and item.slack_ms
                    >= self.config.proactive_scale_up_active_pair_penalty_ms
                    else replace(item, activation_basis=queue_scale_basis)
                    if item.activate else item
                    for item in accepted
                ]
            if route_benefit_basis is not None and queue_scale_basis is None:
                accepted = [
                    replace(
                        item,
                        activation_basis=route_benefit_basis
                    )
                    if item.activate else item
                    for item in accepted
                ]
            if cross_layer_scale_basis is not None:
                accepted = [
                    replace(
                        item,
                        activation_basis=(
                            cross_layer_scale_basis
                            if item.activate else item.activation_basis
                        ),
                    )
                    for item in accepted
                ]
            if shared_fabric_scale_basis is not None:
                accepted = [
                    replace(
                        item,
                        activation_basis=(
                            shared_fabric_scale_basis
                            if item.activate else item.activation_basis
                        ),
                    )
                    for item in accepted
                ]
            if business_isolation_scale_basis is not None:
                accepted = [
                    replace(
                        item,
                        activation_basis=business_isolation_scale_basis,
                    )
                    if (
                        item.activate
                        and self._business_clean_candidate(request, item)
                    )
                    else item
                    for item in accepted
                ]
        elif active:
            accepted = active
        else:
            if not can_scale:
                for item in inactive:
                    rejected.append(RejectedCandidate.from_candidate(
                        item.candidate, "pair_inactive_at_maximum"))
                accepted = []
            else:
                # No active candidate fit.  Capacity/health failure is itself
                # sufficient evidence to use a prewarmed spare.
                accepted = inactive
        accepted = self._prefer_business_clean_evaluations(
            request, accepted, rejected)
        accepted.sort(key=lambda item: (
            item.score_ms,
            not item.candidate.cache_affinity,
            item.candidate.prefill_index,
            item.candidate.pair_index,
            item.candidate.route.value,
        ))
        accepted = self._mesh_near_tie_source_order(accepted)
        rejected.sort(key=lambda item: (
            item.decoder_index,
            item.prefill_index,
            item.route.value,
        ))
        return accepted, rejected

    def _tenant_key(self, request: GlobalRequest, now_ns: int) -> tuple[object, ...]:
        waited = now_ns - request.arrival_ns
        policy = self._tenants[request.tenant_id]
        starved = waited >= min(
            self.config.maximum_queue_wait_ns, policy.maximum_queue_wait_ns)
        # The stored virtual value is already the weighted service debt.  A
        # second division by ``weight`` would turn weighted fair service into
        # weight-squared priority.  Use raw service units for minimum-service
        # guarantees so tenant weights cannot make a tenant appear to have
        # received more (or less) physical work than it actually has.
        virtual = self._tenant_virtual_service[request.tenant_id]
        total_service = sum(self._tenant_service_units.values())
        service_fraction = (
            self._tenant_service_units[request.tenant_id] / total_service
            if total_service > 0.0 else 1.0
        )
        below_minimum = (
            total_service > 0.0
            and service_fraction < policy.minimum_service_fraction
        )
        return (
            0 if starved else 1,
            self._effective_deadline_ns(request)
            if starved else 0 if below_minimum else 1,
            self._effective_deadline_ns(request) if starved else virtual,
            virtual if starved else self._effective_deadline_ns(request),
            request.arrival_ns,
            request.request_id,
        )

    def _dispatch_locked(self, now_ns: int) -> list[GlobalDecision]:
        decisions: list[GlobalDecision] = []
        while self._queued:
            feasible: list[
                tuple[tuple[object, ...], GlobalRequest,
                      _CandidateEvaluation, list[RejectedCandidate]]
            ] = []
            for request in self._queued.values():
                options, rejected = self._options(request, now_ns)
                if options:
                    static_best = min(
                        options,
                        key=lambda item: (
                            item.score_ms,
                            not item.candidate.cache_affinity,
                            item.candidate.prefill_index,
                            item.candidate.pair_index,
                            item.candidate.route.value,
                        ),
                    )
                    feasible.append((
                        self._tenant_key(request, now_ns),
                        request,
                        options[0],
                        rejected + [
                            self._higher_score_rejection(
                                item,
                                selected=options[0],
                                static_best=static_best,
                            )
                            for item in options[1:]
                        ],
                    ))
            if not feasible:
                break
            feasible.sort(key=lambda item: item[0])
            _, request, selected, rejected = feasible[0]
            candidate = selected.candidate
            active_before = tuple(sorted(self._active_pairs))
            if selected.activate:
                self._active_pairs.add(candidate.pair_index)
            active_after = tuple(sorted(self._active_pairs))
            pair = candidate.pair_index
            used_before = selected.effective_used.as_dict()
            tenant_pair_scope_assigned = self._assign_tenant_pair_locked(
                request.tenant_id, pair, now_ns)
            virtual_before = self._tenant_virtual_service[request.tenant_id]
            weight = self._tenants[request.tenant_id].weight
            service_units = candidate.work.dominant_ratio(
                self._capacities[candidate.pair_index])
            virtual_after = virtual_before + service_units / weight
            self._tenant_virtual_service[request.tenant_id] = virtual_after
            self._tenant_service_units[request.tenant_id] += service_units
            self._tenant_admitted_decode_tokens[request.tenant_id] += (
                candidate.work.decode_tokens)
            destination_work = self._destination_work(candidate)
            self._owned[pair] = self._owned[pair] + destination_work
            mesh_stage_held = self._reserve_mesh_stage_locked(candidate)
            self._last_busy_ns[pair] = now_ns
            decision = GlobalDecision(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                kind=GlobalDecisionKind.ADMIT,
                decided_ns=now_ns,
                reason=(
                    "global_protected_service_lane_route_committed"
                    if selected.protected_service_lane
                    else
                    "global_mesh_stale_feedback_fallback_route_committed"
                    if selected.stale_feedback_fallback
                    else "global_tenant_pair_scope_assigned_and_route_committed"
                    if tenant_pair_scope_assigned
                    else (
                        "global_tenant_protected_pair_activated_"
                        "and_route_committed"
                    )
                    if selected.activate and selected.activation_basis == (
                        "tenant_protected_pair_isolation")
                    else (
                        "global_proactive_scale_route_benefit_"
                        "and_route_committed"
                    )
                    if selected.activate
                    and selected.activation_basis == "route_benefit"
                    else (
                        "global_proactive_queue_scale_"
                        f"{selected.activation_basis}_and_route_committed"
                    )
                    if selected.activate and selected.activation_basis in {
                        "queue_occupancy",
                        "tenant_queue_slo_risk",
                    }
                    else "global_pair_activated_and_route_committed"
                    if selected.activate
                    else "global_min_cost_fair_route_committed"
                ),
                pair_index=pair,
                route=candidate.route,
                score_ms=selected.score_ms,
                deadline_slack_ms=selected.slack_ms,
                selected_work=candidate.work.as_dict(),
                predicted_e2e_ms=candidate.predicted_e2e_ms,
                predicted_ttft_ms=candidate.predicted_ttft_ms,
                uncertainty_ms=candidate.uncertainty_ms,
                cache_affinity=candidate.cache_affinity,
                binding_resources=(
                    tuple(
                        value for value in (
                            PROTECTED_SERVICE_LANE_BINDING
                            if selected.protected_service_lane else None,
                            MESH_NEAR_TIE_SOURCE_BALANCE_BINDING
                            if selected.mesh_near_tie_source_balanced else None,
                        )
                        if value is not None
                    )
                ),
                rejected_candidates=tuple(rejected),
                resource_used_before=used_before,
                active_pairs_before=active_before,
                active_pairs_after=active_after,
                pair_activated=selected.activate,
                tenant_virtual_service_before=virtual_before,
                tenant_virtual_service_after=virtual_after,
                telemetry_sequences={
                    index: value.sequence
                    for index, value in self._telemetry.items()
                },
                telemetry_provenance=self._decision_telemetry_provenance(
                    pair, selected.joint_actuation),
                joint_actuation=selected.joint_actuation,
                cache_group_key=request.cache_group_key,
                prefill_index=candidate.prefill_index,
                decoder_index=candidate.decoder_index,
                edge_id=candidate.edge_id,
                receiver_stagger_us=selected.receiver_stagger_us,
                mesh_near_tie_source_balanced=(
                    selected.mesh_near_tie_source_balanced),
                mesh_near_tie_score_window_ms=(
                    selected.mesh_near_tie_score_window_ms),
                mesh_near_tie_score_delta_ms=(
                    selected.mesh_near_tie_score_delta_ms),
                mesh_source_virtual_service_before=(
                    selected.mesh_source_virtual_service_before),
                mesh_edge_virtual_service_before=(
                    selected.mesh_edge_virtual_service_before),
                service_queue_delay_ms=selected.service_queue_delay_ms,
                service_forecast_ms=selected.service_forecast_ms,
                protected_service_lane=selected.protected_service_lane,
                protected_service_lane_key=selected.protected_service_lane_key,
                protected_service_lane_before=(
                    selected.protected_service_lane_before),
                protected_service_lane_after=(
                    selected.protected_service_lane_after),
            )
            self._hold_cache_group_locked(request, candidate)
            self._inflight[request.request_id] = _Reservation(
                request=request,
                candidate=candidate,
                decision=decision,
                held=destination_work,
                committed_ns=now_ns,
                mesh_stage_held=mesh_stage_held,
            )
            del self._queued[request.request_id]
            self._decision_history.setdefault(
                request.request_id, []).append(decision)
            decisions.append(decision)
        return decisions

    def _queue_decision_locked(
        self, request: GlobalRequest, now_ns: int
    ) -> GlobalDecision:
        _, rejected = self._options(request, now_ns)
        binding = sorted({
            name for value in rejected for name in value.binding_resources
        })
        reasons = {value.reason for value in rejected}
        reason = (
            "global_telemetry_unavailable"
            if reasons and reasons <= {"telemetry_missing_or_stale"}
            else "global_resource_or_deadline_queue"
        )
        virtual = self._tenant_virtual_service[request.tenant_id]
        return GlobalDecision(
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            kind=GlobalDecisionKind.QUEUE,
            decided_ns=now_ns,
            reason=reason,
            pair_index=None,
            route=None,
            score_ms=None,
            deadline_slack_ms=None,
            selected_work={},
            predicted_e2e_ms=None,
            predicted_ttft_ms=None,
            uncertainty_ms=None,
            cache_affinity=None,
            binding_resources=tuple(binding),
            rejected_candidates=tuple(rejected),
            resource_used_before={},
            active_pairs_before=tuple(sorted(self._active_pairs)),
            active_pairs_after=tuple(sorted(self._active_pairs)),
            pair_activated=False,
            tenant_virtual_service_before=virtual,
            tenant_virtual_service_after=virtual,
            telemetry_sequences={
                index: value.sequence
                for index, value in self._telemetry.items()
            },
            telemetry_provenance=self._decision_telemetry_provenance(),
            cache_group_key=request.cache_group_key,
        )

    def _reject_locked(
        self,
        request: GlobalRequest,
        now_ns: int,
        *,
        reason: str,
        rejected_candidates: tuple[RejectedCandidate, ...] = (),
    ) -> GlobalDecision:
        virtual = self._tenant_virtual_service[request.tenant_id]
        decision = GlobalDecision(
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            kind=GlobalDecisionKind.REJECT,
            decided_ns=now_ns,
            reason=reason,
            pair_index=None,
            route=None,
            score_ms=None,
            deadline_slack_ms=None,
            selected_work={},
            predicted_e2e_ms=None,
            predicted_ttft_ms=None,
            uncertainty_ms=None,
            cache_affinity=None,
            binding_resources=(),
            rejected_candidates=tuple(rejected_candidates),
            resource_used_before={},
            active_pairs_before=tuple(sorted(self._active_pairs)),
            active_pairs_after=tuple(sorted(self._active_pairs)),
            pair_activated=False,
            tenant_virtual_service_before=virtual,
            tenant_virtual_service_after=virtual,
            telemetry_sequences={
                index: value.sequence
                for index, value in self._telemetry.items()
            },
            telemetry_provenance=self._decision_telemetry_provenance(),
            cache_group_key=request.cache_group_key,
        )
        self._terminal[request.request_id] = GlobalRequestPhase.REJECTED
        self._decision_history.setdefault(request.request_id, []).append(decision)
        return decision

    def _reconcile_pairs_locked(self, now_ns: int) -> None:
        if len(self._active_pairs) <= self.config.minimum_active_pairs:
            return
        queued_pairs = {
            candidate.pair_index
            for request in self._queued.values()
            for candidate in request.candidates
        }
        for pair in sorted(self._active_pairs, reverse=True):
            if len(self._active_pairs) <= self.config.minimum_active_pairs:
                break
            if any(self._owned[pair].as_dict().values()):
                continue
            if pair in queued_pairs:
                continue
            if now_ns - self._last_busy_ns[pair] < self.config.scale_down_idle_ns:
                continue
            self._active_pairs.remove(pair)
