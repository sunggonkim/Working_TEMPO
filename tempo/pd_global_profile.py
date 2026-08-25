"""Frozen, fingerprinted deployment profile for TEMPO-GO."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from tempo.pd_global_orchestrator import (
    GlobalOrchestratorConfig,
    PairCapacity,
    ResourceVector,
    TenantPolicy,
)
from tempo.pd_global_telemetry import (
    EndpointTelemetryContract,
    GlobalTelemetryAdapter,
)


SCHEMA = "tempo-go-profile-v1"
TRANSPORT = "LMCacheConnectorV1:UCX"
DEPLOYMENT_SCOPES = frozenset({"discovery", "frozen_validation"})
_TOP_LEVEL_KEYS = frozenset({
    "schema",
    "profile_id",
    "deployment_scope",
    "transport",
    "topology",
    "causality",
    "identity",
    "telemetry",
    "capacities",
    "tenants",
    "controller",
    "fingerprint_sha256",
})
_TOPOLOGY_KEYS = frozenset({
    "node_count",
    "gpu_count",
    "pair_count",
    "prewarmed_pair_count",
    "native_only",
    "route_immutable",
    "privileged_nic_control",
})
_CAUSALITY_KEYS = frozenset({
    "telemetry_clock",
    "decoder_credit_scope",
    "endpoint_credit_scope",
    "phase_label_policy_input",
    "physical_switch_label_policy_input",
    "future_arrivals_policy_input",
    "oracle_policy_input",
})
_IDENTITY_KEYS = frozenset({
    "router_schema",
    "endpoint_profile_schema",
    "endpoint_profile_id",
    "endpoint_profile_fingerprint_sha256",
    "endpoint_profile_deployment_scope",
    "elastic_profile_fingerprint_sha256",
    "workload_manifest_sha256",
    "model_config_sha256",
})
_TELEMETRY_KEYS = frozenset({
    "agent_epoch_source",
    "freshness_ns",
    "refresh_timeout_ns",
    "maximum_collection_span_ns",
    "tokenizer_timeout_ns",
    "controller_generation",
    "endpoint_feedback_mode",
    "endpoint_routing_policy",
})
_TELEMETRY_OPTIONAL_KEYS = frozenset({
    "scheduler_observation_required",
})
_CAPACITY_KEYS = frozenset({"pair_index", *ResourceVector.names()})
_TENANT_REQUIRED_KEYS = frozenset({"tenant_id", "weight"})
_TENANT_OPTIONAL_KEYS = frozenset({
    "ttft_slo_ms",
    "tpot_slo_ms",
    "e2e_slo_ms",
    "maximum_queue_wait_ns",
    "minimum_service_fraction",
    "queue_reservation_slots",
    "queue_lease_on_timeout",
    "telemetry_stale_grace_ns",
    "admission_priority",
    "protected_capacity_fraction",
    "pair_spread_limit",
})
_CONTROLLER_KEYS = frozenset({
    "queue_capacity",
    "minimum_active_pairs",
    "maximum_active_pairs",
    "scale_up_utilization",
    "scale_down_idle_ns",
    "utilization_penalty_ms",
    "activation_penalty_ms",
    "probe_penalty_ms",
    "maximum_queue_wait_ns",
})
_CONTROLLER_OPTIONAL_KEYS = frozenset({
    "remote_semantic_ops_safety_reserve",
    "proactive_scale_up_queue_fraction",
    "proactive_scale_up_wait_fraction",
    "proactive_scale_up_active_pair_penalty_ms",
    "proactive_scale_up_route_benefit_margin_ms",
    "route_failure_quarantine_mode",
    "telemetry_failure_quarantine_mode",
    "telemetry_failure_quarantine_scope",
    "survivor_capacity_reserve_fraction",
    "survivor_reserve_bypass_min_weight",
    "cross_layer_remote_limit_floor_fraction",
    "cross_layer_local_limit_floor_fraction",
    "cross_layer_stagger_max_us",
    "cross_layer_control_mode",
    "cross_layer_shadow_price_ms",
    "cross_layer_critical_pressure_fraction",
    "shared_fabric_control_mode",
    "shared_remote_requests_capacity",
    "shared_remote_kv_bytes_capacity",
    "shared_remote_semantic_ops_capacity",
    "shared_remote_limit_floor_fraction",
    "shared_remote_stagger_max_us",
    "mesh_control_mode",
    "mesh_receiver_stagger_max_us",
    "mesh_edge_service_ewma_alpha",
    "mesh_near_tie_source_balance_mode",
    "mesh_near_tie_source_balance_uncertainty_fraction",
    "mesh_cool_remote_route_pressure_fraction",
    "telemetry_stale_grace_ns",
    "overload_action",
    "endpoint_queue_debt_mode",
    "endpoint_queue_capacity",
    "endpoint_queue_admission_mode",
    "priority_service_lane_mode",
    "priority_service_lane_capacity",
    "priority_service_lane_min_admission_priority",
    "priority_service_lane_priority",
    "decoder_business_admission_mode",
    "decoder_business_background_max_wait_ns",
    "frozen_service_proxy_policy",
})

SERVICE_PROXY_POLICY_ID = "tempo-go-frozen-service-proxy-v1"
_SERVICE_PROXY_POLICY_KEYS = frozenset({
    "policy_id",
    "endpoint_profile_id",
    "endpoint_profile_fingerprint_sha256",
    "calibration_receipt_sha256",
    "allowed_lookup_modes",
    "allowed_cache_residencies",
    "allowed_remote_cache_residencies",
    "allowed_geometries",
    "proxy_is_not_exact",
    "numeric_rows_unchanged",
    "performance_claim_allowed",
})
_SERVICE_PROXY_LOOKUP_MODES = frozenset({
    "exact",
    "same_residency_geometry_ceiling",
    "miss_via_prefill_only_geometry_ceiling",
})
_SERVICE_PROXY_RESIDENCIES = frozenset({
    "confirmed_miss",
    "prefill_only",
})


def _canonical_sha(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def global_profile_fingerprint(raw: Mapping[str, object]) -> str:
    if not isinstance(raw, Mapping):
        raise TypeError("global profile must be a mapping")
    payload = dict(raw)
    payload.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GlobalTopology:
    node_count: int
    gpu_count: int
    pair_count: int
    prewarmed_pair_count: int
    native_only: bool
    route_immutable: bool
    privileged_nic_control: bool

    def __post_init__(self) -> None:
        if (
            self.node_count != 4
            or self.gpu_count != 16
            or self.pair_count not in (1, 2)
            or not 1 <= self.prewarmed_pair_count <= self.pair_count
        ):
            raise ValueError(
                "TEMPO-GO v1 requires the native 4-node/16-GPU topology "
                "with one or two inference pairs"
            )
        if self.native_only is not True or self.route_immutable is not True:
            raise ValueError("TEMPO-GO requires native execution and immutable routes")
        if self.privileged_nic_control is not False:
            raise ValueError("TEMPO-GO forbids privileged NIC control")


@dataclass(frozen=True)
class GlobalCausality:
    telemetry_clock: str
    decoder_credit_scope: str
    endpoint_credit_scope: str
    phase_label_policy_input: bool
    physical_switch_label_policy_input: bool
    future_arrivals_policy_input: bool
    oracle_policy_input: bool

    def __post_init__(self) -> None:
        if self.telemetry_clock != "frontend_perf_counter_interval_start":
            raise ValueError("global telemetry clock contract differs")
        if self.decoder_credit_scope != "request_start_to_http_eof":
            raise ValueError("global decoder credit scope differs")
        if self.endpoint_credit_scope != "route_commit_to_first_response":
            raise ValueError("global endpoint credit scope differs")
        for name in (
            "phase_label_policy_input",
            "physical_switch_label_policy_input",
            "future_arrivals_policy_input",
            "oracle_policy_input",
        ):
            if getattr(self, name) is not False:
                raise ValueError(f"non-causal global policy input enabled: {name}")


@dataclass(frozen=True)
class GlobalIdentity:
    router_schema: str
    endpoint_profile_schema: str
    endpoint_profile_id: str
    endpoint_profile_fingerprint_sha256: str
    endpoint_profile_deployment_scope: str
    elastic_profile_fingerprint_sha256: str
    workload_manifest_sha256: str
    model_config_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "router_schema",
            "endpoint_profile_schema",
            "endpoint_profile_id",
            "endpoint_profile_deployment_scope",
        ):
            _nonempty(name, getattr(self, name))
        for name in (
            "endpoint_profile_fingerprint_sha256",
            "elastic_profile_fingerprint_sha256",
            "workload_manifest_sha256",
            "model_config_sha256",
        ):
            _canonical_sha(name, getattr(self, name))


@dataclass(frozen=True)
class GlobalTelemetryConfig:
    agent_epoch_source: str
    freshness_ns: int
    refresh_timeout_ns: int
    maximum_collection_span_ns: int
    tokenizer_timeout_ns: int
    controller_generation: int
    endpoint_feedback_mode: str
    endpoint_routing_policy: str
    scheduler_observation_required: bool = False

    def __post_init__(self) -> None:
        for name in (
            "endpoint_feedback_mode",
            "endpoint_routing_policy",
        ):
            _nonempty(name, getattr(self, name))
        if self.agent_epoch_source != "slurm_job_id_frontend_start_ns":
            raise ValueError("global agent epoch source contract differs")
        for name in (
            "freshness_ns",
            "refresh_timeout_ns",
            "maximum_collection_span_ns",
            "tokenizer_timeout_ns",
        ):
            _positive_int(name, getattr(self, name))
        _nonnegative_int("controller_generation", self.controller_generation)
        if self.maximum_collection_span_ns > self.refresh_timeout_ns:
            raise ValueError("collection span cannot exceed refresh timeout")
        if self.refresh_timeout_ns > self.freshness_ns:
            raise ValueError("refresh timeout cannot exceed telemetry freshness")
        if type(self.scheduler_observation_required) is not bool:
            raise TypeError("scheduler_observation_required must be bool")


@dataclass(frozen=True)
class FrozenServiceProxyPolicy:
    """Explicit, non-performance contract for missing endpoint rows.

    A frozen profile may use this contract only for an allowlisted request
    geometry and cache-residency label.  The contract is intentionally kept
    outside ``GlobalOrchestratorConfig``: it authorizes an auditable service
    prior lookup, not a controller capacity or a performance claim.
    """

    policy_id: str
    endpoint_profile_id: str
    endpoint_profile_fingerprint_sha256: str
    calibration_receipt_sha256: str
    allowed_lookup_modes: tuple[str, ...]
    allowed_cache_residencies: tuple[str, ...]
    allowed_remote_cache_residencies: tuple[str, ...]
    allowed_geometries: tuple[tuple[int, int], ...]
    proxy_is_not_exact: bool
    numeric_rows_unchanged: bool
    performance_claim_allowed: bool

    def __post_init__(self) -> None:
        if self.policy_id != SERVICE_PROXY_POLICY_ID:
            raise ValueError("frozen service proxy policy ID differs")
        _nonempty("proxy endpoint_profile_id", self.endpoint_profile_id)
        _canonical_sha(
            "proxy endpoint_profile_fingerprint_sha256",
            self.endpoint_profile_fingerprint_sha256,
        )
        _canonical_sha(
            "proxy calibration_receipt_sha256", self.calibration_receipt_sha256)
        if (
            not self.allowed_lookup_modes
            or len(set(self.allowed_lookup_modes)) != len(self.allowed_lookup_modes)
            or not set(self.allowed_lookup_modes) <= _SERVICE_PROXY_LOOKUP_MODES
        ):
            raise ValueError("frozen service proxy lookup-mode allowlist is invalid")
        if (
            not self.allowed_cache_residencies
            or len(set(self.allowed_cache_residencies))
            != len(self.allowed_cache_residencies)
            or not set(self.allowed_cache_residencies)
            <= _SERVICE_PROXY_RESIDENCIES
        ):
            raise ValueError("frozen service proxy residency allowlist is invalid")
        if (
            len(set(self.allowed_remote_cache_residencies))
            != len(self.allowed_remote_cache_residencies)
            or not set(self.allowed_remote_cache_residencies)
            <= set(self.allowed_cache_residencies)
        ):
            raise ValueError(
                "frozen service proxy remote-residency allowlist is invalid")
        if not self.allowed_geometries:
            raise ValueError("frozen service proxy geometry allowlist is empty")
        if tuple(sorted(set(self.allowed_geometries))) != self.allowed_geometries:
            raise ValueError("frozen service proxy geometry allowlist is not canonical")
        for geometry in self.allowed_geometries:
            if (
                not isinstance(geometry, tuple)
                or len(geometry) != 2
                or any(type(value) is not int or value < 2 for value in geometry)
            ):
                raise ValueError("frozen service proxy geometry is invalid")
        for name in (
            "proxy_is_not_exact",
            "numeric_rows_unchanged",
            "performance_claim_allowed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.proxy_is_not_exact is not True:
            raise ValueError("frozen service proxy must be marked non-exact")
        if self.numeric_rows_unchanged is not True:
            raise ValueError("frozen service proxy must preserve numeric rows")
        if self.performance_claim_allowed is not False:
            raise ValueError("frozen service proxy cannot authorize performance claims")

    @classmethod
    def from_mapping(cls, value: object) -> "FrozenServiceProxyPolicy":
        if not isinstance(value, dict) or set(value) != _SERVICE_PROXY_POLICY_KEYS:
            raise ValueError("frozen service proxy policy inventory is not exact")
        modes = value["allowed_lookup_modes"]
        residencies = value["allowed_cache_residencies"]
        remote_residencies = value["allowed_remote_cache_residencies"]
        geometries = value["allowed_geometries"]
        if (
            not isinstance(modes, list)
            or not all(isinstance(item, str) for item in modes)
            or not isinstance(residencies, list)
            or not all(isinstance(item, str) for item in residencies)
            or not isinstance(remote_residencies, list)
            or not all(isinstance(item, str) for item in remote_residencies)
            or not isinstance(geometries, list)
        ):
            raise ValueError("frozen service proxy allowlists must be JSON lists")
        normalized_geometries = []
        for item in geometries:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or any(type(token) is not int for token in item)
            ):
                raise ValueError("frozen service proxy geometry must be [prompt, output]")
            normalized_geometries.append((item[0], item[1]))
        return cls(
            policy_id=value["policy_id"],
            endpoint_profile_id=value["endpoint_profile_id"],
            endpoint_profile_fingerprint_sha256=(
                value["endpoint_profile_fingerprint_sha256"]),
            calibration_receipt_sha256=value["calibration_receipt_sha256"],
            allowed_lookup_modes=tuple(modes),
            allowed_cache_residencies=tuple(residencies),
            allowed_remote_cache_residencies=tuple(remote_residencies),
            allowed_geometries=tuple(normalized_geometries),
            proxy_is_not_exact=value["proxy_is_not_exact"],
            numeric_rows_unchanged=value["numeric_rows_unchanged"],
            performance_claim_allowed=value["performance_claim_allowed"],
        )

    def allows_geometry(self, prompt_tokens: int, output_tokens: int) -> bool:
        return (prompt_tokens, output_tokens) in self.allowed_geometries

    def allows_residency(self, residency: str) -> bool:
        return residency in self.allowed_cache_residencies

    def allows_remote_residency(self, residency: str) -> bool:
        return residency in self.allowed_remote_cache_residencies

    def allows_lookup_mode(self, lookup_mode: str) -> bool:
        return lookup_mode in self.allowed_lookup_modes

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "endpoint_profile_id": self.endpoint_profile_id,
            "endpoint_profile_fingerprint_sha256": (
                self.endpoint_profile_fingerprint_sha256),
            "calibration_receipt_sha256": self.calibration_receipt_sha256,
            "allowed_lookup_modes": list(self.allowed_lookup_modes),
            "allowed_cache_residencies": list(self.allowed_cache_residencies),
            "allowed_remote_cache_residencies": list(
                self.allowed_remote_cache_residencies),
            "allowed_geometries": [list(item) for item in self.allowed_geometries],
            "proxy_is_not_exact": self.proxy_is_not_exact,
            "numeric_rows_unchanged": self.numeric_rows_unchanged,
            "performance_claim_allowed": self.performance_claim_allowed,
        }


@dataclass(frozen=True)
class GlobalProfile:
    profile_id: str
    deployment_scope: str
    topology: GlobalTopology
    causality: GlobalCausality
    identity: GlobalIdentity
    telemetry: GlobalTelemetryConfig
    capacities: tuple[PairCapacity, ...]
    tenants: tuple[TenantPolicy, ...]
    controller: Mapping[str, object]
    fingerprint_sha256: str
    transport: str = TRANSPORT
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        _nonempty("profile_id", self.profile_id)
        if self.deployment_scope not in DEPLOYMENT_SCOPES:
            raise ValueError("global profile deployment_scope is invalid")
        if self.transport != TRANSPORT:
            raise ValueError("global profile transport contract differs")
        if self.schema != SCHEMA:
            raise ValueError("global profile schema mismatch")
        if not isinstance(self.topology, GlobalTopology):
            raise TypeError("topology is invalid")
        if not isinstance(self.causality, GlobalCausality):
            raise TypeError("causality is invalid")
        if not isinstance(self.identity, GlobalIdentity):
            raise TypeError("identity is invalid")
        if not isinstance(self.telemetry, GlobalTelemetryConfig):
            raise TypeError("telemetry is invalid")
        if tuple(item.pair_index for item in self.capacities) != tuple(
            range(self.topology.pair_count)
        ):
            raise ValueError("pair capacities do not match topology")
        if not self.tenants:
            raise ValueError("global profile requires tenant policies")
        if (
            not _CONTROLLER_KEYS <= set(self.controller)
            or set(self.controller) - _CONTROLLER_KEYS
            - _CONTROLLER_OPTIONAL_KEYS
        ):
            raise ValueError("global controller inventory is not exact")
        _canonical_sha("fingerprint_sha256", self.fingerprint_sha256)
        proxy_policy = self.service_proxy_policy()
        if proxy_policy is not None:
            if (
                proxy_policy.endpoint_profile_id != self.identity.endpoint_profile_id
                or proxy_policy.endpoint_profile_fingerprint_sha256
                != self.identity.endpoint_profile_fingerprint_sha256
            ):
                raise ValueError("frozen service proxy endpoint identity differs")
        self.orchestrator_config()
        if (
            self.deployment_scope == "frozen_validation"
            and self.identity.endpoint_profile_deployment_scope
            != "frozen_validation"
        ):
            raise ValueError(
                "frozen global validation requires a frozen endpoint profile")
        if self.deployment_scope == "frozen_validation" and proxy_policy is None:
            raise ValueError(
                "frozen global validation requires an explicit service proxy policy")

    def service_proxy_policy(self) -> FrozenServiceProxyPolicy | None:
        value = self.controller.get("frozen_service_proxy_policy")
        if value is None:
            return None
        return FrozenServiceProxyPolicy.from_mapping(value)

    def orchestrator_config(self) -> GlobalOrchestratorConfig:
        controller = {
            key: value for key, value in self.controller.items()
            if key != "frozen_service_proxy_policy"
        }
        return GlobalOrchestratorConfig(
            capacities=self.capacities,
            tenants=self.tenants,
            telemetry_fresh_ns=self.telemetry.freshness_ns,
            **controller,
        )

    def endpoint_contracts(self) -> tuple[EndpointTelemetryContract, ...]:
        identity = self.identity
        telemetry = self.telemetry
        return tuple(
            EndpointTelemetryContract(
                pair_index=pair_index,
                router_schema=identity.router_schema,
                endpoint_feedback_mode=telemetry.endpoint_feedback_mode,
                endpoint_routing_policy=telemetry.endpoint_routing_policy,
                profile_schema=identity.endpoint_profile_schema,
                profile_id=identity.endpoint_profile_id,
                profile_fingerprint_sha256=(
                    identity.endpoint_profile_fingerprint_sha256),
                elastic_profile_fingerprint_sha256=(
                    identity.elastic_profile_fingerprint_sha256),
                workload_manifest_sha256=identity.workload_manifest_sha256,
                deployment_scope=identity.endpoint_profile_deployment_scope,
                controller_generation=telemetry.controller_generation,
            )
            for pair_index in range(self.topology.pair_count)
        )

    def telemetry_adapter(self, *, agent_epoch: str) -> GlobalTelemetryAdapter:
        return GlobalTelemetryAdapter(
            self.endpoint_contracts(),
            agent_epoch=agent_epoch,
            maximum_collection_span_ns=(
                self.telemetry.maximum_collection_span_ns),
            require_scheduler_snapshot=(
                self.telemetry.scheduler_observation_required),
        )


def _exact_mapping(
    name: str, value: object, keys: frozenset[str]
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} inventory is not exact")
    return value


def _mapping_with_optional(
    name: str,
    value: object,
    required: frozenset[str],
    optional: frozenset[str],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} inventory is not exact")
    if not required <= set(value) or set(value) - required - optional:
        raise ValueError(f"{name} inventory is not exact")
    return value


def load_global_profile(path: Path) -> GlobalProfile:
    if not isinstance(path, Path):
        raise TypeError("path must be Path")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("failed to read global profile") from exc
    if not isinstance(raw, dict) or set(raw) != _TOP_LEVEL_KEYS:
        raise ValueError("global profile top-level inventory is not exact")
    if raw.get("schema") != SCHEMA:
        raise ValueError("global profile schema mismatch")
    declared = _canonical_sha("fingerprint_sha256", raw["fingerprint_sha256"])
    if declared != global_profile_fingerprint(raw):
        raise ValueError("global profile fingerprint mismatch")

    topology = GlobalTopology(**_exact_mapping(
        "topology", raw["topology"], _TOPOLOGY_KEYS))
    causality = GlobalCausality(**_exact_mapping(
        "causality", raw["causality"], _CAUSALITY_KEYS))
    identity = GlobalIdentity(**_exact_mapping(
        "identity", raw["identity"], _IDENTITY_KEYS))
    telemetry_raw = _mapping_with_optional(
        "telemetry", raw["telemetry"], _TELEMETRY_KEYS,
        _TELEMETRY_OPTIONAL_KEYS)
    telemetry_values = dict(telemetry_raw)
    telemetry_values.setdefault("scheduler_observation_required", False)
    telemetry = GlobalTelemetryConfig(**telemetry_values)

    capacity_raw = raw["capacities"]
    pair_count = topology.pair_count
    if not isinstance(capacity_raw, list) or len(capacity_raw) != pair_count:
        raise ValueError(
            "global profile capacity rows do not match topology.pair_count"
        )
    capacities = []
    for value in capacity_raw:
        item = _exact_mapping("capacity", value, _CAPACITY_KEYS)
        pair_index = item["pair_index"]
        resources = {
            name: item[name] for name in ResourceVector.names()
        }
        capacities.append(PairCapacity(
            pair_index=pair_index,
            resources=ResourceVector(**resources),
        ))

    tenant_raw = raw["tenants"]
    if not isinstance(tenant_raw, list) or not tenant_raw:
        raise ValueError("global profile tenants must be a nonempty list")
    tenants = []
    for item in tenant_raw:
        value = _mapping_with_optional(
            "tenant", item, _TENANT_REQUIRED_KEYS, _TENANT_OPTIONAL_KEYS)
        tenants.append(TenantPolicy(**value))
    tenants = tuple(tenants)
    controller = _mapping_with_optional(
        "controller", raw["controller"], _CONTROLLER_KEYS,
        _CONTROLLER_OPTIONAL_KEYS)
    return GlobalProfile(
        profile_id=raw["profile_id"],
        deployment_scope=raw["deployment_scope"],
        topology=topology,
        causality=causality,
        identity=identity,
        telemetry=telemetry,
        capacities=tuple(capacities),
        tenants=tenants,
        controller=controller,
        fingerprint_sha256=declared,
        transport=raw["transport"],
        schema=raw["schema"],
    )


__all__ = [
    "DEPLOYMENT_SCOPES",
    "GlobalCausality",
    "GlobalIdentity",
    "GlobalProfile",
    "FrozenServiceProxyPolicy",
    "GlobalTelemetryConfig",
    "GlobalTopology",
    "SCHEMA",
    "SERVICE_PROXY_POLICY_ID",
    "TRANSPORT",
    "global_profile_fingerprint",
    "load_global_profile",
]
