"""Causal application-level telemetry assembly for TEMPO-GO.

The frontend owns the only clock used by the global policy.  It records a
conservative interval around one parallel poll of every pair router and uses
the interval start as the policy sample time.  Router-local monotonic clocks
are deliberately never subtracted across Perlmutter nodes.

Only existing application-visible state is consumed: the frontend decoder
ledger and the endpoint feedback controller exposed by each canonical pair
router.  No sysfs counter, privileged NIC control, physical-switch label,
benchmark phase, or future arrival enters the assembled state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import threading
from typing import Any

from tempo.pd_global_orchestrator import (
    CrossLayerSignal,
    CrossLayerTelemetry,
    PairTelemetry,
    PathHealth,
    ResourceVector,
    TELEMETRY_SCHEMA,
)


BATCH_SCHEMA = "tempo-go-telemetry-batch-v1"
FRONTEND_LEDGER_SCHEMA = "tempo-go-frontend-ledger-v1"
ENDPOINT_CONTROLLER_SCHEMA = "tempo-pd-endpoint-controller-v1"
SCHEDULER_SCHEMA = "tempo-go-vllm-scheduler-snapshot-v1"
COMPLETION_SCHEMA = "tempo-go-endpoint-completion-v1"
CROSS_LAYER_SCHEMA = "tempo-go-cross-layer-envelope-v1"

_LOCAL_ROUTE = "decoder_local_chunked_prefill"
_REMOTE_ROUTE = "official_lmcache_remote_prefill"
_ENDPOINT_RESOURCES = (
    "local_token_ms",
    "remote_prefill_token_ms",
    "remote_kv_bytes",
    "remote_semantic_ops",
)
_MESH_REMOTE_RESOURCES = (
    "remote_prefill_token_ms",
    "remote_kv_bytes",
    "remote_semantic_ops",
)
_CAUSAL_POLICY_FLAGS = (
    "phase_label_policy_input",
    "physical_switch_label_policy_input",
    "future_arrivals_policy_input",
    "oracle_policy_input",
)


def _nonnegative_int(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _positive_int(name: str, value: object) -> int:
    result = _nonnegative_int(name, value)
    if result == 0:
        raise ValueError(f"{name} must be a positive int")
    return result


def _nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value


def _sha256(name: str, value: object) -> str:
    result = _nonempty(name, value)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _integer_vector(
    name: str, value: object, *, length: int
) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != length
    ):
        raise ValueError(f"{name} must have exactly {length} entries")
    return tuple(
        _nonnegative_int(f"{name}[{index}]", item)
        for index, item in enumerate(value)
    )


def _endpoint_resource_map(name: str, value: object) -> dict[str, int]:
    raw = _mapping(name, value)
    if set(raw) != set(_ENDPOINT_RESOURCES):
        raise ValueError(f"{name} keys do not match the endpoint schema")
    return {
        resource: _nonnegative_int(f"{name}.{resource}", raw[resource])
        for resource in _ENDPOINT_RESOURCES
    }


def _mesh_remote_map(
    name: str, value: object, *, pair_count: int,
) -> dict[int, dict[str, int]]:
    raw = _mapping(name, value)
    expected = {str(index) for index in range(pair_count)}
    if set(raw) != expected:
        raise ValueError(f"{name} keys do not cover every decoder")
    result = {}
    for key in sorted(raw, key=int):
        item = _mapping(f"{name}.{key}", raw[key])
        if set(item) != set(_MESH_REMOTE_RESOURCES):
            raise ValueError(f"{name}.{key} resource keys differ")
        result[int(key)] = {
            resource: _nonnegative_int(
                f"{name}.{key}.{resource}", item[resource])
            for resource in _MESH_REMOTE_RESOURCES
        }
    return result


@dataclass(frozen=True)
class EndpointTelemetryContract:
    """Frozen identity expected from one prewarmed P/D pair."""

    pair_index: int
    router_schema: str
    endpoint_feedback_mode: str
    endpoint_routing_policy: str
    profile_schema: str
    profile_id: str
    profile_fingerprint_sha256: str
    elastic_profile_fingerprint_sha256: str
    workload_manifest_sha256: str
    deployment_scope: str
    controller_generation: int
    endpoint_controller_schema: str = ENDPOINT_CONTROLLER_SCHEMA

    def __post_init__(self) -> None:
        _nonnegative_int("pair_index", self.pair_index)
        for name in (
            "router_schema",
            "endpoint_feedback_mode",
            "endpoint_routing_policy",
            "profile_schema",
            "profile_id",
            "deployment_scope",
            "endpoint_controller_schema",
        ):
            _nonempty(name, getattr(self, name))
        for name in (
            "profile_fingerprint_sha256",
            "elastic_profile_fingerprint_sha256",
            "workload_manifest_sha256",
        ):
            _sha256(name, getattr(self, name))
        _nonnegative_int("controller_generation", self.controller_generation)


@dataclass(frozen=True)
class GlobalTelemetryBatch:
    """One atomic, all-pair telemetry generation."""

    sequence: int
    sampled_ns: int
    collected_ns: int
    agent_epoch: str
    pairs: tuple[PairTelemetry, ...]
    schema: str = BATCH_SCHEMA

    def __post_init__(self) -> None:
        _positive_int("sequence", self.sequence)
        _nonnegative_int("sampled_ns", self.sampled_ns)
        _nonnegative_int("collected_ns", self.collected_ns)
        if self.collected_ns < self.sampled_ns:
            raise ValueError("telemetry batch collection interval is inverted")
        _nonempty("agent_epoch", self.agent_epoch)
        if self.schema != BATCH_SCHEMA:
            raise ValueError("global telemetry batch schema mismatch")
        if not self.pairs:
            raise ValueError("global telemetry batch has no pairs")
        indices = tuple(item.pair_index for item in self.pairs)
        if indices != tuple(range(len(self.pairs))):
            raise ValueError("global telemetry pairs must be contiguous and ordered")
        for item in self.pairs:
            if not isinstance(item, PairTelemetry):
                raise TypeError("global telemetry batch contains a non-pair item")
            if (
                item.sequence != self.sequence
                or item.sampled_ns != self.sampled_ns
                or item.collected_ns != self.collected_ns
                or item.agent_epoch != self.agent_epoch
            ):
                raise ValueError("pair telemetry does not share the batch epoch")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sequence": self.sequence,
            "sampled_ns": self.sampled_ns,
            "collected_ns": self.collected_ns,
            "collection_span_ns": self.collected_ns - self.sampled_ns,
            "agent_epoch": self.agent_epoch,
            "pairs": [
                {
                    "schema": item.schema,
                    "pair_index": item.pair_index,
                    "sequence": item.sequence,
                    "sampled_ns": item.sampled_ns,
                    "collected_ns": item.collected_ns,
                    "agent_epoch": item.agent_epoch,
                    "profile_fingerprint_sha256": (
                        item.profile_fingerprint_sha256),
                    "controller_generation": item.controller_generation,
                    "observed_total": item.observed_total.as_dict(),
                    "local_health": item.local_health.value,
                    "remote_health": item.remote_health.value,
                    "local_service_multiplier": (
                        item.local_service_multiplier),
                    "remote_service_multiplier": (
                        item.remote_service_multiplier),
                    "route_failures": {
                        "local_count": item.local_failure_count,
                        "remote_count": item.remote_failure_count,
                        "local_last_kind": item.local_last_failure_kind,
                        "remote_last_kind": item.remote_last_failure_kind,
                    },
                    "scheduler": {
                        "schema": item.scheduler_schema,
                        "source": item.scheduler_source,
                        "running_requests": item.scheduler_running_requests,
                        "waiting_requests": item.scheduler_waiting_requests,
                        "kv_cache_usage_fraction": (
                            item.scheduler_kv_cache_usage_fraction),
                    },
                    "completion": {
                        "schema": item.completion_schema,
                        "completed_first_responses": (
                            item.endpoint_completed_first_responses),
                        "residual_inflight": item.endpoint_residual_inflight,
                    },
                    "cross_layer": (
                        item.cross_layer.as_dict()
                        if item.cross_layer is not None else None
                    ),
                    "source": item.source,
                }
                for item in self.pairs
            ],
        }


class GlobalTelemetryAdapter:
    """Validate and atomically assemble frontend/router snapshots."""

    def __init__(
        self,
        contracts: Sequence[EndpointTelemetryContract],
        *,
        agent_epoch: str,
        maximum_collection_span_ns: int,
        require_scheduler_snapshot: bool = False,
    ) -> None:
        if (
            not isinstance(contracts, Sequence)
            or isinstance(contracts, (str, bytes, bytearray))
            or not contracts
            or any(
                not isinstance(item, EndpointTelemetryContract)
                for item in contracts
            )
        ):
            raise TypeError("contracts must contain endpoint contracts")
        self.contracts = tuple(contracts)
        indices = tuple(item.pair_index for item in self.contracts)
        if indices != tuple(range(len(self.contracts))):
            raise ValueError("endpoint contracts must be contiguous and ordered")
        for name in (
            "profile_id",
            "profile_fingerprint_sha256",
            "elastic_profile_fingerprint_sha256",
            "workload_manifest_sha256",
            "deployment_scope",
            "controller_generation",
        ):
            if len({getattr(item, name) for item in self.contracts}) != 1:
                raise ValueError(f"endpoint contracts have mixed {name}")
        self.agent_epoch = _nonempty("agent_epoch", agent_epoch)
        self.maximum_collection_span_ns = _positive_int(
            "maximum_collection_span_ns", maximum_collection_span_ns)
        if type(require_scheduler_snapshot) is not bool:
            raise TypeError("require_scheduler_snapshot must be bool")
        self.require_scheduler_snapshot = require_scheduler_snapshot
        self._sequence = 0
        self._last_collected_ns: int | None = None
        self._lock = threading.Lock()

    def assemble(
        self,
        frontend_snapshot: Mapping[str, Any],
        endpoint_snapshots: Mapping[int, Mapping[str, Any]],
        *,
        collection_started_ns: int,
        collection_finished_ns: int,
        quarantined_pairs: Mapping[int, str] | None = None,
    ) -> GlobalTelemetryBatch:
        """Build one policy-eligible batch or fail without advancing sequence."""

        started = _nonnegative_int(
            "collection_started_ns", collection_started_ns)
        finished = _nonnegative_int(
            "collection_finished_ns", collection_finished_ns)
        if finished < started:
            raise ValueError("telemetry collection interval is inverted")
        if finished - started > self.maximum_collection_span_ns:
            raise ValueError("telemetry collection exceeded its causal span")
        with self._lock:
            if (
                self._last_collected_ns is not None
                and started < self._last_collected_ns
            ):
                raise ValueError("telemetry collection intervals overlap")
            loads, active = self._parse_frontend(frontend_snapshot)
            if not isinstance(endpoint_snapshots, Mapping):
                raise ValueError("endpoint_snapshots must be keyed by pair")
            if quarantined_pairs is None:
                quarantined_pairs = {}
            if not isinstance(quarantined_pairs, Mapping):
                raise ValueError("quarantined_pairs must be keyed by pair")
            expected_pairs = set(range(len(self.contracts)))
            if not set(quarantined_pairs).issubset(expected_pairs):
                raise ValueError("quarantine contains an unknown pair")
            if set(endpoint_snapshots) | set(quarantined_pairs) != expected_pairs:
                raise ValueError(
                    "endpoint snapshots and quarantines do not cover every pair")
            for pair_index, reason in quarantined_pairs.items():
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError("quarantine reason must be nonempty")
            sequence = self._sequence + 1
            parsed_pairs = []
            mesh_maps = []
            for contract in self.contracts:
                if contract.pair_index in quarantined_pairs:
                    parsed_pairs.append(self._quarantined_pair(
                        contract,
                        sequence=sequence,
                        sampled_ns=started,
                        collected_ns=finished,
                        decode_tokens=loads[contract.pair_index],
                        active_sequences=active[contract.pair_index],
                        reason=quarantined_pairs[contract.pair_index],
                    ))
                    mesh_maps.append({})
                    continue
                raw = endpoint_snapshots[contract.pair_index]
                pair, mesh_map = self._parse_endpoint(
                    contract,
                    raw,
                    sequence=sequence,
                    sampled_ns=started,
                    collected_ns=finished,
                    decode_tokens=loads[contract.pair_index],
                    active_sequences=active[contract.pair_index],
                )
                parsed_pairs.append(pair)
                mesh_maps.append(mesh_map)
            mesh_enabled = any(mesh_maps)
            if mesh_enabled and any(not value for value in mesh_maps):
                raise ValueError(
                    "mesh edge telemetry must cover every endpoint")
            receiver_remote = {
                index: {name: 0 for name in _MESH_REMOTE_RESOURCES}
                for index in range(len(self.contracts))
            }
            if mesh_enabled:
                for source_map in mesh_maps:
                    for decoder, values in source_map.items():
                        for name in _MESH_REMOTE_RESOURCES:
                            receiver_remote[decoder][name] += values[name]
            pairs = []
            for pair in parsed_pairs:
                if not mesh_enabled:
                    pairs.append(pair)
                    continue
                receiver = receiver_remote[pair.pair_index]
                scheduler_active = (
                    (pair.scheduler_running_requests or 0)
                    + (pair.scheduler_waiting_requests or 0)
                    + (pair.endpoint_residual_inflight or 0)
                )
                observed = pair.observed_total
                # local_token_ms and remote_prefill_token_ms are producer
                # resources and remain attached to P_i.  Receiver KV and
                # semantic credits are aggregated by their actual D_j target.
                # Decoder scheduler occupancy is the destination-side request
                # count; the frontend pair ledger is ingress/source state.
                destination = ResourceVector(
                    decode_tokens=observed.decode_tokens,
                    active_sequences=max(observed.active_sequences,
                                         scheduler_active),
                    endpoint_requests=max(observed.endpoint_requests,
                                          scheduler_active),
                    local_prefill_token_ms=observed.local_prefill_token_ms,
                    remote_prefill_token_ms=observed.remote_prefill_token_ms,
                    remote_kv_bytes=receiver["remote_kv_bytes"],
                    remote_semantic_ops=receiver["remote_semantic_ops"],
                )
                pairs.append(replace(pair, observed_total=destination))
            pairs = tuple(pairs)
            batch = GlobalTelemetryBatch(
                sequence=sequence,
                sampled_ns=started,
                collected_ns=finished,
                agent_epoch=self.agent_epoch,
                pairs=pairs,
            )
            self._sequence = sequence
            self._last_collected_ns = finished
            return batch

    def _quarantined_pair(
        self,
        contract: EndpointTelemetryContract,
        *,
        sequence: int,
        sampled_ns: int,
        collected_ns: int,
        decode_tokens: int,
        active_sequences: int,
        reason: str,
    ) -> PairTelemetry:
        """Represent an endpoint fetch failure without reusing stale totals.

        The frontend ledger remains useful for accounting requests already in
        flight, but no endpoint resource total is trusted after the router
        fetch fails.  Denying both routes makes the global policy quarantine
        this pair while leaving healthy pairs eligible in the same batch.
        """

        return PairTelemetry(
            pair_index=contract.pair_index,
            sequence=sequence,
            sampled_ns=sampled_ns,
            collected_ns=collected_ns,
            agent_epoch=self.agent_epoch,
            profile_fingerprint_sha256=contract.profile_fingerprint_sha256,
            controller_generation=contract.controller_generation,
            observed_total=ResourceVector(
                decode_tokens=decode_tokens,
                active_sequences=active_sequences,
            ),
            local_health=PathHealth.DENIED,
            remote_health=PathHealth.DENIED,
            quarantine_reason=reason,
        )

    def _parse_frontend(
        self, value: Mapping[str, Any]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        raw = _mapping("frontend_snapshot", value)
        if raw.get("schema") != FRONTEND_LEDGER_SCHEMA:
            raise ValueError("frontend telemetry schema mismatch")
        pair_count = len(self.contracts)
        loads = _integer_vector("frontend.loads", raw.get("loads"), length=pair_count)
        active = _integer_vector(
            "frontend.active_by_pair", raw.get("active_by_pair"),
            length=pair_count,
        )
        total_active = _nonnegative_int("frontend.active", raw.get("active"))
        if total_active != sum(active):
            raise ValueError("frontend active request totals disagree")
        return loads, active

    def _parse_endpoint(
        self,
        contract: EndpointTelemetryContract,
        value: Mapping[str, Any],
        *,
        sequence: int,
        sampled_ns: int,
        collected_ns: int,
        decode_tokens: int,
        active_sequences: int,
    ) -> tuple[PairTelemetry, dict[int, dict[str, int]]]:
        raw = _mapping(f"endpoint[{contract.pair_index}]", value)
        if raw.get("schema") != contract.router_schema:
            raise ValueError("pair router telemetry schema mismatch")
        if raw.get("pair_index") != contract.pair_index:
            raise ValueError("pair router identity mismatch")
        if raw.get("endpoint_feedback_mode") != contract.endpoint_feedback_mode:
            raise ValueError("endpoint feedback mode differs from contract")
        if raw.get("endpoint_routing_policy") != contract.endpoint_routing_policy:
            raise ValueError("endpoint routing policy differs from contract")
        generation = _nonnegative_int(
            "controller_generation", raw.get("controller_generation"))
        if generation != contract.controller_generation:
            raise ValueError("endpoint controller generation differs from contract")

        profile = _mapping(
            "endpoint_service_profile", raw.get("endpoint_service_profile"))
        expected_profile = {
            "schema": contract.profile_schema,
            "profile_id": contract.profile_id,
            "fingerprint_sha256": contract.profile_fingerprint_sha256,
            "elastic_profile_fingerprint_sha256": (
                contract.elastic_profile_fingerprint_sha256),
            "workload_manifest_sha256": contract.workload_manifest_sha256,
            "deployment_scope": contract.deployment_scope,
        }
        for name, expected in expected_profile.items():
            if profile.get(name) != expected:
                raise ValueError(
                    f"endpoint profile {name} differs from contract")
        routing_policy = profile.get("routing_policy")
        if routing_policy is not None:
            policy = _mapping("endpoint profile routing_policy", routing_policy)
            for name in _CAUSAL_POLICY_FLAGS:
                if policy.get(name, False) is not False:
                    raise ValueError(
                        f"non-causal endpoint policy input enabled: {name}")

        controller = _mapping("endpoint controller", raw.get("controller"))
        if controller.get("schema") != contract.endpoint_controller_schema:
            raise ValueError("endpoint controller telemetry schema mismatch")
        resources = _endpoint_resource_map(
            "controller.resources", controller.get("resources"))
        owned = _endpoint_resource_map(
            "controller.owned_resources", controller.get("owned_resources"))
        external = _endpoint_resource_map(
            "controller.external_resources", controller.get("external_resources"))
        if any(
            resources[name] != owned[name] + external[name]
            for name in _ENDPOINT_RESOURCES
        ):
            raise ValueError("endpoint total is not owned plus external state")

        scheduler = self._parse_scheduler(raw.get("vllm_scheduler"))
        completion = self._parse_completion(controller.get("completion"))
        cross_layer = self._parse_cross_layer(
            raw.get("cross_layer"), pair_index=contract.pair_index)

        controller_inflight = _nonnegative_int(
            "controller.inflight", controller.get("inflight"))
        external_inflight = _nonnegative_int(
            "controller.external_inflight", controller.get("external_inflight"))
        queued = _nonnegative_int(
            "endpoint.queued_requests", raw.get("queued_requests"))
        routes = _mapping("controller.routes", controller.get("routes"))
        if set(routes) != {_LOCAL_ROUTE, _REMOTE_ROUTE}:
            raise ValueError("endpoint route state keys differ from contract")
        (
            local_health,
            local_multiplier,
            local_failure_count,
            local_last_failure_kind,
        ) = self._parse_route(
            routes[_LOCAL_ROUTE], name="local")
        (
            remote_health,
            remote_multiplier,
            remote_failure_count,
            remote_last_failure_kind,
        ) = self._parse_route(
            routes[_REMOTE_ROUTE], name="remote")

        pair = PairTelemetry(
            pair_index=contract.pair_index,
            sequence=sequence,
            sampled_ns=sampled_ns,
            collected_ns=collected_ns,
            agent_epoch=self.agent_epoch,
            profile_fingerprint_sha256=(
                contract.profile_fingerprint_sha256),
            controller_generation=generation,
            observed_total=ResourceVector(
                decode_tokens=decode_tokens,
                active_sequences=active_sequences,
                endpoint_requests=(
                    controller_inflight + external_inflight + queued),
                local_prefill_token_ms=resources["local_token_ms"],
                remote_prefill_token_ms=resources[
                    "remote_prefill_token_ms"],
                remote_kv_bytes=resources["remote_kv_bytes"],
                remote_semantic_ops=resources["remote_semantic_ops"],
            ),
            local_health=local_health,
            remote_health=remote_health,
            local_service_multiplier=local_multiplier,
            remote_service_multiplier=remote_multiplier,
            local_failure_count=local_failure_count,
            remote_failure_count=remote_failure_count,
            local_last_failure_kind=local_last_failure_kind,
            remote_last_failure_kind=remote_last_failure_kind,
            scheduler_running_requests=(
                scheduler["running_requests"] if scheduler else None),
            scheduler_waiting_requests=(
                scheduler["waiting_requests"] if scheduler else None),
            scheduler_kv_cache_usage_fraction=(
                scheduler["kv_cache_usage_fraction"] if scheduler else None),
            scheduler_schema=scheduler["schema"] if scheduler else None,
            scheduler_source=scheduler["source"] if scheduler else None,
            endpoint_completed_first_responses=(
                completion["completed_first_responses"] if completion else None),
            endpoint_residual_inflight=(
                completion["residual_inflight"] if completion else None),
            completion_schema=completion["schema"] if completion else None,
            cross_layer=cross_layer,
            schema=TELEMETRY_SCHEMA,
        )
        mesh_remote = raw.get("mesh_remote_by_decoder")
        if mesh_remote is None:
            return pair, {}
        return pair, _mesh_remote_map(
            f"endpoint[{contract.pair_index}].mesh_remote_by_decoder",
            mesh_remote,
            pair_count=len(self.contracts),
        )

    def _parse_cross_layer(
        self, value: object, *, pair_index: int
    ) -> CrossLayerTelemetry | None:
        if value is None:
            return None
        raw = _mapping("cross_layer", value)
        required = {
            "schema", "pair_index", "node_id", "endpoint_id",
            "communicator_id", "source_epoch",
            "topology_fingerprint_sha256", "sequence", "sampled_ns",
            "window_ms", "signals", "cassini_by_nic",
        }
        if set(raw) != required:
            raise ValueError("cross-layer telemetry inventory is not exact")
        if raw["schema"] != CROSS_LAYER_SCHEMA:
            raise ValueError("cross-layer telemetry schema mismatch")
        if raw["pair_index"] != pair_index:
            raise ValueError("cross-layer telemetry pair identity mismatch")
        raw_signals = raw["signals"]
        if (
            not isinstance(raw_signals, list)
            or not raw_signals
        ):
            raise ValueError("cross-layer telemetry signals are missing")
        signals: list[CrossLayerSignal] = []
        for index, item in enumerate(raw_signals):
            signal = _mapping(f"cross_layer.signals[{index}]", item)
            if set(signal) != {
                "name", "value", "unit", "support", "source",
                "uncertainty", "scope",
            }:
                raise ValueError("cross-layer signal inventory is not exact")
            support = signal["support"]
            if support == "supported":
                value = signal["value"]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not float(value) >= 0.0
                ):
                    raise ValueError("supported cross-layer signal is invalid")
            elif signal["value"] is not None:
                raise ValueError("unsupported cross-layer signal has a value")
            signals.append(CrossLayerSignal(
                name=_nonempty("cross_layer.signal.name", signal["name"]),
                value=signal["value"],
                unit=_nonempty("cross_layer.signal.unit", signal["unit"]),
                support=_nonempty("cross_layer.signal.support", support),
                source=_nonempty("cross_layer.signal.source", signal["source"]),
                uncertainty=float(signal["uncertainty"]),
                scope=_nonempty("cross_layer.signal.scope", signal["scope"]),
            ))
        raw_nics = raw["cassini_by_nic"]
        if (
            not isinstance(raw_nics, list)
            or any(not isinstance(nic, list) for nic in raw_nics)
        ):
            raise ValueError("cross-layer Cassini NIC vector is invalid")
        cassini_by_nic = []
        for nic_index, raw_nic in enumerate(raw_nics):
            parsed_nic = []
            for traffic_index, item in enumerate(raw_nic):
                traffic = _mapping(
                    f"cross_layer.cassini_by_nic[{nic_index}][{traffic_index}]",
                    item,
                )
                if set(traffic) != {
                    "traffic_class", "rx_pause_fraction", "tx_pause_fraction",
                }:
                    raise ValueError("cross-layer Cassini TC inventory is invalid")
                if traffic["traffic_class"] != traffic_index:
                    raise ValueError("cross-layer Cassini TC identity is invalid")
                parsed = []
                for name in ("rx_pause_fraction", "tx_pause_fraction"):
                    value = traffic[name]
                    if value is not None and (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not 0.0 <= float(value) <= 1.0
                    ):
                        raise ValueError(
                            f"cross-layer Cassini {name} is invalid")
                    parsed.append(None if value is None else float(value))
                parsed_nic.append((traffic_index, parsed[0], parsed[1]))
            cassini_by_nic.append(tuple(parsed_nic))
        window_ms = raw["window_ms"]
        if (
            isinstance(window_ms, bool)
            or not isinstance(window_ms, (int, float))
            or not float(window_ms) > 0.0
        ):
            raise ValueError("cross-layer telemetry window is invalid")
        return CrossLayerTelemetry(
            pair_index=pair_index,
            node_id=_nonempty("cross_layer.node_id", raw["node_id"]),
            endpoint_id=_nonempty(
                "cross_layer.endpoint_id", raw["endpoint_id"]),
            communicator_id=_nonempty(
                "cross_layer.communicator_id", raw["communicator_id"]),
            source_epoch=_nonempty(
                "cross_layer.source_epoch", raw["source_epoch"]),
            topology_fingerprint_sha256=_sha256(
                "cross_layer.topology_fingerprint_sha256",
                raw["topology_fingerprint_sha256"],
            ),
            sequence=_positive_int(
                "cross_layer.sequence", raw["sequence"]),
            sampled_ns=_nonnegative_int(
                "cross_layer.sampled_ns", raw["sampled_ns"]),
            window_ms=float(window_ms),
            signals=tuple(signals),
            cassini_by_nic=tuple(cassini_by_nic),
        )

    def _parse_scheduler(self, value: object) -> dict[str, object] | None:
        if value is None:
            if self.require_scheduler_snapshot:
                raise ValueError("actual vLLM scheduler telemetry is missing")
            return None
        raw = _mapping("vllm_scheduler", value)
        required = {
            "schema", "source", "decision_mode", "model_name",
            "engine_indices", "num_requests_running",
            "num_requests_waiting", "kv_cache_usage_fraction",
        }
        if set(raw) != required:
            raise ValueError("vLLM scheduler telemetry inventory is not exact")
        if raw["schema"] != SCHEDULER_SCHEMA:
            raise ValueError("vLLM scheduler telemetry schema mismatch")
        if raw["source"] != "router_local_vllm_prometheus_observe_only":
            raise ValueError("vLLM scheduler telemetry source mismatch")
        if raw["decision_mode"] != "observe_only":
            raise ValueError("vLLM scheduler telemetry is not observe-only")
        _nonempty("vllm_scheduler.model_name", raw["model_name"])
        engines = raw["engine_indices"]
        if (
            not isinstance(engines, list)
            or not engines
            or engines != sorted(set(engines))
            or any(type(item) is not int or item < 0 for item in engines)
        ):
            raise ValueError("vLLM scheduler engine set is invalid")
        running = _nonnegative_int(
            "vllm_scheduler.num_requests_running",
            raw["num_requests_running"],
        )
        waiting = _nonnegative_int(
            "vllm_scheduler.num_requests_waiting",
            raw["num_requests_waiting"],
        )
        usage = raw["kv_cache_usage_fraction"]
        if (
            isinstance(usage, bool)
            or not isinstance(usage, (int, float))
            or not 0.0 <= float(usage) <= 1.0
        ):
            raise ValueError("vLLM scheduler KV-cache usage is invalid")
        return {
            "schema": raw["schema"],
            "source": raw["source"],
            "running_requests": running,
            "waiting_requests": waiting,
            "kv_cache_usage_fraction": float(usage),
        }

    def _parse_completion(self, value: object) -> dict[str, object] | None:
        if value is None:
            if self.require_scheduler_snapshot:
                raise ValueError("endpoint completion telemetry is missing")
            return None
        raw = _mapping("controller.completion", value)
        if set(raw) != {
            "schema", "completed_first_responses", "residual_inflight",
        }:
            raise ValueError("endpoint completion telemetry inventory is not exact")
        if raw["schema"] != COMPLETION_SCHEMA:
            raise ValueError("endpoint completion telemetry schema mismatch")
        return {
            "schema": raw["schema"],
            "completed_first_responses": _nonnegative_int(
                "completed_first_responses", raw["completed_first_responses"]),
            "residual_inflight": _nonnegative_int(
                "residual_inflight", raw["residual_inflight"]),
        }

    @staticmethod
    def _parse_route(
        value: object, *, name: str
    ) -> tuple[PathHealth, float, int, str | None]:
        route = _mapping(f"controller.routes.{name}", value)
        try:
            health = PathHealth(route.get("state"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"endpoint {name} route health is invalid") from exc
        multiplier = route.get("service_multiplier")
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, (int, float))
            or not 1.0 <= float(multiplier) < float("inf")
        ):
            raise ValueError(
                f"endpoint {name} service multiplier is invalid")
        failure_count = _nonnegative_int(
            f"endpoint {name} failures", route.get("failures", 0))
        last_failure_kind = route.get("last_failure_kind")
        if last_failure_kind is not None:
            last_failure_kind = _nonempty(
                f"endpoint {name} last_failure_kind", last_failure_kind)
        return health, float(multiplier), failure_count, last_failure_kind


__all__ = [
    "BATCH_SCHEMA",
    "ENDPOINT_CONTROLLER_SCHEMA",
    "COMPLETION_SCHEMA",
    "EndpointTelemetryContract",
    "FRONTEND_LEDGER_SCHEMA",
    "GlobalTelemetryAdapter",
    "GlobalTelemetryBatch",
    "SCHEDULER_SCHEMA",
]
