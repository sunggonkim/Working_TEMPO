from __future__ import annotations

from copy import deepcopy

import pytest

from tempo.pd_global_orchestrator import GlobalOrchestrator
from tempo.pd_global_telemetry import (
    ENDPOINT_CONTROLLER_SCHEMA,
    FRONTEND_LEDGER_SCHEMA,
    EndpointTelemetryContract,
    GlobalTelemetryAdapter,
)
from tempo.test_pd_global_orchestrator import controller


ROUTER_SCHEMA = "tempo-elastic-pd-router-canonical"
PROFILE_SCHEMA = "tempo-pd-endpoint-service-profile-v2"
PROFILE_ID = "tempo-go-test-profile"
PROFILE_SHA = "a" * 64
ELASTIC_SHA = "b" * 64
WORKLOAD_SHA = "c" * 64


def contract(pair_index: int) -> EndpointTelemetryContract:
    return EndpointTelemetryContract(
        pair_index=pair_index,
        router_schema=ROUTER_SCHEMA,
        endpoint_feedback_mode="adaptive",
        endpoint_routing_policy="global_commit_v1",
        profile_schema=PROFILE_SCHEMA,
        profile_id=PROFILE_ID,
        profile_fingerprint_sha256=PROFILE_SHA,
        elastic_profile_fingerprint_sha256=ELASTIC_SHA,
        workload_manifest_sha256=WORKLOAD_SHA,
        deployment_scope="discovery",
        controller_generation=3,
    )


def adapter(*, require_scheduler: bool = False) -> GlobalTelemetryAdapter:
    return GlobalTelemetryAdapter(
        (contract(0), contract(1)),
        agent_epoch="slurm-1234-frontend",
        maximum_collection_span_ns=1_000,
        require_scheduler_snapshot=require_scheduler,
    )


def frontend() -> dict[str, object]:
    return {
        "schema": FRONTEND_LEDGER_SCHEMA,
        "loads": [120, 40],
        "active": 3,
        "active_by_pair": [2, 1],
    }


def endpoint(pair_index: int) -> dict[str, object]:
    resources = {
        "local_token_ms": 10 + pair_index,
        "remote_prefill_token_ms": 20 + pair_index,
        "remote_kv_bytes": 300 + pair_index,
        "remote_semantic_ops": 2,
    }
    owned = {
        "local_token_ms": 4,
        "remote_prefill_token_ms": 5,
        "remote_kv_bytes": 100,
        "remote_semantic_ops": 1,
    }
    external = {
        name: resources[name] - owned[name] for name in resources
    }
    return {
        "schema": ROUTER_SCHEMA,
        "pair_index": pair_index,
        "endpoint_feedback_mode": "adaptive",
        "endpoint_routing_policy": "global_commit_v1",
        "endpoint_service_profile": {
            "schema": PROFILE_SCHEMA,
            "profile_id": PROFILE_ID,
            "fingerprint_sha256": PROFILE_SHA,
            "elastic_profile_fingerprint_sha256": ELASTIC_SHA,
            "workload_manifest_sha256": WORKLOAD_SHA,
            "deployment_scope": "discovery",
            "routing_policy": {
                "phase_label_policy_input": False,
                "physical_switch_label_policy_input": False,
                "future_arrivals_policy_input": False,
                "oracle_policy_input": False,
            },
        },
        "controller": {
            "schema": ENDPOINT_CONTROLLER_SCHEMA,
            "resources": resources,
            "owned_resources": owned,
            "external_resources": external,
            "inflight": 1,
            "external_inflight": 2,
            "routes": {
                "decoder_local_chunked_prefill": {
                    "state": "good",
                    "service_multiplier": 1.25,
                },
                "official_lmcache_remote_prefill": {
                    "state": "probe" if pair_index == 0 else "good",
                    "service_multiplier": 1.5,
                },
            },
        },
        "controller_generation": 3,
        "queued_requests": 1,
        "passive_registered_requests": 0,
    }


def endpoints() -> dict[int, dict[str, object]]:
    return {0: endpoint(0), 1: endpoint(1)}


def live_endpoints() -> dict[int, dict[str, object]]:
    values = endpoints()
    for pair_index, value in values.items():
        value["vllm_scheduler"] = {
            "schema": "tempo-go-vllm-scheduler-snapshot-v1",
            "source": "router_local_vllm_prometheus_observe_only",
            "decision_mode": "observe_only",
            "model_name": "Qwen2.5-7B-Instruct",
            "engine_indices": [0],
            "num_requests_running": 2 + pair_index,
            "num_requests_waiting": pair_index,
            "kv_cache_usage_fraction": 0.25,
        }
        value["controller"]["completion"] = {
            "schema": "tempo-go-endpoint-completion-v1",
            "completed_first_responses": 10 + pair_index,
            "residual_inflight": 3 + pair_index,
        }
    return values


def test_valid_batch_joins_eof_and_first_response_ownership() -> None:
    batch = adapter().assemble(
        frontend(), endpoints(),
        collection_started_ns=10_000,
        collection_finished_ns=10_200,
    )
    assert batch.sequence == 1
    assert batch.sampled_ns == 10_000
    first = batch.pairs[0]
    assert first.observed_total.decode_tokens == 120
    assert first.observed_total.active_sequences == 2
    assert first.observed_total.endpoint_requests == 4
    assert first.observed_total.local_prefill_token_ms == 10
    assert first.observed_total.remote_prefill_token_ms == 20
    assert first.observed_total.remote_kv_bytes == 300
    assert first.local_service_multiplier == 1.25
    assert first.remote_service_multiplier == 1.5
    assert first.remote_health.value == "probe"
    evidence = batch.as_dict()
    assert evidence["collection_span_ns"] == 200
    assert evidence["pairs"][0]["source"] == "application_endpoint_agent"


def test_endpoint_quarantine_is_atomic_and_does_not_reuse_stale_totals() -> None:
    batch = adapter().assemble(
        frontend(), {0: endpoint(0)},
        collection_started_ns=11_000,
        collection_finished_ns=11_100,
        quarantined_pairs={1: "endpoint_fetch:ConnectError"},
    )
    healthy, denied = batch.pairs
    assert healthy.local_health.value == "good"
    assert denied.local_health.value == "denied"
    assert denied.remote_health.value == "denied"
    assert denied.quarantine_reason == "endpoint_fetch:ConnectError"
    assert denied.observed_total.endpoint_requests == 0
    assert denied.observed_total.remote_kv_bytes == 0
    assert denied.scheduler_running_requests is None
    assert batch.agent_epoch == "slurm-1234-frontend"


def test_route_failure_provenance_survives_telemetry_assembly() -> None:
    raw = endpoints()
    remote = raw[0]["controller"]["routes"][
        "official_lmcache_remote_prefill"]
    remote.update({
        "state": "denied",
        "failures": 2,
        "last_failure_kind": "active_upstream_failure",
    })
    batch = adapter().assemble(
        frontend(), raw,
        collection_started_ns=12_000,
        collection_finished_ns=12_100,
    )
    pair = batch.pairs[0]
    assert pair.remote_health.value == "denied"
    assert pair.remote_failure_count == 2
    assert pair.remote_last_failure_kind == "active_upstream_failure"
    encoded = batch.as_dict()["pairs"][0]["route_failures"]
    assert encoded["remote_count"] == 2
    assert encoded["remote_last_kind"] == "active_upstream_failure"


def test_complete_batch_installs_atomically_in_global_controller() -> None:
    value: GlobalOrchestrator = controller()
    batch = adapter().assemble(
        frontend(), endpoints(),
        collection_started_ns=10,
        collection_finished_ns=11,
    )
    value.update_telemetry_batch(batch.pairs)
    state = value.snapshot(now_ns=12)
    assert state["telemetry_sequences"] == {"0": 1, "1": 1}
    assert state["telemetry_provenance"]["0"][
        "profile_fingerprint_sha256"] == PROFILE_SHA
    with pytest.raises(ValueError, match="every configured pair"):
        value.update_telemetry_batch((batch.pairs[0],))


def test_required_live_scheduler_and_completion_telemetry_is_captured() -> None:
    batch = adapter(require_scheduler=True).assemble(
        frontend(), live_endpoints(),
        collection_started_ns=10_000,
        collection_finished_ns=10_200,
    )
    pair = batch.pairs[1]
    assert pair.scheduler_running_requests == 3
    assert pair.scheduler_waiting_requests == 1
    assert pair.scheduler_kv_cache_usage_fraction == 0.25
    assert pair.endpoint_completed_first_responses == 11
    assert pair.endpoint_residual_inflight == 4
    assert batch.as_dict()["pairs"][1]["scheduler"]["source"] == (
        "router_local_vllm_prometheus_observe_only")


def test_required_live_scheduler_missing_fails_closed() -> None:
    with pytest.raises(ValueError, match="scheduler telemetry is missing"):
        adapter(require_scheduler=True).assemble(
            frontend(), endpoints(),
            collection_started_ns=10,
            collection_finished_ns=11,
        )


def test_mesh_edge_resources_are_aggregated_by_decoder_destination() -> None:
    raw = live_endpoints()
    for source in (0, 1):
        raw[source]["mesh_remote_by_decoder"] = {
            "0": {
                "remote_prefill_token_ms": 100 + source,
                "remote_kv_bytes": 1_000 + source,
                "remote_semantic_ops": 2,
            },
            "1": {
                "remote_prefill_token_ms": 0,
                "remote_kv_bytes": 0,
                "remote_semantic_ops": 0,
            },
        }
    batch = adapter(require_scheduler=True).assemble(
        frontend(), raw,
        collection_started_ns=10_000,
        collection_finished_ns=10_200,
    )
    # Source P_i keeps its outbound prefill credit; D0 receives the summed
    # inbound receiver resources from both P0 and P1.  D1 stays clean.
    assert batch.pairs[0].observed_total.remote_prefill_token_ms == 20
    assert batch.pairs[1].observed_total.remote_prefill_token_ms == 21
    assert batch.pairs[0].observed_total.remote_kv_bytes == 2_001
    assert batch.pairs[0].observed_total.remote_semantic_ops == 4
    assert batch.pairs[1].observed_total.remote_kv_bytes == 0
    assert batch.pairs[1].observed_total.remote_semantic_ops == 0
    assert batch.pairs[0].observed_total.endpoint_requests == 5
    assert batch.pairs[1].observed_total.endpoint_requests == 8


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda raw: raw.update(schema="wrong"), "router telemetry schema"),
        (lambda raw: raw.update(pair_index=1), "identity mismatch"),
        (
            lambda raw: raw["endpoint_service_profile"].update(
                fingerprint_sha256="d" * 64),
            "profile fingerprint_sha256",
        ),
        (
            lambda raw: raw.update(controller_generation=4),
            "generation differs",
        ),
        (
            lambda raw: raw["controller"].update(schema="wrong"),
            "controller telemetry schema",
        ),
    ),
)
def test_identity_and_schema_mismatch_fail_closed(mutation, message: str) -> None:
    raw = endpoints()
    mutation(raw[0])
    with pytest.raises(ValueError, match=message):
        adapter().assemble(
            frontend(), raw,
            collection_started_ns=10,
            collection_finished_ns=11,
        )


def test_endpoint_total_must_equal_owned_plus_external() -> None:
    raw = endpoints()
    raw[0]["controller"]["resources"]["remote_kv_bytes"] += 1
    with pytest.raises(ValueError, match="owned plus external"):
        adapter().assemble(
            frontend(), raw,
            collection_started_ns=10,
            collection_finished_ns=11,
        )


def test_frontend_snapshot_must_be_coherent_and_versioned() -> None:
    for name, value, message in (
        ("schema", "wrong", "frontend telemetry schema"),
        ("active", 4, "active request totals"),
        ("loads", [1], "exactly 2 entries"),
    ):
        raw = frontend()
        raw[name] = value
        with pytest.raises(ValueError, match=message):
            adapter().assemble(
                raw, endpoints(),
                collection_started_ns=10,
                collection_finished_ns=11,
            )


def test_noncausal_profile_input_is_rejected() -> None:
    raw = endpoints()
    raw[1]["endpoint_service_profile"]["routing_policy"][
        "physical_switch_label_policy_input"] = True
    with pytest.raises(ValueError, match="non-causal"):
        adapter().assemble(
            frontend(), raw,
            collection_started_ns=10,
            collection_finished_ns=11,
        )


def test_failed_assembly_does_not_advance_sequence() -> None:
    value = adapter()
    bad = endpoints()
    bad[1]["controller_generation"] = 4
    with pytest.raises(ValueError):
        value.assemble(
            frontend(), bad,
            collection_started_ns=10,
            collection_finished_ns=11,
        )
    good = value.assemble(
        frontend(), endpoints(),
        collection_started_ns=12,
        collection_finished_ns=13,
    )
    assert good.sequence == 1


def test_collection_span_and_order_are_conservative() -> None:
    value = adapter()
    with pytest.raises(ValueError, match="causal span"):
        value.assemble(
            frontend(), endpoints(),
            collection_started_ns=10,
            collection_finished_ns=1_011,
        )
    first = value.assemble(
        frontend(), endpoints(),
        collection_started_ns=2_000,
        collection_finished_ns=2_100,
    )
    assert first.sequence == 1
    with pytest.raises(ValueError, match="overlap"):
        value.assemble(
            frontend(), endpoints(),
            collection_started_ns=2_099,
            collection_finished_ns=2_101,
        )


def test_contracts_must_describe_one_coherent_deployment() -> None:
    second = deepcopy(contract(1))
    object.__setattr__(second, "workload_manifest_sha256", "d" * 64)
    with pytest.raises(ValueError, match="mixed workload_manifest_sha256"):
        GlobalTelemetryAdapter(
            (contract(0), second),
            agent_epoch="epoch",
            maximum_collection_span_ns=10,
        )
