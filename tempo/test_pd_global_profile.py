from __future__ import annotations

from copy import deepcopy
import json

import pytest

from tempo.pd_global_profile import (
    SCHEMA,
    SERVICE_PROXY_POLICY_ID,
    TRANSPORT,
    global_profile_fingerprint,
    load_global_profile,
)


def raw_profile(*, scope: str = "discovery") -> dict[str, object]:
    value = {
        "schema": SCHEMA,
        "profile_id": "tempo-go-test-v1",
        "deployment_scope": scope,
        "transport": TRANSPORT,
        "topology": {
            "node_count": 4,
            "gpu_count": 16,
            "pair_count": 2,
            "prewarmed_pair_count": 2,
            "native_only": True,
            "route_immutable": True,
            "privileged_nic_control": False,
        },
        "causality": {
            "telemetry_clock": "frontend_perf_counter_interval_start",
            "decoder_credit_scope": "request_start_to_http_eof",
            "endpoint_credit_scope": "route_commit_to_first_response",
            "phase_label_policy_input": False,
            "physical_switch_label_policy_input": False,
            "future_arrivals_policy_input": False,
            "oracle_policy_input": False,
        },
        "identity": {
            "router_schema": "tempo-elastic-pd-router-canonical",
            "endpoint_profile_schema": "tempo-pd-endpoint-service-profile-v2",
            "endpoint_profile_id": "endpoint-test-v1",
            "endpoint_profile_fingerprint_sha256": "a" * 64,
            "endpoint_profile_deployment_scope": (
                "frozen_validation" if scope == "frozen_validation"
                else "calibration_only"),
            "elastic_profile_fingerprint_sha256": "b" * 64,
            "workload_manifest_sha256": "c" * 64,
            "model_config_sha256": "d" * 64,
        },
        "telemetry": {
            "agent_epoch_source": "slurm_job_id_frontend_start_ns",
            "freshness_ns": 100_000_000,
            "refresh_timeout_ns": 50_000_000,
            "maximum_collection_span_ns": 20_000_000,
            "tokenizer_timeout_ns": 5_000_000_000,
            "controller_generation": 0,
            "endpoint_feedback_mode": "adaptive",
            "endpoint_routing_policy": "instant_score_v1",
        },
        "capacities": [
            {
                "pair_index": pair,
                "decode_tokens": 2_048,
                "active_sequences": 8,
                "endpoint_requests": 8,
                "local_prefill_token_ms": 16_000_000,
                "remote_prefill_token_ms": 12_000_000,
                "remote_kv_bytes": 900_000_000,
                "remote_semantic_ops": 4,
            }
            for pair in range(2)
        ],
        "tenants": [
            {"tenant_id": "latency", "weight": 2.0},
            {"tenant_id": "batch", "weight": 1.0},
        ],
        "controller": {
            "queue_capacity": 64,
            "minimum_active_pairs": 1,
            "maximum_active_pairs": 2,
            "scale_up_utilization": 0.75,
            "scale_down_idle_ns": 5_000_000_000,
            "utilization_penalty_ms": 100.0,
            "activation_penalty_ms": 1.0,
            "probe_penalty_ms": 10.0,
            "maximum_queue_wait_ns": 5_000_000_000,
        },
    }
    value["fingerprint_sha256"] = global_profile_fingerprint(value)
    return value


def write_profile(tmp_path, value: dict[str, object]):
    path = tmp_path / "global-profile.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_exact_profile_builds_controller_and_endpoint_contracts(tmp_path) -> None:
    raw = raw_profile()
    profile = load_global_profile(write_profile(tmp_path, raw))
    assert profile.fingerprint_sha256 == raw["fingerprint_sha256"]
    config = profile.orchestrator_config()
    assert [item.pair_index for item in config.capacities] == [0, 1]
    assert config.capacities[0].resources.endpoint_requests == 8
    contracts = profile.endpoint_contracts()
    assert [item.pair_index for item in contracts] == [0, 1]
    assert contracts[0].profile_fingerprint_sha256 == "a" * 64
    assert profile.telemetry_adapter(agent_epoch="slurm-123-start-1").contracts == (
        contracts)


def test_single_inference_pair_profile_supports_external_cojob_scope(tmp_path) -> None:
    raw = raw_profile()
    raw["topology"].update({"pair_count": 1, "prewarmed_pair_count": 1})
    raw["capacities"] = raw["capacities"][:1]
    raw["controller"].update({
        "maximum_active_pairs": 1,
    })
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    profile = load_global_profile(write_profile(tmp_path, raw))
    assert profile.topology.pair_count == 1
    assert [item.pair_index for item in profile.endpoint_contracts()] == [0]
    assert profile.orchestrator_config().maximum_active_pairs == 1


def test_optional_scheduler_and_business_tenant_contract_round_trip(tmp_path) -> None:
    raw = raw_profile()
    raw["telemetry"]["scheduler_observation_required"] = True
    raw["tenants"][0].update({
        "ttft_slo_ms": 1000.0,
        "tpot_slo_ms": 100.0,
        "e2e_slo_ms": 4000.0,
        "maximum_queue_wait_ns": 500_000_000,
        "minimum_service_fraction": 0.15,
        "telemetry_stale_grace_ns": 2_000_000_000,
        "pair_spread_limit": 1,
    })
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    profile = load_global_profile(write_profile(tmp_path, raw))
    assert profile.telemetry.scheduler_observation_required is True
    assert profile.tenants[0].e2e_slo_ms == 4000.0
    assert profile.tenants[0].minimum_service_fraction == 0.15
    assert profile.tenants[0].telemetry_stale_grace_ns == 2_000_000_000
    assert profile.tenants[0].pair_spread_limit == 1
    assert profile.telemetry_adapter(
        agent_epoch="slurm-123-start-1").require_scheduler_snapshot is True


def test_remote_semantic_ops_safety_reserve_round_trips(tmp_path) -> None:
    raw = raw_profile()
    raw["controller"]["remote_semantic_ops_safety_reserve"] = 1
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    profile = load_global_profile(write_profile(tmp_path, raw))
    config = profile.orchestrator_config()
    assert config.remote_semantic_ops_safety_reserve == 1
    assert config.capacities[0].resources.remote_semantic_ops == 4


def _proxy_policy() -> dict[str, object]:
    return {
        "policy_id": SERVICE_PROXY_POLICY_ID,
        "endpoint_profile_id": "endpoint-test-v1",
        "endpoint_profile_fingerprint_sha256": "a" * 64,
        "calibration_receipt_sha256": "e" * 64,
        "allowed_lookup_modes": [
            "exact", "miss_via_prefill_only_geometry_ceiling",
        ],
        "allowed_cache_residencies": ["confirmed_miss", "prefill_only"],
        "allowed_remote_cache_residencies": ["prefill_only"],
        "allowed_geometries": [[10, 64]],
        "proxy_is_not_exact": True,
        "numeric_rows_unchanged": True,
        "performance_claim_allowed": False,
    }


def test_frozen_service_proxy_policy_is_audited_and_not_controller_input(
    tmp_path,
) -> None:
    raw = raw_profile()
    raw["controller"]["frozen_service_proxy_policy"] = _proxy_policy()
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    profile = load_global_profile(write_profile(tmp_path, raw))
    policy = profile.service_proxy_policy()
    assert policy is not None
    assert policy.allows_geometry(10, 64)
    assert profile.orchestrator_config().queue_capacity == 64


def test_frozen_profile_requires_explicit_proxy_contract(tmp_path) -> None:
    raw = raw_profile(scope="frozen_validation")
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    with pytest.raises(ValueError, match="explicit service proxy policy"):
        load_global_profile(write_profile(tmp_path, raw))


def test_proxy_policy_cannot_authorize_a_performance_claim(tmp_path) -> None:
    raw = raw_profile()
    policy = _proxy_policy()
    policy["performance_claim_allowed"] = True
    raw["controller"]["frozen_service_proxy_policy"] = policy
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    with pytest.raises(ValueError, match="performance claims"):
        load_global_profile(write_profile(tmp_path, raw))


def test_queue_slo_pair_scale_controls_round_trip(tmp_path) -> None:
    raw = raw_profile()
    raw["controller"].update({
        "proactive_scale_up_queue_fraction": 0.25,
        "proactive_scale_up_wait_fraction": 0.25,
        "proactive_scale_up_active_pair_penalty_ms": 25.0,
    })
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    profile = load_global_profile(write_profile(tmp_path, raw))
    config = profile.orchestrator_config()
    assert config.proactive_scale_up_queue_fraction == 0.25
    assert config.proactive_scale_up_wait_fraction == 0.25
    assert config.proactive_scale_up_active_pair_penalty_ms == 25.0


def test_priority_remote_cache_service_lane_round_trips(tmp_path) -> None:
    raw = raw_profile()
    raw["controller"].update({
        "overload_action": "endpoint_queue_lease",
        "priority_service_lane_mode": "vllm_priority_remote_cache_v1",
        "priority_service_lane_capacity": 8,
        "priority_service_lane_min_admission_priority": 800,
        "priority_service_lane_priority": -2,
    })
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    config = load_global_profile(
        write_profile(tmp_path, raw)).orchestrator_config()
    assert config.priority_service_lane_mode == (
        "vllm_priority_remote_cache_v1")
    assert config.priority_service_lane_capacity == 8
    assert config.priority_service_lane_min_admission_priority == 800
    assert config.priority_service_lane_priority == -2


def test_mesh_near_tie_source_balance_round_trips(tmp_path) -> None:
    raw = raw_profile()
    raw["controller"].update({
        "mesh_control_mode": "receiver_credit_pxd_v1",
        "mesh_near_tie_source_balance_mode": (
            "telemetry_uncertainty_virtual_service_v1"),
        "mesh_near_tie_source_balance_uncertainty_fraction": 1.0,
    })
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    config = load_global_profile(
        write_profile(tmp_path, raw)).orchestrator_config()
    assert config.mesh_near_tie_source_balance_mode == (
        "telemetry_uncertainty_virtual_service_v1")
    assert config.mesh_near_tie_source_balance_uncertainty_fraction == 1.0


def test_mesh_near_tie_source_balance_requires_mesh_and_bound(tmp_path) -> None:
    raw = raw_profile()
    raw["controller"].update({
        "mesh_near_tie_source_balance_mode": (
            "telemetry_uncertainty_virtual_service_v1"),
        "mesh_near_tie_source_balance_uncertainty_fraction": 1.0,
    })
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    with pytest.raises(ValueError, match="near-tie source balance requires"):
        load_global_profile(write_profile(tmp_path, raw))


def test_priority_service_lane_requires_real_queue_and_scheduler_action(
    tmp_path,
) -> None:
    raw = raw_profile()
    raw["controller"].update({
        "priority_service_lane_mode": "vllm_priority_remote_cache_v1",
        "priority_service_lane_capacity": 8,
        "priority_service_lane_min_admission_priority": 800,
        "priority_service_lane_priority": 0,
    })
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    with pytest.raises(ValueError, match="priority service lane requires"):
        load_global_profile(write_profile(tmp_path, raw))


def test_telemetry_failure_survivor_controls_round_trip(tmp_path) -> None:
    raw = raw_profile()
    raw["controller"].update({
        "telemetry_failure_quarantine_mode": "deny_until_probe",
        "telemetry_failure_quarantine_scope": "pair",
        "survivor_capacity_reserve_fraction": 0.25,
        "survivor_reserve_bypass_min_weight": 2.0,
    })
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    profile = load_global_profile(write_profile(tmp_path, raw))
    config = profile.orchestrator_config()
    assert config.telemetry_failure_quarantine_mode == "deny_until_probe"
    assert config.telemetry_failure_quarantine_scope == "pair"
    assert config.survivor_capacity_reserve_fraction == 0.25
    assert config.survivor_reserve_bypass_min_weight == 2.0


def test_remote_semantic_ops_safety_reserve_cannot_consume_whole_window(
    tmp_path,
) -> None:
    raw = raw_profile()
    raw["controller"]["remote_semantic_ops_safety_reserve"] = 4
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    with pytest.raises(ValueError, match="leaves no admission slot"):
        load_global_profile(write_profile(tmp_path, raw))


def test_any_profile_mutation_requires_a_new_fingerprint(tmp_path) -> None:
    raw = raw_profile()
    raw["controller"]["queue_capacity"] = 65
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_global_profile(write_profile(tmp_path, raw))


@pytest.mark.parametrize(
    ("section", "name", "value", "message"),
    (
        ("topology", "native_only", False, "native execution"),
        ("topology", "privileged_nic_control", True, "privileged NIC"),
        ("causality", "phase_label_policy_input", True, "non-causal"),
        (
            "causality", "physical_switch_label_policy_input", True,
            "non-causal",
        ),
        ("causality", "future_arrivals_policy_input", True, "non-causal"),
        ("causality", "oracle_policy_input", True, "non-causal"),
    ),
)
def test_safety_and_causality_are_hard_contracts(
    tmp_path, section: str, name: str, value: object, message: str,
) -> None:
    raw = raw_profile()
    raw[section][name] = value
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    with pytest.raises(ValueError, match=message):
        load_global_profile(write_profile(tmp_path, raw))


def test_inventory_is_exact(tmp_path) -> None:
    raw = raw_profile()
    raw["telemetry"]["phase_label"] = "hot"
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    with pytest.raises(ValueError, match="telemetry inventory"):
        load_global_profile(write_profile(tmp_path, raw))


def test_frozen_validation_requires_frozen_endpoint_identity(tmp_path) -> None:
    raw = raw_profile(scope="frozen_validation")
    raw["identity"]["endpoint_profile_deployment_scope"] = "calibration_only"
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    with pytest.raises(ValueError, match="frozen endpoint"):
        load_global_profile(write_profile(tmp_path, raw))


def test_two_pair_contract_cannot_be_silently_reordered(tmp_path) -> None:
    raw = raw_profile()
    raw["capacities"] = list(reversed(raw["capacities"]))
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    with pytest.raises(ValueError, match="pair capacities"):
        load_global_profile(write_profile(tmp_path, raw))


def test_unknown_top_level_field_fails_even_with_matching_hash(tmp_path) -> None:
    raw = deepcopy(raw_profile())
    raw["physical_switch_label"] = "future-oracle"
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    with pytest.raises(ValueError, match="top-level inventory"):
        load_global_profile(write_profile(tmp_path, raw))
