from __future__ import annotations

import copy

import pytest

from eval.sota_4node import analyze_tempo_pd_c4_semantic_epoch_screen as analyzer


def _contract():
    return {
        "semantic_credit_contract": dict(
            analyzer.contract_builder.semantic_profile_builder.SEMANTIC_ROUTING_POLICY
        ),
        "endpoint_service_profile": {
            "fingerprint_sha256": "a" * 64,
        },
    }


def _decision():
    return {
        "route": analyzer.REMOTE_ROUTE,
        "reason": "semantic_epoch_open_remote_high_water",
        "endpoint_routing_policy": "semantic_epoch_v1",
        "semantic_epoch_applied": True,
        "semantic_epoch_schema": analyzer.EPOCH_SCHEMA,
        "semantic_epoch_policy": "semantic_epoch_v1",
        "semantic_epoch_profile_fingerprint_sha256": "a" * 64,
        "endpoint_service_profile_fingerprint_sha256": "a" * 64,
        "frontend_semantic_load_schema": analyzer.LOAD_SCHEMA,
        "frontend_semantic_load_source": analyzer.LOAD_SOURCE,
        "semantic_epoch_active_requests_before": 9,
        "semantic_epoch_decode_tokens_before": 1024,
        "semantic_epoch_max_num_seqs": 16,
        "frontend_semantic_active_requests_before": 9,
        "frontend_semantic_decode_tokens_before": 1024,
        "frontend_semantic_max_num_seqs": 16,
        "semantic_epoch_decoder_high_water": True,
        "semantic_epoch_decoder_low_water": False,
        "semantic_epoch_decoder_high_water_numerator": 1,
        "semantic_epoch_decoder_high_water_denominator": 2,
        "semantic_epoch_decoder_low_water_numerator": 1,
        "semantic_epoch_decoder_low_water_denominator": 4,
        "semantic_epoch_confirmation_requests": 2,
        "semantic_epoch_overload_multiplier": 2.0,
        "semantic_epoch_remote_external_credit_close_fraction": 1.0,
        "semantic_epoch_route_after": analyzer.REMOTE_ROUTE,
        "endpoint_decision_route": analyzer.REMOTE_ROUTE,
        "semantic_epoch_reason": "semantic_epoch_open_remote_high_water",
        "endpoint_request_local_allowed": False,
        "endpoint_request_remote_allowed": True,
        "semantic_epoch_generation": 1,
        "admission_credit_release_event": "first_response_chunk",
        "admission_credit_released_ns": 123,
        "endpoint_feedback_event": "first_response_chunk",
        "endpoint_external_credit_registered": False,
    }


def test_semantic_decision_is_exactly_bound_to_pair_load_and_route():
    route, reason, generation = analyzer._validate_semantic_decision(
        _decision(), contract=_contract())
    assert route == analyzer.REMOTE_ROUTE
    assert reason == "semantic_epoch_open_remote_high_water"
    assert generation == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "semantic_epoch_profile_fingerprint_sha256",
            "b" * 64,
            "profile binding",
        ),
        ("semantic_epoch_decoder_high_water", False, "watermarks"),
        ("semantic_epoch_decoder_high_water_numerator", 2, "watermarks"),
        ("semantic_epoch_confirmation_requests", 3, "confirmation"),
        ("semantic_epoch_overload_multiplier", 1.9, "overload"),
        ("endpoint_request_remote_allowed", False, "route latch"),
        ("endpoint_feedback_event", "complete", "credit lifecycle"),
    ),
)
def test_semantic_decision_drift_fails_closed(field, value, message):
    decision = copy.deepcopy(_decision())
    decision[field] = value
    with pytest.raises(ValueError, match=message):
        analyzer._validate_semantic_decision(decision, contract=_contract())


def test_external_proxy_provenance_is_geometry_bounded_and_explicit():
    request_id = "epd-local-cache-miss-endpoint-observed-background"
    decision = {
        "route": analyzer.LOCAL_ROUTE,
        "endpoint_passive_feedback_enabled": True,
        "endpoint_passive_registered": True,
        "endpoint_external_credit_registered": True,
        "endpoint_feedback_passive": True,
        "endpoint_feedback_event": (
            "external_credit_passive_first_response_chunk"),
        "admission_credit_scope": None,
        "admission_credit_release_event": None,
        "endpoint_passive_service_lookup_mode": (
            "miss_via_prefill_only_geometry_ceiling"),
        "endpoint_passive_service_source_prompt_tokens": 4094,
        "endpoint_passive_service_source_output_tokens": 16,
        "endpoint_passive_service_source_cache_residency": "prefill_only",
    }
    metadata = {
        "prompt_tokens": 4094,
        "output_tokens": 2,
        "cache_state": "miss",
    }
    route, mode = analyzer._validate_external_decision(
        decision, metadata=metadata, request_id=request_id)
    assert route == analyzer.LOCAL_ROUTE
    assert mode == "miss_via_prefill_only_geometry_ceiling"

    decision["endpoint_passive_service_source_output_tokens"] = 1
    with pytest.raises(ValueError, match="proxy geometry"):
        analyzer._validate_external_decision(
            decision, metadata=metadata, request_id=request_id)
