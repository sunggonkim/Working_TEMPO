from __future__ import annotations

import json

import pytest

from tempo.pd_elastic_controller_v443 import CacheResidency
from tempo.pd_endpoint_profile import (
    SCHEMA,
    SCHEMA_V2,
    endpoint_service_profile_fingerprint,
    load_endpoint_service_profile,
)
from tempo.pd_endpoint_controller import EndpointRoute


def _payload() -> dict[str, object]:
    payload = {
        "schema": SCHEMA,
        "profile_id": "endpoint-profile-test",
        "elastic_profile_fingerprint_sha256": "1" * 64,
        "workload_manifest_sha256": "2" * 64,
        "deployment_scope": "calibration_only",
        "default_e2e_deadline_ms": 5_000.0,
        "controller": {
            "local_token_ms_window": 1_000,
            "remote_prefill_token_ms_window": 1_000,
            "remote_kv_bytes_window": 1_000_000,
            "remote_semantic_ops_window": 4,
            "feedback_history": 8,
            "feedback_quantile": 0.75,
            "minimum_feedback": 2,
            "route_margin_ms": 5.0,
            "feedback_fresh_ns": 100_000_000,
            "probe_after_ns": 50_000_000,
            "denied_probe_after_ns": 500_000_000,
        },
        "rows": [
            {
                "prompt_tokens": 4094,
                "output_tokens": 16,
                "cache_residency": "confirmed_miss",
                "local_ttft_prior_ms": 150.0,
                "remote_ttft_prior_ms": 300.0,
                "local_token_ms": 614_100,
                "remote_prefill_token_ms": 1_228_200,
                "samples_local": 4,
                "samples_remote": 4,
                "outputs_equivalent": True,
                "evidence_valid": True,
            }
        ],
    }
    payload["fingerprint_sha256"] = endpoint_service_profile_fingerprint(payload)
    return payload


def _semantic_payload() -> dict[str, object]:
    payload = _payload()
    payload["schema"] = SCHEMA_V2
    payload["routing_policy"] = {
        "policy": "semantic_epoch_v1",
        "pair_local": True,
        "decoder_load_scope": "frontend_request_start_to_http_eof",
        "endpoint_credit_scope": (
            "all_route_pinned_and_tempo_work_to_first_response"),
        "decoder_high_water_numerator": 1,
        "decoder_high_water_denominator": 2,
        "decoder_low_water_numerator": 1,
        "decoder_low_water_denominator": 4,
        "epoch_confirmation_requests": 2,
        "remote_overload_service_stretch": 2.0,
        "remote_external_credit_close_fraction": 1.0,
        "phase_label_policy_input": False,
        "physical_switch_label_policy_input": False,
    }
    payload["fingerprint_sha256"] = endpoint_service_profile_fingerprint(
        payload)
    return payload


def test_profile_is_fingerprint_bound_and_exact(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    profile = load_endpoint_service_profile(path)
    row = profile.exact_row(4094, 16, CacheResidency.MISS)
    assert row.local_ttft_prior_ms == 150.0
    assert profile.exact_row(
        4094, 16, CacheResidency.UNKNOWN, cold_unknown_as_miss=True
    ) is row
    with pytest.raises(ValueError, match="no exact"):
        profile.exact_row(4094, 16, CacheResidency.UNKNOWN)


def test_profile_rejects_tampering_and_unsafe_rows(tmp_path) -> None:
    payload = _payload()
    payload["rows"][0]["local_ttft_prior_ms"] = 151.0
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_endpoint_service_profile(path)

    payload = _payload()
    payload["rows"][0]["evidence_valid"] = False
    payload["fingerprint_sha256"] = endpoint_service_profile_fingerprint(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="valid equivalent"):
        load_endpoint_service_profile(path)


def test_profile_inventory_is_fail_closed(tmp_path) -> None:
    payload = _payload()
    payload["unexpected"] = 1
    payload["fingerprint_sha256"] = endpoint_service_profile_fingerprint(payload)
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="top-level inventory"):
        load_endpoint_service_profile(path)


def test_external_credit_proxy_is_explicit_and_bounded(tmp_path) -> None:
    payload = _payload()
    payload["rows"][0]["cache_residency"] = "prefill_only"
    payload["fingerprint_sha256"] = endpoint_service_profile_fingerprint(payload)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    profile = load_endpoint_service_profile(path)

    p_only = profile.external_credit_proxy(
        4094, 2, CacheResidency.P_ONLY, route=EndpointRoute.REMOTE)
    assert p_only.lookup_mode == "same_residency_geometry_ceiling"
    assert p_only.row.output_tokens == 16
    miss = profile.external_credit_proxy(
        4094, 2, CacheResidency.MISS, route=EndpointRoute.LOCAL)
    assert miss.lookup_mode == "miss_via_prefill_only_geometry_ceiling"
    assert miss.requested_cache_residency is CacheResidency.MISS

    with pytest.raises(ValueError, match="no safe external"):
        profile.external_credit_proxy(
            4095, 2, CacheResidency.MISS, route=EndpointRoute.LOCAL)
    with pytest.raises(ValueError, match="no safe external"):
        profile.external_credit_proxy(
            4094, 2, CacheResidency.D_ONLY, route=EndpointRoute.REMOTE)


def test_v2_semantic_policy_is_fingerprint_bound_and_exact(tmp_path) -> None:
    payload = _semantic_payload()
    path = tmp_path / "semantic.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    profile = load_endpoint_service_profile(path)
    policy = profile.routing_policy
    assert profile.schema == SCHEMA_V2
    assert policy is not None
    assert policy.decoder_high_water_numerator == 1
    assert policy.decoder_high_water_denominator == 2
    assert policy.decoder_low_water_numerator == 1
    assert policy.decoder_low_water_denominator == 4
    assert policy.epoch_confirmation_requests == 2
    assert policy.remote_overload_service_stretch == 2.0
    assert policy.remote_external_credit_close_fraction == 1.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("pair_local", False, "pair-local"),
        ("decoder_low_water_numerator", 2, "below high"),
        ("epoch_confirmation_requests", 0, "positive int"),
        ("remote_overload_service_stretch", 0.9, "safe range"),
        ("remote_external_credit_close_fraction", 1.1, "safe range"),
        ("phase_label_policy_input", True, "phase labels"),
    ),
)
def test_v2_semantic_policy_drift_fails_closed(
    tmp_path, field, value, message,
) -> None:
    payload = _semantic_payload()
    payload["routing_policy"][field] = value
    payload["fingerprint_sha256"] = endpoint_service_profile_fingerprint(
        payload)
    path = tmp_path / "semantic-invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_endpoint_service_profile(path)
