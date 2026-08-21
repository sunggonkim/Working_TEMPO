from __future__ import annotations

import json
from pathlib import Path
import time
from unittest.mock import patch

import pytest

from eval.sota_4node import tempo_pd_elastic_router as router
from eval.sota_4node.test_tempo_pd_elastic_router_v445 import (
    config,
    profile_payload,
)
from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_endpoint_profile import (
    SCHEMA as ENDPOINT_PROFILE_SCHEMA,
    SCHEMA_V2 as ENDPOINT_PROFILE_V2_SCHEMA,
    endpoint_service_profile_fingerprint,
    load_endpoint_service_profile,
)


WORKLOAD_SHA256 = "a" * 64


def _load_elastic(tmp_path: Path, *, local_ms: float = 20.0,
                  remote_ms: float = 25.0):
    payload = profile_payload()
    payload["rows"][0]["local_upper_bound_ms"] = local_ms
    payload["rows"][0]["remote_upper_bound_ms"] = remote_ms
    path = tmp_path / "elastic.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_elastic_profile(path)


def _write_endpoint_profile(
    tmp_path: Path,
    elastic_fingerprint: str,
    *,
    resource_window: int = 100,
    semantic_epoch: bool = False,
    semantic_policy_changes: dict[str, object] | None = None,
) -> Path:
    payload = {
        "schema": ENDPOINT_PROFILE_SCHEMA,
        "profile_id": "endpoint-router-test",
        "elastic_profile_fingerprint_sha256": elastic_fingerprint,
        "workload_manifest_sha256": WORKLOAD_SHA256,
        "deployment_scope": "calibration_only",
        "default_e2e_deadline_ms": 1_000.0,
        "controller": {
            "local_token_ms_window": resource_window,
            "remote_prefill_token_ms_window": resource_window,
            "remote_kv_bytes_window": 2_000,
            "remote_semantic_ops_window": 2,
            "feedback_history": 4,
            "feedback_quantile": 0.75,
            "minimum_feedback": 1,
            "route_margin_ms": 1.0,
            "feedback_fresh_ns": 10_000_000_000,
            "probe_after_ns": 10_000_000_000,
            "denied_probe_after_ns": 30_000_000_000,
        },
        "rows": [{
            "prompt_tokens": 10,
            "output_tokens": 64,
            "cache_residency": "confirmed_miss",
            "local_ttft_prior_ms": 1.0,
            "remote_ttft_prior_ms": 2.0,
            "local_token_ms": 50,
            "remote_prefill_token_ms": 50,
            "samples_local": 3,
            "samples_remote": 3,
            "outputs_equivalent": True,
            "evidence_valid": True,
        }],
    }
    if semantic_epoch:
        payload["schema"] = ENDPOINT_PROFILE_V2_SCHEMA
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
        if semantic_policy_changes:
            payload["routing_policy"].update(semantic_policy_changes)
    payload["fingerprint_sha256"] = endpoint_service_profile_fingerprint(
        payload)
    path = tmp_path / "endpoint.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _environment(endpoint_path: Path, **changes: str) -> dict[str, str]:
    values = {
        router.ENDPOINT_FEEDBACK_MODE_ENV:
            router.ENDPOINT_FEEDBACK_ADAPTIVE_MODE,
        router.ENDPOINT_SERVICE_PROFILE_ENV: str(endpoint_path),
        router.ENDPOINT_WORKLOAD_MANIFEST_SHA256_ENV: WORKLOAD_SHA256,
        router.COLD_MEASURED_ENV: "1",
        router.PRESSURE_MODE_ENV: router.PRESSURE_DISABLED_MODE,
        router.VLLM_LOAD_SNAPSHOT_MODE_ENV: router.VLLM_LOAD_DISABLED_MODE,
        "TEMPO_PD_REMOTE_CATCHUP_PRIORITY": "0",
        "TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY": "0",
        "TEMPO_PD_LONG_REMOTE_CATCHUP_PRIORITY": "0",
        "TEMPO_PD_MEDIAN_GUARD_PRIORITY": "0",
        "TEMPO_PD_MEDIUM_REMOTE_CATCHUP_PRIORITY": "0",
    }
    values.update(changes)
    return values


def _core(
    tmp_path: Path, *, resource_window: int = 100,
    semantic_policy_changes: dict[str, object] | None = None,
    **environment_changes: str,
):
    elastic = _load_elastic(tmp_path)
    endpoint_path = _write_endpoint_profile(
        tmp_path, elastic.fingerprint_sha256,
        resource_window=resource_window,
        semantic_epoch=(
            environment_changes.get(router.ENDPOINT_ROUTING_POLICY_ENV)
            == router.ENDPOINT_SEMANTIC_EPOCH_POLICY
        ),
        semantic_policy_changes=semantic_policy_changes,
    )
    with patch.dict(
        "os.environ",
        _environment(endpoint_path, **environment_changes),
        clear=True,
    ):
        core = router.ElasticPDRouterCore(
            config(), elastic, allow_screen_profile=True)
    return core, elastic, endpoint_path


def _prepare_semantic(core, request_id: str, *, active: int, decode: int = 512):
    return core.prepare_frontend_semantic_load(
        request_id=request_id,
        pair_index="0",
        decode_tokens_before=str(decode),
        active_requests_before=str(active),
        max_num_seqs="8",
    )


def test_real_c4_profile_accounts_all_route_pinned_background_geometries() -> None:
    profile = load_endpoint_service_profile(
        Path(__file__).with_name(
            "real_tempo_pd_endpoint_service_profile_c4_screen_v1.json"))
    cases = (
        (router.CacheResidency.MISS, router.EndpointRoute.LOCAL),
        (router.CacheResidency.MISS, router.EndpointRoute.REMOTE),
        (router.CacheResidency.P_ONLY, router.EndpointRoute.REMOTE),
    )
    for residency, route in cases:
        proxy = profile.external_credit_proxy(
            4094, 2, residency, route=route)
        assert proxy.row.prompt_tokens == 4094
        assert proxy.row.output_tokens == 16
        assert proxy.row.cache_residency is router.CacheResidency.P_ONLY
        assert proxy.lookup_mode != "exact"


def test_route_pinned_tenant_passively_updates_service_without_credit(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path, **{router.ENDPOINT_PASSIVE_FEEDBACK_ENV: "1"})
    background_id = (
        "epd-local-r0-measured-endpoint-observed-background-local")
    background = core.decide(
        request_id=background_id,
        prompt_tokens=10,
        output_tokens=64,
    )
    assert background.route is router.ElasticRoute.LOCAL
    core.mark_upstream_started(background_id)
    time.sleep(0.01)
    core.mark_first_response_chunk(background_id)
    core.complete(background_id)

    state = core.endpoint_controller_state()
    assert state["passive_registered_requests"] == 1
    assert state["controller"]["resources"] == {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }
    local = state["controller"]["routes"][
        router.EndpointRoute.LOCAL.value]
    assert local["passive_samples"] == 1
    assert local["active_samples"] == 0

    foreground = core.decide(
        request_id="epd-tempo-r0-measured-after-passive",
        prompt_tokens=10,
        output_tokens=64,
    )
    assert foreground.route is router.ElasticRoute.REMOTE
    rows = {row["request_id"]: row for row in core.records()}
    row = rows[background_id]
    assert row["endpoint_feedback_event"] == "passive_first_response_chunk"
    assert row["endpoint_feedback_passive"] is True
    assert row["endpoint_feedback_passive_accepted"] is True
    assert row["endpoint_feedback_service_stretch"] > 1.0
    assert row["endpoint_passive_registered"] is True
    assert row["admission_credit_scope"] is None
    assert row["admission_credit_release_event"] is None
    assert row["admission_credit_released_ns"] is None


def test_semantic_epoch_opens_remote_after_decoder_high_water_confirmation(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        **{
            router.ENDPOINT_PASSIVE_FEEDBACK_ENV: "1",
            router.ENDPOINT_ROUTING_POLICY_ENV:
                router.ENDPOINT_SEMANTIC_EPOCH_POLICY,
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            "TEMPO_VLLM_MAX_NUM_SEQS": "8",
        },
    )
    background_id = (
        "epd-local-r0-measured-endpoint-observed-semantic-background")
    background = core.decide(
        request_id=background_id, prompt_tokens=10, output_tokens=64)
    assert background.route is router.ElasticRoute.LOCAL
    state = core.endpoint_controller_state()
    assert state["endpoint_routing_policy"] == "semantic_epoch_v1"
    assert state["endpoint_service_profile"]["routing_policy"] == {
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
    assert state["controller"]["external_inflight"] == 1
    assert state["controller"]["external_resources"]["local_token_ms"] == 50

    first_id = "epd-tempo-r0-measured-semantic-first"
    _prepare_semantic(core, first_id, active=4)
    first = core.decide(
        request_id=first_id, prompt_tokens=10, output_tokens=64)
    assert first.route is router.ElasticRoute.LOCAL
    assert first.reason == "semantic_epoch_local_high_water_confirmation"

    second_id = "epd-tempo-r0-measured-semantic-second"
    _prepare_semantic(core, second_id, active=5)
    second = core.decide(
        request_id=second_id, prompt_tokens=10, output_tokens=64)
    assert second.route is router.ElasticRoute.REMOTE
    assert second.reason == "semantic_epoch_open_remote_high_water"
    rows = {row["request_id"]: row for row in core.records()}
    assert rows[first_id]["semantic_epoch_decoder_high_water"] is True
    assert rows[first_id]["semantic_epoch_high_streak_after"] == 1
    assert rows[second_id]["semantic_epoch_route_before"] == (
        router.EndpointRoute.LOCAL.value)
    assert rows[second_id]["semantic_epoch_route_after"] == (
        router.EndpointRoute.REMOTE.value)
    assert rows[second_id]["semantic_epoch_generation"] == 1
    assert rows[background_id]["endpoint_external_credit_registered"] is True

    core.mark_upstream_started(background_id)
    core.mark_first_response_chunk(background_id)
    core.complete(background_id)
    state = core.endpoint_controller_state()["controller"]
    assert state["external_inflight"] == 0
    assert all(value == 0 for value in state["external_resources"].values())
    row = {value["request_id"]: value for value in core.records()}[
        background_id]
    assert row["endpoint_feedback_event"] == (
        "external_credit_passive_first_response_chunk")


def test_semantic_epoch_decision_thresholds_come_only_from_v2_profile(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        semantic_policy_changes={
            "decoder_high_water_numerator": 3,
            "decoder_high_water_denominator": 4,
            "epoch_confirmation_requests": 3,
        },
        **{
            router.ENDPOINT_PASSIVE_FEEDBACK_ENV: "1",
            router.ENDPOINT_ROUTING_POLICY_ENV:
                router.ENDPOINT_SEMANTIC_EPOCH_POLICY,
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            "TEMPO_VLLM_MAX_NUM_SEQS": "8",
        },
    )
    routes = []
    for index, active in enumerate((4, 5, 6, 7, 6)):
        request_id = f"epd-tempo-r0-measured-profile-bound-{index}"
        _prepare_semantic(core, request_id, active=active)
        record = core.decide(
            request_id=request_id, prompt_tokens=10, output_tokens=64)
        routes.append(record.route)
        core.mark_upstream_started(request_id)
        core.mark_first_response_chunk(request_id)
        core.complete(request_id)
    assert routes == [
        router.ElasticRoute.LOCAL,
        router.ElasticRoute.LOCAL,
        router.ElasticRoute.LOCAL,
        router.ElasticRoute.LOCAL,
        router.ElasticRoute.REMOTE,
    ]
    row = {value["request_id"]: value for value in core.records()}[
        "epd-tempo-r0-measured-profile-bound-4"]
    assert row["semantic_epoch_decoder_high_water_numerator"] == 3
    assert row["semantic_epoch_decoder_high_water_denominator"] == 4
    assert row["semantic_epoch_confirmation_requests"] == 3


def test_semantic_epoch_closes_remote_when_external_remote_credit_is_full(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        **{
            router.ENDPOINT_PASSIVE_FEEDBACK_ENV: "1",
            router.ENDPOINT_ROUTING_POLICY_ENV:
                router.ENDPOINT_SEMANTIC_EPOCH_POLICY,
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            "TEMPO_VLLM_MAX_NUM_SEQS": "8",
        },
    )
    for index, active in enumerate((4, 5)):
        request_id = f"epd-tempo-r0-measured-open-{index}"
        _prepare_semantic(core, request_id, active=active)
        record = core.decide(
            request_id=request_id, prompt_tokens=10, output_tokens=64)
        core.mark_upstream_started(request_id)
        core.mark_first_response_chunk(request_id)
        core.complete(request_id)
    assert record.route is router.ElasticRoute.REMOTE

    for index in range(2):
        request_id = (
            "epd-remote-r0-measured-endpoint-observed-"
            f"remote-pressure-{index}")
        fixed = core.decide(
            request_id=request_id, prompt_tokens=10, output_tokens=64)
        assert fixed.route is router.ElasticRoute.REMOTE
    state = core.endpoint_controller_state()["controller"]
    assert state["external_resources"]["remote_semantic_ops"] == 2

    foreground_id = "epd-tempo-r0-measured-close-remote"
    _prepare_semantic(core, foreground_id, active=6)
    foreground = core.decide(
        request_id=foreground_id, prompt_tokens=10, output_tokens=64)
    assert foreground.route is router.ElasticRoute.LOCAL
    assert foreground.reason == "semantic_epoch_close_remote_unavailable"
    row = {value["request_id"]: value for value in core.records()}[
        foreground_id]
    assert row["semantic_epoch_remote_external_utilization"] == 1.0
    assert row["semantic_epoch_remote_available"] is False


def test_semantic_credit_epoch_uses_external_local_credit_not_watermark(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        semantic_policy_changes={
            "local_external_credit_opens_epoch": True,
            "frontend_decoder_watermarks_policy_input": False,
        },
        **{
            router.ENDPOINT_PASSIVE_FEEDBACK_ENV: "1",
            router.ENDPOINT_ROUTING_POLICY_ENV:
                router.ENDPOINT_SEMANTIC_EPOCH_POLICY,
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            "TEMPO_VLLM_MAX_NUM_SEQS": "8",
        },
    )
    background_id = (
        "epd-local-r0-measured-endpoint-observed-credit-background")
    background = core.decide(
        request_id=background_id, prompt_tokens=10, output_tokens=64)
    assert background.route is router.ElasticRoute.LOCAL

    first_id = "epd-tempo-r0-measured-credit-first"
    _prepare_semantic(core, first_id, active=0, decode=0)
    first = core.decide(
        request_id=first_id, prompt_tokens=10, output_tokens=64)
    assert first.route is router.ElasticRoute.LOCAL
    assert first.reason == "semantic_credit_epoch_local_credit_confirmation"
    core.mark_upstream_started(first_id)
    core.mark_first_response_chunk(first_id)
    core.complete(first_id)

    second_id = "epd-tempo-r0-measured-credit-second"
    _prepare_semantic(core, second_id, active=0, decode=0)
    second = core.decide(
        request_id=second_id, prompt_tokens=10, output_tokens=64)
    assert second.route is router.ElasticRoute.REMOTE
    assert second.reason == "semantic_credit_epoch_open_remote_local_credit"
    row = {value["request_id"]: value for value in core.records()}[second_id]
    assert row["semantic_epoch_decoder_high_water"] is False
    assert row["semantic_epoch_local_external_credit_pressure"] is True
    assert row["semantic_epoch_decision_basis"] == (
        "local_external_credit_nonzero")


def test_endpoint_completion_feedback_flips_route_and_is_auditable(
    tmp_path: Path,
) -> None:
    core, elastic, endpoint_path = _core(tmp_path)
    first = core.decide(
        request_id="epd-tempo-r0-measured-a",
        prompt_tokens=10,
        output_tokens=64,
    )
    assert first.route is router.ElasticRoute.LOCAL
    core.mark_upstream_started(first.request_id)
    time.sleep(0.01)
    core.mark_first_response_chunk(first.request_id)
    core.complete(first.request_id)

    second = core.decide(
        request_id="epd-tempo-r0-measured-b",
        prompt_tokens=10,
        output_tokens=64,
    )
    assert second.route is router.ElasticRoute.REMOTE

    rows = {row["request_id"]: row for row in core.records()}
    first_row = rows[first.request_id]
    second_row = rows[second.request_id]
    assert first_row["endpoint_feedback_event"] == "first_response_chunk"
    assert first_row["endpoint_feedback_passive"] is False
    assert first_row["endpoint_feedback_accepted"] is True
    assert first_row["endpoint_feedback_passive_accepted"] is None
    assert first_row["admission_credit_release_event"] == "first_response_chunk"
    assert first_row["endpoint_feedback_observed_ttft_ms"] >= 5.0
    assert first_row["endpoint_feedback_prior_ttft_ms"] == 1.0
    assert first_row["endpoint_feedback_service_stretch"] >= 5.0
    assert first_row["endpoint_resource_local_token_ms_used_after_feedback"] == 0
    assert second_row["endpoint_decision_route"] == second.route.value
    assert second_row["endpoint_decision_local_multiplier"] > 1.0
    assert second_row["endpoint_decision_local_state"] == "skip"
    assert second_row["endpoint_decision_attempts"] == 1
    assert second_row[
        "endpoint_service_profile_elastic_fingerprint_sha256"
    ] == elastic.fingerprint_sha256

    state = core.endpoint_controller_state()
    assert state["endpoint_feedback_mode"] == "adaptive"
    assert state["controller"]["inflight"] == 1
    assert state["endpoint_service_profile"]["workload_manifest_sha256"] == (
        WORKLOAD_SHA256)

    with patch.dict("os.environ", _environment(endpoint_path), clear=True):
        app = router.build_app(config(), elastic, allow_screen_profile=True)
    assert "/tempo/endpoint_controller" in {
        route.path for route in app.routes
    }
    assert "/tempo/endpoint_controller" in router._CanonicalWireMiddleware._JSON_PATHS


def test_endpoint_controller_reset_requires_quiescence_and_increments_generation(
    tmp_path: Path,
) -> None:
    core, elastic, endpoint_path = _core(tmp_path)
    first = core.reset_endpoint_controller()
    assert first["success"] is True
    assert first["controller_generation"] == 1
    assert all(value == 0 for value in first["controller"]["resources"].values())

    record = core.decide(
        request_id="epd-tempo-r0-measured-reset-inflight",
        prompt_tokens=10,
        output_tokens=64,
    )
    with pytest.raises(ValueError, match="not quiescent"):
        core.reset_endpoint_controller()
    core.mark_upstream_started(record.request_id)
    core.mark_first_response_chunk(record.request_id)
    core.complete(record.request_id)
    second = core.reset_endpoint_controller()
    assert second["controller_generation"] == 2
    assert core.endpoint_controller_state()["controller_generation"] == 2

    with patch.dict("os.environ", _environment(endpoint_path), clear=True):
        app = router.build_app(
            config(), elastic, allow_screen_profile=True)
    assert "/tempo/reset_endpoint_controller" in {
        route.path for route in app.routes
    }
    assert (
        "/tempo/reset_endpoint_controller"
        in router._CanonicalWireMiddleware._JSON_PATHS
    )


def test_static_predictor_matches_replay_decoder_residency_constraint(
    tmp_path: Path,
) -> None:
    elastic = _load_elastic(tmp_path, local_ms=25.0, remote_ms=20.0)
    endpoint_path = _write_endpoint_profile(
        tmp_path, elastic.fingerprint_sha256)
    with patch.dict("os.environ", _environment(endpoint_path), clear=True):
        decoder_hot = router.ElasticPDRouterCore(
            config(),
            elastic,
            cache_residency=lambda _request_id: router.CacheResidency.D_ONLY,
            allow_screen_profile=True,
        )
        prefill_hot = router.ElasticPDRouterCore(
            config(),
            elastic,
            cache_residency=lambda _request_id: router.CacheResidency.P_ONLY,
            allow_screen_profile=True,
        )

    d_only = decoder_hot.decide(
        request_id="epd-predictor-cache-d-only-measured-test",
        prompt_tokens=10,
        output_tokens=64,
    )
    p_only = prefill_hot.decide(
        request_id="epd-predictor-cache-p-only-measured-test",
        prompt_tokens=10,
        output_tokens=64,
    )
    assert d_only.route is router.ElasticRoute.LOCAL
    assert d_only.reason == "predictor_decoder_residency_local"
    assert p_only.route is router.ElasticRoute.REMOTE
    assert p_only.reason == "predictor_remote_lower_bound"


def test_endpoint_failure_releases_once_and_records_failure(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(tmp_path)
    record = core.decide(
        request_id="epd-tempo-r0-measured-fail",
        prompt_tokens=10,
        output_tokens=64,
    )
    core.mark_upstream_started(record.request_id)
    core.fail(record.request_id, "injected upstream failure")
    assert core.endpoint_controller_state()["controller"]["resources"] == {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }
    row = core.records()[0]
    assert row["endpoint_feedback_event"] == "upstream_failure"
    assert row["admission_credit_release_event"] == "upstream_failure"
    assert row["admission_credit_released_ns"] is not None
    with pytest.raises(ValueError, match="already terminal"):
        core.fail(record.request_id, "second failure")


def test_route_pinned_failure_denies_without_credit(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path, **{router.ENDPOINT_PASSIVE_FEEDBACK_ENV: "1"})
    request_id = (
        "epd-remote-r0-measured-endpoint-observed-background-fail")
    record = core.decide(
        request_id=request_id,
        prompt_tokens=10,
        output_tokens=64,
    )
    assert record.route is router.ElasticRoute.REMOTE
    core.mark_upstream_started(request_id)
    core.fail(request_id, "injected route-pinned upstream failure")

    state = core.endpoint_controller_state()["controller"]
    assert state["resources"] == {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }
    remote = state["routes"][router.EndpointRoute.REMOTE.value]
    assert remote["state"] == "denied"
    assert remote["passive_failures"] == 1
    row = core.records()[0]
    assert row["endpoint_feedback_event"] == "passive_upstream_failure"
    assert row["endpoint_feedback_passive"] is True
    assert row["endpoint_feedback_passive_accepted"] is True
    assert row["admission_credit_scope"] is None
    assert row["admission_credit_release_event"] is None
    assert row["admission_credit_released_ns"] is None


def test_endpoint_queue_retry_preserves_frozen_deadline_and_history(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(tmp_path, resource_window=49)
    record = core.decide(
        request_id="epd-tempo-r0-measured-queued",
        prompt_tokens=10,
        output_tokens=64,
    )
    assert record.route is router.ElasticRoute.QUEUE
    retried = core.retry(record.request_id, 1_000_000_000.0)
    assert retried.route is router.ElasticRoute.QUEUE
    row = core.records()[0]
    assert row["endpoint_request_e2e_deadline_ms"] == 1_000.0
    assert row["endpoint_decision_attempts"] == 2
    assert len(row["endpoint_decision_history"]) == 2
    core.fail(record.request_id, "bounded ingress queue timeout")
    row = core.records()[0]
    assert row["endpoint_feedback_event"] == "queue_failure_no_reservation"
    assert row["admission_credit_released_ns"] is None


def test_semantic_epoch_queue_retry_preserves_ingress_policy_provenance(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        resource_window=49,
        **{
            router.ENDPOINT_PASSIVE_FEEDBACK_ENV: "1",
            router.ENDPOINT_ROUTING_POLICY_ENV:
                router.ENDPOINT_SEMANTIC_EPOCH_POLICY,
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            "TEMPO_VLLM_MAX_NUM_SEQS": "8",
        },
    )
    request_id = "epd-tempo-r0-measured-semantic-queued"
    _prepare_semantic(core, request_id, active=0)
    record = core.decide(
        request_id=request_id, prompt_tokens=10, output_tokens=64)
    assert record.route is router.ElasticRoute.QUEUE
    assert record.reason == "semantic_epoch_local_default"
    assert record.regime == router.ENDPOINT_SEMANTIC_EPOCH_POLICY

    retried = core.retry(request_id, 1_000_000_000.0)
    assert retried.route is router.ElasticRoute.QUEUE
    assert retried.reason == "semantic_epoch_local_default"
    assert retried.regime == router.ENDPOINT_SEMANTIC_EPOCH_POLICY
    row = core.records()[0]
    assert row["semantic_epoch_route_after"] == (
        router.EndpointRoute.LOCAL.value)
    assert row["endpoint_decision_attempts"] == 2
    retry_reason = row["endpoint_decision_history"][-1]["reason"]
    assert retry_reason.startswith("endpoint_")
    assert retry_reason != retried.reason


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({router.ENDPOINT_WORKLOAD_MANIFEST_SHA256_ENV: "b" * 64},
         "workload binding differs"),
        ({router.PRESSURE_MODE_ENV: router.PRESSURE_OBSERVE_MODE},
         "forbids the scalar Cassini pressure policy"),
        ({router.VLLM_LOAD_SNAPSHOT_MODE_ENV: router.VLLM_LOAD_DECISION_MODE},
         "forbids synchronous request-start /metrics"),
        ({"TEMPO_PD_REMOTE_CATCHUP_PRIORITY": "-1",
          "TEMPO_VLLM_SCHEDULING_POLICY": "priority"},
         "forbids shape-specific priority exceptions"),
    ],
)
def test_endpoint_mode_rejects_unbound_or_confounded_policy_inputs(
    tmp_path: Path,
    change: dict[str, str],
    message: str,
) -> None:
    elastic = _load_elastic(tmp_path)
    endpoint_path = _write_endpoint_profile(
        tmp_path, elastic.fingerprint_sha256)
    with patch.dict(
        "os.environ", _environment(endpoint_path, **change), clear=True,
    ), pytest.raises(ValueError, match=message):
        router.ElasticPDRouterCore(
            config(), elastic, allow_screen_profile=True)


def test_endpoint_profile_must_bind_the_exact_elastic_profile(
    tmp_path: Path,
) -> None:
    first = _load_elastic(tmp_path, local_ms=20.0, remote_ms=25.0)
    endpoint_path = _write_endpoint_profile(
        tmp_path, first.fingerprint_sha256)
    second_payload = profile_payload()
    second_payload["rows"][0]["local_upper_bound_ms"] = 21.0
    second_path = tmp_path / "elastic-second.json"
    second_path.write_text(json.dumps(second_payload), encoding="utf-8")
    second = load_elastic_profile(second_path)
    with patch.dict(
        "os.environ", _environment(endpoint_path), clear=True,
    ), pytest.raises(ValueError, match="fingerprints differ"):
        router.ElasticPDRouterCore(
            config(), second, allow_screen_profile=True)
