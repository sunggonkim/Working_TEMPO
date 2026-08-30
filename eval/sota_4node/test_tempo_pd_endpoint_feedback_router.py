from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time
from unittest.mock import patch

import httpx
import pytest

from eval.sota_4node import tempo_pd_elastic_router as router
from eval.sota_4node.tempo_pd_elastic_frontend import (
    tempo_route_failure_kind,
    tempo_route_failure_scope,
)
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
from eval.sota_4node.tempo_pd_elastic_frontend import PairLoadLedger


WORKLOAD_SHA256 = "a" * 64


def test_lmcache_cache_key_failure_is_pair_scoped_global_failure():
    request = httpx.Request("POST", "http://pair/v1/completions")
    response = httpx.Response(
        500,
        request=request,
        text=(
            "AssertionError: Key CacheEngineKey(world_size=4, worker_id=0) "
            "not found in local data"
        ),
    )
    error = httpx.HTTPStatusError(
        "upstream 500; body=" + response.text,
        request=request,
        response=response,
    )

    assert tempo_route_failure_kind(error) == (
        "lmcache_cache_key_ownership_failure"
    )
    assert tempo_route_failure_scope(error) == "pair"


def test_unrelated_http_500_remains_route_scoped():
    request = httpx.Request("POST", "http://pair/v1/completions")
    response = httpx.Response(500, request=request, text="worker failed")
    error = httpx.HTTPStatusError(
        "upstream 500; body=worker failed",
        request=request,
        response=response,
    )

    assert tempo_route_failure_kind(error) == "upstream_http_status_500"
    assert tempo_route_failure_scope(error) == "route"


def test_pair_failure_invalidates_only_stale_cache_affinity_owner():
    async def run():
        ledger = PairLoadLedger(2)
        affinity_key = "b" * 64
        await ledger.register_affinity_replicas(
            affinity_key,
            {0, 1},
            evidence_request_ids={
                0: "epd-tempo-warm-seed-owner0",
                1: "epd-tempo-warm-seed-owner1",
            },
        )
        request_id = "epd-tempo-c3-cache-p-only-measured-owner0"
        await ledger.reserve(
            request_id,
            2,
            preferred=0,
            dynamic=True,
            affinity_key=affinity_key,
            committed_pair=0,
        )
        await ledger.record_global_failure(
            request_id,
            failure={
                "request_id": request_id,
                "terminal_phase": "failed",
                "pair_index": 0,
                "quarantine_scope": "pair",
                "failure_kind": "lmcache_cache_key_ownership_failure",
                "reason": "global_route_failure_quarantine",
            },
            failure_sha256="c" * 64,
        )

        states = await ledger.cache_states(
            affinity_key,
            explicit_cache_reset_miss=False,
        )
        assert states[0].residency.value == "confirmed_miss"
        assert states[1].residency.value == "prefill_only"
        snapshot = await ledger.snapshot()
        row = snapshot["rows"][request_id]
        assert row["frontend_pair_affinity_invalidated"] is True

    asyncio.run(run())


def test_pair_ledger_preserves_queue_promotion_rejection_receipt():
    async def run():
        ledger = PairLoadLedger(2)
        request_id = "epd-tempo-background-service-lane-reject"
        await ledger.reserve(
            request_id,
            2,
            preferred=0,
            dynamic=True,
            committed_pair=0,
        )
        await ledger.record_global_decision(
            request_id,
            decision={
                "request_id": request_id,
                "pair_index": 0,
                "route": router.ElasticRoute.LOCAL.value,
                "queue_lease": False,
            },
            decision_sha256="a" * 64,
            tokenizer_ms=1.0,
        )
        await ledger.record_service_lane_queue_promotion_rejection(
            request_id,
            promotion={
                "schema": "tempo-go-service-lane-queue-promotion-v1",
                "request_id": request_id,
                "status": "rejected",
                "reason": "queue_lease_policy_disabled",
                "route": router.ElasticRoute.LOCAL.value,
            },
            promotion_sha256="b" * 64,
        )
        snapshot = await ledger.snapshot()
        row = snapshot["rows"][request_id]
        assert row[
            "frontend_tempo_go_service_lane_queue_promotion"
        ]["status"] == "rejected"
        assert row[
            "frontend_tempo_go_service_lane_queue_promotion_sha256"
        ] == "b" * 64
        await ledger.release(request_id)

    asyncio.run(run())


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
    cache_residency: str = "confirmed_miss",
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
            "cache_residency": cache_residency,
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
    endpoint_cache_residency: str = "confirmed_miss",
    semantic_policy_changes: dict[str, object] | None = None,
    **environment_changes: str,
):
    elastic = _load_elastic(tmp_path)
    endpoint_path = _write_endpoint_profile(
        tmp_path, elastic.fingerprint_sha256,
        resource_window=resource_window,
        cache_residency=endpoint_cache_residency,
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


def _prepare_global(
    core, request_id: str, *, route: str, pair: str = "0",
    profile_sha: str = "f" * 64, sequence: str = "7",
    queue_lease: bool = False, service_forecast: bool = False,
):
    return core.prepare_global_commit(
        request_id=request_id,
        schema=router.GLOBAL_COMMIT_SCHEMA,
        pair_index=pair,
        route=route,
        profile_sha256=profile_sha,
        decision_sha256="e" * 64,
        telemetry_sequence=sequence,
        queue_lease="1" if queue_lease else "0",
        service_queue_delay_ms="12.5" if service_forecast else None,
        service_forecast_ms="37.5" if service_forecast else None,
    )


def _prepare_joint_global(
    core, request_id: str, *, route: str, v2: bool = False,
    v3: bool = False, enforced_local_limit: int = 100,
    queue_lease: bool = False,
    soft_overage_resources: tuple[str, ...] = ("remote_prefill_token_ms",),
):
    plan = {
        "schema": (
            router.JOINT_ACTUATION_SCHEMA_V3
            if v3 else
            router.JOINT_ACTUATION_SCHEMA_V2
            if v2 else router.JOINT_ACTUATION_SCHEMA
        ),
        "pair_index": 0,
        "route": route,
        "local_prefill_token_ms_limit": 100,
        "remote_prefill_token_ms_limit": 100,
        "remote_kv_bytes_limit": 2_000,
        "remote_semantic_ops_limit": 1,
        "dispatch_stagger_us": 250,
        "telemetry_sequence": 7,
        "confidence": 1.0,
        "signal_contributions": [
            {"name": "nccl_collective_p99_ms", "pressure": 0.5},
        ],
    }
    if v2 or v3:
        plan.update({
            "action_mode": (
                "shared_budget_v3" if v3 else "soft_shadow_price_v2"),
            "critical_guard": False,
            "enforced_local_prefill_token_ms_limit": enforced_local_limit,
            "enforced_remote_prefill_token_ms_limit": 100,
            "enforced_remote_kv_bytes_limit": 2_000,
            "enforced_remote_semantic_ops_limit": 1,
            "overage_fraction": 0.25,
            "overage_penalty_ms": 250.0,
            "soft_overage_resources": list(soft_overage_resources),
        })
    if v3:
        plan.update({
            "shared_fabric_group": "epoch|topology|communicator",
            "shared_remote_requests_limit": 32,
            "shared_remote_kv_bytes_limit": 4_000,
            "shared_remote_semantic_ops_limit": 2,
            "shared_remote_requests_used_before": 1,
            "shared_remote_kv_bytes_used_before": 400,
            "shared_remote_semantic_ops_used_before": 1,
            "shared_budget_action": "global_remote_stagger",
            "shared_budget_contributions": [
                {"name": "shared.requests.nccl_collective_p99_ms",
                 "pressure": 0.5},
            ],
        })
    return core.prepare_global_commit(
        request_id=request_id,
        schema=router.GLOBAL_JOINT_COMMIT_SCHEMA,
        pair_index="0",
        route=route,
        profile_sha256="f" * 64,
        decision_sha256="e" * 64,
        telemetry_sequence="7",
        actuation_plan=json.dumps(plan, separators=(",", ":")),
        queue_lease="1" if queue_lease else "0",
    )


def test_global_commit_forces_one_route_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    request_id = "epd-tempo-r0-measured-global-remote"
    evidence = _prepare_global(
        core, request_id, route=router.ElasticRoute.REMOTE.value)
    assert evidence["phase_label_policy_input"] is False
    record = core.decide(
        request_id=request_id, prompt_tokens=10, output_tokens=64)
    assert record.route is router.ElasticRoute.REMOTE
    assert record.reason == "tempo_go_global_route_committed"
    assert record.regime == "tempo_go_global_commit_v1"
    core.mark_upstream_started(request_id)
    core.mark_first_response_chunk(request_id)
    core.complete(request_id)
    row = {item["request_id"]: item for item in core.records()}[request_id]
    assert row["tempo_go_global_commit_applied"] is True
    assert row["tempo_go_global_commit_route"] == (
        router.ElasticRoute.REMOTE.value)
    assert row["tempo_go_global_commit_profile_sha256"] == "f" * 64
    assert row["tempo_go_global_commit_decision_sha256"] == "e" * 64
    assert row["tempo_go_global_commit_telemetry_sequence"] == 7
    assert row["tempo_go_global_commit_service_queue_delay_ms"] is None
    assert row["tempo_go_global_commit_service_forecast_ms"] is None
    assert row["semantic_epoch_applied"] is False


def test_global_commit_preserves_service_forecast_provenance(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    request_id = "epd-tempo-r0-measured-global-service-forecast"
    evidence = _prepare_global(
        core, request_id, route=router.ElasticRoute.LOCAL.value,
        service_forecast=True)
    assert evidence["service_queue_delay_ms"] == 12.5
    assert evidence["service_forecast_ms"] == 37.5
    record = core.decide(
        request_id=request_id, prompt_tokens=10, output_tokens=64)
    assert record.route is router.ElasticRoute.LOCAL
    core.mark_upstream_started(request_id)
    core.mark_first_response_chunk(request_id)
    core.complete(request_id)
    row = {item["request_id"]: item for item in core.records()}[request_id]
    assert row["tempo_go_global_commit_service_queue_delay_ms"] == 12.5
    assert row["tempo_go_global_commit_service_forecast_ms"] == 37.5


def test_global_mesh_commit_routes_p0_to_d1_with_immutable_decoder_header(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            "TEMPO_PD_REMOTE_DECODE_PLACEMENT": "global_mesh",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    request_id = "epd-tempo-r0-measured-global-mesh-p0-d1"
    evidence = core.prepare_global_commit(
        request_id=request_id,
        schema=router.GLOBAL_MESH_COMMIT_SCHEMA,
        pair_index="1",
        prefill_index="0",
        decoder_index="1",
        edge_id="remote:p0->d1",
        route=router.ElasticRoute.REMOTE.value,
        profile_sha256="f" * 64,
        decision_sha256="e" * 64,
        telemetry_sequence="7",
        queue_lease="0",
    )
    assert evidence["pair_index"] == evidence["decoder_index"] == 1
    assert evidence["prefill_index"] == 0
    assert evidence["edge_id"] == "remote:p0->d1"
    semantic = core.prepare_frontend_semantic_load(
        request_id=request_id,
        pair_index="1",
        decode_tokens_before="0",
        active_requests_before="0",
        max_num_seqs="8",
    )
    assert semantic["pair_index"] == 1
    record = core.decide(
        request_id=request_id, prompt_tokens=10, output_tokens=64)
    assert record.route is router.ElasticRoute.REMOTE
    headers = core.prepare_upstream_headers(record, {})
    assert headers[router.DECODER_INDEX_HEADER] == "1"
    with pytest.raises(ValueError, match="changed after preflight"):
        core.prepare_global_commit(
            request_id=request_id,
            schema=router.GLOBAL_MESH_COMMIT_SCHEMA,
            pair_index="0",
            prefill_index="0",
            decoder_index="0",
            edge_id="remote:p0->d0",
            route=router.ElasticRoute.REMOTE.value,
            profile_sha256="f" * 64,
            decision_sha256="d" * 64,
            telemetry_sequence="8",
            queue_lease="0",
        )


def test_global_mesh_allows_only_marker_exact_exogenous_fixed_remote(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            "TEMPO_PD_REMOTE_DECODE_PLACEMENT": "global_mesh",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    request_id = (
        "epd-remote-background-cache-miss-measured-endpoint-observed-"
        "tempo-go-exogenous-fixed-remote-d1-c7-000000-0"
    )
    record = core.decide(
        request_id=request_id, prompt_tokens=4094, output_tokens=2)
    assert record.route is router.ElasticRoute.REMOTE
    headers = core.prepare_upstream_headers(record, {})
    assert headers[router.DECODER_INDEX_HEADER] == "1"

    ordinary = "epd-remote-background-cache-miss-measured-c7-000001-0"
    ordinary_record = core.decide(
        request_id=ordinary, prompt_tokens=4094, output_tokens=2)
    assert ordinary_record.route is router.ElasticRoute.REMOTE
    with pytest.raises(ValueError, match="lacks a global edge commitment"):
        core.prepare_upstream_headers(ordinary_record, {})

    malformed = (
        "epd-remote-background-cache-miss-measured-"
        "tempo-go-exogenous-fixed-remote-d2-c7-000002-0"
    )
    malformed_record = core.decide(
        request_id=malformed, prompt_tokens=4094, output_tokens=2)
    with pytest.raises(ValueError, match="lacks a global edge commitment"):
        core.prepare_upstream_headers(malformed_record, {})

    paired = (
        "epd-predictor-interactive-cache-miss-measured-"
        "c7-joint-paired-baseline-000003-0"
    )
    paired_record = core.decide(
        request_id=paired, prompt_tokens=10, output_tokens=64)
    paired_record = replace(
        paired_record, route=router.ElasticRoute.REMOTE)
    paired_headers = core.prepare_upstream_headers(paired_record, {})
    assert paired_headers[router.DECODER_INDEX_HEADER] == "0"


def test_global_mesh_allows_only_physical_p_only_cache_preparation(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "1",
            "TEMPO_PD_REMOTE_DECODE_PLACEMENT": "global_mesh",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    physical = (
        "epd-tempo-interactive-c4-cache-p-only-warm-seed-o128-physical-"
        "c8-05_p_only_dual_decoder_hot-pool-00-owner-0-"
        "affinity-shadow-p1-item-000000"
    )
    physical_record = core.decide(
        request_id=physical, prompt_tokens=4094, output_tokens=128)
    assert physical_record.route is router.ElasticRoute.REMOTE
    assert physical_record.reason == "unmeasured_p_only_seed_remote"
    physical_headers = core.prepare_upstream_headers(physical_record, {})
    assert physical_headers[router.DECODER_INDEX_HEADER] == "1"

    fixed_physical = physical.replace("epd-tempo-", "epd-remote-", 1)
    fixed_record = core.decide(
        request_id=fixed_physical, prompt_tokens=4094, output_tokens=128)
    assert fixed_record.reason == "unmeasured_p_only_seed_remote"
    fixed_headers = core.prepare_upstream_headers(fixed_record, {})
    assert fixed_headers[router.DECODER_INDEX_HEADER] == "1"

    nonphysical = (
        "epd-tempo-interactive-c4-cache-p-only-warm-seed-o128-"
        "c8-05_p_only_dual_decoder_hot-item-000000"
    )
    nonphysical_record = core.decide(
        request_id=nonphysical, prompt_tokens=4094, output_tokens=128)
    with pytest.raises(ValueError, match="lacks a global edge commitment"):
        core.prepare_upstream_headers(nonphysical_record, {})


def test_global_queue_lease_extends_only_the_endpoint_service_wait(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    request_id = "epd-tempo-r0-measured-global-lease"
    evidence = _prepare_global(
        core, request_id, route=router.ElasticRoute.LOCAL.value,
        queue_lease=True,
    )
    assert evidence["queue_lease"] is True
    assert core.global_queue_wait_ms(
        request_id, default_queue_wait_ms=1_000.0,
        remaining_deadline_ms=8_000.0,
    ) == 8_000.0
    record = core.decide(
        request_id=request_id, prompt_tokens=10, output_tokens=64)
    assert record.route is router.ElasticRoute.LOCAL
    reservation = core.service_lane_reservation(request_id)
    assert reservation is not None
    assert reservation["status"] == "accepted"
    assert reservation["reason"] == "endpoint_service_lane_reserved"
    core.mark_upstream_started(request_id)
    core.mark_first_response_chunk(request_id)
    core.complete(request_id)
    row = {item["request_id"]: item for item in core.records()}[request_id]
    assert row["tempo_go_global_commit_queue_lease"] is True
    assert row["tempo_go_global_queue_wait_ms"] == 8_000.0


def test_global_queue_lease_accepts_bounded_endpoint_queue(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        resource_window=50,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    holder_id = "epd-tempo-r0-measured-service-holder"
    _prepare_global(
        core, holder_id, route=router.ElasticRoute.LOCAL.value)
    holder = core.decide(
        request_id=holder_id, prompt_tokens=10, output_tokens=64)
    assert holder.route is router.ElasticRoute.LOCAL
    core.mark_upstream_started(holder_id)

    lease_id = "epd-tempo-r0-measured-service-lease"
    _prepare_global(
        core, lease_id, route=router.ElasticRoute.LOCAL.value,
        queue_lease=True,
    )
    leased = core.decide(
        request_id=lease_id, prompt_tokens=10, output_tokens=64)
    assert leased.route is router.ElasticRoute.QUEUE
    reservation = core.service_lane_reservation(lease_id)
    assert reservation is not None
    assert reservation["status"] == "accepted"
    assert reservation["reason"] == (
        "endpoint_bounded_queue_lease_accepted")
    # A plain queue lease still consumes only the residual budget after the
    # current global-route service estimate.  A shared-budget soft-overage
    # lease is tested separately: it is explicitly work-conserving and may
    # use the complete remaining deadline while waiting for endpoint credit.
    bounded_wait = core.global_queue_wait_ms(
        lease_id,
        default_queue_wait_ms=1_000.0,
        remaining_deadline_ms=1_000.0,
    )
    assert 0.0 <= bounded_wait < 1_000.0

    core.fail(lease_id, "bounded ingress queue timeout")
    row = {item["request_id"]: item for item in core.records()}[lease_id]
    assert row["phase"] == router.ElasticPhase.FAILED.value
    assert row["tempo_go_service_lane_reservation_status"] == "accepted"
    assert row["tempo_go_service_lane_reservation_reason"] == (
        "endpoint_bounded_queue_lease_accepted")
    core.mark_first_response_chunk(holder_id)
    core.complete(holder_id)


def test_service_lane_preflight_promotes_same_endpoint_queue_offer(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        resource_window=50,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    holder_id = "epd-tempo-r0-measured-preflight-holder"
    _prepare_global(
        core, holder_id, route=router.ElasticRoute.LOCAL.value)
    holder = core.decide(
        request_id=holder_id, prompt_tokens=10, output_tokens=64)
    assert holder.route is router.ElasticRoute.LOCAL
    core.mark_upstream_started(holder_id)

    request_id = "epd-tempo-r0-measured-preflight-promote"
    commit = {
        "schema": router.GLOBAL_COMMIT_SCHEMA,
        "pair_index": "0",
        "prefill_index": "0",
        "decoder_index": "0",
        "edge_id": "local:d0",
        "route": router.ElasticRoute.LOCAL.value,
        "profile_sha256": "f" * 64,
        "decision_sha256": "e" * 64,
        "telemetry_sequence": "7",
        "actuation_plan": None,
        "queue_lease": "0",
    }
    offered = core.preflight_global_service_lane(
        request_id=request_id,
        prompt_key="a" * 64,
        prompt_tokens=10,
        output_tokens=64,
        remaining_deadline_ms=1_000.0,
        commit=commit,
    )
    assert offered["status"] == "queue_required"
    assert offered["endpoint_route"] == router.ElasticRoute.QUEUE.value
    assert core.service_lane_reservation(request_id) is None

    promoted = dict(commit)
    promoted["decision_sha256"] = "d" * 64
    promoted["queue_lease"] = "1"
    accepted = core.preflight_global_service_lane(
        request_id=request_id,
        prompt_key="a" * 64,
        prompt_tokens=10,
        output_tokens=64,
        remaining_deadline_ms=900.0,
        commit=promoted,
    )
    assert accepted["status"] == "accepted"
    assert accepted["endpoint_route"] == router.ElasticRoute.QUEUE.value
    reservation = core.service_lane_reservation(request_id)
    assert reservation is not None
    assert reservation["queue_lease"] is True
    assert reservation["reason"] == "endpoint_bounded_queue_lease_accepted"
    # The real /v1/completions middleware derives the same token-ID hash
    # again after preflight.  Re-registering it must be idempotent and the
    # preflighted record must be reused rather than double-admitted.
    core.prepare_prompt_namespace(request_id, "a" * 64)
    reused = core.decide(
        request_id=request_id, prompt_tokens=10, output_tokens=64)
    assert reused.route is router.ElasticRoute.QUEUE

    core.mark_first_response_chunk(holder_id)
    core.complete(holder_id)
    admitted = core.retry(request_id, 800.0)
    assert admitted.route is router.ElasticRoute.LOCAL
    core.mark_upstream_started(request_id)
    core.mark_first_response_chunk(request_id)
    core.complete(request_id)


def test_service_lane_preflight_rejects_json_integer_identity_before_admission(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    request_id = "epd-tempo-r0-measured-preflight-integer-alias"
    commit = {
        "schema": router.GLOBAL_COMMIT_SCHEMA,
        "pair_index": 0,
        "prefill_index": 0,
        "decoder_index": 0,
        "edge_id": "local:d0",
        "route": router.ElasticRoute.LOCAL.value,
        "profile_sha256": "f" * 64,
        "decision_sha256": "e" * 64,
        "telemetry_sequence": 7,
        "actuation_plan": None,
        "queue_lease": "0",
    }
    with pytest.raises(ValueError, match="canonical non-negative integer"):
        core.preflight_global_service_lane(
            request_id=request_id,
            prompt_key="a" * 64,
            prompt_tokens=10,
            output_tokens=64,
            remaining_deadline_ms=1_000.0,
            commit=commit,
        )
    assert request_id not in {
        row["request_id"] for row in core.records()
    }


def test_soft_overage_queue_lease_preserves_full_remaining_budget(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        resource_window=50,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    holder_id = "epd-tempo-r0-measured-soft-overage-holder"
    _prepare_global(core, holder_id, route=router.ElasticRoute.LOCAL)
    holder = core.decide(
        request_id=holder_id, prompt_tokens=10, output_tokens=64)
    assert holder.route is router.ElasticRoute.LOCAL
    core.mark_upstream_started(holder_id)

    lease_id = "epd-tempo-r0-measured-soft-overage-lease"
    _prepare_joint_global(
        core,
        lease_id,
        route=router.ElasticRoute.LOCAL.value,
        queue_lease=True,
        v3=True,
        soft_overage_resources=("local_prefill_token_ms",),
    )
    leased = core.decide(
        request_id=lease_id, prompt_tokens=10, output_tokens=64)
    assert leased.route is router.ElasticRoute.QUEUE
    assert core.global_queue_wait_ms(
        lease_id,
        default_queue_wait_ms=1_000.0,
        remaining_deadline_ms=8_000.0,
    ) == 8_000.0


def test_native_router_bounds_global_endpoint_queue_without_global_lease(
    tmp_path: Path,
) -> None:
    elastic = _load_elastic(tmp_path)
    endpoint_path = _write_endpoint_profile(
        tmp_path, elastic.fingerprint_sha256, resource_window=50)
    environment = _environment(
        endpoint_path,
        TEMPO_PD_LOCAL_DECODER_INDEX="0",
        **{router.GLOBAL_PROFILE_SHA_ENV: "f" * 64},
    )
    with patch.dict("os.environ", environment, clear=True):
        app = router.build_app(config(), elastic, allow_screen_profile=True)

    async def scenario():
        async with app.router.lifespan_context(app):
            core = app.state.tempo_core
            holder_id = "epd-tempo-r0-native-service-holder"
            _prepare_global(
                core, holder_id, route=router.ElasticRoute.LOCAL.value)
            holder = core.decide(
                request_id=holder_id, prompt_tokens=10, output_tokens=64)
            assert holder.route is router.ElasticRoute.LOCAL
            core.mark_upstream_started(holder_id)

            request_id = "epd-tempo-r0-native-service-unavailable"
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://router.test"
            ) as client:
                response = await client.post(
                    "/v1/completions",
                    headers={
                        "x-tempo-request-id": request_id,
                        "x-tempo-go-schema": router.GLOBAL_COMMIT_SCHEMA,
                        "x-tempo-go-pair-index": "0",
                        "x-tempo-go-route": router.ElasticRoute.LOCAL.value,
                        "x-tempo-go-profile-sha256": "f" * 64,
                        "x-tempo-go-decision-sha256": "e" * 64,
                        "x-tempo-go-telemetry-sequence": "7",
                        # A global route commit owns the request even when the
                        # endpoint's physical service lane is temporarily
                        # full.  The frontend must attempt the bounded native
                        # queue before releasing that global reservation.
                        "x-tempo-go-queue-lease": "0",
                    },
                    json={
                        "model": config().served_model_name,
                        "prompt": [1] * 10,
                        "max_tokens": 64,
                    },
                )
            assert response.status_code == 503
            assert response.headers[
                "x-tempo-service-lane-reservation"] == "timeout"
            assert response.headers["x-tempo-service-lane-reason"] == (
                "endpoint_bounded_global_route_timeout")
            assert response.json()["detail"] == (
                "elastic ingress queue timeout")
            row = {
                item["request_id"]: item for item in core.records()
            }[request_id]
            assert row["phase"] == router.ElasticPhase.FAILED.value
            assert row["tempo_go_service_lane_reservation_status"] == (
                "accepted")
            assert row["tempo_go_service_lane_reservation_reason"] == (
                "endpoint_bounded_global_route_accepted")

            timeout_id = "epd-tempo-r0-native-service-queue-timeout"
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://router.test"
            ) as client:
                response = await client.post(
                    "/v1/completions",
                    headers={
                        "x-tempo-request-id": timeout_id,
                        "x-tempo-go-schema": router.GLOBAL_COMMIT_SCHEMA,
                        "x-tempo-go-pair-index": "0",
                        "x-tempo-go-route": router.ElasticRoute.LOCAL.value,
                        "x-tempo-go-profile-sha256": "f" * 64,
                        "x-tempo-go-decision-sha256": "e" * 64,
                        "x-tempo-go-telemetry-sequence": "7",
                        "x-tempo-go-queue-lease": "1",
                    },
                    json={
                        "model": config().served_model_name,
                        "prompt": [1] * 10,
                        "max_tokens": 64,
                    },
                )
            assert response.status_code == 503
            assert response.headers[
                "x-tempo-service-lane-reservation"] == "timeout"
            assert response.headers["x-tempo-service-lane-reason"] == (
                "endpoint_bounded_queue_lease_timeout")
            assert response.json()["detail"] == (
                "elastic ingress queue timeout")
            timeout_row = {
                item["request_id"]: item for item in core.records()
            }[timeout_id]
            assert timeout_row["phase"] == router.ElasticPhase.FAILED.value
            assert timeout_row["tempo_go_service_lane_reservation_status"] == (
                "accepted")
            core.mark_first_response_chunk(holder_id)
            core.complete(holder_id)

    asyncio.run(scenario())


def test_global_commit_fails_closed_on_partial_or_wrong_identity(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    with pytest.raises(ValueError, match="incomplete"):
        core.prepare_global_commit(
            request_id="epd-tempo-r0-measured-partial",
            schema=router.GLOBAL_COMMIT_SCHEMA,
            pair_index="0",
            route=None,
            profile_sha256="f" * 64,
            decision_sha256="e" * 64,
            telemetry_sequence="1",
        )
    with pytest.raises(ValueError, match="pair/router"):
        _prepare_global(
            core,
            "epd-tempo-r0-measured-wrong-pair",
            route=router.ElasticRoute.LOCAL.value,
            pair="1",
        )
    with pytest.raises(ValueError, match="profile SHA"):
        _prepare_global(
            core,
            "epd-tempo-r0-measured-wrong-profile",
            route=router.ElasticRoute.LOCAL.value,
            profile_sha="d" * 64,
        )


def test_global_commit_queue_retry_cannot_switch_route(tmp_path: Path) -> None:
    core, _, _ = _core(
        tmp_path,
        resource_window=50,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    first_id = "epd-tempo-r0-measured-global-holder"
    _prepare_global(
        core, first_id, route=router.ElasticRoute.REMOTE.value)
    first = core.decide(
        request_id=first_id, prompt_tokens=10, output_tokens=64)
    assert first.route is router.ElasticRoute.REMOTE
    core.mark_upstream_started(first_id)

    queued_id = "epd-tempo-r0-measured-global-queued"
    _prepare_global(
        core, queued_id, route=router.ElasticRoute.REMOTE.value)
    queued = core.decide(
        request_id=queued_id, prompt_tokens=10, output_tokens=64)
    assert queued.route is router.ElasticRoute.QUEUE
    retry = core.retry(queued_id, 900.0)
    assert retry.route is router.ElasticRoute.QUEUE
    core.mark_first_response_chunk(first_id)
    core.complete(first_id)
    admitted = core.retry(queued_id, 800.0)
    assert admitted.route is router.ElasticRoute.REMOTE
    assert admitted.reason == "tempo_go_global_route_committed"


def test_joint_commit_enforces_remote_transfer_limit_and_preserves_plan(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        resource_window=100,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    first_id = "epd-tempo-r0-measured-joint-holder"
    evidence = _prepare_joint_global(
        core, first_id, route=router.ElasticRoute.REMOTE.value)
    assert evidence["schema"] == router.GLOBAL_JOINT_COMMIT_SCHEMA
    assert evidence["actuation_plan"]["remote_semantic_ops_limit"] == 1
    first = core.decide(
        request_id=first_id, prompt_tokens=10, output_tokens=64)
    assert first.route is router.ElasticRoute.REMOTE
    core.mark_upstream_started(first_id)

    queued_id = "epd-tempo-r0-measured-joint-queued"
    _prepare_joint_global(
        core, queued_id, route=router.ElasticRoute.REMOTE.value)
    queued = core.decide(
        request_id=queued_id, prompt_tokens=10, output_tokens=64)
    assert queued.route is router.ElasticRoute.QUEUE
    core.mark_first_response_chunk(first_id)
    core.complete(first_id)
    admitted = core.retry(queued_id, 800.0)
    assert admitted.route is router.ElasticRoute.REMOTE
    row = {item["request_id"]: item for item in core.records()}[first_id]
    assert row["tempo_go_global_commit_actuation_plan"][
        "dispatch_stagger_us"] == 250


def test_joint_commit_v2_accepts_shadow_price_lease_inventory(
    tmp_path: Path,
) -> None:
    core, _, _ = _core(
        tmp_path,
        resource_window=100,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    evidence = _prepare_joint_global(
        core, "epd-tempo-r0-measured-joint-v2",
        route=router.ElasticRoute.REMOTE.value,
        v2=True,
    )
    assert evidence["actuation_plan"]["schema"] == (
        router.JOINT_ACTUATION_SCHEMA_V2
    )
    assert evidence["actuation_plan"][
        "enforced_remote_prefill_token_ms_limit"] == 100


def test_joint_commit_v3_accepts_shared_budget_inventory(tmp_path: Path) -> None:
    core, _, _ = _core(
        tmp_path,
        resource_window=100,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    evidence = _prepare_joint_global(
        core, "epd-tempo-r0-measured-joint-v3",
        route=router.ElasticRoute.REMOTE.value,
        v3=True,
    )
    assert evidence["actuation_plan"]["schema"] == (
        router.JOINT_ACTUATION_SCHEMA_V3
    )
    assert evidence["actuation_plan"]["shared_budget_action"] == (
        "global_remote_stagger"
    )
    assert evidence["actuation_plan"]["shared_remote_requests_limit"] == 32


def test_joint_commit_queue_lease_clamps_endpoint_window(tmp_path: Path) -> None:
    core, _, _ = _core(
        tmp_path,
        resource_window=100,
        **{
            "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
            router.GLOBAL_PROFILE_SHA_ENV: "f" * 64,
        },
    )
    holder_id = "epd-tempo-r0-measured-queue-lease-holder"
    _prepare_joint_global(
        core, holder_id, route=router.ElasticRoute.LOCAL.value,
        v2=True,
    )
    holder = core.decide(
        request_id=holder_id, prompt_tokens=10, output_tokens=64)
    assert holder.route is router.ElasticRoute.LOCAL
    core.mark_upstream_started(holder_id)

    first_queued_id = "epd-tempo-r0-measured-queue-lease-first"
    _prepare_joint_global(
        core, first_queued_id, route=router.ElasticRoute.LOCAL.value,
        v2=True, enforced_local_limit=200,
    )
    first_queued = core.decide(
        request_id=first_queued_id, prompt_tokens=10, output_tokens=64)
    assert first_queued.route is router.ElasticRoute.LOCAL
    core.mark_upstream_started(first_queued_id)

    second_queued_id = "epd-tempo-r0-measured-queue-lease-second"
    _prepare_joint_global(
        core, second_queued_id, route=router.ElasticRoute.LOCAL.value,
        v2=True, enforced_local_limit=200,
    )
    second_queued = core.decide(
        request_id=second_queued_id, prompt_tokens=10, output_tokens=64)
    assert second_queued.route is router.ElasticRoute.QUEUE


def test_global_commit_middleware_forwards_only_complete_contract() -> None:
    class Core:
        def __init__(self):
            self.values = None

        def prepare_global_commit(self, **values):
            if values["telemetry_sequence"] is None:
                raise ValueError("TEMPO-GO route-commit headers are incomplete")
            self.values = values

    async def scenario(headers):
        core = Core()
        messages = []

        async def inner(_scope, _receive, send):
            await send({
                "type": "http.response.start", "status": 204,
                "headers": [],
            })
            await send({"type": "http.response.body", "body": b""})

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            messages.append(message)

        middleware = router._GlobalCommitMiddleware(inner, core=core)
        await middleware({
            "type": "http",
            "method": "POST",
            "path": "/v1/completions",
            "headers": [
                (name.encode(), value.encode())
                for name, value in headers.items()
            ],
        }, receive, send)
        return core, messages

    base_headers = {
        "x-tempo-request-id": "epd-tempo-r0-measured-middleware",
        "x-tempo-go-schema": router.GLOBAL_COMMIT_SCHEMA,
        "x-tempo-go-pair-index": "0",
        "x-tempo-go-route": router.ElasticRoute.LOCAL.value,
        "x-tempo-go-profile-sha256": "f" * 64,
        "x-tempo-go-decision-sha256": "e" * 64,
        "x-tempo-go-telemetry-sequence": "1",
    }
    core, messages = asyncio.run(scenario(base_headers))
    assert core.values["request_id"] == base_headers["x-tempo-request-id"]
    assert messages[0]["status"] == 204

    partial = dict(base_headers)
    del partial["x-tempo-go-telemetry-sequence"]
    core, messages = asyncio.run(scenario(partial))
    assert core.values is None
    assert messages[0]["status"] == 400
    assert b"headers are incomplete" in messages[1]["body"]


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


def test_tempo_uses_bounded_endpoint_proxy_for_unmeasured_anchor_geometry(
    tmp_path: Path,
) -> None:
    # The C5 output=2 endpoint profile intentionally carries a measured
    # P_ONLY geometry ceiling rather than pretending that MISS/output=2 is an
    # exact service sample.  TEMPO adaptive admission must use that explicit
    # proxy and retain its provenance in the request ledger.
    core, _, _ = _core(
        tmp_path, endpoint_cache_residency="prefill_only")
    request_id = "epd-tempo-r0-measured-cache-miss-proxy"
    record = core.decide(
        request_id=request_id, prompt_tokens=10, output_tokens=64)
    assert record.route is router.ElasticRoute.LOCAL
    row = {value["request_id"]: value for value in core.records()}[request_id]
    assert row["endpoint_service_lookup_mode"] == (
        "miss_via_prefill_only_geometry_ceiling")
    assert row["endpoint_service_requested_cache_residency"] == (
        "unknown")
    assert row["endpoint_service_effective_cache_residency"] == (
        "confirmed_miss")
    assert row["endpoint_service_source_cache_residency"] == "prefill_only"
    core.mark_upstream_started(request_id)
    core.mark_first_response_chunk(request_id)
    core.complete(request_id)


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
