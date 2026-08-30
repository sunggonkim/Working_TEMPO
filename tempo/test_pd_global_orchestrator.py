from __future__ import annotations

import pytest

from tempo.pd_global_orchestrator import (
    GlobalFailureReceipt,
    GlobalDecisionKind,
    GlobalOrchestrator,
    GlobalOrchestratorConfig,
    GlobalRequest,
    GlobalRoute,
    PairCapacity,
    PairTelemetry,
    PathHealth,
    ResourceVector,
    RouteCandidate,
    RejectedCandidate,
    TenantPolicy,
    global_decision_dict,
    global_failure_dict,
    global_failure_fingerprint,
    global_service_lane_queue_promotion_dict,
    global_service_lane_queue_promotion_fingerprint,
    global_service_lane_reservation_failure_dict,
    global_service_lane_reservation_failure_fingerprint,
)


CAPACITY = ResourceVector(
    decode_tokens=100,
    active_sequences=2,
    endpoint_requests=2,
    local_prefill_token_ms=100,
    remote_prefill_token_ms=100,
    remote_kv_bytes=1_000,
    remote_semantic_ops=2,
)


def controller(**overrides) -> GlobalOrchestrator:
    values = dict(
        capacities=(PairCapacity(0, CAPACITY), PairCapacity(1, CAPACITY)),
        tenants=(TenantPolicy("latency", 2.0), TenantPolicy("batch", 1.0)),
        telemetry_fresh_ns=1_000,
        queue_capacity=16,
        minimum_active_pairs=1,
        maximum_active_pairs=2,
        utilization_penalty_ms=10.0,
        activation_penalty_ms=1.0,
        scale_down_idle_ns=100,
        maximum_queue_wait_ns=500,
    )
    values.update(overrides)
    return GlobalOrchestrator(GlobalOrchestratorConfig(**values))


def telemetry(
    pair: int, *, sequence: int = 1, sampled_ns: int = 10,
    observed: ResourceVector = ResourceVector(),
    local_health: PathHealth = PathHealth.GOOD,
    remote_health: PathHealth = PathHealth.GOOD,
    local_multiplier: float = 1.0,
    remote_multiplier: float = 1.0,
    local_failure_count: int = 0,
    remote_failure_count: int = 0,
    local_last_failure_kind: str | None = None,
    remote_last_failure_kind: str | None = None,
    scheduler_running: int | None = None,
    scheduler_waiting: int | None = None,
    scheduler_kv: float | None = None,
    completion_residual: int | None = None,
    completion_completed: int = 0,
) -> PairTelemetry:
    return PairTelemetry(
        pair_index=pair,
        sequence=sequence,
        sampled_ns=sampled_ns,
        collected_ns=sampled_ns + 1,
        agent_epoch="test-allocation-epoch",
        profile_fingerprint_sha256="a" * 64,
        controller_generation=0,
        observed_total=observed,
        local_health=local_health,
        remote_health=remote_health,
        local_service_multiplier=local_multiplier,
        remote_service_multiplier=remote_multiplier,
        local_failure_count=local_failure_count,
        remote_failure_count=remote_failure_count,
        local_last_failure_kind=local_last_failure_kind,
        remote_last_failure_kind=remote_last_failure_kind,
        scheduler_running_requests=scheduler_running,
        scheduler_waiting_requests=scheduler_waiting,
        scheduler_kv_cache_usage_fraction=scheduler_kv,
        scheduler_schema=(
            "tempo-go-vllm-scheduler-snapshot-v1"
            if scheduler_running is not None else None),
        scheduler_source=(
            "router_local_vllm_prometheus_observe_only"
            if scheduler_running is not None else None),
        endpoint_completed_first_responses=(
            completion_completed if completion_residual is not None else None),
        endpoint_residual_inflight=completion_residual,
        completion_schema=(
            "tempo-go-endpoint-completion-v1"
            if completion_residual is not None else None),
    )


def work(
    route: GlobalRoute, *, decode: int = 40, local_prefill: int = 40,
) -> ResourceVector:
    if route is GlobalRoute.LOCAL:
        return ResourceVector(
            decode_tokens=decode,
            active_sequences=1,
            endpoint_requests=1,
            local_prefill_token_ms=local_prefill,
        )
    return ResourceVector(
        decode_tokens=decode,
        active_sequences=1,
        endpoint_requests=1,
        remote_prefill_token_ms=30,
        remote_kv_bytes=400,
        remote_semantic_ops=1,
    )


def candidate(
    pair: int, route: GlobalRoute, *, e2e: float, decode: int = 40,
    local_prefill: int = 40,
) -> RouteCandidate:
    return RouteCandidate(
        pair_index=pair,
        route=route,
        work=work(
            route, decode=decode, local_prefill=local_prefill),
        predicted_e2e_ms=e2e,
        predicted_ttft_ms=e2e / 2,
        uncertainty_ms=1.0,
    )


def mesh_candidate(
    prefill: int,
    decoder: int,
    *,
    e2e: float,
    remote_prefill: int = 30,
    kv_bytes: int = 400,
    cache_affinity: bool = False,
) -> RouteCandidate:
    return RouteCandidate(
        pair_index=decoder,
        prefill_index=prefill,
        decoder_index=decoder,
        route=GlobalRoute.REMOTE,
        work=ResourceVector(
            decode_tokens=40,
            active_sequences=1,
            endpoint_requests=1,
            remote_prefill_token_ms=remote_prefill,
            remote_kv_bytes=kv_bytes,
            remote_semantic_ops=1,
        ),
        predicted_e2e_ms=e2e,
        predicted_ttft_ms=e2e / 2,
        uncertainty_ms=1.0,
        cache_affinity=cache_affinity,
    )


def request(
    request_id: str, tenant: str, candidates: tuple[RouteCandidate, ...],
    *, arrival_ns: int = 10, deadline_ns: int = 1_000_000_000,
    cache_group_key: str | None = None,
) -> GlobalRequest:
    return GlobalRequest(
        request_id=request_id,
        tenant_id=tenant,
        arrival_ns=arrival_ns,
        deadline_ns=deadline_ns,
        candidates=candidates,
        cache_group_key=cache_group_key,
    )


def seed(value: GlobalOrchestrator, *, sampled_ns: int = 10) -> None:
    value.update_telemetry(telemetry(0, sampled_ns=sampled_ns))
    value.update_telemetry(telemetry(1, sampled_ns=sampled_ns))


def test_missing_or_stale_telemetry_fails_closed_to_queue() -> None:
    value = controller()
    result = value.submit(request(
        "r0", "latency", (candidate(0, GlobalRoute.LOCAL, e2e=20),)
    ), now_ns=10)
    assert result.kind is GlobalDecisionKind.QUEUE
    assert result.reason == "global_telemetry_unavailable"
    value.update_telemetry(telemetry(0, sampled_ns=10))
    late = value.submit(request(
        "r1", "batch", (candidate(0, GlobalRoute.LOCAL, e2e=20),),
        arrival_ns=2_000,
    ), now_ns=2_000)
    assert late.kind is GlobalDecisionKind.QUEUE


def test_protected_service_lane_reserves_decoder_and_endpoint_slots() -> None:
    value = controller(
        tenants=(
            TenantPolicy("latency", 2.0, admission_priority=100),
            TenantPolicy("batch", 1.0, admission_priority=0),
        ),
        mesh_control_mode="receiver_credit_pxd_v1",
        protected_service_lane_mode="tenant_pair_edge_reservation_v1",
        protected_service_lane_capacity=1,
        protected_service_lane_min_admission_priority=100,
    )
    seed(value)

    background = value.submit(request(
        "lane-background-0", "batch",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=10)
    assert background.kind is GlobalDecisionKind.ADMIT

    blocked = value.submit(request(
        "lane-background-1", "batch",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=11)
    assert blocked.kind is GlobalDecisionKind.QUEUE
    assert any(
        item.reason == "protected_service_lane_reserve"
        for item in blocked.rejected_candidates
    )

    protected = value.submit(request(
        "lane-protected-0", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=12)
    assert protected.kind is GlobalDecisionKind.ADMIT
    assert protected.protected_service_lane is True
    assert protected.protected_service_lane_key == "local:d0"
    assert protected.protected_service_lane_before == 0
    assert protected.protected_service_lane_after == 1
    assert "global_protected_service_lane_reservation_v1" in (
        protected.binding_resources)
    snapshot = value.snapshot(now_ns=12)
    assert snapshot["protected_service_lane_debt"]["local:d0"] == 1


def test_protected_service_lane_v2_is_reserve_not_protected_ceiling() -> None:
    value = controller(
        tenants=(
            TenantPolicy("latency", 2.0, admission_priority=100),
            TenantPolicy("batch", 1.0, admission_priority=0),
        ),
        mesh_control_mode="receiver_credit_pxd_v1",
        protected_service_lane_mode="tenant_pair_edge_reservation_v2",
        protected_service_lane_capacity=1,
        protected_service_lane_min_admission_priority=100,
    )
    seed(value)

    first = value.submit(request(
        "reserve-protected-0", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=10)
    assert first.kind is GlobalDecisionKind.ADMIT

    # The physical pair has two slots.  v2 must allow protected work to use
    # the second slot even though the protected reserve is only one slot.
    second = value.submit(request(
        "reserve-protected-1", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=11)
    assert second.kind is GlobalDecisionKind.ADMIT
    assert second.protected_service_lane is True
    assert second.protected_service_lane_before == 1
    assert second.protected_service_lane_after == 2

    # Lower-priority work still cannot consume the reserved slot once both
    # physical slots are busy.
    blocked = value.submit(request(
        "reserve-background-0", "batch",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=12)
    assert blocked.kind is GlobalDecisionKind.QUEUE
    assert any(
        item.reason in {"capacity", "protected_service_lane_reserve"}
        for item in blocked.rejected_candidates
    )


def test_protected_service_lane_queue_promotion_does_not_double_count_owned_request() -> None:
    value = controller(
        tenants=(
            TenantPolicy(
                "latency", 2.0, admission_priority=100,
                queue_lease_on_timeout=True,
            ),
            TenantPolicy("batch", 1.0),
        ),
        mesh_control_mode="receiver_credit_pxd_v1",
        protected_service_lane_mode="tenant_pair_edge_reservation_v1",
        protected_service_lane_capacity=1,
        protected_service_lane_min_admission_priority=100,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode="completion_credit_mesh_endpoint_queue_v1",
    )
    value.update_telemetry(telemetry(
        0, scheduler_running=0, scheduler_waiting=0,
        scheduler_kv=0.0, completion_residual=0, completion_completed=1,
    ))
    value.update_telemetry(telemetry(
        1, scheduler_running=0, scheduler_waiting=0,
        scheduler_kv=0.0, completion_residual=0,
    ))
    request_id = "protected-queue-promotion"
    admitted = value.submit(request(
        request_id, "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=10)
    assert admitted.kind is GlobalDecisionKind.ADMIT
    assert admitted.protected_service_lane is True

    promoted = value.promote_service_lane_queue_lease(
        request_id, now_ns=11)
    assert promoted.receipt.status == "promoted"
    assert promoted.decision is not None


def test_protected_service_lane_is_edge_scoped_for_remote_mesh() -> None:
    value = controller(
        tenants=(
            TenantPolicy("latency", 2.0, admission_priority=100),
            TenantPolicy("batch", 1.0, admission_priority=0),
        ),
        mesh_control_mode="receiver_credit_pxd_v1",
        protected_service_lane_mode="tenant_pair_edge_reservation_v1",
        protected_service_lane_capacity=1,
        protected_service_lane_min_admission_priority=100,
    )
    seed(value)
    first = value.submit(request(
        "edge-lane-0", "latency",
        (mesh_candidate(0, 1, e2e=20),),
    ), now_ns=10)
    assert first.kind is GlobalDecisionKind.ADMIT
    assert first.protected_service_lane_key == "remote:p0->d1"

    second = value.submit(request(
        "edge-lane-1", "latency",
        (mesh_candidate(0, 1, e2e=20), mesh_candidate(1, 1, e2e=20)),
    ), now_ns=11)
    assert second.kind is GlobalDecisionKind.ADMIT
    assert second.protected_service_lane_key == "remote:p1->d1"


def test_bounded_stale_grace_keeps_admission_work_conserving() -> None:
    value = controller(telemetry_stale_grace_ns=1_000)
    seed(value, sampled_ns=10)
    decision = value.submit(request(
        "stale-grace",
        "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
        arrival_ns=1_500,
    ), now_ns=1_500)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert value.snapshot(now_ns=1_500)["telemetry_stale_grace_ns"] == 1_000

    expired = controller(telemetry_stale_grace_ns=1_000)
    seed(expired, sampled_ns=10)
    decision = expired.submit(request(
        "stale-grace-expired",
        "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
        arrival_ns=2_500,
    ), now_ns=2_500)
    assert decision.kind is GlobalDecisionKind.QUEUE
    assert decision.reason == "global_telemetry_unavailable"


def test_service_feasibility_lease_rejects_observed_decoder_backlog() -> None:
    value = controller(service_feasibility_mode="deadline_residual_v1")
    value.update_telemetry(telemetry(
        0,
        scheduler_running=2,
        scheduler_waiting=2,
        scheduler_kv=0.5,
        completion_residual=2,
    ))
    value.update_telemetry(telemetry(1))
    decision = value.submit(request(
        "service-infeasible",
        "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
        deadline_ns=20_000_010,
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.QUEUE
    assert any(
        item.reason == "global_service_lane_slo_infeasible"
        for item in decision.rejected_candidates
    )
    assert value.snapshot(now_ns=10)["service_feasibility_mode"] == (
        "deadline_residual_v1")


def test_service_feasibility_receipt_records_observed_wave_forecast() -> None:
    value = controller(service_feasibility_mode="deadline_residual_v1")
    seed(value)
    decision = value.submit(request(
        "service-feasible",
        "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
        deadline_ns=1_000_000_000,
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.service_queue_delay_ms == 0.0
    assert decision.service_forecast_ms == 11.0
    payload = global_decision_dict(decision)
    assert payload["service_forecast_ms"] == 11.0


def test_remote_cache_chunk_group_is_serialized_until_first_response() -> None:
    value = controller()
    seed(value)
    cache_group_key = "b" * 64
    first = value.submit(request(
        "cache-owner", "latency",
        (candidate(0, GlobalRoute.REMOTE, e2e=10),),
        cache_group_key=cache_group_key,
    ), now_ns=10)
    assert first.kind is GlobalDecisionKind.ADMIT
    assert first.cache_group_key == cache_group_key
    assert value.snapshot(now_ns=11)["cache_group_holds"] == [{
        "pair_index": 0,
        "cache_group_key": cache_group_key,
        "request_id": "cache-owner",
    }]

    duplicate = value.submit(request(
        "cache-duplicate", "latency",
        (candidate(0, GlobalRoute.REMOTE, e2e=5),),
        cache_group_key=cache_group_key,
    ), now_ns=11)
    assert duplicate.kind is GlobalDecisionKind.QUEUE
    assert any(
        item.reason == "cache_chunk_transfer_serialization"
        for item in duplicate.rejected_candidates
    )

    dispatched = value.mark_first_response("cache-owner", now_ns=12)
    assert len(dispatched) == 1
    assert dispatched[0].request_id == "cache-duplicate"
    assert dispatched[0].cache_group_key == cache_group_key


def test_route_failure_is_terminal_and_quarantines_without_same_id_migration() -> None:
    value = controller(
        maximum_active_pairs=2,
        route_failure_quarantine_mode="deny_until_probe",
    )
    seed(value)
    admitted = value.submit(request(
        "failed-route", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=10)
    assert admitted.kind is GlobalDecisionKind.ADMIT
    report = value.report_route_failure(
        "failed-route",
        failure_kind="upstream_transport_error",
        now_ns=20,
        route=GlobalRoute.LOCAL,
    )
    assert isinstance(report.receipt, GlobalFailureReceipt)
    assert report.receipt.phase_before.value == "route_committed"
    assert report.receipt.terminal_phase.value == "failed"
    assert report.receipt.reassignment_policy == "new_request_id_required"
    assert report.receipt.released_work["endpoint_requests"] == 1
    assert value.snapshot(now_ns=21)["route_failure_quarantines"][0][
        "failure_kind"] == "upstream_transport_error"
    with pytest.raises(ValueError, match="duplicate request_id"):
        value.submit(request(
            "failed-route", "latency",
            (candidate(1, GlobalRoute.LOCAL, e2e=20),),
        ), now_ns=22)
    retry = value.submit(request(
        "retry-new-id", "latency",
        (
            candidate(0, GlobalRoute.LOCAL, e2e=20),
            candidate(1, GlobalRoute.LOCAL, e2e=20),
        ),
    ), now_ns=23)
    assert retry.kind is GlobalDecisionKind.ADMIT
    assert retry.pair_index == 1
    assert any(
        item.reason == "route_failure_quarantine"
        for item in retry.rejected_candidates
    )
    payload = global_failure_dict(report.receipt)
    assert payload["route"] == GlobalRoute.LOCAL.value
    assert len(global_failure_fingerprint(report.receipt)) == 64


def test_route_failure_recovers_only_after_new_probe_telemetry() -> None:
    value = controller(route_failure_quarantine_mode="deny_until_probe")
    seed(value)
    value.submit(request(
        "r0", "latency", (candidate(0, GlobalRoute.REMOTE, e2e=20),)
    ), now_ns=10)
    value.report_route_failure(
        "r0", failure_kind="lmcache_transfer_error", now_ns=20,
        route=GlobalRoute.REMOTE,
    )
    good = telemetry(0, sequence=2, sampled_ns=20)
    value.update_telemetry(good)
    blocked = value.submit(request(
        "blocked-good", "latency",
        (candidate(0, GlobalRoute.REMOTE, e2e=20),),
    ), now_ns=21)
    assert blocked.kind is GlobalDecisionKind.QUEUE
    probe = telemetry(
        0, sequence=3, sampled_ns=21, remote_health=PathHealth.PROBE,
    )
    value.update_telemetry(probe)
    dispatched = value.dispatch(now_ns=22)
    assert [item.request_id for item in dispatched] == ["blocked-good"]


def test_pair_scope_quarantines_local_and_remote_paths() -> None:
    value = controller(route_failure_quarantine_mode="deny_until_probe")
    seed(value)
    admitted = value.submit(request(
        "pair-failed", "latency", (
            candidate(0, GlobalRoute.LOCAL, e2e=20),
        ),
    ), now_ns=10)
    assert admitted.kind is GlobalDecisionKind.ADMIT
    report = value.report_route_failure(
        "pair-failed",
        failure_kind="endpoint_transport_unavailable",
        now_ns=20,
        scope="pair",
    )
    assert set(report.receipt.quarantined_routes) == {
        (0, GlobalRoute.LOCAL),
        (0, GlobalRoute.REMOTE),
    }
    next_request = value.submit(request(
        "pair-retry", "latency", (
            candidate(0, GlobalRoute.LOCAL, e2e=1),
            candidate(0, GlobalRoute.REMOTE, e2e=1),
            candidate(1, GlobalRoute.LOCAL, e2e=20),
        ),
    ), now_ns=21)
    assert next_request.kind is GlobalDecisionKind.ADMIT
    assert next_request.pair_index == 1
    assert {
        item.reason for item in next_request.rejected_candidates
    } >= {"route_failure_quarantine"}


def test_queue_overload_is_a_terminal_policy_rejection() -> None:
    value = controller(queue_capacity=1, maximum_active_pairs=1)
    seed(value)
    active = value.submit(request(
        "active", "latency", (candidate(0, GlobalRoute.LOCAL, e2e=20, decode=100),)
    ), now_ns=10)
    assert active.kind is GlobalDecisionKind.ADMIT
    queued = value.submit(request(
        "queued", "batch", (candidate(0, GlobalRoute.LOCAL, e2e=20),)
    ), now_ns=11)
    assert queued.kind is GlobalDecisionKind.QUEUE
    rejected = value.submit(request(
        "overloaded", "batch", (candidate(0, GlobalRoute.LOCAL, e2e=20),)
    ), now_ns=12)
    assert rejected.kind is GlobalDecisionKind.REJECT
    assert rejected.reason == "global_ingress_overload_reject"
    assert value.snapshot(now_ns=13)["phases"]["overloaded"] == "rejected"
    value.fail("active", now_ns=14)


def test_tenant_queue_reservation_protects_later_business_class() -> None:
    value = controller(
        queue_capacity=3,
        maximum_active_pairs=1,
        tenants=(
            TenantPolicy("background", 0.5, queue_reservation_slots=0),
            TenantPolicy("latency", 4.0, queue_reservation_slots=1),
        ),
    )
    seed(value)
    active = value.submit(request(
        "active", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=20, decode=100),),
    ), now_ns=10)
    assert active.kind is GlobalDecisionKind.ADMIT

    first = value.submit(request(
        "background-1", "background",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=11)
    second = value.submit(request(
        "background-2", "background",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=12)
    assert first.kind is second.kind is GlobalDecisionKind.QUEUE

    protected = value.submit(request(
        "background-3", "background",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=13)
    assert protected.kind is GlobalDecisionKind.REJECT
    assert protected.reason == "global_tenant_queue_reservation"

    latency = value.submit(request(
        "latency-1", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=14)
    assert latency.kind is GlobalDecisionKind.QUEUE
    assert value.snapshot(now_ns=15)["admission_guards"][
        "tenant_queue_reservation_slots"] == {"background": 0, "latency": 1}


def test_protected_capacity_reserve_keeps_background_off_interactive_lane() -> None:
    value = controller(
        maximum_active_pairs=1,
        tenants=(
            TenantPolicy(
                "interactive", 2.0, admission_priority=100,
                protected_capacity_fraction=0.5,
            ),
            TenantPolicy("background", 0.5, admission_priority=0),
        ),
    )
    seed(value)
    first = value.submit(request(
        "background-1", "background",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=10)
    assert first.kind is GlobalDecisionKind.ADMIT

    second = value.submit(request(
        "background-2", "background",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=11)
    assert second.kind is GlobalDecisionKind.QUEUE
    assert {
        item.reason for item in second.rejected_candidates
    } == {"tenant_protected_capacity_reserve"}

    interactive = value.submit(request(
        "interactive-1", "interactive",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=12)
    assert interactive.kind is GlobalDecisionKind.ADMIT
    policies = value.snapshot(now_ns=12)["tenant_policies"]
    assert policies["interactive"]["protected_capacity_fraction"] == 0.5


def test_telemetry_is_monotonic_and_application_scoped() -> None:
    value = controller()
    value.update_telemetry(telemetry(0, sequence=2, sampled_ns=20))
    with pytest.raises(ValueError, match="monotonic"):
        value.update_telemetry(telemetry(0, sequence=2, sampled_ns=21))
    with pytest.raises(ValueError, match="policy-eligible"):
        PairTelemetry(
            pair_index=0,
            sequence=3,
            sampled_ns=30,
            collected_ns=31,
            agent_epoch="test-allocation-epoch",
            profile_fingerprint_sha256="a" * 64,
            controller_generation=0,
            observed_total=ResourceVector(),
            source="physical_switch_label",
        )


def test_global_choice_uses_pair_route_cost_and_service_feedback() -> None:
    value = controller()
    value.update_telemetry(telemetry(0, local_multiplier=2.0))
    value.update_telemetry(telemetry(1))
    decision = value.submit(request(
        "r0",
        "latency",
        (
            candidate(0, GlobalRoute.LOCAL, e2e=20),
            candidate(0, GlobalRoute.REMOTE, e2e=25),
        ),
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.route is GlobalRoute.REMOTE
    assert any(
        item.reason == "higher_global_score"
        for item in decision.rejected_candidates
    )


def test_rejected_candidate_score_delta_can_explain_semantic_override() -> None:
    receipt = RejectedCandidate(
        pair_index=0,
        route=GlobalRoute.LOCAL,
        reason="higher_global_score",
        evaluated_score_ms=9.0,
        score_delta_ms=-1.0,
        uncertainty_ms=1.0,
        mesh_near_tie_eligible=False,
    )
    assert receipt.score_delta_ms == -1.0


def test_queue_lease_preserves_live_route_service_feedback() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    seed(value)
    holder = value.submit(request(
        "lease-feedback-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=20, decode=100),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT
    # Remote is nominally faster, but the live endpoint feedback says its
    # service stretch is four times worse at the queue-lease boundary.
    value.update_telemetry(telemetry(
        0, sequence=2, sampled_ns=12, remote_multiplier=4.0))
    queued = value.submit(request(
        "lease-feedback-waiter", "latency",
        (
            candidate(0, GlobalRoute.LOCAL, e2e=30),
            candidate(0, GlobalRoute.REMOTE, e2e=20),
        ),
    ), now_ns=13)
    assert queued.kind is GlobalDecisionKind.QUEUE
    leased = value.lease_queued_to_endpoint(
        "lease-feedback-waiter", now_ns=20)
    assert leased is not None and leased.queue_lease is True
    assert leased.route is GlobalRoute.LOCAL
    assert any(
        item.route is GlobalRoute.REMOTE
        and item.reason == "higher_global_score"
        for item in leased.rejected_candidates
    )


def test_remote_capacity_binds_without_blocking_local() -> None:
    value = controller()
    value.update_telemetry(telemetry(
        0,
        observed=ResourceVector(remote_kv_bytes=800),
    ))
    value.update_telemetry(telemetry(1))
    decision = value.submit(request(
        "r0",
        "latency",
        (
            candidate(0, GlobalRoute.REMOTE, e2e=10),
            candidate(0, GlobalRoute.LOCAL, e2e=30),
        ),
    ), now_ns=10)
    assert decision.route is GlobalRoute.LOCAL
    rejected = next(
        item for item in decision.rejected_candidates
        if item.route is GlobalRoute.REMOTE
    )
    assert rejected.reason == "capacity"
    assert "remote_kv_bytes" in rejected.binding_resources


def test_remote_semantic_operation_safety_reserve_is_fail_closed() -> None:
    value = controller(remote_semantic_ops_safety_reserve=1)
    value.update_telemetry(telemetry(
        0, observed=ResourceVector(remote_semantic_ops=1)))
    value.update_telemetry(telemetry(1))
    decision = value.submit(request(
        "semantic-guard-r0", "latency", (
            candidate(0, GlobalRoute.REMOTE, e2e=1),
            candidate(0, GlobalRoute.LOCAL, e2e=30),
        )
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.route is GlobalRoute.LOCAL
    rejected = next(
        item for item in decision.rejected_candidates
        if item.route is GlobalRoute.REMOTE
    )
    assert rejected.reason == "remote_semantic_ops_admission_guard"
    assert rejected.binding_resources == (
        "remote_semantic_ops_safety_reserve",
    )


def test_path_health_excludes_denied_route() -> None:
    value = controller()
    value.update_telemetry(telemetry(0, remote_health=PathHealth.DENIED))
    value.update_telemetry(telemetry(1))
    decision = value.submit(request(
        "r0", "latency", (
            candidate(0, GlobalRoute.REMOTE, e2e=1),
            candidate(0, GlobalRoute.LOCAL, e2e=20),
        )
    ), now_ns=10)
    assert decision.route is GlobalRoute.LOCAL
    assert any(
        item.reason == "path_denied" for item in decision.rejected_candidates)


def test_remote_failure_guard_is_provenanced_and_uses_surviving_pair() -> None:
    value = controller()
    value.update_telemetry(telemetry(
        0,
        remote_health=PathHealth.DENIED,
        remote_failure_count=1,
        remote_last_failure_kind="active_upstream_failure",
    ))
    value.update_telemetry(telemetry(1))
    decision = value.submit(request(
        "remote-guard-r0", "latency", (
            candidate(0, GlobalRoute.REMOTE, e2e=1),
            candidate(1, GlobalRoute.LOCAL, e2e=20),
        )
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.pair_index == 1
    assert decision.pair_activated is True
    rejected = next(
        item for item in decision.rejected_candidates
        if item.pair_index == 0 and item.route is GlobalRoute.REMOTE
    )
    assert rejected.reason == "remote_pre_admission_guard"
    provenance = decision.telemetry_provenance[0]
    assert provenance["route_failures"]["remote_count"] == 1
    assert provenance["route_failures"]["remote_last_kind"] == (
        "active_upstream_failure")


def test_telemetry_failure_delta_quarantines_pair_before_new_admission() -> None:
    value = controller(
        telemetry_failure_quarantine_mode="deny_until_probe",
        telemetry_failure_quarantine_scope="pair",
        survivor_capacity_reserve_fraction=0.5,
        survivor_reserve_bypass_min_weight=2.0,
    )
    seed(value)
    value.update_telemetry(telemetry(
        0,
        sequence=2,
        sampled_ns=20,
        remote_failure_count=1,
        remote_last_failure_kind="lmcache_engine_dead",
    ))
    snapshot = value.snapshot(now_ns=21)
    assert snapshot["admission_guards"]["fully_quarantined_pairs"] == [0]
    quarantine = snapshot["route_failure_quarantines"]
    assert {item["route"] for item in quarantine} == {
        GlobalRoute.LOCAL.value,
        GlobalRoute.REMOTE.value,
    }
    assert all(item["trigger"] == "telemetry_failure_delta" for item in quarantine)

    urgent = value.submit(request(
        "survivor-urgent", "latency", (
            candidate(0, GlobalRoute.LOCAL, e2e=1),
            candidate(1, GlobalRoute.LOCAL, e2e=2, decode=40),
        ),
    ), now_ns=22)
    assert urgent.kind is GlobalDecisionKind.ADMIT
    assert urgent.pair_index == 1
    assert urgent.pair_activated is True
    assert any(
        item.reason == "route_failure_quarantine"
        for item in urgent.rejected_candidates
    )

    normal = value.submit(request(
        "survivor-normal", "batch", (
            candidate(1, GlobalRoute.LOCAL, e2e=2, decode=60),
        ),
    ), now_ns=23)
    assert normal.kind is GlobalDecisionKind.QUEUE
    assert normal.rejected_candidates[0].reason == "survivor_capacity_reserve"

    value.update_telemetry(telemetry(
        0,
        sequence=3,
        sampled_ns=24,
        remote_health=PathHealth.PROBE,
        local_health=PathHealth.PROBE,
        remote_failure_count=1,
        remote_last_failure_kind="lmcache_engine_dead",
    ))
    assert value.snapshot(now_ns=25)["admission_guards"][
        "fully_quarantined_pairs"] == []


def test_telemetry_failure_delta_route_scope_preserves_healthy_sibling() -> None:
    value = controller(
        telemetry_failure_quarantine_mode="deny_until_probe",
        telemetry_failure_quarantine_scope="route",
    )
    seed(value)
    value.update_telemetry(telemetry(
        0,
        sequence=2,
        sampled_ns=20,
        remote_failure_count=1,
        remote_last_failure_kind="lmcache_transfer_failure",
    ))
    snapshot = value.snapshot(now_ns=21)
    assert snapshot["admission_guards"]["fully_quarantined_pairs"] == []
    quarantine = snapshot["route_failure_quarantines"]
    assert [(item["pair_index"], item["route"]) for item in quarantine] == [
        (0, GlobalRoute.REMOTE.value),
    ]
    decision = value.submit(request(
        "route-isolated-sibling",
        "latency",
        (
            candidate(0, GlobalRoute.REMOTE, e2e=1),
            candidate(0, GlobalRoute.LOCAL, e2e=2),
            candidate(1, GlobalRoute.LOCAL, e2e=20),
        ),
    ), now_ns=22)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.pair_index == 0
    assert decision.route is GlobalRoute.LOCAL
    assert any(
        item.reason == "route_failure_quarantine"
        and item.route is GlobalRoute.REMOTE
        for item in decision.rejected_candidates
    )


def test_explicit_route_failure_quarantines_path_and_reassigns_waiter() -> None:
    value = controller(route_failure_quarantine_mode="deny_until_probe")
    value.update_telemetry(telemetry(0, sampled_ns=10_000))
    value.update_telemetry(telemetry(1, sampled_ns=0))
    first = value.submit(request(
        "route-failure-active", "latency", (
            candidate(0, GlobalRoute.REMOTE, e2e=10),
        ),
    ), now_ns=10_000)
    assert first.kind is GlobalDecisionKind.ADMIT
    second = value.submit(request(
        "route-failure-waiter", "batch", (
            candidate(0, GlobalRoute.REMOTE, e2e=10),
        ),
    ), now_ns=10_001)
    assert second.kind is GlobalDecisionKind.ADMIT
    # Fill the active pair's endpoint/semantic-op slots so the next request
    # remains queued until the explicit route failure releases one reservation.
    fill = value.submit(request(
        "route-failure-fill", "batch", (
            candidate(0, GlobalRoute.REMOTE, e2e=10),
        ),
    ), now_ns=10_002)
    assert fill.kind is GlobalDecisionKind.QUEUE
    queued = value.submit(request(
        "route-failure-reassign", "batch", (
            candidate(0, GlobalRoute.REMOTE, e2e=10),
            candidate(1, GlobalRoute.LOCAL, e2e=12),
        ),
    ), now_ns=10_003)
    assert queued.kind is GlobalDecisionKind.QUEUE

    # Refresh the previously stale spare without dispatching; the failure
    # transaction below must observe and use this newly eligible survivor.
    value.update_telemetry(telemetry(1, sequence=2, sampled_ns=10_014))
    report = value.report_route_failure(
        "route-failure-active",
        failure_kind="active_upstream_failure",
        now_ns=10_014,
        scope="route",
    )
    assert report.receipt.schema == "tempo-go-global-failure-v1"
    assert report.receipt.terminal_phase.value == "failed"
    assert report.receipt.quarantined_routes == ((0, GlobalRoute.REMOTE),)
    assert [item.request_id for item in report.dispatched] == [
        "route-failure-reassign"
    ]
    assert report.dispatched[0].pair_index == 1
    assert report.dispatched[0].route is GlobalRoute.LOCAL
    rejected = value.submit(request(
        "route-failure-blocked", "batch", (
            candidate(0, GlobalRoute.REMOTE, e2e=10),
        ),
    ), now_ns=10_015)
    assert rejected.kind is GlobalDecisionKind.QUEUE
    assert rejected.rejected_candidates[0].reason == "route_failure_quarantine"

    value.fail("route-failure-waiter", now_ns=10_021)
    value.cancel_queued(
        "route-failure-fill", now_ns=10_022, reason="test_cleanup")
    value.cancel_queued(
        "route-failure-blocked", now_ns=10_023, reason="test_cleanup")
    value.update_telemetry(telemetry(
        0, sequence=2, sampled_ns=10_020, remote_health=PathHealth.PROBE))
    value.update_telemetry(telemetry(1, sequence=3, sampled_ns=10_020))
    recovered = value.submit(request(
        "route-failure-recovered", "batch", (
            candidate(0, GlobalRoute.REMOTE, e2e=10),
        ),
    ), now_ns=10_024)
    assert recovered.kind is GlobalDecisionKind.ADMIT


def test_quarantined_pair_is_skipped_and_spare_pair_is_activated() -> None:
    value = controller()
    value.update_telemetry(
        telemetry(
            0,
            local_health=PathHealth.DENIED,
            remote_health=PathHealth.DENIED,
        )
    )
    value.update_telemetry(telemetry(1))
    decision = value.submit(request(
        "quarantine-r0", "latency", (
            candidate(0, GlobalRoute.LOCAL, e2e=1),
            candidate(1, GlobalRoute.LOCAL, e2e=20),
        )
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.pair_index == 1
    assert decision.pair_activated is True
    assert {item.reason for item in decision.rejected_candidates} >= {
        "path_denied",
    }


def test_first_response_releases_prefill_but_eof_owns_decode() -> None:
    value = controller()
    seed(value)
    admitted = value.submit(request(
        "r0", "latency", (candidate(0, GlobalRoute.REMOTE, e2e=10),)
    ), now_ns=10)
    assert admitted.kind is GlobalDecisionKind.ADMIT
    before = value.snapshot(now_ns=11)["owned_by_pair"]["0"]
    assert before["remote_kv_bytes"] == 400
    assert before["decode_tokens"] == 40
    value.mark_first_response("r0", now_ns=20)
    middle = value.snapshot(now_ns=21)["owned_by_pair"]["0"]
    assert middle["remote_kv_bytes"] == 0
    assert middle["remote_semantic_ops"] == 0
    assert middle["decode_tokens"] == 40
    assert middle["active_sequences"] == 1
    value.complete("r0", now_ns=30)
    after = value.snapshot(now_ns=31)["owned_by_pair"]["0"]
    assert not any(after.values())
    with pytest.raises(ValueError, match="in flight"):
        value.complete("r0", now_ns=32)


def test_pre_warmed_pair_scales_up_when_active_pair_is_full() -> None:
    value = controller()
    seed(value)
    first = value.submit(request(
        "r0", "latency", (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=100),)
    ), now_ns=10)
    assert first.pair_index == 0
    second = value.submit(request(
        "r1", "batch", (
            candidate(0, GlobalRoute.LOCAL, e2e=9),
            candidate(1, GlobalRoute.LOCAL, e2e=12),
        )
    ), now_ns=11)
    assert second.kind is GlobalDecisionKind.ADMIT
    assert second.pair_index == 1
    assert second.pair_activated is True
    assert second.active_pairs_before == (0,)
    assert second.active_pairs_after == (0, 1)


def test_scheduler_waiting_pressure_proactively_considers_spare_pair() -> None:
    value = controller()
    value.update_telemetry(telemetry(
        0, scheduler_running=2, scheduler_waiting=0, scheduler_kv=0.5))
    value.update_telemetry(telemetry(1))
    decision = value.submit(request(
        "scheduler-pressure", "latency", (
            candidate(0, GlobalRoute.LOCAL, e2e=50),
            candidate(1, GlobalRoute.LOCAL, e2e=1),
        ),
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.pair_index == 1
    assert decision.pair_activated is True


def test_endpoint_completion_residual_proactively_considers_spare_pair() -> None:
    value = controller()
    value.update_telemetry(telemetry(0, completion_residual=2))
    value.update_telemetry(telemetry(1))
    decision = value.submit(request(
            "endpoint-residual-pressure", "latency", (
                candidate(0, GlobalRoute.LOCAL, e2e=1),
                candidate(1, GlobalRoute.LOCAL, e2e=4),
            ),
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.pair_index == 1
    assert decision.pair_activated is True


def test_queue_occupancy_proactively_scales_a_spare_pair() -> None:
    value = controller(
        queue_capacity=4,
        proactive_scale_up_queue_fraction=0.25,
        proactive_scale_up_wait_fraction=1.0,
        proactive_scale_up_active_pair_penalty_ms=25.0,
    )
    value.update_telemetry(telemetry(0))
    value.update_telemetry(telemetry(1))
    decision = value.submit(request(
        "queue-scale", "latency", (
            candidate(0, GlobalRoute.LOCAL, e2e=10),
            candidate(1, GlobalRoute.LOCAL, e2e=11),
        ),
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.pair_index == 1
    assert decision.pair_activated is True
    assert decision.reason == (
        "global_proactive_queue_scale_queue_occupancy_and_route_committed")


def test_tenant_queue_wait_risk_proactively_scales_a_spare_pair() -> None:
    value = controller(
        proactive_scale_up_queue_fraction=1.0,
        proactive_scale_up_wait_fraction=0.1,
    )
    first = value.submit(request(
        "queue-risk", "batch", (
            candidate(0, GlobalRoute.LOCAL, e2e=10),
            candidate(1, GlobalRoute.LOCAL, e2e=11),
        ),
    ), now_ns=10)
    assert first.kind is GlobalDecisionKind.QUEUE
    value.update_telemetry(telemetry(0, sampled_ns=10, local_multiplier=4.0))
    value.update_telemetry(telemetry(1, sampled_ns=10))
    decisions = value.dispatch(now_ns=100)
    assert len(decisions) == 1
    assert decisions[0].pair_index == 1
    assert decisions[0].pair_activated is True
    assert decisions[0].reason == (
        "global_proactive_queue_scale_tenant_queue_slo_risk_"
        "and_route_committed")


def test_fairness_service_unit_is_dominant_resource_not_decode_tokens() -> None:
    value = controller()
    seed(value)
    decision = value.submit(request(
        "dominant-unit", "latency", (
            candidate(0, GlobalRoute.LOCAL, e2e=10, decode=40),
        ),
    ), now_ns=10)
    assert decision.tenant_virtual_service_after == pytest.approx(0.25)
    assert value.snapshot(now_ns=11)["fairness_basis"] == (
        "weighted_dominant_resource_service")


def test_fairness_keeps_weighted_debt_and_raw_service_units_distinct() -> None:
    value = controller(tenants=(
        TenantPolicy("background", 0.5, minimum_service_fraction=0.4),
        TenantPolicy("latency", 4.0, minimum_service_fraction=0.4),
    ))
    seed(value)
    value.submit(request(
        "background-work", "background",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=10)
    value.submit(request(
        "latency-work", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=11)
    snapshot = value.snapshot(now_ns=12)
    assert snapshot["tenant_virtual_service"] == {
        "background": pytest.approx(1.0),
        "latency": pytest.approx(0.125),
    }
    assert snapshot["tenant_service_units"] == {
        "background": pytest.approx(0.5),
        "latency": pytest.approx(0.5),
    }


def test_business_pair_packing_activates_a_clean_pair_for_priority() -> None:
    value = controller(tenants=(
        TenantPolicy(
            "latency", 2.0, admission_priority=800,
            protected_capacity_fraction=0.2),
        TenantPolicy(
            "background", 0.5, admission_priority=0,
            pair_spread_limit=1),
    ))
    seed(value)
    packed = value.submit(request(
        "packed-background", "background", (
            candidate(0, GlobalRoute.LOCAL, e2e=10),
            candidate(1, GlobalRoute.LOCAL, e2e=20),
        ),
    ), now_ns=10)
    assert packed.pair_index == 0
    assert packed.reason == (
        "global_tenant_pair_scope_assigned_and_route_committed")

    urgent = value.submit(request(
        "urgent-clean-pair", "latency", (
            candidate(0, GlobalRoute.LOCAL, e2e=1),
            candidate(1, GlobalRoute.LOCAL, e2e=200),
        ),
    ), now_ns=11)
    assert urgent.pair_index == 1
    assert urgent.pair_activated is True
    assert urgent.reason == (
        "global_tenant_protected_pair_activated_and_route_committed")
    assert any(
        item.pair_index == 0
        and item.reason == "higher_priority_clean_pair_available"
        for item in urgent.rejected_candidates
    )
    value.mark_first_response("packed-background", now_ns=12)
    value.complete("packed-background", now_ns=13)

    second_background = value.submit(request(
        "packed-background-second", "background", (
            candidate(0, GlobalRoute.LOCAL, e2e=200),
            candidate(1, GlobalRoute.LOCAL, e2e=1),
        ),
        arrival_ns=14,
    ), now_ns=14)
    assert second_background.pair_index == 0
    assert any(
        item.pair_index == 1
        and item.reason == "tenant_pair_spread_limit"
        for item in second_background.rejected_candidates
    )
    snapshot = value.snapshot(now_ns=14)
    assert snapshot["tenant_pair_assignments"] == {
        "background": [0], "latency": [],
    }
    assert snapshot["tenant_policies"]["background"][
        "pair_spread_limit"] == 1


def test_business_pair_packing_spills_when_clean_pair_is_under_pressure() -> None:
    capacity = ResourceVector(
        decode_tokens=100, active_sequences=10, endpoint_requests=10,
        local_prefill_token_ms=100, remote_prefill_token_ms=100,
        remote_kv_bytes=1_000, remote_semantic_ops=2,
    )
    value = controller(
        capacities=(PairCapacity(0, capacity), PairCapacity(1, capacity)),
        business_clean_pair_pressure_fraction=0.5,
        tenants=(
            TenantPolicy(
                "latency", 2.0, admission_priority=800,
                protected_capacity_fraction=0.2),
            TenantPolicy(
                "background", 0.5, admission_priority=0,
                pair_spread_limit=1),
        ),
    )
    seed(value)
    packed = value.submit(request(
        "packed-background-spill", "background", (
            candidate(1, GlobalRoute.LOCAL, e2e=10),
        ),
    ), now_ns=10)
    assert packed.pair_index == 1
    value.update_telemetry(telemetry(
        0, sequence=2, sampled_ns=20,
        observed=ResourceVector(active_sequences=6, endpoint_requests=6),
    ))
    value.update_telemetry(telemetry(
        1, sequence=2, sampled_ns=20,
        observed=ResourceVector(active_sequences=1, endpoint_requests=1),
    ))
    urgent = value.submit(request(
        "urgent-spill", "latency", (
            candidate(0, GlobalRoute.LOCAL, e2e=100),
            candidate(1, GlobalRoute.LOCAL, e2e=2),
        ),
    ), now_ns=21)
    assert urgent.pair_index == 1
    assert not any(
        item.pair_index == 1
        and item.reason == "higher_priority_clean_pair_available"
        for item in urgent.rejected_candidates
    )


def test_admission_wait_budget_obeys_tenant_queue_slo() -> None:
    value = controller(tenants=(
        TenantPolicy("latency", 2.0, maximum_queue_wait_ns=100),
        TenantPolicy("batch", 1.0, maximum_queue_wait_ns=400),
    ))
    assert value.admission_wait_budget_ns("latency") == 100
    assert value.admission_wait_budget_ns("batch") == 400


def test_endpoint_queue_lease_is_explicit_and_preserves_global_ownership() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        tenants=(
            TenantPolicy(
                "latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    seed(value)
    holder = value.submit(request(
        "lease-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=100),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT
    queued = value.submit(request(
        "lease-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=40),),
    ), now_ns=12)
    assert queued.kind is GlobalDecisionKind.QUEUE
    leased = value.lease_queued_to_endpoint("lease-waiter", now_ns=20)
    assert leased is not None
    assert leased.kind is GlobalDecisionKind.ADMIT
    assert leased.queue_lease is True
    assert leased.reason == "global_endpoint_queue_lease_route_committed"
    assert "decode_tokens" in leased.binding_resources
    assert value.snapshot(now_ns=20)["phases"]["lease-waiter"] == "route_committed"
    value.mark_first_response("lease-holder", now_ns=21)
    value.complete("lease-holder", now_ns=22)
    value.mark_first_response("lease-waiter", now_ns=23)
    value.complete("lease-waiter", now_ns=24)


def test_endpoint_queue_lease_fails_closed_when_route_window_is_full() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    seed(value)
    holder = value.submit(request(
        "endpoint-window-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, local_prefill=100),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT
    queued = value.submit(request(
        "endpoint-window-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=12)
    assert queued.kind is GlobalDecisionKind.QUEUE

    # The decoder has a bounded waiting queue, but the endpoint local
    # prefill window is already full.  Do not commit a global route that the
    # endpoint can only reject after the HTTP request crosses the boundary.
    assert value.lease_queued_to_endpoint(
        "endpoint-window-waiter", now_ns=20) is None
    terminal = value.reject_queued(
        "endpoint-window-waiter",
        now_ns=21,
        reason="global_admission_queue_timeout",
    )
    assert terminal.kind is GlobalDecisionKind.REJECT
    assert [item.reason for item in terminal.rejected_candidates] == [
        "endpoint_service_window_full",
    ]


def test_work_conserving_endpoint_queue_debt_uses_bounded_native_queue() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode="work_conserving_endpoint_queue_v1",
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    value.update_telemetry(
        telemetry(0, scheduler_running=0, scheduler_waiting=0, scheduler_kv=0.0)
    )
    value.update_telemetry(
        telemetry(1, scheduler_running=0, scheduler_waiting=0, scheduler_kv=0.0)
    )
    holder = value.submit(request(
        "endpoint-debt-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, local_prefill=100),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT
    queued = value.submit(request(
        "endpoint-debt-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=12)
    assert queued.kind is GlobalDecisionKind.QUEUE

    leased = value.lease_queued_to_endpoint(
        "endpoint-debt-waiter", now_ns=20)
    assert leased is not None and leased.queue_lease is True
    assert "local_prefill_token_ms" in leased.binding_resources
    value.mark_first_response("endpoint-debt-holder", now_ns=21)
    value.complete("endpoint-debt-holder", now_ns=22)
    value.mark_first_response("endpoint-debt-waiter", now_ns=23)
    value.complete("endpoint-debt-waiter", now_ns=24)


def test_queue_lease_prefers_serviceable_alternate_route_over_endpoint_debt() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode="work_conserving_endpoint_queue_v1",
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    seed(value)
    holder = value.submit(request(
        "alternate-route-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, local_prefill=100),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT

    # Both routes exceed the global decoder window, so the request waits.
    # Only local also exceeds its endpoint service window; remote remains a
    # usable downstream route even though its static score is slightly higher.
    local = candidate(0, GlobalRoute.LOCAL, e2e=10, decode=80)
    remote = RouteCandidate(
        pair_index=0,
        route=GlobalRoute.REMOTE,
        work=ResourceVector(
            decode_tokens=80,
            active_sequences=1,
            endpoint_requests=1,
            remote_prefill_token_ms=30,
            remote_kv_bytes=400,
            remote_semantic_ops=1,
        ),
        predicted_e2e_ms=11,
        predicted_ttft_ms=5.5,
        uncertainty_ms=1.0,
        cache_affinity=True,
    )
    queued = value.submit(request(
        "alternate-route-waiter", "latency", (local, remote),
    ), now_ns=12)
    assert queued.kind is GlobalDecisionKind.QUEUE

    leased = value.lease_queued_to_endpoint(
        "alternate-route-waiter", now_ns=20)
    assert leased is not None and leased.queue_lease is True
    assert leased.route is GlobalRoute.REMOTE
    assert "local_prefill_token_ms" not in leased.binding_resources


def test_endpoint_queue_lease_uses_observed_scheduler_headroom() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    seed(value)
    value.update_telemetry(telemetry(
        0,
        sequence=2,
        scheduler_running=2,
        scheduler_waiting=0,
        scheduler_kv=0.25,
    ))
    holder = value.submit(request(
        "scheduler-headroom-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, local_prefill=100),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT
    queued = value.submit(request(
        "scheduler-headroom-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=12)
    assert queued.kind is GlobalDecisionKind.QUEUE

    leased = value.lease_queued_to_endpoint(
        "scheduler-headroom-waiter", now_ns=20)
    assert leased is not None and leased.queue_lease is True
    assert leased.reason == "global_endpoint_queue_lease_route_committed"
    assert leased.telemetry_provenance[0]["scheduler"][
        "waiting_requests"] == 0
    value.mark_first_response("scheduler-headroom-holder", now_ns=21)
    value.complete("scheduler-headroom-holder", now_ns=22)
    value.mark_first_response("scheduler-headroom-waiter", now_ns=23)
    value.complete("scheduler-headroom-waiter", now_ns=24)


def test_endpoint_queue_debt_uses_explicit_native_waiting_capacity() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode="work_conserving_endpoint_queue_v1",
        endpoint_queue_capacity=128,
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    seed(value)
    value.update_telemetry(telemetry(
        0,
        sequence=2,
        scheduler_running=16,
        scheduler_waiting=65,
        scheduler_kv=0.75,
        completion_residual=0,
    ))
    holder = value.submit(request(
        "native-queue-capacity-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, local_prefill=100),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT
    queued = value.submit(request(
        "native-queue-capacity-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=12)
    assert queued.kind is GlobalDecisionKind.QUEUE

    # Active endpoint_requests remains a two-request reservation in this
    # fixture, while the observed vLLM waiting queue is bounded separately by
    # the explicit native queue capacity of 128.
    leased = value.lease_queued_to_endpoint(
        "native-queue-capacity-waiter", now_ns=20)
    assert leased is not None and leased.queue_lease is True
    assert "local_prefill_token_ms" in leased.binding_resources
    value.mark_first_response("native-queue-capacity-holder", now_ns=21)
    value.complete("native-queue-capacity-holder", now_ns=22)
    value.mark_first_response("native-queue-capacity-waiter", now_ns=23)
    value.complete("native-queue-capacity-waiter", now_ns=24)


def test_endpoint_queue_lease_rejects_scheduler_and_completion_overload() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode="work_conserving_endpoint_queue_v1",
        endpoint_queue_capacity=2,
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    seed(value)
    value.update_telemetry(telemetry(
        0,
        sequence=2,
        scheduler_running=2,
        scheduler_waiting=1,
        scheduler_kv=0.9,
        completion_residual=1,
    ))
    holder = value.submit(request(
        "service-lane-overloaded-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, local_prefill=100),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT
    queued = value.submit(request(
        "service-lane-overloaded-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=12)
    assert queued.kind is GlobalDecisionKind.QUEUE
    assert value.lease_queued_to_endpoint(
        "service-lane-overloaded-waiter", now_ns=20) is None


def test_endpoint_queue_lease_blocks_v104_native_service_window_backlog() -> None:
    """The v104 native receipt must not become a 16-second HTTP 503."""
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode="work_conserving_endpoint_queue_v1",
        endpoint_queue_capacity=16,
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    seed(value)
    # These are the measured v104 failure-window values, not a workload
    # oracle: the native pair reported 16 running, 14 waiting, and 13
    # endpoint-residual requests when the lease was committed.
    value.update_telemetry(telemetry(
        0,
        sequence=2,
        scheduler_running=16,
        scheduler_waiting=14,
        scheduler_kv=0.95,
        completion_residual=13,
    ))
    holder = value.submit(request(
        "v104-service-window-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, local_prefill=100),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT
    queued = value.submit(request(
        "v104-service-window-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=12)
    assert queued.kind is GlobalDecisionKind.QUEUE

    assert value.lease_queued_to_endpoint(
        "v104-service-window-waiter", now_ns=20) is None
    terminal = value.reject_queued(
        "v104-service-window-waiter",
        now_ns=21,
        reason="endpoint_queue_capacity_full",
    )
    assert terminal.kind is GlobalDecisionKind.REJECT
    assert [item.reason for item in terminal.rejected_candidates] == [
        "endpoint_queue_capacity_full",
    ]


def test_completion_liveness_v2_recovers_failure_free_stale_route() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode="completion_liveness_endpoint_queue_v2",
        endpoint_queue_capacity=16,
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    value.update_telemetry(telemetry(
        0,
        local_health=PathHealth.SKIP,
        local_multiplier=50.0,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    value.update_telemetry(telemetry(
        1,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    queued = value.submit(request(
        "failure-free-stale", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=11)
    assert queued.kind is GlobalDecisionKind.QUEUE

    leased = value.lease_queued_to_endpoint(
        "failure-free-stale", now_ns=20)
    assert leased is not None
    assert leased.queue_lease is True
    assert leased.route is GlobalRoute.LOCAL
    assert leased.reason == (
        "global_endpoint_completion_liveness_probe_route_committed")
    assert leased.score_ms is not None and leased.score_ms < 100.0


def test_completion_liveness_v2_never_probes_explicit_failure() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode="completion_liveness_endpoint_queue_v2",
        endpoint_queue_capacity=16,
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    value.update_telemetry(telemetry(
        0,
        local_health=PathHealth.SKIP,
        local_multiplier=50.0,
        local_failure_count=1,
        local_last_failure_kind="active_upstream_failure",
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    value.update_telemetry(telemetry(
        1,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    queued = value.submit(request(
        "explicit-failure", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=11)
    assert queued.kind is GlobalDecisionKind.QUEUE
    assert value.lease_queued_to_endpoint(
        "explicit-failure", now_ns=20) is None


def test_completion_liveness_v2_queues_remote_semantic_debt_safely() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode="completion_liveness_endpoint_queue_v2",
        endpoint_queue_capacity=16,
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    value.update_telemetry(telemetry(
        0,
        observed=ResourceVector(remote_semantic_ops=2),
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    value.update_telemetry(telemetry(
        1,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    queued = value.submit(request(
        "remote-semantic-debt", "latency",
        (candidate(0, GlobalRoute.REMOTE, e2e=10),),
    ), now_ns=11)
    assert queued.kind is GlobalDecisionKind.QUEUE
    leased = value.lease_queued_to_endpoint(
        "remote-semantic-debt", now_ns=20)
    assert leased is not None and leased.queue_lease is True
    assert leased.route is GlobalRoute.REMOTE
    assert "remote_semantic_ops" in leased.binding_resources


def test_completion_credit_v3_leases_once_per_observed_first_response() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode="completion_credit_endpoint_queue_v3",
        endpoint_queue_capacity=16,
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    value.update_telemetry(telemetry(
        0,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    value.update_telemetry(telemetry(
        1,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    value.update_telemetry(telemetry(
        0,
        sequence=2,
        sampled_ns=12,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=5,
    ))
    assert value.snapshot(now_ns=12)["completion_credit_balance"]["0"] == 1

    holder = value.submit(request(
        "credit-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, local_prefill=100),),
    ), now_ns=13)
    assert holder.kind is GlobalDecisionKind.ADMIT
    first_waiter = value.submit(request(
        "credit-first-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=14)
    assert first_waiter.kind is GlobalDecisionKind.QUEUE
    leased = value.lease_queued_to_endpoint(
        "credit-first-waiter", now_ns=20)
    assert leased is not None and leased.queue_lease is True
    assert leased.reason == "global_endpoint_completion_credit_route_committed"
    assert "completion_first_response_credit" in leased.binding_resources
    assert value.snapshot(now_ns=20)["completion_credit_balance"]["0"] == 0

    second_waiter = value.submit(request(
        "credit-second-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=21)
    assert second_waiter.kind is GlobalDecisionKind.QUEUE
    assert value.lease_queued_to_endpoint(
        "credit-second-waiter", now_ns=22) is None
    terminal = value.reject_queued(
        "credit-second-waiter",
        now_ns=23,
        reason="completion_credit_unavailable",
    )
    assert [item.reason for item in terminal.rejected_candidates] == [
        "completion_credit_unavailable",
    ]


def test_completion_credit_v3_refills_one_remote_semantic_slot() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode="completion_credit_endpoint_queue_v3",
        endpoint_queue_capacity=128,
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    value.update_telemetry(telemetry(
        0,
        observed=ResourceVector(remote_semantic_ops=2),
        scheduler_running=2,
        scheduler_waiting=0,
        scheduler_kv=0.5,
        completion_residual=1,
        completion_completed=4,
    ))
    value.update_telemetry(telemetry(
        1,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    queued = value.submit(request(
        "credit-remote-refill", "latency",
        (candidate(0, GlobalRoute.REMOTE, e2e=10),),
    ), now_ns=11)
    assert queued.kind is GlobalDecisionKind.QUEUE
    value.update_telemetry(telemetry(
        0,
        sequence=2,
        sampled_ns=12,
        observed=ResourceVector(remote_semantic_ops=2),
        scheduler_running=2,
        scheduler_waiting=0,
        scheduler_kv=0.5,
        completion_residual=1,
        completion_completed=5,
    ))

    leased = value.lease_queued_to_endpoint(
        "credit-remote-refill", now_ns=13)
    assert leased is not None and leased.queue_lease is True
    assert leased.route is GlobalRoute.REMOTE
    assert leased.reason == "global_endpoint_completion_credit_route_committed"
    assert "remote_semantic_ops" in leased.binding_resources
    assert value.snapshot(now_ns=13)["completion_credit_balance"]["0"] == 0


def test_mesh_completion_credit_lease_preserves_destination_and_edge_ownership() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=1,
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode=(
            "completion_credit_mesh_endpoint_queue_v1"),
        endpoint_queue_capacity=16,
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    value.update_telemetry(telemetry(
        0,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    value.update_telemetry(telemetry(
        1,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    value.update_telemetry(telemetry(
        0,
        sequence=2,
        sampled_ns=12,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=5,
    ))
    holder = value.submit(request(
        "mesh-queue-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, local_prefill=100),),
    ), now_ns=13)
    assert holder.kind is GlobalDecisionKind.ADMIT
    waiter = value.submit(request(
        "mesh-queue-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=14)
    assert waiter.kind is GlobalDecisionKind.QUEUE
    leased = value.lease_queued_to_endpoint(
        "mesh-queue-waiter", now_ns=20)
    assert leased is not None and leased.queue_lease is True
    assert leased.reason == "global_endpoint_completion_credit_route_committed"
    assert "completion_first_response_credit" in leased.binding_resources
    assert value.snapshot(now_ns=20)["completion_credit_balance"]["0"] == 0
    assert value.snapshot(now_ns=20)["owned_by_pair"]["0"][
        "local_prefill_token_ms"] == 140


def test_priority_remote_cache_lane_bypasses_only_ordinary_decoder_backlog() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=1,
        maximum_active_pairs=2,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode=(
            "completion_credit_mesh_endpoint_queue_v1"),
        endpoint_queue_admission_mode="headroom_first_v1",
        endpoint_queue_capacity=32,
        priority_service_lane_mode="vllm_priority_remote_cache_v1",
        priority_service_lane_capacity=1,
        priority_service_lane_min_admission_priority=800,
        priority_service_lane_priority=-2,
        tenants=(
            TenantPolicy(
                "latency", 4.0, admission_priority=1_000,
                protected_capacity_fraction=0.2,
            ),
            TenantPolicy(
                "interactive", 2.0, e2e_slo_ms=8_000.0,
                queue_lease_on_timeout=True, admission_priority=800,
            ),
            TenantPolicy("background", 0.5),
        ),
    )
    value.update_telemetry(telemetry(
        0,
        scheduler_running=2,
        scheduler_waiting=68,
        scheduler_kv=0.9,
        completion_residual=8,
        completion_completed=10,
    ))
    value.update_telemetry(telemetry(
        1,
        observed=ResourceVector(
            decode_tokens=100,
            active_sequences=2,
            endpoint_requests=2,
        ),
        scheduler_running=2,
        scheduler_waiting=68,
        scheduler_kv=0.9,
        completion_residual=8,
        completion_completed=10,
    ))
    queued = value.submit(request(
        "priority-p0-d1", "interactive",
        (mesh_candidate(
            0, 1, e2e=10, cache_affinity=True),),
    ), now_ns=11)
    assert queued.kind is GlobalDecisionKind.QUEUE

    leased = value.lease_queued_to_endpoint(
        "priority-p0-d1", now_ns=12)
    assert leased is not None and leased.queue_lease is True
    assert leased.reason == (
        "global_priority_remote_cache_service_lane_route_committed")
    assert leased.prefill_index == 0
    assert leased.decoder_index == leased.pair_index == 1
    assert leased.edge_id == "remote:p0->d1"
    assert "vllm_priority_remote_cache_service_lane" in (
        leased.binding_resources)
    assert "completion_first_response_credit" not in (
        leased.binding_resources)
    state = value.snapshot(now_ns=12)
    assert state["priority_service_lane_priority"] == -2
    assert state["priority_service_lane_debt"] == {"0": 0, "1": 1}

    second = value.submit(request(
        "priority-lane-full", "interactive",
        (mesh_candidate(
            0, 1, e2e=10, cache_affinity=True),),
        arrival_ns=13,
    ), now_ns=13)
    assert second.kind is GlobalDecisionKind.QUEUE
    assert value.lease_queued_to_endpoint(
        "priority-lane-full", now_ns=14) is None
    rejected = value.reject_queued(
        "priority-lane-full", now_ns=15, reason="test_terminal")
    assert any(
        item.edge_id == "remote:p0->d1"
        for item in rejected.rejected_candidates
    )


def test_business_dual_route_lane_uses_local_priority_when_fabric_route_is_unusable() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=1,
        maximum_active_pairs=2,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode=(
            "completion_credit_mesh_endpoint_queue_v1"),
        endpoint_queue_admission_mode="headroom_first_v1",
        endpoint_queue_capacity=32,
        priority_service_lane_mode=(
            "vllm_priority_business_dual_route_v2"),
        priority_service_lane_capacity=1,
        priority_service_lane_min_admission_priority=800,
        priority_service_lane_priority=-2,
        tenants=(
            TenantPolicy(
                "latency", 4.0, admission_priority=1_000,
                protected_capacity_fraction=0.2,
            ),
            TenantPolicy(
                "interactive", 2.0, e2e_slo_ms=8_000.0,
                queue_lease_on_timeout=True, admission_priority=800,
            ),
            TenantPolicy("background", 0.5),
        ),
    )
    value.update_telemetry(telemetry(
        0,
        observed=ResourceVector(
            decode_tokens=100,
            active_sequences=2,
            endpoint_requests=2,
            local_prefill_token_ms=100,
        ),
        scheduler_running=2,
        scheduler_waiting=68,
        scheduler_kv=0.9,
        completion_residual=8,
        completion_completed=10,
    ))
    value.update_telemetry(telemetry(
        1,
        scheduler_running=2,
        scheduler_waiting=68,
        scheduler_kv=0.9,
        completion_residual=8,
        completion_completed=10,
    ))
    queued = value.submit(request(
        "priority-local-d0", "interactive",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=11)
    assert queued.kind is GlobalDecisionKind.QUEUE

    leased = value.lease_queued_to_endpoint(
        "priority-local-d0", now_ns=12)
    assert leased is not None and leased.queue_lease is True
    assert leased.route is GlobalRoute.LOCAL
    assert leased.reason == (
        "global_priority_business_dual_route_service_lane_route_committed")
    assert "vllm_priority_business_dual_route_service_lane" in (
        leased.binding_resources)
    assert value.snapshot(now_ns=12)["priority_service_lane_debt"] == {
        "0": 1, "1": 0,
    }


def test_priority_lane_balances_only_near_tie_remote_sources() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=2,
        maximum_active_pairs=2,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode=(
            "completion_credit_mesh_endpoint_queue_v1"),
        endpoint_queue_admission_mode="headroom_first_v1",
        endpoint_queue_capacity=32,
        priority_service_lane_mode="vllm_priority_remote_cache_v1",
        priority_service_lane_capacity=1,
        priority_service_lane_min_admission_priority=800,
        priority_service_lane_priority=-2,
        mesh_near_tie_source_balance_mode=(
            "telemetry_uncertainty_virtual_service_v1"),
        mesh_near_tie_source_balance_uncertainty_fraction=1.0,
        tenants=(
            TenantPolicy(
                "interactive", 2.0, e2e_slo_ms=8_000.0,
                queue_lease_on_timeout=True, admission_priority=800,
                protected_capacity_fraction=0.2,
            ),
            TenantPolicy("background", 0.5),
        ),
    )
    for pair in (0, 1):
        value.update_telemetry(telemetry(
            pair,
            observed=(
                ResourceVector(
                    decode_tokens=100,
                    active_sequences=2,
                    endpoint_requests=2,
                )
                if pair == 0 else ResourceVector()
            ),
            scheduler_running=2,
            scheduler_waiting=68,
            scheduler_kv=0.9,
            completion_residual=8,
            completion_completed=10,
        ))

    def run_one(request_id: str, now_ns: int):
        queued = value.submit(request(
            request_id,
            "interactive",
            (
                mesh_candidate(0, 0, e2e=10, cache_affinity=True),
                mesh_candidate(1, 0, e2e=10, cache_affinity=True),
            ),
            arrival_ns=now_ns,
        ), now_ns=now_ns)
        assert queued.kind is GlobalDecisionKind.QUEUE
        leased = value.lease_queued_to_endpoint(
            request_id, now_ns=now_ns + 1)
        assert leased is not None
        assert leased.mesh_near_tie_source_balanced is True
        assert leased.mesh_near_tie_score_window_ms == 1.0
        assert leased.mesh_near_tie_score_delta_ms == 0.0
        assert "mesh_telemetry_uncertainty_source_virtual_service" in (
            leased.binding_resources)
        value.mark_first_response(request_id, now_ns=now_ns + 2)
        value.complete(request_id, now_ns=now_ns + 3)
        return leased

    first = run_one("near-tie-first", 11)
    second = run_one("near-tie-second", 21)
    assert first.edge_id == "remote:p0->d0"
    assert second.edge_id == "remote:p1->d0"
    rejected_p0 = next(
        item for item in second.rejected_candidates
        if item.edge_id == "remote:p0->d0"
    )
    assert rejected_p0.reason == "mesh_near_tie_source_virtual_service"
    assert rejected_p0.evaluated_score_ms == second.score_ms
    assert rejected_p0.score_delta_ms == 0.0
    assert rejected_p0.mesh_near_tie_eligible is True
    state = value.snapshot(now_ns=25)
    assert state["mesh_source_virtual_service"] == {"0": 0.3, "1": 0.3}


def test_mesh_source_balance_never_overrides_score_outside_uncertainty() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=2,
        maximum_active_pairs=2,
        mesh_near_tie_source_balance_mode=(
            "telemetry_uncertainty_virtual_service_v1"),
        mesh_near_tie_source_balance_uncertainty_fraction=1.0,
    )
    seed(value)
    first = value.submit(request(
        "outside-window-seed",
        "latency",
        (mesh_candidate(0, 0, e2e=10),),
    ), now_ns=11)
    assert first.kind is GlobalDecisionKind.ADMIT
    value.mark_first_response("outside-window-seed", now_ns=12)
    value.complete("outside-window-seed", now_ns=13)

    decision = value.submit(request(
        "outside-window-choice",
        "latency",
        (
            mesh_candidate(0, 0, e2e=10),
            mesh_candidate(1, 0, e2e=12),
        ),
        arrival_ns=20,
    ), now_ns=20)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.edge_id == "remote:p0->d0"
    assert decision.mesh_near_tie_source_balanced is False
    rejected_p1 = next(
        item for item in decision.rejected_candidates
        if item.edge_id == "remote:p1->d0"
    )
    assert rejected_p1.reason == "higher_global_score"
    assert rejected_p1.score_delta_ms > rejected_p1.uncertainty_ms
    assert rejected_p1.mesh_near_tie_eligible is False


def test_priority_lane_does_not_admit_local_or_unproven_cache_work() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode="completion_credit_endpoint_queue_v3",
        endpoint_queue_capacity=2,
        priority_service_lane_mode="vllm_priority_remote_cache_v1",
        priority_service_lane_capacity=1,
        priority_service_lane_min_admission_priority=800,
        priority_service_lane_priority=-2,
        tenants=(
            TenantPolicy(
                "interactive", 2.0, queue_lease_on_timeout=True,
                admission_priority=800,
            ),
            TenantPolicy("background", 0.5),
        ),
    )
    value.update_telemetry(telemetry(
        0,
        observed=ResourceVector(
            decode_tokens=100,
            active_sequences=2,
            endpoint_requests=2,
        ),
        scheduler_running=2,
        scheduler_waiting=68,
        scheduler_kv=0.9,
        completion_residual=8,
        completion_completed=10,
    ))
    value.update_telemetry(telemetry(
        1,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=10,
    ))
    for offset, (request_id, route_candidate) in enumerate((
        ("priority-local-denied", candidate(
            0, GlobalRoute.LOCAL, e2e=10)),
        ("priority-cold-remote-denied", candidate(
            0, GlobalRoute.REMOTE, e2e=10)),
    )):
        now_ns = 11 + offset * 3
        queued = value.submit(request(
            request_id, "interactive", (route_candidate,),
            arrival_ns=now_ns,
        ), now_ns=now_ns)
        assert queued.kind is GlobalDecisionKind.QUEUE
        assert value.lease_queued_to_endpoint(
            request_id, now_ns=now_ns + 1) is None
        value.reject_queued(
            request_id, now_ns=now_ns + 2, reason="test_terminal")


def test_endpoint_queue_lease_timeout_preserves_global_rejection_causes() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        tenants=(
            TenantPolicy(
                "latency",
                2.0,
                e2e_slo_ms=15.0,
                queue_lease_on_timeout=True,
            ),
            TenantPolicy("batch", 1.0),
        ),
    )
    seed(value)
    holder = value.submit(request(
        "cause-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=5, decode=40),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT
    queued = value.submit(request(
        "cause-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=20),),
    ), now_ns=12)
    assert queued.kind is GlobalDecisionKind.QUEUE

    assert value.lease_queued_to_endpoint("cause-waiter", now_ns=20) is None
    terminal = value.reject_queued(
        "cause-waiter",
        now_ns=21,
        reason="global_admission_queue_timeout",
    )
    assert terminal.kind is GlobalDecisionKind.REJECT
    assert [item.reason for item in terminal.rejected_candidates] == [
        "tenant_e2e_slo",
    ]
    assert terminal.rejected_candidates[0].pair_index == 0


def test_endpoint_queue_lease_carries_decoder_window_debt_to_native_queue() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    seed(value)
    holder = value.submit(request(
        "hard-cap-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT
    holder_two = value.submit(request(
        "hard-cap-holder-two", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=12)
    assert holder_two.kind is GlobalDecisionKind.ADMIT
    # The test capacity has two active sequences.  The native vLLM waiting
    # queue is precisely where a third request belongs after the global
    # reservation window expires.  TEMPO keeps the overage in its ownership
    # ledger and receipt instead of turning it into an ingress reject.
    queued = value.submit(request(
        "hard-cap-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, local_prefill=20),),
    ), now_ns=13)
    assert queued.kind is GlobalDecisionKind.QUEUE
    leased = value.lease_queued_to_endpoint("hard-cap-waiter", now_ns=20)
    assert leased is not None and leased.queue_lease is True
    assert {
        "active_sequences", "endpoint_requests",
    } <= set(leased.binding_resources)
    assert value.snapshot(now_ns=20)["phases"]["hard-cap-waiter"] == (
        "route_committed"
    )
    value.mark_first_response("hard-cap-holder", now_ns=21)
    value.complete("hard-cap-holder", now_ns=22)
    value.mark_first_response("hard-cap-holder-two", now_ns=23)
    value.complete("hard-cap-holder-two", now_ns=24)
    value.mark_first_response("hard-cap-waiter", now_ns=25)
    value.complete("hard-cap-waiter", now_ns=26)


def test_service_lane_reservation_failure_releases_global_debt_without_quarantine() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    seed(value)
    holder = value.submit(request(
        "reservation-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=100),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT
    queued = value.submit(request(
        "reservation-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=40),),
    ), now_ns=12)
    assert queued.kind is GlobalDecisionKind.QUEUE
    leased = value.lease_queued_to_endpoint("reservation-waiter", now_ns=20)
    assert leased is not None and leased.queue_lease is True

    report = value.fail_service_lane_reservation(
        "reservation-waiter",
        failure_kind="endpoint_service_lane_reservation_unavailable",
        reason="endpoint_service_lane_capacity_unavailable",
        now_ns=21,
    )
    assert report.receipt.phase_before.value == "route_committed"
    assert report.receipt.terminal_phase.value == "failed"
    assert report.receipt.released_work["decode_tokens"] == 40
    assert report.receipt.schema == "tempo-go-service-lane-reservation-v1"
    assert len(global_service_lane_reservation_failure_fingerprint(
        report.receipt)) == 64
    assert global_service_lane_reservation_failure_dict(
        report.receipt)["terminal_phase"] == "failed"
    state = value.snapshot(now_ns=22)
    assert state["phases"]["reservation-waiter"] == "failed"
    assert state["owned_by_pair"]["0"] == {
        "decode_tokens": 100,
        "active_sequences": 1,
        "endpoint_requests": 1,
        "local_prefill_token_ms": 40,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }
    assert not state["route_failure_quarantines"]
    value.mark_first_response("reservation-holder", now_ns=23)
    value.complete("reservation-holder", now_ns=24)


def test_service_lane_queue_offer_promotes_existing_ownership_once() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode="completion_credit_endpoint_queue_v3",
        endpoint_queue_capacity=16,
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    value.update_telemetry(telemetry(
        0,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    value.update_telemetry(telemetry(
        1,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    admitted = value.submit(request(
        "service-lane-promote", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=11)
    assert admitted.kind is GlobalDecisionKind.ADMIT
    before = value.snapshot(now_ns=11)["owned_by_pair"]["0"]

    report = value.promote_service_lane_queue_lease(
        "service-lane-promote", now_ns=12)
    assert report.decision is not None
    assert report.decision.queue_lease is True
    assert report.receipt.status == "promoted"
    assert report.receipt.completion_liveness_probe is True
    assert report.receipt.completion_credit_consumed is False
    assert report.receipt.endpoint_queue_debt_before == 0
    assert report.receipt.endpoint_queue_debt_after == 1
    assert value.snapshot(now_ns=12)["owned_by_pair"]["0"] == before
    payload = global_service_lane_queue_promotion_dict(report.receipt)
    assert payload["reason"] == (
        "global_endpoint_service_lane_completion_liveness_promoted")
    assert len(global_service_lane_queue_promotion_fingerprint(
        report.receipt)) == 64

    with pytest.raises(ValueError, match="already promoted"):
        value.promote_service_lane_queue_lease(
            "service-lane-promote", now_ns=13)
    value.mark_first_response("service-lane-promote", now_ns=14)
    value.complete("service-lane-promote", now_ns=15)


def test_mesh_shared_liveness_probe_reuses_one_failure_free_probe_with_headroom() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=1,
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode=(
            "completion_credit_mesh_endpoint_queue_v1"),
        completion_liveness_shared_probe_mode="headroom_shared_v1",
        endpoint_queue_capacity=16,
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    value.update_telemetry(telemetry(
        0,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    value.update_telemetry(telemetry(
        1,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    holder = value.submit(request(
        "shared-probe-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=100),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT
    waiter = value.submit(request(
        "shared-probe-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=12)
    assert waiter.kind is GlobalDecisionKind.QUEUE

    # The first queue lease consumes the one failure-free recovery probe.
    # The second lease arrives while that probe is still in flight.  It may
    # share the evidence only because native endpoint queue headroom remains.
    value.update_telemetry(telemetry(
        0,
        sequence=2,
        sampled_ns=13,
        local_health=PathHealth.SKIP,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    first = value.promote_service_lane_queue_lease(
        "shared-probe-holder", now_ns=14)
    assert first.decision is not None
    assert first.receipt.completion_liveness_probe is True
    second = value.lease_queued_to_endpoint(
        "shared-probe-waiter", now_ns=15)
    assert second is not None
    assert second.queue_lease is True
    assert "completion_liveness_shared_probe" in second.binding_resources
    assert second.reason == (
        "global_endpoint_completion_liveness_shared_probe_route_committed")

    value.mark_first_response("shared-probe-holder", now_ns=16)
    value.complete("shared-probe-holder", now_ns=17)
    value.mark_first_response("shared-probe-waiter", now_ns=18)
    value.complete("shared-probe-waiter", now_ns=19)


def test_mesh_headroom_admission_can_lease_initial_waiter_without_credit() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=1,
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode=(
            "completion_credit_mesh_endpoint_queue_v1"),
        endpoint_queue_admission_mode="headroom_first_v1",
        endpoint_queue_headroom_admission_mode="completion_progress_v1",
        endpoint_queue_capacity=16,
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    for pair in (0, 1):
        value.update_telemetry(telemetry(
            pair,
            scheduler_running=0,
            scheduler_waiting=0,
            scheduler_kv=0.0,
            completion_residual=0,
            completion_completed=4,
        ))
    holder = value.submit(request(
        "headroom-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=100),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT
    waiter = value.submit(request(
        "headroom-waiter", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=12)
    assert waiter.kind is GlobalDecisionKind.QUEUE
    leased = value.lease_queued_to_endpoint("headroom-waiter", now_ns=13)
    assert leased is not None
    assert leased.queue_lease is True
    assert leased.reason == (
        "global_endpoint_completion_progress_headroom_route_committed")
    assert "completion_progress_headroom" in leased.binding_resources
    assert "completion_first_response_credit" not in leased.binding_resources
    value.mark_first_response("headroom-holder", now_ns=14)
    value.complete("headroom-holder", now_ns=15)
    value.mark_first_response("headroom-waiter", now_ns=16)
    value.complete("headroom-waiter", now_ns=17)


def test_service_lane_queue_promotion_consumes_causal_completion_credit() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        endpoint_queue_debt_mode="completion_credit_endpoint_queue_v3",
        endpoint_queue_capacity=16,
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    for pair in (0, 1):
        value.update_telemetry(telemetry(
            pair,
            scheduler_running=0,
            scheduler_waiting=0,
            scheduler_kv=0.0,
            completion_residual=0,
            completion_completed=4,
        ))
    value.update_telemetry(telemetry(
        0,
        sequence=2,
        sampled_ns=12,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=5,
    ))
    admitted = value.submit(request(
        "service-lane-credit", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=13)
    assert admitted.kind is GlobalDecisionKind.ADMIT
    report = value.promote_service_lane_queue_lease(
        "service-lane-credit", now_ns=14)
    assert report.decision is not None
    assert report.receipt.completion_credit_consumed is True
    assert report.receipt.completion_liveness_probe is False
    assert value.snapshot(now_ns=14)["completion_credit_balance"]["0"] == 0
    value.mark_first_response("service-lane-credit", now_ns=15)
    value.complete("service-lane-credit", now_ns=16)


def test_endpoint_queue_timeout_cools_only_future_queue_leases_until_drain() -> None:
    value = controller(
        maximum_active_pairs=1,
        overload_action="endpoint_queue_lease",
        tenants=(
            TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
            TenantPolicy("batch", 1.0),
        ),
    )
    seed(value)
    holder = value.submit(request(
        "cooldown-holder", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=100),),
    ), now_ns=11)
    assert holder.kind is GlobalDecisionKind.ADMIT
    first_waiter = value.submit(request(
        "cooldown-first", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=40),),
    ), now_ns=12)
    assert first_waiter.kind is GlobalDecisionKind.QUEUE
    leased = value.lease_queued_to_endpoint("cooldown-first", now_ns=20)
    assert leased is not None and leased.queue_lease is True
    value.fail_service_lane_reservation(
        "cooldown-first",
        failure_kind="endpoint_bounded_queue_lease_timeout",
        reason="endpoint_bounded_queue_lease_timeout",
        now_ns=21,
    )

    second_waiter = value.submit(request(
        "cooldown-second", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=40),),
    ), now_ns=22)
    assert second_waiter.kind is GlobalDecisionKind.QUEUE
    assert value.lease_queued_to_endpoint("cooldown-second", now_ns=23) is None
    terminal = value.reject_queued(
        "cooldown-second",
        now_ns=24,
        reason="global_admission_queue_timeout",
    )
    assert [item.reason for item in terminal.rejected_candidates] == [
        "endpoint_queue_lease_cooldown",
    ]

    value.update_telemetry(telemetry(
        0,
        sequence=2,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
    ))
    third_waiter = value.submit(request(
        "cooldown-third", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=40),),
    ), now_ns=30)
    assert third_waiter.kind is GlobalDecisionKind.QUEUE
    recovered = value.lease_queued_to_endpoint("cooldown-third", now_ns=31)
    assert recovered is not None and recovered.queue_lease is True


def test_tenant_ttft_and_e2e_slo_are_admission_constraints() -> None:
    value = controller(tenants=(TenantPolicy(
        "latency", 2.0, ttft_slo_ms=10.0, e2e_slo_ms=15.0),))
    value.update_telemetry(telemetry(0))
    result = value.submit(request(
        "slo-bound", "latency", (
            candidate(0, GlobalRoute.LOCAL, e2e=20),
        ),
    ), now_ns=10)
    assert result.kind is GlobalDecisionKind.QUEUE
    assert result.rejected_candidates[0].reason == "tenant_e2e_slo"


def test_queue_order_caps_external_deadline_by_tenant_e2e_slo() -> None:
    value = controller(maximum_active_pairs=1)
    seed(value)
    active = value.submit(request(
        "active", "batch",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=100),),
        deadline_ns=100_000_000_000,
    ), now_ns=10)
    assert active.kind is GlobalDecisionKind.ADMIT
    latency = value.submit(request(
        "z-latency", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=100),),
        deadline_ns=100_000_000_000,
    ), now_ns=11)
    batch = value.submit(request(
        "a-batch", "batch",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=100),),
        deadline_ns=100_000_000_000,
    ), now_ns=12)
    assert latency.kind is batch.kind is GlobalDecisionKind.QUEUE

    first = value.fail("active", now_ns=20)
    assert [item.request_id for item in first] == ["z-latency"]
    second = value.fail("z-latency", now_ns=21)
    assert [item.request_id for item in second] == ["a-batch"]


def test_pair_scale_down_waits_for_eof_and_idle_epoch() -> None:
    value = controller()
    seed(value)
    value.submit(request(
        "fill", "latency", (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=100),)
    ), now_ns=10)
    value.submit(request(
        "scaled", "batch", (candidate(1, GlobalRoute.LOCAL, e2e=10),)
    ), now_ns=11)
    assert value.reconcile_pairs(now_ns=1_000) == (0, 1)
    value.mark_first_response("scaled", now_ns=20)
    assert value.reconcile_pairs(now_ns=1_000) == (0, 1)
    value.complete("scaled", now_ns=30)
    assert value.reconcile_pairs(now_ns=129) == (0, 1)
    assert value.reconcile_pairs(now_ns=130) == (0,)


def test_fair_dispatch_prefers_tenant_with_less_weighted_service() -> None:
    value = controller(maximum_active_pairs=1)
    seed(value)
    value.submit(request(
        "active", "latency", (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=100),)
    ), now_ns=10)
    queued_a = value.submit(request(
        "queued-a", "latency", (candidate(0, GlobalRoute.LOCAL, e2e=10),)
    ), now_ns=11)
    queued_b = value.submit(request(
        "queued-b", "batch", (candidate(0, GlobalRoute.LOCAL, e2e=10),)
    ), now_ns=12)
    assert queued_a.kind is queued_b.kind is GlobalDecisionKind.QUEUE
    value.mark_first_response("active", now_ns=20)
    dispatched = value.complete("active", now_ns=21)
    assert [item.request_id for item in dispatched] == ["queued-b", "queued-a"]
    assert value.snapshot(now_ns=22)["phases"]["queued-b"] == "route_committed"


def test_route_is_immutable_and_fail_releases_all_held_resources() -> None:
    value = controller()
    seed(value)
    decision = value.submit(request(
        "r0", "latency", (candidate(0, GlobalRoute.REMOTE, e2e=10),)
    ), now_ns=10)
    assert decision.route is GlobalRoute.REMOTE
    with pytest.raises(ValueError, match="duplicate"):
        value.submit(request(
            "r0", "latency", (candidate(0, GlobalRoute.LOCAL, e2e=1),)
        ), now_ns=11)
    value.fail("r0", now_ns=12)
    assert not any(value.snapshot(now_ns=13)["owned_by_pair"]["0"].values())


def test_effective_use_does_not_double_count_owned_and_observed_total() -> None:
    value = controller()
    seed(value)
    value.submit(request(
        "r0", "latency", (candidate(0, GlobalRoute.LOCAL, e2e=10),)
    ), now_ns=10)
    value.update_telemetry(telemetry(
        0,
        sequence=2,
        sampled_ns=20,
        observed=ResourceVector(
            decode_tokens=40,
            active_sequences=1,
            local_prefill_token_ms=40,
        ),
    ))
    second = value.submit(request(
        "r1", "batch", (candidate(0, GlobalRoute.LOCAL, e2e=10),)
    ), now_ns=20)
    assert second.kind is GlobalDecisionKind.ADMIT
    assert second.resource_used_before["decode_tokens"] == 40


def test_mesh_cross_edge_conserves_source_receiver_and_decoder_credits() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=2,
        maximum_active_pairs=2,
    )
    seed(value)
    decision = value.submit(request(
        "mesh-p0-d1",
        "latency",
        (mesh_candidate(0, 1, e2e=20),),
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.pair_index == decision.decoder_index == 1
    assert decision.prefill_index == 0
    assert decision.edge_id == "remote:p0->d1"

    admitted = value.snapshot(now_ns=11)
    assert admitted["mesh_source_prefill_owned"]["0"] == 30
    assert admitted["mesh_edges"]["p0->d1"]["inflight_transfers"] == 1
    assert admitted["owned_by_pair"]["1"]["remote_prefill_token_ms"] == 0
    assert admitted["owned_by_pair"]["1"]["remote_kv_bytes"] == 400
    assert admitted["owned_by_pair"]["1"]["decode_tokens"] == 40

    value.mark_first_response("mesh-p0-d1", now_ns=20_000_010)
    first_response = value.snapshot(now_ns=20_000_011)
    assert first_response["mesh_source_prefill_owned"]["0"] == 0
    edge = first_response["mesh_edges"]["p0->d1"]
    assert edge["inflight_transfers"] == 0
    assert edge["completed_first_responses"] == 1
    assert edge["first_response_ewma_ms"] == pytest.approx(20.0)
    assert first_response["owned_by_pair"]["1"]["remote_kv_bytes"] == 0
    assert first_response["owned_by_pair"]["1"]["decode_tokens"] == 40

    value.complete("mesh-p0-d1", now_ns=21_000_010)
    terminal = value.snapshot(now_ns=21_000_011)
    assert not any(terminal["owned_by_pair"]["1"].values())
    assert terminal["mesh_source_prefill_owned"]["0"] == 0


def test_mesh_receiver_hotspot_selects_alternate_decoder() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=2,
        maximum_active_pairs=2,
    )
    value.update_telemetry(telemetry(
        0,
        observed=ResourceVector(
            remote_kv_bytes=1_000,
            remote_semantic_ops=2,
        ),
    ))
    value.update_telemetry(telemetry(1))
    decision = value.submit(request(
        "mesh-alternate-decoder",
        "latency",
        (
            mesh_candidate(0, 0, e2e=10),
            mesh_candidate(0, 1, e2e=30),
        ),
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.prefill_index == 0
    assert decision.decoder_index == 1
    assert decision.edge_id == "remote:p0->d1"
    assert any(
        item.edge_id == "remote:p0->d0"
        and item.reason in {"capacity", "remote_semantic_ops_admission_guard"}
        for item in decision.rejected_candidates
    )


def test_mesh_recovers_failure_free_stale_feedback_with_fresh_global_state() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=1,
        maximum_active_pairs=2,
    )
    value.update_telemetry(telemetry(
        0,
        local_health=PathHealth.SKIP,
        local_multiplier=50.0,
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    value.update_telemetry(telemetry(
        1,
        local_health=PathHealth.SKIP,
        local_multiplier=50.0,
        scheduler_running=1,
        scheduler_waiting=0,
        scheduler_kv=0.1,
        completion_residual=1,
        completion_completed=4,
    ))
    decision = value.submit(request(
        "mesh-stale-feedback-fallback",
        "latency",
        (
            candidate(0, GlobalRoute.LOCAL, e2e=10),
            candidate(1, GlobalRoute.LOCAL, e2e=10),
        ),
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.pair_index == 0
    assert decision.pair_activated is False
    assert decision.reason == (
        "global_mesh_stale_feedback_fallback_route_committed")
    assert decision.score_ms == pytest.approx(26.0)


def test_scheduler_occupancy_can_select_cool_remote_destination() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=1,
        maximum_active_pairs=2,
    )
    value.update_telemetry(telemetry(
        0, scheduler_running=0, scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0, completion_completed=4,
    ))
    value.update_telemetry(telemetry(
        1, scheduler_running=1, scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0, completion_completed=4,
    ))
    decision = value.submit(request(
        "mesh-cool-remote-destination",
        "latency",
        (
            candidate(1, GlobalRoute.LOCAL, e2e=10),
            mesh_candidate(0, 0, e2e=11),
        ),
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.route is GlobalRoute.REMOTE
    assert decision.prefill_index == 0
    assert decision.decoder_index == 0


def test_live_route_benefit_can_activate_cool_remote_destination() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=1,
        maximum_active_pairs=2,
    )
    value.update_telemetry(telemetry(
        0, scheduler_running=1, scheduler_waiting=0,
        scheduler_kv=0.5,
        completion_residual=1, completion_completed=4,
    ))
    value.update_telemetry(telemetry(
        1, scheduler_running=0, scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0, completion_completed=4,
    ))
    decision = value.submit(request(
        "mesh-route-benefit-activates-spare",
        "latency",
        (
            candidate(0, GlobalRoute.LOCAL, e2e=10),
            candidate(1, GlobalRoute.LOCAL, e2e=50),
            mesh_candidate(0, 0, e2e=50),
            mesh_candidate(1, 0, e2e=50),
            mesh_candidate(0, 1, e2e=10),
            mesh_candidate(1, 1, e2e=50),
        ),
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.route is GlobalRoute.REMOTE
    assert decision.prefill_index == 0
    assert decision.decoder_index == 1
    assert decision.pair_activated is True
    assert decision.reason == (
        "global_proactive_scale_route_benefit_and_route_committed")


def test_mesh_stale_feedback_fallback_never_overrides_explicit_failure() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=1,
        maximum_active_pairs=2,
    )
    value.update_telemetry(telemetry(
        0,
        local_health=PathHealth.SKIP,
        local_failure_count=1,
        local_last_failure_kind="active_upstream_failure",
        scheduler_running=0,
        scheduler_waiting=0,
        scheduler_kv=0.0,
        completion_residual=0,
        completion_completed=4,
    ))
    value.update_telemetry(telemetry(1))
    decision = value.submit(request(
        "mesh-stale-feedback-explicit-failure",
        "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.QUEUE
    assert [item.reason for item in decision.rejected_candidates] == [
        "path_skip"]


def test_mesh_stale_feedback_requires_fresh_scheduler_and_completion_state() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=1,
        maximum_active_pairs=2,
    )
    value.update_telemetry(telemetry(
        0,
        local_health=PathHealth.SKIP,
        completion_completed=4,
    ))
    value.update_telemetry(telemetry(1))
    decision = value.submit(request(
        "mesh-stale-feedback-no-global-state",
        "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10),),
    ), now_ns=10)
    assert decision.kind is GlobalDecisionKind.QUEUE
    assert [item.reason for item in decision.rejected_candidates] == [
        "path_skip"]


def test_mesh_source_credit_is_shared_across_destination_edges() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=2,
        maximum_active_pairs=2,
    )
    seed(value)
    first = value.submit(request(
        "mesh-source-holder",
        "latency",
        (mesh_candidate(0, 1, e2e=10, remote_prefill=80),),
    ), now_ns=10)
    assert first.kind is GlobalDecisionKind.ADMIT
    second = value.submit(request(
        "mesh-source-blocked",
        "batch",
        (mesh_candidate(0, 0, e2e=10, remote_prefill=30),),
        arrival_ns=11,
    ), now_ns=11)
    assert second.kind is GlobalDecisionKind.QUEUE
    assert any(
        item.reason == "mesh_source_prefill_credit"
        and item.edge_id == "remote:p0->d0"
        for item in second.rejected_candidates
    )
    dispatched = value.mark_first_response(
        "mesh-source-holder", now_ns=20)
    assert [item.request_id for item in dispatched] == [
        "mesh-source-blocked"]


def test_mesh_edge_failure_quarantines_only_edge_and_requires_two_end_probe() -> None:
    value = controller(
        mesh_control_mode="receiver_credit_pxd_v1",
        minimum_active_pairs=2,
        maximum_active_pairs=2,
        route_failure_quarantine_mode="deny_until_probe",
    )
    seed(value)
    admitted = value.submit(request(
        "mesh-edge-failed",
        "latency",
        (mesh_candidate(0, 1, e2e=10),),
    ), now_ns=10)
    assert admitted.kind is GlobalDecisionKind.ADMIT
    report = value.report_route_failure(
        "mesh-edge-failed",
        failure_kind="lmcache_edge_transport_error",
        scope="edge",
        now_ns=20,
    )
    assert report.receipt.edge_id == "remote:p0->d1"
    assert report.receipt.quarantined_edges == ((0, 1),)
    assert report.receipt.quarantined_routes == ()

    alternate = value.submit(request(
        "mesh-edge-alternate",
        "latency",
        (
            mesh_candidate(0, 1, e2e=5),
            mesh_candidate(1, 1, e2e=20),
        ),
        arrival_ns=21,
    ), now_ns=21)
    assert alternate.kind is GlobalDecisionKind.ADMIT
    assert alternate.edge_id == "remote:p1->d1"
    assert any(
        item.edge_id == "remote:p0->d1"
        and item.reason == "mesh_edge_failure_quarantine"
        for item in alternate.rejected_candidates
    )

    value.update_telemetry(telemetry(
        0,
        sequence=2,
        sampled_ns=22,
        remote_health=PathHealth.PROBE,
    ))
    still_quarantined = value.snapshot(now_ns=22)[
        "mesh_edge_failure_quarantines"]
    assert [item["edge_id"] for item in still_quarantined] == [
        "remote:p0->d1"]
    value.update_telemetry(telemetry(
        1,
        sequence=2,
        sampled_ns=22,
        remote_health=PathHealth.PROBE,
    ))
    assert value.snapshot(now_ns=22)["mesh_edge_failure_quarantines"] == []
