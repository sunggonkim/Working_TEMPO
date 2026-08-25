from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from tempo.pd_global_agent import RequestTriggeredTelemetryAgent
from tempo.pd_global_coordinator import GlobalAdmissionCoordinator
from tempo.pd_global_hierarchy import HierarchicalCandidateReducer
from tempo.pd_global_orchestrator import (
    GlobalDecisionKind,
    GlobalRoute,
    TenantPolicy,
    global_decision_dict,
    global_decision_fingerprint,
)
from tempo.test_pd_global_agent import Clock
from tempo.test_pd_global_orchestrator import candidate, controller, request
from tempo.test_pd_global_telemetry import adapter, endpoint
from tempo.pd_global_telemetry import FRONTEND_LEDGER_SCHEMA


def coordinator(
    *, admission_wait_ns: int = 1_000_000_000, **controller_overrides
):
    clock = Clock()

    async def fetch_frontend():
        return {
            "schema": FRONTEND_LEDGER_SCHEMA,
            "loads": [0, 0],
            "active": 0,
            "active_by_pair": [0, 0],
        }

    def fetch_endpoint(pair_index: int):
        async def fetch():
            raw = endpoint(pair_index)
            zeros = {
                "local_token_ms": 0,
                "remote_prefill_token_ms": 0,
                "remote_kv_bytes": 0,
                "remote_semantic_ops": 0,
            }
            raw["controller"]["resources"] = dict(zeros)
            raw["controller"]["owned_resources"] = dict(zeros)
            raw["controller"]["external_resources"] = dict(zeros)
            raw["controller"]["inflight"] = 0
            raw["controller"]["external_inflight"] = 0
            raw["queued_requests"] = 0
            return raw

        return fetch

    agent = RequestTriggeredTelemetryAgent(
        adapter(),
        frontend_fetcher=fetch_frontend,
        endpoint_fetchers=(fetch_endpoint(0), fetch_endpoint(1)),
        freshness_ns=1_000_000_000,
        refresh_timeout_ns=100_000_000,
        clock_ns=clock,
    )
    result = GlobalAdmissionCoordinator(
        controller(maximum_active_pairs=1, **controller_overrides),
        agent,
        admission_wait_ns=admission_wait_ns,
        clock_ns=clock,
    )
    result.clock = clock
    return result


def global_request(request_id: str, *, decode: int = 100):
    return request(
        request_id,
        "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10, decode=decode),),
    )


def test_decode_credit_is_held_until_eof_before_queue_dispatch() -> None:
    async def scenario() -> None:
        value = coordinator()
        first = await value.admit(global_request("first"))
        assert first.kind is GlobalDecisionKind.ADMIT
        waiting = asyncio.create_task(value.admit(global_request("waiting")))
        await asyncio.sleep(0)
        assert not waiting.done()
        first_response_dispatch = await value.mark_first_response("first")
        assert first_response_dispatch == ()
        await asyncio.sleep(0)
        assert not waiting.done()
        eof_dispatch = await value.complete("first")
        assert [item.request_id for item in eof_dispatch] == ["waiting"]
        second = await waiting
        assert second.request_id == "waiting"
        assert second.route is GlobalRoute.LOCAL
        await value.mark_first_response("waiting")
        await value.complete("waiting")
        state = value.status()
        assert state["waiters"] == 0
        assert state["delivered_from_queue"] == 1
        assert state["queue_timeouts"] == 0

    asyncio.run(scenario())


def test_failure_before_first_response_releases_all_global_credit() -> None:
    async def scenario() -> None:
        value = coordinator()
        await value.admit(global_request("failed"))
        waiting = asyncio.create_task(value.admit(global_request("after-fail")))
        await asyncio.sleep(0)
        dispatched = await value.fail("failed")
        assert [item.request_id for item in dispatched] == ["after-fail"]
        admitted = await waiting
        assert admitted.kind is GlobalDecisionKind.ADMIT

    asyncio.run(scenario())


def test_telemetry_fetch_does_not_block_completion_lifecycle() -> None:
    async def scenario() -> None:
        value = coordinator()
        first = await value.admit(global_request("nonblocking-first"))
        assert first.kind is GlobalDecisionKind.ADMIT
        await value.mark_first_response(first.request_id)

        original_fetch = value.telemetry_agent.frontend_fetcher
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked_fetch():
            started.set()
            await release.wait()
            return await original_fetch()

        value.telemetry_agent.frontend_fetcher = blocked_fetch
        value.telemetry_agent.freshness_ns = 1
        value.telemetry_agent._per_fetch_timeout_ns = 1_000_000_000
        pending = asyncio.create_task(
            value.admit(global_request("nonblocking-second")))
        await asyncio.wait_for(started.wait(), 0.1)
        await asyncio.wait_for(value.complete(first.request_id), 0.1)
        release.set()
        second = await asyncio.wait_for(pending, 0.1)
        assert second.kind is GlobalDecisionKind.ADMIT
        await value.mark_first_response(second.request_id)
        await value.complete(second.request_id)

    asyncio.run(scenario())


def test_queue_timeout_is_terminal_and_never_forwards() -> None:
    async def scenario() -> None:
        value = coordinator(admission_wait_ns=1_000_000)
        await value.admit(global_request("holder"))
        decision = await value.admit(global_request("timeout"))
        assert decision.kind is GlobalDecisionKind.REJECT
        assert decision.reason == "global_admission_queue_timeout"
        state = value.orchestrator.snapshot(now_ns=value.clock_ns())
        assert state["phases"]["timeout"] == "rejected"
        assert value.status()["queue_timeouts"] == 1
        await value.fail("holder")

    asyncio.run(scenario())


def test_endpoint_queue_lease_timeout_is_a_forwarded_global_admission() -> None:
    async def scenario() -> None:
        value = coordinator(
            admission_wait_ns=1_000_000,
            overload_action="endpoint_queue_lease",
            tenants=(
                TenantPolicy(
                    "latency", 2.0, queue_lease_on_timeout=True),
                TenantPolicy("batch", 1.0),
            ),
        )
        holder = await value.admit(global_request("lease-holder", decode=100))
        assert holder.kind is GlobalDecisionKind.ADMIT
        leased = await value.admit(global_request("lease-waiter", decode=40))
        assert leased.kind is GlobalDecisionKind.ADMIT
        assert leased.queue_lease is True
        assert leased.reason == "global_endpoint_queue_lease_route_committed"
        assert value.status()["queue_leases"] == 1
        await value.mark_first_response("lease-holder")
        await value.complete("lease-holder")
        await value.mark_first_response("lease-waiter")
        await value.complete("lease-waiter")

    asyncio.run(scenario())


def test_headroom_first_queue_lease_enters_native_lane_without_global_wait() -> None:
    async def scenario() -> None:
        value = coordinator(
            admission_wait_ns=10_000_000_000,
            overload_action="endpoint_queue_lease",
            endpoint_queue_admission_mode="headroom_first_v1",
            tenants=(
                TenantPolicy(
                    "latency", 2.0, queue_lease_on_timeout=True),
                TenantPolicy("batch", 1.0),
            ),
        )
        holder = await value.admit(global_request("headroom-holder", decode=100))
        assert holder.kind is GlobalDecisionKind.ADMIT
        leased = await value.admit(global_request("headroom-waiter", decode=40))
        assert leased.kind is GlobalDecisionKind.ADMIT
        assert leased.queue_lease is True
        assert value.status()["queue_leases"] == 1
        await value.mark_first_response("headroom-holder")
        await value.complete("headroom-holder")
        await value.mark_first_response("headroom-waiter")
        await value.complete("headroom-waiter")

    asyncio.run(scenario())


def test_queue_timeout_refreshes_stale_telemetry_before_endpoint_lease() -> None:
    async def scenario() -> None:
        value = coordinator(
            admission_wait_ns=1_000_000,
            overload_action="endpoint_queue_lease",
            tenants=(
                TenantPolicy(
                    "latency", 2.0, queue_lease_on_timeout=True),
                TenantPolicy("batch", 1.0),
            ),
        )
        # Force the queue boundary to cross the telemetry freshness window.
        # The timeout path must obtain one fresh batch before deciding whether
        # a native endpoint queue lease is safe.
        value.telemetry_agent.freshness_ns = 1
        holder = await value.admit(global_request("refresh-holder"))
        assert holder.kind is GlobalDecisionKind.ADMIT
        leased = await value.admit(global_request("refresh-waiter", decode=40))
        assert leased.kind is GlobalDecisionKind.ADMIT
        assert leased.queue_lease is True
        assert value.status()["queue_leases"] == 1
        assert value.telemetry_agent.status()["refreshes"] >= 2
        await value.mark_first_response("refresh-holder")
        await value.complete("refresh-holder")
        await value.mark_first_response("refresh-waiter")
        await value.complete("refresh-waiter")

    asyncio.run(scenario())


def test_queue_timeout_forces_a_new_telemetry_batch_at_lease_boundary() -> None:
    async def scenario() -> None:
        value = coordinator(
            admission_wait_ns=1_000_000,
            overload_action="endpoint_queue_lease",
            tenants=(
                TenantPolicy(
                    "latency", 2.0, queue_lease_on_timeout=True),
                TenantPolicy("batch", 1.0),
            ),
        )
        force_values: list[bool] = []
        original_get = value.telemetry_agent.get

        async def recording_get(*, force: bool = False):
            force_values.append(force)
            return await original_get(force=force)

        value.telemetry_agent.get = recording_get
        holder = await value.admit(global_request("force-holder"))
        assert holder.kind is GlobalDecisionKind.ADMIT
        leased = await value.admit(global_request("force-waiter", decode=40))
        assert leased.kind is GlobalDecisionKind.ADMIT
        assert leased.queue_lease is True
        assert True in force_values
        await value.mark_first_response("force-holder")
        await value.complete("force-holder")
        await value.mark_first_response("force-waiter")
        await value.complete("force-waiter")

    asyncio.run(scenario())


def test_refresh_timeout_uses_bounded_last_snapshot_grace() -> None:
    async def scenario() -> None:
        value = coordinator(telemetry_stale_grace_ns=1_000_000_000)
        first = await value.admit(global_request("stale-fallback-first"))
        assert first.kind is GlobalDecisionKind.ADMIT
        await value.mark_first_response(first.request_id)
        await value.complete(first.request_id)

        async def fail_get(*, force: bool = False):
            raise RuntimeError("global telemetry refresh timed out")

        value.telemetry_agent.get = fail_get
        second = await value.admit(global_request("stale-fallback-second"))
        assert second.kind is GlobalDecisionKind.ADMIT
        assert value.status()["stale_snapshot_fallbacks"] == 1
        await value.mark_first_response(second.request_id)
        await value.complete(second.request_id)

    asyncio.run(scenario())


def test_interactive_tenant_can_use_business_scoped_stale_grace() -> None:
    async def scenario() -> None:
        value = coordinator(
            tenants=(
                TenantPolicy(
                    "interactive", 2.0,
                    telemetry_stale_grace_ns=3_000_000_000),
                TenantPolicy("batch", 1.0),
            ),
        )
        interactive = request(
            "interactive-stale", "interactive",
            (candidate(0, GlobalRoute.LOCAL, e2e=10),),
            deadline_ns=10_000_000_000,
        )
        first = await value.admit(interactive)
        assert first.kind is GlobalDecisionKind.ADMIT
        await value.mark_first_response(first.request_id)
        await value.complete(first.request_id)
        value.clock.advance(2_000_000_000)

        async def fail_get(*, force: bool = False):
            raise RuntimeError("global telemetry validation failed")

        value.telemetry_agent.get = fail_get
        second = await value.admit(request(
            "interactive-stale-second", "interactive",
            (candidate(0, GlobalRoute.LOCAL, e2e=10),),
            deadline_ns=10_000_000_000,
        ))
        assert second.kind is GlobalDecisionKind.ADMIT
        assert value.status()["stale_snapshot_fallbacks"] == 1
        await value.mark_first_response(second.request_id)
        await value.complete(second.request_id)
        batch = await value.admit(request(
            "batch-stale", "batch",
            (candidate(0, GlobalRoute.LOCAL, e2e=10),),
            deadline_ns=10_000_000_000,
        ))
        assert batch.kind is GlobalDecisionKind.REJECT
        assert batch.reason == "global_telemetry_validation_failed"

    asyncio.run(scenario())


def test_validation_failure_gets_one_fresh_foreground_retry() -> None:
    async def scenario() -> None:
        value = coordinator()
        original_get = value.telemetry_agent.get
        attempts = 0

        async def fail_validation_once(*, force: bool = False):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("global telemetry validation failed")
            return await original_get(force=force)

        value.telemetry_agent.get = fail_validation_once
        decision = await value.admit(global_request("validation-retry"))
        assert decision.kind is GlobalDecisionKind.ADMIT
        assert attempts == 2
        assert value.status()["telemetry_validation_retries"] == 1
        assert value.status()["telemetry_validation_retry_failures"] == 0
        assert value.status()["telemetry_rejections"] == 0
        await value.mark_first_response(decision.request_id)
        await value.complete(decision.request_id)

    asyncio.run(scenario())


def test_validation_retry_still_fails_closed_after_second_failure() -> None:
    async def scenario() -> None:
        value = coordinator()
        attempts = 0

        async def always_invalid(*, force: bool = False):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("global telemetry validation failed")

        value.telemetry_agent.get = always_invalid
        decision = await value.admit(global_request("validation-retry-fails"))
        assert decision.kind is GlobalDecisionKind.REJECT
        assert decision.reason == "global_telemetry_validation_failed"
        assert attempts == 2
        assert value.status()["telemetry_validation_retries"] == 1
        assert value.status()["telemetry_validation_retry_failures"] == 1
        assert value.status()["telemetry_rejections"] == 1

    asyncio.run(scenario())


def test_prepared_telemetry_is_consumed_without_a_second_collection() -> None:
    async def scenario() -> None:
        value = coordinator()
        preparation = await value.prepare_admission()
        receipt = preparation.as_dict()
        assert receipt["schema"] == "tempo-go-admission-preparation-v1"
        assert receipt["status"] == "batch"
        assert receipt["batch_sequence"] == 1
        assert receipt["attempts_used"] == 1
        assert receipt["retry_triggered"] is False
        assert receipt["collection_elapsed_ns"] >= 0
        requests_before = value.telemetry_agent.status()["requests"]
        decision = await value.admit(
            global_request("prepared-admission"),
            preparation=preparation,
        )
        assert decision.kind is GlobalDecisionKind.ADMIT
        assert value.telemetry_agent.status()["requests"] == requests_before
        state = value.status()
        assert state["telemetry_preparations"] == 1
        assert state["prepared_admission_uses"] == 1
        await value.mark_first_response(decision.request_id)
        await value.complete(decision.request_id)

    asyncio.run(scenario())


def test_prepared_telemetry_never_reinstalls_an_older_sequence() -> None:
    async def scenario() -> None:
        value = coordinator()
        older = await value.prepare_admission()
        first = await value.admit(
            global_request("prepared-first"), preparation=older)
        assert first.kind is GlobalDecisionKind.ADMIT
        await value.mark_first_response(first.request_id)
        await value.complete(first.request_id)

        value.telemetry_agent.freshness_ns = 1
        newer_request = global_request("prepared-newer")
        newer = await value.prepare_admission()
        second = await value.admit(newer_request, preparation=newer)
        assert second.kind is GlobalDecisionKind.ADMIT
        await value.mark_first_response(second.request_id)
        await value.complete(second.request_id)

        reused = await value.admit(
            global_request("prepared-superseded"), preparation=older)
        assert reused.kind is GlobalDecisionKind.ADMIT
        assert value.status()["prepared_telemetry_superseded"] == 1
        await value.mark_first_response(reused.request_id)
        await value.complete(reused.request_id)

    asyncio.run(scenario())


def test_service_lane_reservation_failure_is_not_route_quarantine() -> None:
    async def scenario() -> None:
        value = coordinator(
            admission_wait_ns=1_000_000,
            overload_action="endpoint_queue_lease",
            tenants=(
                TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
                TenantPolicy("batch", 1.0),
            ),
        )
        await value.admit(global_request("reservation-holder", decode=100))
        leased = await value.admit(global_request("reservation-waiter", decode=40))
        assert leased.queue_lease is True
        report = await value.fail_service_lane_reservation(
            "reservation-waiter",
            failure_kind="endpoint_service_lane_reservation_unavailable",
            reason="endpoint_service_lane_capacity_unavailable",
        )
        assert report.receipt.schema == "tempo-go-service-lane-reservation-v1"
        state = value.orchestrator.snapshot(now_ns=value.clock_ns())
        assert state["phases"]["reservation-waiter"] == "failed"
        assert not state["route_failure_quarantines"]
        assert value.status()["service_lane_reservation_failures"] == 1
        await value.mark_first_response("reservation-holder")
        await value.complete("reservation-holder")

    asyncio.run(scenario())


def test_service_lane_queue_promotion_is_serialized_by_coordinator() -> None:
    async def scenario() -> None:
        value = coordinator(
            overload_action="endpoint_queue_lease",
            endpoint_queue_debt_mode=(
                "completion_credit_endpoint_queue_v3"),
            endpoint_queue_capacity=16,
            tenants=(
                TenantPolicy("latency", 2.0, queue_lease_on_timeout=True),
                TenantPolicy("batch", 1.0),
            ),
        )
        admitted = await value.admit(global_request("service-lane-promote"))
        assert admitted.queue_lease is False
        current = value.orchestrator._telemetry[0]
        value.orchestrator.update_telemetry(replace(
            current,
            sequence=current.sequence + 1,
            sampled_ns=current.sampled_ns + 1,
            collected_ns=current.collected_ns + 1,
            scheduler_running_requests=0,
            scheduler_waiting_requests=0,
            scheduler_kv_cache_usage_fraction=0.0,
            scheduler_schema="tempo-go-vllm-scheduler-snapshot-v1",
            scheduler_source=(
                "router_local_vllm_prometheus_observe_only"),
            endpoint_completed_first_responses=10,
            endpoint_residual_inflight=0,
            completion_schema="tempo-go-endpoint-completion-v1",
        ))
        report = await value.promote_service_lane_queue_lease(
            "service-lane-promote")
        assert report.decision is not None
        assert report.decision.queue_lease is True
        assert report.receipt.status == "promoted"
        status = value.status()
        assert status["queue_leases"] == 1
        assert status["service_lane_queue_promotions"] == 1
        assert status["service_lane_queue_promotion_rejections"] == 0
        await value.mark_first_response("service-lane-promote")
        await value.complete("service-lane-promote")

    asyncio.run(scenario())


def test_telemetry_refresh_timeout_is_an_explicit_terminal_reject() -> None:
    async def scenario() -> None:
        value = coordinator()

        async def blocked():
            await asyncio.Event().wait()

        value.telemetry_agent.frontend_fetcher = blocked
        value.telemetry_agent.refresh_timeout_ns = 1_000_000
        decision = await value.admit(global_request("telemetry-timeout"))
        assert decision.kind is GlobalDecisionKind.REJECT
        assert decision.reason == "global_telemetry_refresh_timeout"
        assert value.orchestrator.snapshot(now_ns=value.clock_ns())["phases"][
            "telemetry-timeout"] == "rejected"
        assert value.status()["telemetry_rejections"] == 1
        assert global_decision_dict(decision)["kind"] == "reject"

    asyncio.run(scenario())


def test_decision_digest_covers_work_rejections_and_telemetry() -> None:
    async def scenario() -> None:
        value = coordinator()
        decision = await value.admit(global_request("digest", decode=40))
        payload = global_decision_dict(decision)
        digest = global_decision_fingerprint(decision)
        assert len(digest) == 64
        assert payload["selected_work"]["endpoint_requests"] == 1
        assert payload["predicted_e2e_ms"] == 10
        assert payload["telemetry_sequences"] == {"0": 1, "1": 1}
        assert payload["telemetry_provenance"]["0"][
            "profile_fingerprint_sha256"] == "a" * 64
        assert digest == global_decision_fingerprint(decision)

    asyncio.run(scenario())


def test_hierarchical_receipt_is_consumable_after_global_admission() -> None:
    async def scenario() -> None:
        value = coordinator()
        value.hierarchical_reducer = HierarchicalCandidateReducer(
            shard_count=2,
            max_pairs_per_shard=1,
            max_routes_per_pair=2,
            telemetry_fresh_ns=1_000_000_000,
        )
        decision = await value.admit(global_request("hierarchical"))
        assert decision.kind is GlobalDecisionKind.ADMIT
        receipt = value.take_hierarchy_receipt("hierarchical")
        assert receipt is not None
        assert receipt["receipt"]["schema"] == "tempo-go-reduction-receipt-v1"
        assert receipt["receipt"]["forwarded_candidate_count"] == 1
        assert value.take_hierarchy_receipt("hierarchical") is None
        await value.mark_first_response("hierarchical")
        await value.complete("hierarchical")

    asyncio.run(scenario())
