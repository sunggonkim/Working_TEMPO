from __future__ import annotations

from dataclasses import replace

from tempo.pd_global_orchestrator import (
    CrossLayerSignal,
    CrossLayerTelemetry,
    GlobalRoute,
    PathHealth,
    ResourceVector,
)
from tempo.test_pd_global_telemetry import adapter, frontend, live_endpoints
from tempo.test_pd_global_orchestrator import candidate, controller, request


def _envelope(pair_index: int = 0) -> dict[str, object]:
    supported = {
        "nccl_collective_p99_ms": (18.0, "milliseconds"),
        "nccl_arrival_spread_ms": (3.0, "milliseconds"),
        "lmcache_transfer_p99_ms": (40.0, "milliseconds"),
        "lmcache_remote_ops_inflight": (3, "operations"),
        "lmcache_remote_kv_bytes_inflight": (512 * 1024 * 1024, "bytes"),
        "cassini_rx_pause_fraction_max": (0.2, "fraction"),
        "cassini_tx_pause_fraction_max": (0.1, "fraction"),
        "cassini_host_posted_cycles_per_packet_max": (2.0, "cycles_per_packet"),
        "cassini_ecn_fraction_max": (0.05, "fraction"),
        "cassini_retries": (2, "events"),
        "cassini_timeouts": (1, "events"),
    }
    return {
        "schema": "tempo-go-cross-layer-envelope-v1",
        "pair_index": pair_index,
        "node_id": "nid00001",
        "endpoint_id": f"pair-{pair_index}",
        "communicator_id": "nccl-test-0",
        "source_epoch": "slurm-1234",
        "topology_fingerprint_sha256": "d" * 64,
        "sequence": 4,
        "sampled_ns": 100,
        "window_ms": 20.0,
        "cassini_by_nic": [
            [
                {
                    "traffic_class": traffic_class,
                    "rx_pause_fraction": 0.2 if traffic_class == 0 else 0.0,
                    "tx_pause_fraction": 0.1 if traffic_class == 0 else 0.0,
                }
                for traffic_class in range(8)
            ]
        ],
        "signals": [
            {
                "name": name,
                "value": value,
                "unit": unit,
                "support": "supported",
                "source": "test",
                "uncertainty": 0.0,
                "scope": "pair",
            }
            for name, (value, unit) in supported.items()
        ],
    }


def test_cross_layer_vector_is_parsed_and_serialized() -> None:
    raw = live_endpoints()
    raw[0]["cross_layer"] = _envelope()
    batch = adapter(require_scheduler=True).assemble(
        frontend(), raw, collection_started_ns=10_000, collection_finished_ns=10_200
    )
    telemetry = batch.pairs[0].cross_layer
    assert telemetry is not None
    assert telemetry.signal("cassini_rx_pause_fraction_max").value == 0.2
    local_cost, local_contributions, _ = telemetry.route_externality(
        GlobalRoute.LOCAL)
    remote_cost, remote_contributions, _ = telemetry.route_externality(
        GlobalRoute.REMOTE)
    assert remote_cost > local_cost
    assert not any(name.startswith("cassini_") for name in local_contributions)
    assert any(name.startswith("cassini_") for name in remote_contributions)
    encoded = batch.as_dict()["pairs"][0]["cross_layer"]
    assert encoded["signals"]
    assert encoded["derived_route_externality"][GlobalRoute.REMOTE.value][
        "confidence"
    ] == 1.0


def test_unsupported_signal_is_not_treated_as_zero() -> None:
    signal = CrossLayerSignal(
        name="nccl_collective_p99_ms",
        value=None,
        unit="milliseconds",
        support="not_collected",
        source="cuda_collective_observer_unavailable",
    )
    telemetry = CrossLayerTelemetry(
        pair_index=0,
        node_id="nid00001",
        endpoint_id="pair-0",
        communicator_id="nccl-unobserved",
        source_epoch="slurm-1234",
        topology_fingerprint_sha256="e" * 64,
        sequence=1,
        sampled_ns=0,
        window_ms=1.0,
        signals=(signal,),
    )
    assert telemetry.route_externality(GlobalRoute.REMOTE) == (0.0, {}, 0.0)


def test_cross_layer_state_changes_global_route_and_is_provenanced() -> None:
    raw = live_endpoints()
    raw[0]["cross_layer"] = _envelope()
    zero_resources = {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }
    raw[0]["controller"].update({
        "resources": dict(zero_resources),
        "owned_resources": dict(zero_resources),
        "external_resources": dict(zero_resources),
        "inflight": 0,
        "external_inflight": 0,
    })
    low_load_frontend = frontend()
    low_load_frontend.update({"loads": [0, 0], "active": 0, "active_by_pair": [0, 0]})
    batch = adapter(require_scheduler=True).assemble(
        low_load_frontend, raw,
        collection_started_ns=10_000,
        collection_finished_ns=10_200,
    )
    value = controller()
    value.update_telemetry_batch((
        batch.pairs[0],
        batch.pairs[1],
    ))
    decision = value.submit(request(
        "cross-layer-route",
        "latency",
        (
            candidate(0, GlobalRoute.LOCAL, e2e=100.0),
            candidate(0, GlobalRoute.REMOTE, e2e=60.0),
        ),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)
    assert decision.route is GlobalRoute.LOCAL
    assert decision.telemetry_provenance[0]["cross_layer"] is not None
    assert decision.telemetry_provenance[0]["cross_layer"][
        "derived_route_externality"][GlobalRoute.REMOTE.value]["confidence"] == 1.0
    assert decision.joint_actuation is not None
    assert decision.joint_actuation.pair_index == 0
    assert decision.joint_actuation.route is GlobalRoute.LOCAL
    assert decision.joint_actuation.dispatch_stagger_us > 0
    assert decision.joint_actuation.signal_contributions
    assert decision.joint_actuation.telemetry_sequence == batch.pairs[0].sequence
    assert decision.joint_actuation.telemetry_sequence != (
        batch.pairs[0].cross_layer.sequence
    )
    assert value.snapshot(now_ns=11_000)["telemetry_provenance"]["0"][
        "cross_layer"] is not None


def test_joint_limits_preserve_capacity_inside_safe_envelope() -> None:
    raw = live_endpoints()
    raw[0]["cross_layer"] = _envelope()
    zero_resources = {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }
    raw[0]["controller"].update({
        "resources": dict(zero_resources),
        "owned_resources": dict(zero_resources),
        "external_resources": dict(zero_resources),
        "inflight": 0,
        "external_inflight": 0,
    })
    low_load_frontend = frontend()
    low_load_frontend.update({"loads": [0, 0], "active": 0, "active_by_pair": [0, 0]})
    batch = adapter(require_scheduler=True).assemble(
        low_load_frontend, raw,
        collection_started_ns=10_000,
        collection_finished_ns=10_200,
    )
    value = controller()
    value.update_telemetry_batch((batch.pairs[0], batch.pairs[1]))
    decision = value.submit(request(
        "safe-envelope", "latency", (
            candidate(0, GlobalRoute.LOCAL, e2e=100.0),
            candidate(0, GlobalRoute.REMOTE, e2e=60.0),
        ),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)
    assert decision.joint_actuation is not None
    assert decision.joint_actuation.local_prefill_token_ms_limit == (
        value._capacities[0].local_prefill_token_ms
    )


def _pressure_batch(signal_name: str, value: float):
    raw = live_endpoints()
    envelope = _envelope()
    for signal in envelope["signals"]:
        if signal["name"] == signal_name:
            signal["value"] = value
    raw[0]["cross_layer"] = envelope
    zero_resources = {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }
    raw[0]["controller"].update({
        "resources": dict(zero_resources),
        "owned_resources": dict(zero_resources),
        "external_resources": dict(zero_resources),
        "inflight": 0,
        "external_inflight": 0,
    })
    low_load_frontend = frontend()
    low_load_frontend.update({
        "loads": [0, 0], "active": 0, "active_by_pair": [0, 0],
    })
    return adapter(require_scheduler=True).assemble(
        low_load_frontend, raw,
        collection_started_ns=10_000,
        collection_finished_ns=10_200,
    )


def _shared_batch(signal_name: str | None = None, value: float | None = None):
    raw = live_endpoints()
    for index in (0, 1):
        envelope = _envelope(index)
        if signal_name is not None:
            for signal in envelope["signals"]:
                if signal["name"] == signal_name:
                    signal["value"] = value
        raw[index]["cross_layer"] = envelope
        raw[index]["controller"].update({
            "resources": {
                "local_token_ms": 0,
                "remote_prefill_token_ms": 0,
                "remote_kv_bytes": 0,
                "remote_semantic_ops": 0,
            },
            "owned_resources": {
                "local_token_ms": 0,
                "remote_prefill_token_ms": 0,
                "remote_kv_bytes": 0,
                "remote_semantic_ops": 0,
            },
            "external_resources": {
                "local_token_ms": 0,
                "remote_prefill_token_ms": 0,
                "remote_kv_bytes": 0,
                "remote_semantic_ops": 0,
            },
            "inflight": 0,
            "external_inflight": 0,
        })
    low_load_frontend = frontend()
    low_load_frontend.update({
        "loads": [0, 0], "active": 0, "active_by_pair": [0, 0],
    })
    return adapter(require_scheduler=True).assemble(
        low_load_frontend, raw,
        collection_started_ns=10_000,
        collection_finished_ns=10_200,
    )


def test_v2_uses_shadow_price_and_commit_lease_for_noncritical_overage() -> None:
    batch = _pressure_batch("lmcache_transfer_p99_ms", 200.0)
    value = controller(
        cross_layer_control_mode="soft_shadow_price_v2",
        cross_layer_shadow_price_ms=1_000.0,
    )
    value.update_telemetry_batch((batch.pairs[0], batch.pairs[1]))
    decision = value.submit(request(
        "soft-overage", "latency",
        (candidate(0, GlobalRoute.REMOTE, e2e=100.0),),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)
    assert decision.kind.value == "admit"
    plan = decision.joint_actuation
    assert plan is not None
    assert plan.schema == "tempo-go-joint-actuation-v2"
    assert plan.action_mode == "soft_shadow_price_v2"
    assert plan.critical_guard is False
    assert "remote_prefill_token_ms" in plan.soft_overage_resources
    assert plan.overage_penalty_ms > 0.0
    assert plan.enforced_remote_prefill_token_ms_limit is not None
    assert plan.enforced_remote_prefill_token_ms_limit >= (
        decision.selected_work["remote_prefill_token_ms"]
    )
    assert not any(
        item.reason == "cross_layer_joint_actuation_limit"
        for item in decision.rejected_candidates
    )


def test_joint_commit_keeps_endpoint_limits_within_physical_window() -> None:
    """Global queue debt must not become an invalid endpoint limit."""
    batch = _pressure_batch("lmcache_transfer_p99_ms", 200.0)
    value = controller(
        cross_layer_control_mode="soft_shadow_price_v2",
        cross_layer_shadow_price_ms=1_000.0,
    )
    capacity = value._capacities[0]
    after = ResourceVector(
        decode_tokens=100,
        active_sequences=2,
        endpoint_requests=2,
        local_prefill_token_ms=capacity.local_prefill_token_ms + 40,
    )
    plan = value._joint_actuation_plan(
        candidate(0, GlobalRoute.LOCAL, e2e=100.0),
        telemetry=batch.pairs[0],
        capacity=capacity,
        after=after,
    )
    assert plan is not None
    assert plan.enforced_local_prefill_token_ms_limit == (
        capacity.local_prefill_token_ms
    )


def test_v2_retains_hard_guard_for_critical_cassini_pressure() -> None:
    batch = _pressure_batch("cassini_timeouts", 5.0)
    value = controller(
        cross_layer_control_mode="soft_shadow_price_v2",
        cross_layer_shadow_price_ms=1_000.0,
    )
    value.update_telemetry_batch((batch.pairs[0], batch.pairs[1]))
    decision = value.submit(request(
        "critical-overage", "latency",
        (candidate(0, GlobalRoute.REMOTE, e2e=100.0),),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)
    assert decision.kind.value == "queue"
    assert any(
        item.reason == "cross_layer_joint_actuation_limit"
        for item in decision.rejected_candidates
    )


def test_critical_cassini_pressure_preserves_local_fabric_escape_path() -> None:
    batch = _pressure_batch("cassini_timeouts", 5.0)
    value = controller(
        cross_layer_control_mode="soft_shadow_price_v2",
        cross_layer_shadow_price_ms=1_000.0,
    )
    value.update_telemetry_batch((batch.pairs[0], batch.pairs[1]))
    decision = value.submit(request(
        "cassini-local-escape", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=100.0),),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)

    assert decision.kind.value == "admit"
    assert decision.route is GlobalRoute.LOCAL
    assert decision.joint_actuation is not None
    assert decision.joint_actuation.critical_guard is False
    assert not any(
        name.startswith("cassini_")
        for name, _value in decision.joint_actuation.signal_contributions
    )


def test_soft_cross_layer_overage_opens_spare_pair_in_same_decision() -> None:
    """A hot remote envelope must be able to trigger global pair scaling.

    The request remains work-conserving under v2, but the over-target
    resource is not ignored: a cool prewarmed pair is admitted as a real
    candidate instead of leaving all work on the externally congested pair.
    """
    batch = _pressure_batch("lmcache_transfer_p99_ms", 200.0)
    value = controller(
        cross_layer_control_mode="soft_shadow_price_v2",
        cross_layer_shadow_price_ms=1_000.0,
    )
    cool_pair = replace(
        batch.pairs[1],
        observed_total=ResourceVector(),
    )
    value.update_telemetry_batch((batch.pairs[0], cool_pair))
    decision = value.submit(request(
        "scale-on-envelope",
        "latency",
        (
            candidate(0, GlobalRoute.REMOTE, e2e=10.0),
            candidate(1, GlobalRoute.REMOTE, e2e=12.0),
        ),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)
    assert decision.kind.value == "admit"
    assert decision.pair_index == 1
    assert decision.pair_activated is True
    assert decision.reason == (
        "global_pair_activated_and_route_committed"
    )


def test_v3_enforces_remote_budget_across_compatible_pairs() -> None:
    batch = _shared_batch()
    value = controller(
        shared_fabric_control_mode="global_budget_v3",
        shared_remote_requests_capacity=1,
        shared_remote_kv_bytes_capacity=2_000_000_000,
        shared_remote_semantic_ops_capacity=1,
        cross_layer_control_mode="soft_shadow_price_v2",
        cross_layer_shadow_price_ms=1_000.0,
    )
    value.update_telemetry_batch((batch.pairs[0], batch.pairs[1]))
    first = value.submit(request(
        "shared-first", "latency",
        (candidate(0, GlobalRoute.REMOTE, e2e=10.0),),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)
    assert first.kind.value == "admit"
    plan = first.joint_actuation
    assert plan is not None
    assert plan.schema == "tempo-go-joint-actuation-v3"
    assert plan.action_mode == "shared_budget_v3"
    assert plan.shared_remote_requests_limit == 1
    assert plan.shared_remote_requests_used_before == 0
    assert plan.shared_fabric_group is not None
    assert plan.shared_budget_action in {
        "global_remote_budget", "global_remote_stagger",
    }

    second = value.submit(request(
        "shared-second", "latency",
        (candidate(1, GlobalRoute.REMOTE, e2e=10.0),),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)
    assert second.kind.value == "queue"
    assert any(
        item.reason == "shared_remote_budget"
        and "shared_remote_requests" in item.binding_resources
        for item in second.rejected_candidates
    )
    snapshot = value.snapshot(now_ns=11_000)
    assert snapshot["shared_fabric_control_mode"] == "global_budget_v3"
    assert len(snapshot["shared_remote_budgets"]) == 1


def test_v3_soft_shadow_price_keeps_kv_and_ops_work_conserving() -> None:
    """Latency-only pressure must not become a remote hard-cap rejection."""

    batch = _shared_batch("lmcache_transfer_p99_ms", 200.0)
    value = controller(
        shared_fabric_control_mode="global_budget_v3",
        shared_remote_requests_capacity=32,
        shared_remote_kv_bytes_capacity=1,
        shared_remote_semantic_ops_capacity=1,
        cross_layer_control_mode="soft_shadow_price_v2",
        cross_layer_shadow_price_ms=1_000.0,
    )
    value.update_telemetry_batch((batch.pairs[0], batch.pairs[1]))
    first = value.submit(request(
        "soft-first", "latency",
        (candidate(0, GlobalRoute.REMOTE, e2e=10.0),),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)
    assert first.kind.value == "admit"

    second = value.submit(request(
        "soft-second", "latency",
        (candidate(1, GlobalRoute.REMOTE, e2e=10.0),),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)
    assert second.kind.value == "admit"
    assert second.joint_actuation is not None
    assert second.joint_actuation.shared_budget_action == (
        "global_remote_stagger"
    )


def test_v3_shared_pressure_does_not_activate_spare_pair() -> None:
    batch = _shared_batch("lmcache_transfer_p99_ms", 200.0)
    value = controller(
        shared_fabric_control_mode="global_budget_v3",
        cross_layer_control_mode="soft_shadow_price_v2",
        cross_layer_shadow_price_ms=1_000.0,
    )
    value.update_telemetry_batch((batch.pairs[0], batch.pairs[1]))
    decision = value.submit(request(
        "shared-no-scale", "latency",
        (
            candidate(0, GlobalRoute.REMOTE, e2e=10.0),
            candidate(1, GlobalRoute.REMOTE, e2e=11.0),
        ),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)
    assert decision.kind.value == "admit"
    assert decision.pair_index == 0
    assert decision.pair_activated is False
    assert decision.active_pairs_after == (0,)
    assert decision.joint_actuation is not None
    # Latency-only shared pressure preserves remote capacity in soft-shadow
    # mode; it requests a stagger, not a hard shared-budget shrink.
    assert decision.joint_actuation.shared_budget_action == (
        "global_remote_stagger"
    )


def test_v3_remote_stagger_does_not_delay_local_prefill() -> None:
    """Remote-fabric pressure must not sleep an unrelated local request."""
    batch = _shared_batch("lmcache_transfer_p99_ms", 200.0)
    pairs = []
    for pair in batch.pairs:
        cross_layer = pair.cross_layer
        assert cross_layer is not None
        signals = tuple(
            replace(signal, value=0.0)
            if signal.name != "lmcache_transfer_p99_ms"
            else signal
            for signal in cross_layer.signals
        )
        pairs.append(replace(
            pair,
            cross_layer=replace(
                cross_layer,
                signals=signals,
                cassini_by_nic=(),
            ),
        ))
    batch = replace(batch, pairs=tuple(pairs))
    value = controller(
        shared_fabric_control_mode="global_budget_v3",
        cross_layer_control_mode="soft_shadow_price_v2",
        cross_layer_shadow_price_ms=1_000.0,
    )
    value.update_telemetry_batch((batch.pairs[0], batch.pairs[1]))
    decision = value.submit(request(
        "shared-local-no-sleep", "latency",
        (candidate(0, GlobalRoute.LOCAL, e2e=10.0),),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)
    assert decision.kind.value == "admit"
    assert decision.joint_actuation is not None
    assert decision.joint_actuation.shared_budget_action == (
        "global_remote_stagger"
    )
    assert decision.joint_actuation.dispatch_stagger_us == 0

    remote_value = controller(
        shared_fabric_control_mode="global_budget_v3",
        cross_layer_control_mode="soft_shadow_price_v2",
        cross_layer_shadow_price_ms=1_000.0,
    )
    remote_value.update_telemetry_batch((batch.pairs[0], batch.pairs[1]))
    remote_decision = remote_value.submit(request(
        "shared-remote-stagger", "latency",
        (candidate(0, GlobalRoute.REMOTE, e2e=10.0),),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)
    assert remote_decision.kind.value == "admit"
    assert remote_decision.joint_actuation is not None
    assert remote_decision.joint_actuation.dispatch_stagger_us == 2_000


def test_v3_pair_local_failure_can_activate_spare_pair() -> None:
    batch = _shared_batch()
    failed_pair = replace(batch.pairs[0], remote_health=PathHealth.DENIED)
    value = controller(
        shared_fabric_control_mode="global_budget_v3",
        cross_layer_control_mode="soft_shadow_price_v2",
        cross_layer_shadow_price_ms=1_000.0,
    )
    value.update_telemetry_batch((failed_pair, batch.pairs[1]))
    decision = value.submit(request(
        "pair-local-failure", "latency",
        (
            candidate(0, GlobalRoute.REMOTE, e2e=10.0),
            candidate(1, GlobalRoute.REMOTE, e2e=11.0),
        ),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)
    assert decision.kind.value == "admit"
    assert decision.pair_index == 1
    assert decision.pair_activated is True


def test_v3_nic_vector_load_balances_cool_spare_pair() -> None:
    """A hot shared communicator may scale when NIC evidence differs."""

    batch = _shared_batch("lmcache_transfer_p99_ms", 200.0)
    hot_nic = (
        (0, 0.60, 0.10),
        (1, 0.0, 0.0),
    )
    cool_nic = (
        (0, 0.05, 0.02),
        (1, 0.0, 0.0),
    )
    hot = replace(
        batch.pairs[0],
        cross_layer=replace(
            batch.pairs[0].cross_layer,
            cassini_by_nic=(hot_nic,),
        ),
    )
    cool = replace(
        batch.pairs[1],
        cross_layer=replace(
            batch.pairs[1].cross_layer,
            cassini_by_nic=(cool_nic,),
        ),
    )
    value = controller(
        shared_fabric_control_mode="global_budget_v3",
        cross_layer_control_mode="soft_shadow_price_v2",
        cross_layer_shadow_price_ms=1_000.0,
    )
    value.update_telemetry_batch((hot, cool))
    decision = value.submit(request(
        "shared-nic-imbalance", "latency",
        (
            candidate(0, GlobalRoute.REMOTE, e2e=10.0),
            candidate(1, GlobalRoute.REMOTE, e2e=11.0),
        ),
        deadline_ns=2_000_000_000,
    ), now_ns=11_000)
    assert decision.kind.value == "admit"
    assert decision.pair_index == 1
    assert decision.pair_activated is True
    assert decision.reason == "global_pair_activated_and_route_committed"
    assert decision.joint_actuation is not None
    assert any(
        name == "cassini_by_nic_pause_fraction_max"
        for name, _value in decision.joint_actuation.signal_contributions
    )
    hot_cost = hot.cross_layer.route_externality(GlobalRoute.REMOTE)[0]
    cool_cost = cool.cross_layer.route_externality(GlobalRoute.REMOTE)[0]
    assert hot_cost > cool_cost
