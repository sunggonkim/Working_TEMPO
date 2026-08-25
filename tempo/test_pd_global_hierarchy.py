from __future__ import annotations

from dataclasses import replace

import pytest

from tempo.pd_global_hierarchy import (
    HierarchyCandidateUnavailableError,
    HierarchyIdentityError,
    HierarchicalCandidateReducer,
    HierarchicalRequestHeader,
    submit_hierarchical,
)
from tempo.pd_global_orchestrator import (
    CrossLayerSignal,
    CrossLayerTelemetry,
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
    TenantPolicy,
)


PROFILE = "a" * 64
CAPACITY = ResourceVector(
    decode_tokens=100,
    active_sequences=2,
    endpoint_requests=2,
    local_prefill_token_ms=100,
    remote_prefill_token_ms=100,
    remote_kv_bytes=10_000,
    remote_semantic_ops=10,
)


def candidate(pair: int, route: GlobalRoute, e2e: float) -> RouteCandidate:
    if route is GlobalRoute.LOCAL:
        selected_work = ResourceVector(
            decode_tokens=20,
            active_sequences=1,
            endpoint_requests=1,
            local_prefill_token_ms=20,
        )
    else:
        selected_work = ResourceVector(
            decode_tokens=20,
            active_sequences=1,
            endpoint_requests=1,
            remote_prefill_token_ms=20,
            remote_kv_bytes=100,
            remote_semantic_ops=1,
        )
    return RouteCandidate(
        pair_index=pair,
        route=route,
        work=selected_work,
        predicted_e2e_ms=e2e,
        predicted_ttft_ms=e2e / 2,
        uncertainty_ms=1.0,
    )


def request(pair_count: int = 8) -> GlobalRequest:
    return GlobalRequest(
        request_id="hierarchical-request",
        tenant_id="latency",
        arrival_ns=10,
        deadline_ns=1_000_000_000,
        candidates=tuple(
            candidate(pair, route, 10.0 + pair + (0.5 if route is GlobalRoute.REMOTE else 0.0))
            for pair in range(pair_count)
            for route in (GlobalRoute.LOCAL, GlobalRoute.REMOTE)
        ),
    )


def telemetry(pair: int, *, sequence: int = 1, sampled_ns: int = 10) -> PairTelemetry:
    return PairTelemetry(
        pair_index=pair,
        sequence=sequence,
        sampled_ns=sampled_ns,
        collected_ns=sampled_ns + 1,
        agent_epoch="allocation-epoch",
        profile_fingerprint_sha256=PROFILE,
        controller_generation=0,
        observed_total=ResourceVector(),
    )


def test_reducer_preserves_identity_and_bounds_shard_fan_in() -> None:
    value = HierarchicalCandidateReducer(
        shard_count=2,
        max_pairs_per_shard=2,
        max_routes_per_pair=2,
        telemetry_fresh_ns=100,
    ).reduce(
        request(),
        telemetry=tuple(telemetry(pair) for pair in range(8)),
        now_ns=10,
    )

    assert value.receipt.raw_pair_count == 8
    assert value.receipt.raw_candidate_count == 16
    assert value.receipt.forwarded_candidate_count == 8
    assert value.receipt.omitted_pair_count == 4
    assert value.receipt.identity_mode == "endpoint_profile_only"
    assert all(len(item.candidates) <= 4 for item in value.shards)
    assert all(item.forwarded_candidate_count == 2 * len(item.forwarded_pair_indices)
               for item in value.shards)
    assert len(value.nodes) == 8
    assert len(value.fingerprint) == 64


def test_unbounded_configuration_is_candidate_exact() -> None:
    original = request(8)
    reduced = HierarchicalCandidateReducer(
        shard_count=2,
        max_pairs_per_shard=8,
        max_routes_per_pair=2,
    ).reduce(
        original,
        telemetry=tuple(telemetry(pair) for pair in range(8)),
        now_ns=10,
    )
    assert reduced.receipt.omitted_pair_count == 0
    assert {
        (item.pair_index, item.route) for item in reduced.request.candidates
    } == {
        (item.pair_index, item.route) for item in original.candidates
    }
    assert reduced.receipt.forwarded_candidate_fingerprint != "0" * 64


def test_1024_pair_population_has_bounded_global_fan_in() -> None:
    original = request(1024)
    reduced = HierarchicalCandidateReducer(
        shard_count=64,
        max_pairs_per_shard=2,
        max_routes_per_pair=2,
    ).reduce(
        original,
        telemetry=tuple(telemetry(pair) for pair in range(1024)),
        now_ns=10,
    )
    assert reduced.receipt.raw_pair_count == 1024
    assert reduced.receipt.raw_candidate_count == 2048
    assert reduced.receipt.forwarded_candidate_count == 256
    assert len(reduced.request.candidates) <= 64 * 2 * 2
    assert reduced.receipt.omitted_pair_count == 896


def test_pair_agent_frontiers_reduce_without_raw_population_at_global() -> None:
    original = request(8)
    observations = tuple(telemetry(pair) for pair in range(8))
    reducer = HierarchicalCandidateReducer(
        shard_count=2,
        max_pairs_per_shard=2,
        max_routes_per_pair=2,
    )
    frontiers = tuple(
        reducer.build_pair_frontier(
            pair_index=pair,
            candidates=tuple(
                item for item in original.candidates if item.pair_index == pair
            ),
            telemetry=observations[pair],
        )
        for pair in range(8)
    )
    reduced = reducer.reduce_frontiers(
        HierarchicalRequestHeader.from_request(original),
        frontiers=frontiers,
        telemetry=observations,
        now_ns=10,
    )

    assert reduced.receipt.raw_pair_count == 8
    assert reduced.receipt.raw_candidate_count == 16
    assert reduced.receipt.forwarded_candidate_count == 8
    assert reduced.receipt.omitted_pair_count == 4
    assert reduced.receipt.identity_mode == "endpoint_profile_frontier"
    assert len(reduced.request.candidates) == 8
    assert all(item.raw_candidate_count == 2 for item in reduced.pairs)
    assert all(
        item.raw_candidate_count == 8 for item in reduced.shards if item.pair_indices
    )


def test_pair_frontier_identity_mismatch_fails_closed() -> None:
    original = request(2)
    observations = tuple(telemetry(pair) for pair in range(2))
    reducer = HierarchicalCandidateReducer(
        shard_count=1,
        max_pairs_per_shard=2,
    )
    frontiers = [
        reducer.build_pair_frontier(
            pair_index=pair,
            candidates=tuple(
                item for item in original.candidates if item.pair_index == pair
            ),
            telemetry=observations[pair],
        )
        for pair in range(2)
    ]
    frontiers[1] = replace(frontiers[1], sequence=2)
    with pytest.raises(HierarchyIdentityError, match="frontier identity differs"):
        reducer.reduce_frontiers(
            HierarchicalRequestHeader.from_request(original),
            frontiers=frontiers,
            telemetry=observations,
            now_ns=10,
        )


def test_global_consumes_shard_frontiers_and_preserves_omission_receipt() -> None:
    original = request(8)
    observations = tuple(telemetry(pair) for pair in range(8))
    reducer = HierarchicalCandidateReducer(
        shard_count=2,
        max_pairs_per_shard=2,
        max_routes_per_pair=2,
    )
    shard_stage = reducer.reduce(
        original,
        telemetry=observations,
        now_ns=10,
    )
    global_stage = reducer.reduce_shard_frontiers(
        HierarchicalRequestHeader.from_request(original),
        shards=shard_stage.shards,
        pairs=tuple(
            item for item in shard_stage.pairs
            if item.forwarded_candidate_count > 0
        ),
        telemetry=observations,
        now_ns=10,
    )

    assert global_stage.receipt.raw_pair_count == 8
    assert global_stage.receipt.raw_candidate_count == 16
    assert global_stage.receipt.forwarded_candidate_count == 8
    assert global_stage.receipt.omitted_pair_count == 4
    assert global_stage.receipt.identity_mode == "cross_layer_shard_frontier"
    assert len(global_stage.request.candidates) == 8
    assert len(global_stage.pairs) == 4
    assert len(global_stage.shards) == 2


def test_mixed_epoch_fails_closed_before_global_submission() -> None:
    observations = [telemetry(pair) for pair in range(8)]
    observations[-1] = telemetry(7, sequence=2)
    reducer = HierarchicalCandidateReducer(
        shard_count=2,
        max_pairs_per_shard=2,
    )
    with pytest.raises(HierarchyIdentityError, match="mixed hierarchy identity"):
        reducer.reduce(request(), telemetry=observations, now_ns=10)


def test_cross_layer_producer_sequences_may_differ_inside_one_atomic_batch() -> None:
    observations = [telemetry(pair) for pair in range(2)]
    signal = CrossLayerSignal(
        name="nccl_collective_p99_ms",
        value=1.0,
        unit="milliseconds",
        support="supported",
        source="test",
    )
    observations[0] = replace(
        observations[0],
        sequence=9,
        cross_layer=CrossLayerTelemetry(
            pair_index=0,
            node_id="nid00001",
            endpoint_id="pair-0",
            communicator_id="nccl-0",
            source_epoch="allocation-epoch",
            topology_fingerprint_sha256="b" * 64,
            sequence=101,
            sampled_ns=10,
            window_ms=1.0,
            signals=(signal,),
        ),
    )
    observations[1] = replace(
        observations[1],
        sequence=9,
        cross_layer=CrossLayerTelemetry(
            pair_index=1,
            node_id="nid00002",
            endpoint_id="pair-1",
            communicator_id="nccl-0",
            source_epoch="allocation-epoch",
            topology_fingerprint_sha256="b" * 64,
            sequence=202,
            sampled_ns=10,
            window_ms=1.0,
            signals=(signal,),
        ),
    )
    reduced = HierarchicalCandidateReducer(
        shard_count=1, max_pairs_per_shard=2, max_routes_per_pair=2,
    ).reduce(request(2), telemetry=observations, now_ns=10)
    assert reduced.receipt.identity_mode == "cross_layer"
    assert reduced.receipt.forwarded_candidate_count == 4


def test_quarantined_pair_is_omitted_without_rejecting_healthy_cross_layer_pair() -> None:
    observations = [telemetry(pair, sequence=9) for pair in range(2)]
    observations[0] = replace(
        observations[0],
        local_health=PathHealth.DENIED,
        remote_health=PathHealth.DENIED,
        quarantine_reason="endpoint_fetch:TimeoutError",
    )
    signal = CrossLayerSignal(
        name="cassini_rx_pause_fraction_max",
        value=0.0,
        unit="fraction",
        support="supported",
        source="test",
    )
    observations[1] = replace(
        observations[1],
        cross_layer=CrossLayerTelemetry(
            pair_index=1,
            node_id="nid00002",
            endpoint_id="pair-1",
            communicator_id="nccl-0",
            source_epoch="allocation-epoch",
            topology_fingerprint_sha256="b" * 64,
            sequence=202,
            sampled_ns=10,
            window_ms=1.0,
            signals=(signal,),
        ),
    )
    reduced = HierarchicalCandidateReducer(
        shard_count=2, max_pairs_per_shard=1, max_routes_per_pair=2,
    ).reduce(request(2), telemetry=observations, now_ns=10)
    assert reduced.receipt.identity_mode == (
        "cross_layer_with_quarantined_pairs")
    assert {item.pair_index for item in reduced.request.candidates} == {1}
    pair_receipts = {item.pair_index: item for item in reduced.pairs}
    assert pair_receipts[0].forwarded_candidate_count == 0
    assert pair_receipts[1].forwarded_candidate_count == 2
    assert reduced.receipt.omitted_pair_count == 1


def test_all_quarantined_pairs_are_admission_unavailable_not_zero_receipt() -> None:
    observations = [
        replace(
            telemetry(pair),
            local_health=PathHealth.DENIED,
            remote_health=PathHealth.DENIED,
            quarantine_reason="endpoint_fetch:TimeoutError",
        )
        for pair in range(2)
    ]
    reducer = HierarchicalCandidateReducer(
        shard_count=2, max_pairs_per_shard=1, max_routes_per_pair=2,
    )
    with pytest.raises(
        HierarchyCandidateUnavailableError,
        match="no policy-eligible candidate",
    ):
        reducer.reduce(request(2), telemetry=observations, now_ns=10)


def test_per_pair_topology_is_preserved_without_rejecting_global_batch() -> None:
    observations = [telemetry(pair) for pair in range(2)]
    signal = CrossLayerSignal(
        name="cassini_rx_pause_fraction_max",
        value=0.1,
        unit="fraction",
        support="supported",
        source="test",
    )
    for pair, node_id, topology in (
        (0, "nid00001", "b" * 64),
        (1, "nid00002", "c" * 64),
    ):
        observations[pair] = replace(
            observations[pair],
            sequence=9,
            cross_layer=CrossLayerTelemetry(
                pair_index=pair,
                node_id=node_id,
                endpoint_id=f"pair-{pair}",
                communicator_id="nccl-0",
                source_epoch="allocation-epoch",
                topology_fingerprint_sha256=topology,
                sequence=100 + pair,
                sampled_ns=10,
                window_ms=1.0,
                signals=(signal,),
            ),
        )

    reduced = HierarchicalCandidateReducer(
        shard_count=1, max_pairs_per_shard=2, max_routes_per_pair=2,
    ).reduce(request(2), telemetry=observations, now_ns=10)

    assert reduced.pairs[0].topology_fingerprint_sha256 == "b" * 64
    assert reduced.pairs[1].topology_fingerprint_sha256 == "c" * 64
    assert reduced.nodes[0].topology_fingerprint_sha256 == "b" * 64
    assert reduced.nodes[1].topology_fingerprint_sha256 == "c" * 64
    assert reduced.receipt.topology_fingerprint_sha256 not in {
        "b" * 64, "c" * 64,
    }


def test_stale_hierarchy_observation_fails_closed() -> None:
    reducer = HierarchicalCandidateReducer(
        shard_count=2,
        max_pairs_per_shard=2,
        telemetry_fresh_ns=5,
    )
    with pytest.raises(HierarchyIdentityError, match="telemetry stale"):
        reducer.reduce(
            request(),
            telemetry=tuple(telemetry(pair, sampled_ns=1) for pair in range(8)),
            now_ns=10,
        )


def test_hierarchical_reduction_submits_to_global_lifecycle() -> None:
    controller = GlobalOrchestrator(GlobalOrchestratorConfig(
        capacities=(PairCapacity(0, CAPACITY), PairCapacity(1, CAPACITY)),
        tenants=(TenantPolicy("latency"),),
        telemetry_fresh_ns=100,
        queue_capacity=8,
        maximum_active_pairs=2,
    ))
    controller.update_telemetry(telemetry(0))
    controller.update_telemetry(telemetry(1))
    decision, reduction = submit_hierarchical(
        controller,
        HierarchicalCandidateReducer(
            shard_count=1,
            max_pairs_per_shard=2,
            telemetry_fresh_ns=100,
        ),
        request(2),
        now_ns=10,
    )
    assert decision.kind is GlobalDecisionKind.ADMIT
    assert decision.pair_index == 0
    assert reduction.receipt.forwarded_candidate_count == 4
    assert controller.snapshot(now_ns=11)["inflight"] == 1
