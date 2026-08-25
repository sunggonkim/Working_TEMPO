import asyncio
import json

from tempo.pd_global_orchestrator import (
    CROSS_LAYER_SCHEMA,
    TELEMETRY_SCHEMA,
    CrossLayerSignal,
    CrossLayerTelemetry,
    GlobalRoute,
    PairTelemetry,
    ResourceVector,
    RouteCandidate,
)
from tempo.pd_paper_baselines import kairos_score_ms, netkv_score_ms
from eval.sota_4node.tempo_pd_paper_baseline_frontend import (
    _NoTempoBusinessAdmissionGate,
)
from eval.sota_4node.analyze_tempo_go_c10_paper_sota import (
    _paper_baseline_admission_evidence,
)
from eval.sota_4node.run_tempo_go_c10_paper_sota_client import (
    _validated_cold_global_edge,
)


def _cross(pair, *, congestion=0.0, inflight=0):
    return CrossLayerTelemetry(
        pair_index=pair,
        node_id=f"n{pair}",
        endpoint_id=f"p{pair}",
        communicator_id="c",
        source_epoch="e",
        topology_fingerprint_sha256="a" * 64,
        sequence=1,
        sampled_ns=1,
        window_ms=1.0,
        signals=(
            CrossLayerSignal(
                "cassini_ecn_fraction_max", congestion, "fraction",
                "supported", "fixture"),
            CrossLayerSignal(
                "lmcache_remote_semantic_ops_inflight", inflight,
                "operations", "supported", "fixture"),
        ),
        schema=CROSS_LAYER_SCHEMA,
    )


def _telemetry(pair, *, running=4, waiting=0, congestion=0.0, inflight=0):
    return PairTelemetry(
        pair_index=pair,
        sequence=1,
        sampled_ns=1,
        collected_ns=1,
        agent_epoch="e",
        profile_fingerprint_sha256="b" * 64,
        controller_generation=0,
        observed_total=ResourceVector(),
        scheduler_running_requests=running,
        scheduler_waiting_requests=waiting,
        scheduler_kv_cache_usage_fraction=0.25,
        scheduler_schema="fixture",
        scheduler_source="fixture",
        cross_layer=_cross(
            pair, congestion=congestion, inflight=inflight),
        schema=TELEMETRY_SCHEMA,
    )


def _candidate(route=GlobalRoute.REMOTE):
    return RouteCandidate(
        pair_index=1,
        prefill_index=0 if route is GlobalRoute.REMOTE else 1,
        decoder_index=1,
        route=route,
        work=(
            ResourceVector(
                decode_tokens=128,
                active_sequences=1,
                endpoint_requests=1,
                remote_prefill_token_ms=2_000_000,
                remote_kv_bytes=1_000_000_000,
                remote_semantic_ops=1,
            )
            if route is GlobalRoute.REMOTE else
            ResourceVector(
                decode_tokens=128,
                active_sequences=1,
                endpoint_requests=1,
                local_prefill_token_ms=4_096 * 600,
            )
        ),
        predicted_e2e_ms=3_000.0,
        predicted_ttft_ms=600.0,
    )


def test_netkv_charges_congestion_and_self_contention():
    candidate = _candidate()
    idle, idle_receipt = netkv_score_ms(
        candidate,
        source=_telemetry(0),
        destination=_telemetry(1),
        beta_max=16,
    )
    loaded, loaded_receipt = netkv_score_ms(
        candidate,
        source=_telemetry(0, congestion=0.5, inflight=2),
        destination=_telemetry(1),
        beta_max=16,
    )
    assert loaded > idle
    assert loaded_receipt["effective_bytes_per_s"] < idle_receipt[
        "effective_bytes_per_s"]
    assert loaded_receipt["source_self_inflight"] == 2


def test_netkv_decoder_queue_term_matches_paper_free_slot_rule():
    candidate = _candidate()
    no_queue, _ = netkv_score_ms(
        candidate,
        source=_telemetry(0),
        destination=_telemetry(1, running=4, waiting=12),
        beta_max=16,
    )
    blocked, receipt = netkv_score_ms(
        candidate,
        source=_telemetry(0),
        destination=_telemetry(1, running=4, waiting=13),
        beta_max=16,
    )
    assert blocked > no_queue
    assert receipt["queue_ms"] > 0


def test_kairos_local_x512_exposes_mixed_step_cost():
    local = _candidate(GlobalRoute.LOCAL)
    score, receipt = kairos_score_ms(
        local, destination=_telemetry(1), beta_max=16)
    remote_score, remote_receipt = kairos_score_ms(
        _candidate(), destination=_telemetry(1), beta_max=16)
    assert score == local.predicted_ttft_ms
    assert receipt["chunk_tokens"] == 512
    assert receipt["mixed_step_ms"] > receipt["first_decode_step_ms"]
    assert remote_receipt["mixed_step_ms"] == remote_receipt[
        "first_decode_step_ms"]
    assert remote_score == _candidate().predicted_ttft_ms


def test_paper_baseline_gate_emits_receipt_without_policy_effect():
    async def exercise():
        gate = _NoTempoBusinessAdmissionGate(
            background_limits=(8, 8),
            background_max_wait_ns=60_000_000_000,
            protected_tenants=frozenset({"interactive"}),
        )
        held = await gate.acquire(
            request_id="r", pair_index=1, tenant_id="interactive",
            globally_committed=True,
        )
        assert held["status"] == "held"
        assert held["admission_class"] == "protected"
        assert held["wait_ns"] == 0
        assert held["policy_effect"] == "none"
        released = await gate.release("r")
        assert released["status"] == "released"
        assert released["policy_effect"] == "none"
        snapshot = await gate.snapshot()
        assert snapshot["mode"] == "evidence_only_no_throttle"
        assert snapshot["leases"] == 0

    asyncio.run(exercise())


def test_analyzer_proves_no_policy_effect(tmp_path):
    artifact_path = tmp_path / "measured.raw.json"
    artifact_path.write_text(json.dumps({
        "router_decision_endpoint": {
            "decoder_business_admission": {
                "mode": "evidence_only_no_throttle",
                "policy_effect": "none",
                "leases": 0,
            },
        },
        "router_decisions": [
            {
                "frontend_decoder_business_admission": {
                    "mode": "evidence_only_no_throttle",
                    "policy_effect": "none",
                    "wait_ns": 0,
                    "starvation_escape": False,
                    "status": "released",
                    "admission_class": admission_class,
                },
            }
            for admission_class in ("protected", "background")
        ],
    }), encoding="utf-8")
    evidence = _paper_baseline_admission_evidence({
        "artifacts": {"measured": str(artifact_path)},
    })
    assert evidence["gate"] is True
    assert evidence["receipt_count"] == 2
    assert evidence["max_wait_ns"] == 0


def test_cold_mesh_validator_separates_prefill_source_and_destination():
    source, decoder = _validated_cold_global_edge({
        "route": "official_lmcache_remote_prefill",
        "tempo_go_global_commit_applied": True,
        "tempo_go_global_commit_prefill_index": 0,
        "tempo_go_global_commit_decoder_index": 1,
        "tempo_go_global_commit_pair_index": 1,
        "tempo_go_global_commit_edge_id": "remote:p0->d1",
        "frontend_pair_index": 1,
        "remote_decoder_index": 1,
    })
    assert (source, decoder) == (0, 1)
