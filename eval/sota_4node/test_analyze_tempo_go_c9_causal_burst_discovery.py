from __future__ import annotations

from eval.sota_4node import analyze_tempo_go_c9_causal_burst_discovery as analyzer


def test_valid_failure_receipt_is_not_a_completed_request() -> None:
    terminal = {
        "http_status": 503,
        "terminal_kind": "service_lane_failure",
        "terminal_error_kind": "endpoint_service_lane_preflight_unavailable",
        "valid": True,
        "done_seen": False,
        "output_token_values": [],
    }

    assert analyzer._business_terminal_outcome(
        terminal, expected_output_tokens=128) == "failure"


def test_completion_requires_done_and_exact_output_tokens() -> None:
    terminal = {
        "http_status": 200,
        "terminal_kind": None,
        "valid": True,
        "done_seen": True,
        "output_token_values": ["x", "y"],
    }

    assert analyzer._business_terminal_outcome(
        terminal, expected_output_tokens=2) == "completed"
    assert analyzer._business_terminal_outcome(
        terminal, expected_output_tokens=3) == "failure"


def test_global_reject_remains_distinct_from_failure() -> None:
    terminal = {
        "http_status": 503,
        "terminal_kind": "global_reject",
        "valid": True,
        "done_seen": False,
        "output_token_values": [],
    }

    assert analyzer._business_terminal_outcome(
        terminal, expected_output_tokens=2) == "global_reject"


def test_empty_offered_regime_preserves_null_latency_metrics() -> None:
    analysis = {
        "remote_favorable": {
            "offered_victims": 30,
            "completed_victims": 0,
            "slo_good_victims": 0,
            "failures": 0,
            "global_rejects": 30,
            "victim": {
                "e2e_ms": {"p50": None, "p99": None},
                "ttft_ms": {"p99": None},
                "tpot_ms": {"p99": None},
            },
            "route_counts": {},
            "edge_counts": {},
        }
    }

    summary = analyzer._regime_summary(analysis, "remote_favorable")

    assert summary["offered"] == 30
    assert summary["completed"] == 0
    assert summary["global_rejects"] == 30
    assert summary["e2e_p50_ms"] is None
    assert summary["e2e_p99_ms"] is None
    assert summary["ttft_p99_ms"] is None
    assert summary["tpot_p99_ms"] is None


def test_effect_fails_closed_when_full_arm_has_no_completed_victims() -> None:
    full = {
        "slo_good_fraction": 0.0,
        "mean_e2e_p50_ms": None,
        "mean_e2e_p99_ms": None,
    }
    baseline = {
        "slo_good_fraction": 0.0,
        "mean_e2e_p50_ms": 30_000.0,
        "mean_e2e_p99_ms": 40_000.0,
    }

    effect = analyzer._effect(full, baseline)

    assert effect["full_minus_blind_p50_fraction"] is None
    assert effect["full_p99_reduction_fraction"] is None
    assert effect["full_slo_good_ratio"] == 1.0
