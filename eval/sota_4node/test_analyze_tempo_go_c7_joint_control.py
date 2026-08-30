from __future__ import annotations

from eval.sota_4node import analyze_tempo_go_c7_joint_control as analyzer


def _arm(normal: float, hot: float, hot_slo: int) -> dict[str, object]:
    def group(p50: float, p99: float, slo: int) -> dict[str, object]:
        return {
            "slo_good_victims": slo,
            "victim": {"e2e_ms": {"p50": p50, "p99": p99}},
        }
    return {
        "normal": group(normal, normal * 1.1, 60),
        "hot": group(hot, hot * 1.5, hot_slo),
        "all": group((normal + hot) / 2, hot * 1.5, hot_slo + 60),
    }


def test_effect_reports_large_receiver_protection() -> None:
    full = _arm(3000.0, 3500.0, 58)
    baseline = _arm(3000.0, 14000.0, 10)
    effect = analyzer._effect(full, baseline)
    assert effect["hot_slo_good_ratio"] == 5.8
    assert effect["hot_e2e_p50_reduction_fraction"] == 0.75
    assert effect["hot_e2e_p99_reduction_fraction"] == 0.75
    assert effect["normal_e2e_p50_regression_fraction"] == 0.0


def test_zero_completion_population_is_analyzable_negative_evidence() -> None:
    summary = analyzer._summary([])
    assert summary["count"] == 0
    assert summary["e2e_ms"]["p50"] is None
    assert summary["ttft_ms"]["p99"] is None
