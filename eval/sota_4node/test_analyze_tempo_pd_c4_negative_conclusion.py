from __future__ import annotations

import hashlib
import json

import pytest

from eval.sota_4node import analyze_tempo_pd_c4_negative_conclusion as analyzer


def _summary(
    label: str,
    mechanism: str,
    *,
    predictor_pass: bool = True,
    median_pass: bool = False,
    tail_pass: bool = False,
    oracle_pass: bool = False,
):
    return {
        "label": label,
        "mechanism": mechanism,
        "predictor_5pct_pass": predictor_pass,
        "median_10pct_pass": median_pass,
        "tail_bundle_pass": tail_pass,
        "diagnostic_phase_oracle_full_gate_pass": oracle_pass,
    }


def _three_candidates():
    return [
        _summary("A", "instant", predictor_pass=False),
        _summary("B", "active_epoch"),
        _summary("C", "credit_epoch"),
    ]


def test_independent_joint_median_tail_failures_authorize_negative_conclusion():
    verdict = analyzer._evaluate_stop_rule(
        _three_candidates(), semantic_correctness_and_exercise_pass=True)

    assert verdict["predictor_5pct_failure_count"] == 1
    assert verdict["two_predictor_failures"] is False
    assert verdict["median_and_tail_joint_pass_count"] == 0
    assert verdict["median_and_tail_cannot_be_jointly_met"] is True
    assert verdict["diagnostic_phase_oracle_full_gate_pass_count"] == 0
    assert verdict["reproducible_negative_conclusion_allowed"] is True
    assert verdict["threshold_retuning_allowed"] is False


@pytest.mark.parametrize("failure_mode", ("joint_pass", "oracle_pass", "semantic"))
def test_positive_or_unvalidated_evidence_blocks_negative_conclusion(failure_mode):
    summaries = _three_candidates()
    semantic_pass = True
    if failure_mode == "joint_pass":
        summaries[2]["median_10pct_pass"] = True
        summaries[2]["tail_bundle_pass"] = True
    elif failure_mode == "oracle_pass":
        summaries[2]["diagnostic_phase_oracle_full_gate_pass"] = True
    else:
        semantic_pass = False

    verdict = analyzer._evaluate_stop_rule(
        summaries, semantic_correctness_and_exercise_pass=semantic_pass)

    assert verdict["reproducible_negative_conclusion_allowed"] is False


def test_bound_artifact_rejects_digest_drift(tmp_path):
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps({"schema": "fixture"}) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert analyzer._load_bound(path, digest, name="fixture")["schema"] == "fixture"
    with pytest.raises(ValueError, match="digest differs"):
        analyzer._load_bound(path, "0" * 64, name="fixture")
