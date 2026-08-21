#!/usr/bin/env python3
"""Bind the independent C4 candidates and evaluate the frozen stop rule."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from eval.sota_4node import analyze_tempo_pd_c4_phase_screen as phase_analysis


SCHEMA = "tempo-pd-c4-negative-conclusion-v1"
SEMANTIC_SCHEMA = "tempo-pd-c4-semantic-epoch-screen-analysis-v1"
CANDIDATES = (
    ("A", "instant_score_v1"),
    ("B", "frontend_active_watermark_epoch"),
    ("C", "local_external_endpoint_credit_epoch"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object, *, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256",
    )
    return value


def _load_bound(path: Path, expected_sha256: str, *, name: str) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file(), f"{name} is missing")
    _require(
        _sha256(path) == _canonical_sha(expected_sha256, name=f"{name} SHA"),
        f"{name} digest differs",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _candidate_summary(
    label: str, mechanism: str, path: Path, value: dict[str, Any],
) -> dict[str, Any]:
    _require(
        value.get("schema") == phase_analysis.SCHEMA
        and value.get("live_screen_correctness_pass") is True
        and value.get("performance_claim_allowed") is False
        and value.get("unchanged_pd_data_plane") is True
        and value.get("transport") == "LMCacheConnectorV1:UCX"
        and value.get("paired_foreground_requests") == 360,
        f"Candidate {label} phase analysis differs",
    )
    fixed = value["tempo_vs_strongest_fixed"]
    predictor = value["tempo_vs_predictor"]
    gates = value["original_goal_gate_on_live_tempo"]
    tail_pass = all((
        gates["every_phase_e2e_p99_regression_at_most_5pct"],
        gates["every_phase_tpot_p99_regression_at_most_5pct"],
        gates["worst_regression_at_most_100ms"],
    ))
    route_quality = value["tempo_selected_route_counterfactual"]
    oracle = value["cross_fit_phase_router"]
    return {
        "label": label,
        "mechanism": mechanism,
        "analysis": str(path.resolve()),
        "analysis_sha256": _sha256(path.resolve()),
        "slurm_job_id": value["slurm_job_id"],
        "strongest_fixed_arm": value["strongest_fixed_arm"],
        "tempo_e2e_median_ms": value["pooled_arm_metrics"]["tempo"][
            "e2e_median_ms"],
        "strongest_fixed_e2e_median_ms": fixed["baseline"][
            "e2e_median_ms"],
        "predictor_e2e_median_ms": predictor["baseline"]["e2e_median_ms"],
        "e2e_median_gain_vs_fixed": fixed[
            "e2e_median_improvement_fraction"],
        "e2e_median_gain_vs_predictor": predictor[
            "e2e_median_improvement_fraction"],
        "goodput_gain_vs_fixed": fixed[
            "goodput_relative_improvement_fraction"],
        "paired_win_fraction_vs_fixed": fixed["paired_win_fraction"],
        "tpot_p99_regression_vs_fixed": fixed[
            "tpot_p99_regression_fraction"],
        "worst_paired_e2e_regression_ms": fixed[
            "worst_paired_e2e_regression_ms"],
        "selected_local_counterfactual_gain": route_quality["local"][
            "selected_route_median_improvement_fraction"],
        "selected_remote_counterfactual_gain": route_quality["remote"][
            "selected_route_median_improvement_fraction"],
        "median_10pct_pass": gates["vs_strongest_fixed_median_10pct"],
        "predictor_5pct_pass": gates["vs_predictor_median_5pct"],
        "tail_bundle_pass": tail_pass,
        "all_original_gates_pass": gates["all_pass"],
        "diagnostic_phase_oracle_full_gate_pass": oracle[
            "full_goal_gate"]["all_pass"],
        "diagnostic_phase_oracle": {
            "median_gain_vs_fixed": oracle["vs_strongest_fixed"][
                "e2e_median_improvement_fraction"],
            "goodput_gain_vs_fixed": oracle["vs_strongest_fixed"][
                "goodput_relative_improvement_fraction"],
            "paired_win_fraction_vs_fixed": oracle["vs_strongest_fixed"][
                "paired_win_fraction"],
            "tpot_p99_regression_vs_fixed": oracle["vs_strongest_fixed"][
                "tpot_p99_regression_fraction"],
            "worst_paired_e2e_regression_ms": oracle[
                "vs_strongest_fixed"]["worst_paired_e2e_regression_ms"],
        },
    }


def _nearest_rank(values: list[float], fraction: float) -> float:
    _require(bool(values), "noise sample is empty")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _same_schedule_replica_noise(candidate_c: dict[str, Any]) -> dict[str, Any]:
    raw_path = Path(str(candidate_c["source_raw"])).resolve()
    _require(
        raw_path.is_file() and _sha256(raw_path) == candidate_c["source_raw_sha256"],
        "Candidate C raw binding differs",
    )
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    bindings = {
        row["key"]: row for row in candidate_c["child_artifact_bindings"]}
    artifacts = raw.get("artifacts")
    _require(
        isinstance(artifacts, dict) and set(artifacts) == set(bindings),
        "Candidate C child inventory differs",
    )
    samples: dict[tuple[str, int, str], tuple[str, float]] = {}
    for key, raw_child in artifacts.items():
        child_path = Path(str(raw_child)).resolve()
        binding = bindings[key]
        _require(
            child_path.is_file()
            and str(child_path) == str(Path(binding["path"]).resolve())
            and _sha256(child_path) == binding["sha256"],
            f"Candidate C child binding differs: {key}",
        )
        child = json.loads(child_path.read_text(encoding="utf-8"))
        contract = child["c4_phase_screen_contract"]
        arm = str(contract["arm"])
        replicate = int(contract["replicate"])
        rows = {row["request_id"]: row for row in child["requests"]}
        for request_id, metadata in contract["request_index"].items():
            pair_key = metadata.get("pair_key")
            if pair_key is None:
                continue
            row = rows[request_id]
            e2e_ms = (
                int(row["stream_end_offset_ns"])
                - int(row["dispatch_offset_ns"])
            ) / 1_000_000.0
            samples[(arm, replicate, str(pair_key))] = (
                str(metadata["phase"]), e2e_ms)
    _require(len(samples) == 4 * 2 * 180,
             "Candidate C paired sample inventory differs")
    result = {}
    for arm in phase_analysis.ARMS:
        absolute = []
        by_phase: dict[str, list[float]] = defaultdict(list)
        for (sample_arm, replicate, pair_key), (phase, first) in samples.items():
            if sample_arm != arm or replicate != 0:
                continue
            second = samples[(arm, 1, pair_key)][1]
            delta = abs(second - first)
            absolute.append(delta)
            by_phase[phase].append(delta)
        _require(len(absolute) == 180, f"{arm} replica coverage differs")
        result[arm] = {
            "pairs": len(absolute),
            "absolute_e2e_delta_median_ms": statistics.median(absolute),
            "absolute_e2e_delta_p90_ms": _nearest_rank(absolute, 0.90),
            "absolute_e2e_delta_p99_ms": _nearest_rank(absolute, 0.99),
            "absolute_e2e_delta_max_ms": max(absolute),
            "phase_absolute_e2e_delta_max_ms": {
                phase: max(values)
                for phase, values in sorted(by_phase.items())
            },
        }
    return {
        "scope": (
            "same arm and semantic schedule cell across the two "
            "counterbalanced internal replicates; diagnostic noise floor only"),
        "cross_replicate_clock_subtraction_used": False,
        "raw": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "by_arm": result,
    }


def _evaluate_stop_rule(
    summaries: list[dict[str, Any]],
    *,
    semantic_correctness_and_exercise_pass: bool,
) -> dict[str, Any]:
    _require(len(summaries) == len(CANDIDATES),
             "candidate summary inventory differs")
    distinct_mechanisms = len({row["mechanism"] for row in summaries}) == 3
    predictor_failures = sum(
        not row["predictor_5pct_pass"] for row in summaries)
    median_tail_joint_passes = sum(
        row["median_10pct_pass"] and row["tail_bundle_pass"]
        for row in summaries)
    oracle_full_passes = sum(
        row["diagnostic_phase_oracle_full_gate_pass"] for row in summaries)
    stop_rule = {
        "independent_candidate_mechanisms_exact": distinct_mechanisms,
        "predictor_5pct_failure_count": predictor_failures,
        "two_predictor_failures": predictor_failures >= 2,
        "median_and_tail_joint_pass_count": median_tail_joint_passes,
        "median_and_tail_cannot_be_jointly_met": (
            distinct_mechanisms and median_tail_joint_passes == 0),
        "diagnostic_phase_oracle_full_gate_pass_count": oracle_full_passes,
        "threshold_retuning_allowed": False,
    }
    stop_rule["reproducible_negative_conclusion_allowed"] = (
        stop_rule["median_and_tail_cannot_be_jointly_met"]
        and oracle_full_passes == 0
        and semantic_correctness_and_exercise_pass
    )
    return stop_rule


def analyze(
    *, candidate_paths: tuple[Path, Path, Path],
    candidate_sha256: tuple[str, str, str], semantic_c_path: Path,
    semantic_c_sha256: str,
) -> dict[str, Any]:
    values = []
    summaries = []
    for (label, mechanism), path, expected in zip(
        CANDIDATES, candidate_paths, candidate_sha256, strict=True,
    ):
        value = _load_bound(path, expected, name=f"Candidate {label} analysis")
        values.append(value)
        summaries.append(_candidate_summary(label, mechanism, path, value))
    semantic_c = _load_bound(
        semantic_c_path, semantic_c_sha256, name="Candidate C semantic analysis")
    _require(
        semantic_c.get("schema") == SEMANTIC_SCHEMA
        and semantic_c.get("semantic_correctness_and_exercise_pass") is True
        and semantic_c.get("original_screen_performance_gate_pass") is False
        and semantic_c.get("authorizes_candidate_for_final_c4_integration")
        is False
        and semantic_c.get("unchanged_pd_data_plane") is True,
        "Candidate C semantic verdict differs",
    )
    stop_rule = _evaluate_stop_rule(
        summaries,
        semantic_correctness_and_exercise_pass=(
            semantic_c["semantic_correctness_and_exercise_pass"] is True),
    )
    _require(stop_rule["reproducible_negative_conclusion_allowed"],
             "frozen negative stop rule is not satisfied")
    analyzer_path = Path(__file__).resolve()
    return {
        "schema": SCHEMA,
        "analyzer": str(analyzer_path),
        "analyzer_sha256": _sha256(analyzer_path),
        "candidates": summaries,
        "candidate_c_semantic_analysis": str(semantic_c_path.resolve()),
        "candidate_c_semantic_analysis_sha256": _sha256(
            semantic_c_path.resolve()),
        "same_schedule_replica_noise": _same_schedule_replica_noise(values[2]),
        "stop_rule": stop_rule,
        "conclusion_scope": (
            "negative for dynamic contention admission/routing with unchanged "
            "vLLM/LMCache P/D data plane on the frozen four-node C4 workload; "
            "not a universal LMCache or orchestration result"),
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "unchanged_pd_data_plane": True,
        "transport": "LMCacheConnectorV1:UCX",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("a", "b", "c"):
        parser.add_argument(f"--candidate-{label}", type=Path, required=True)
        parser.add_argument(
            f"--candidate-{label}-sha256", required=True)
    parser.add_argument("--semantic-c", type=Path, required=True)
    parser.add_argument("--semantic-c-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "refusing to overwrite negative analysis")
    value = analyze(
        candidate_paths=(
            args.candidate_a, args.candidate_b, args.candidate_c),
        candidate_sha256=(
            args.candidate_a_sha256,
            args.candidate_b_sha256,
            args.candidate_c_sha256,
        ),
        semantic_c_path=args.semantic_c,
        semantic_c_sha256=args.semantic_c_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "sha256": _sha256(args.output.resolve()),
        "negative_conclusion": value["stop_rule"][
            "reproducible_negative_conclusion_allowed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
