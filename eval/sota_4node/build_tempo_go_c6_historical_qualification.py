#!/usr/bin/env python3
"""Rebind existing native C1/C2/P_ONLY/C3 evidence to the C6 gates.

This is a provenance/qualification compiler, not a performance simulator.
It never fills missing decoder-TPOT or NCCL victim cells with a proxy and it
never promotes the historical 15-second blocks to the C6 >=30-second phase
contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "tempo-go-c6-historical-qualification-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read qualification source: {path}") from exc
    _require(isinstance(value, dict), f"qualification source is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(value: object, name: str) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value)),
        f"{name} must be finite",
    )
    return float(value)


def _rate_rows(characterization: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = characterization.get("paired_rate_summary")
    _require(isinstance(rows, list) and rows, "P_ONLY rate summary is missing")
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict), "P_ONLY rate row is invalid")
        rate_value = _finite_number(
            row.get("background_rate_per_s"), "P_ONLY offered rate")
        rate = int(rate_value)
        _require(rate_value == rate and rate > 0, "P_ONLY rate must be a positive int")
        _require(rate not in indexed, "P_ONLY rate rows duplicate")
        indexed[rate] = row
    return indexed


def build_receipt(
    *,
    crossover_result_path: Path,
    p_only_characterization_path: Path,
    c3_gate_path: Path,
) -> dict[str, Any]:
    crossover_result_path = crossover_result_path.resolve()
    p_only_characterization_path = p_only_characterization_path.resolve()
    c3_gate_path = c3_gate_path.resolve()
    crossover = _load(crossover_result_path)
    p_only = _load(p_only_characterization_path)
    c3 = _load(c3_gate_path)

    _require(
        crossover.get("schema") == "tempo-pd-contention-node-result-v7",
        "C1/C2 result schema differs",
    )
    gate = crossover.get("crossover_gate")
    _require(isinstance(gate, dict), "C1/C2 crossover gate is missing")
    _require(
        gate.get("workload_valid_for_controller_tuning") is True,
        "C1/C2 source did not pass its native gate",
    )
    _require(
        p_only.get("schema") == "tempo-pd-kv-only-characterization-v2",
        "P_ONLY characterization schema differs",
    )
    _require(
        p_only.get("all_measured_requests_valid") is True,
        "P_ONLY source contains invalid measured requests",
    )
    invariants = p_only.get("invariants")
    _require(isinstance(invariants, dict), "P_ONLY invariants are missing")
    _require(
        invariants.get("synthetic_network_background") is False
        and invariants.get("preseed_outside_measurement_window") is True
        and invariants.get("background_full_source_hits_exact") is True,
        "P_ONLY actual-victim invariants failed",
    )
    _require(
        c3.get("schema") == "tempo-pd-c3-coupled-abba-gate-v1"
        and c3.get("c3_coupled_characterization_valid") is True,
        "C3 ABBA gate is invalid",
    )

    phase_results = gate.get("phase_results")
    _require(isinstance(phase_results, dict), "C1/C2 phase results are missing")
    c1 = phase_results.get("c1_decoder_hot")
    c2 = phase_results.get("c2_remote_hot")
    _require(isinstance(c1, dict) and isinstance(c2, dict), "C1/C2 rows are missing")
    c1_gain = min(
        _finite_number(c1.get("pooled_median_gain"), "C1 pooled gain"),
        _finite_number(c1.get("paired_median_gain"), "C1 paired gain"),
    )
    c2_gain = min(
        _finite_number(c2.get("pooled_median_gain"), "C2 pooled gain"),
        _finite_number(c2.get("paired_median_gain"), "C2 paired gain"),
    )
    _require(c1.get("winner") == "remote", "C1 native winner differs")
    _require(c2.get("winner") == "local", "C2 native winner differs")

    rows = _rate_rows(p_only)
    _require({4, 8, 12, 16} <= set(rows), "P_ONLY Q0/Q1 rates are incomplete")
    ceiling = _finite_number(
        p_only.get("max_observed_remote_background_completion_rate_per_s"),
        "P_ONLY completion ceiling",
    )
    _require(ceiling > 0.0, "P_ONLY completion ceiling must be positive")
    baseline = _finite_number(
        rows[4].get("remote_foreground_median_ms"), "P_ONLY baseline victim p50")
    overload = _finite_number(
        rows[12].get("remote_foreground_median_ms"), "P_ONLY overload victim p50")
    victim_ratio = overload / baseline
    victim_degradation = victim_ratio - 1.0
    lmcache_victim_pass = victim_degradation >= 0.25

    c3_control = _finite_number(
        c3.get("remote_control_median_gain"), "C3 remote-control gain")
    c3_overload = _finite_number(
        c3.get("local_overload_median_gain"), "C3 local-overload gain")
    q2_pass = (
        c1_gain >= 0.15
        and c2_gain >= 0.15
        and max(c1_gain, c2_gain, c3_overload) >= 0.30
        and c3_control >= 0.15
        and c3_overload >= 0.20
    )

    source_paths = (
        crossover_result_path,
        p_only_characterization_path,
        c3_gate_path,
    )
    return {
        "schema": SCHEMA,
        "sources": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in source_paths
        ],
        "q0_capacity_normalized_load": {
            "pass_for_existing_4094_output2_tier": True,
            "official_lmcache_completion_ceiling_per_s": ceiling,
            "measured_points": [
                {
                    "offered_rate_per_s": rate,
                    "offered_over_completion_ceiling": rate / ceiling,
                    "achieved_remote_completion_rate_per_s": _finite_number(
                        rows[rate].get(
                            "remote_background_completion_rate_per_s"),
                        f"P_ONLY achieved rate {rate}",
                    ),
                    "victim_remote_p50_ms": _finite_number(
                        rows[rate].get("remote_foreground_median_ms"),
                        f"P_ONLY victim p50 {rate}",
                    ),
                }
                for rate in (4, 8, 12, 16, 24, 32)
                if rate in rows
            ],
            "tier_mapping": {
                "normal_0p8_nearest_measured_rate_per_s": 8,
                "knee_1p0_is_bracketed_not_interpolated": [8, 12],
                "overload_1p2_nearest_measured_rate_per_s": 12,
                "severe_1p5_nearest_measured_rate_per_s": 16,
            },
            "new_geometry_requires_requalification": True,
        },
        "q1_actual_victim_aggressor": {
            "official_lmcache_receiver": {
                "pass": lmcache_victim_pass,
                "criterion": "victim_p50_degradation_at_least_25pct",
                "baseline_offered_rate_per_s": 4,
                "overload_offered_rate_per_s": 12,
                "baseline_victim_p50_ms": baseline,
                "overload_victim_p50_ms": overload,
                "victim_p50_ratio": victim_ratio,
                "victim_p50_degradation": victim_degradation,
                "actual_route_pinned_vllm": True,
                "official_lmcache_nixl_ucx": True,
                "synthetic_aggressor": False,
            },
            "decoder_tpot_output_completion": {
                "pass": False,
                "status": "missing_output16_128_256_long_phase_victim_matrix",
            },
            "real_nccl_collective_completion": {
                "pass": False,
                "status": "missing_same_population_no_aggressor_vs_aggressor_victim_matrix",
            },
            "all_required_victims_pass": False,
        },
        "q2_opposite_action_opportunity": {
            "pass": q2_pass,
            "minimum_phase_winner_margin": 0.15,
            "minimum_one_overload_margin": 0.30,
            "minimum_alternate_recovery": 0.20,
            "c1_remote_gain_conservative": c1_gain,
            "c2_local_gain_conservative": c2_gain,
            "c3_remote_control_gain": c3_control,
            "c3_local_overload_gain": c3_overload,
            "replicate_direction_exact": (
                all(c1.get("replicate_direction_correct", ()))
                and all(c2.get("replicate_direction_correct", ()))
                and all(c3.get("remote_control_replicate_direction_correct", ()))
                and all(c3.get("local_overload_replicate_direction_correct", ()))
            ),
        },
        "q3_service_horizon_and_geometry": {
            "pass": False,
            "historical_phase_duration_ms": 15_000,
            "required_minimum_phase_duration_ms": 30_000,
            "measured_p95_first_response_horizon_bound": False,
            "output16_128_256_matrix": False,
            "reason": "historical evidence proves opportunity but not the frozen C6 horizon",
        },
        "promotion": {
            "controller_performance_run_allowed": False,
            "reason": "Q1 decoder/NCCL cells and Q3 service horizon remain open",
            "next_native_work": [
                "decoder_output_completion_victim_no_aggressor_vs_hot",
                "real_nccl_collective_victim_no_aggressor_vs_lmcache_and_decode_hot",
                "at_least_30s_or_3xp95_phase_horizon",
                "fixed_pxd_edge_alternate_decoder_capacity_recovery",
            ],
            "performance_claim_allowed": False,
        },
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossover-result", type=Path, required=True)
    parser.add_argument("--p-only-characterization", type=Path, required=True)
    parser.add_argument("--c3-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    output = args.output.resolve()
    _require(not output.exists(), "refusing to overwrite qualification receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = build_receipt(
        crossover_result_path=args.crossover_result,
        p_only_characterization_path=args.p_only_characterization,
        c3_gate_path=args.c3_gate,
    )
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(output),
        "q0_pass": receipt["q0_capacity_normalized_load"][
            "pass_for_existing_4094_output2_tier"],
        "q1_lmcache_pass": receipt["q1_actual_victim_aggressor"][
            "official_lmcache_receiver"]["pass"],
        "q2_pass": receipt["q2_opposite_action_opportunity"]["pass"],
        "q3_pass": receipt["q3_service_horizon_and_geometry"]["pass"],
        "promotion": receipt["promotion"]["controller_performance_run_allowed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
