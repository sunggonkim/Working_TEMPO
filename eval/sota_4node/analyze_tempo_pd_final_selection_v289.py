#!/usr/bin/env python3
"""Select the frozen two-regime controller from completed actual-vLLM evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def analyze(frontier_path: Path, policy8_rate52_path: Path,
            policy11_rate52_path: Path, allocation: int) -> dict:
    frontier = json.loads(frontier_path.resolve().read_text())
    policy8 = json.loads(policy8_rate52_path.resolve().read_text())
    policy11 = json.loads(policy11_rate52_path.resolve().read_text())
    _require(frontier.get("schema") == "tempo-pd-mixed-frontier-analysis-280",
             "frontier schema")
    _require(frontier.get("allocation_id") == allocation and frontier.get("passes") is True,
             "frontier validity")
    _require(policy8.get("schema") == "tempo-pd-mixed-request-crossover-analysis-263",
             "policy8 schema")
    _require(policy11.get("schema") == "tempo-pd-policy11-highload-mixed-analysis-286",
             "policy11 schema")
    _require(policy8.get("allocation_id") == allocation == policy11.get("allocation_id"),
             "allocation mismatch")
    _require(policy8.get("route_counts") == {
        "lmcache_remote": 24, "tempo_local": 19, "tempo_remote": 5},
        "policy8 routes")
    _require(policy11.get("route_counts") == {
        "lmcache_remote": 24, "tempo_local": 24, "tempo_remote": 0},
        "policy11 routes")
    _require(policy8.get("passes") is True and policy11.get("passes") is True,
             "same-window gates")
    p8 = policy8["summary"]
    p11 = policy11["summary"]
    gates = {
        "normal_load_frontier_passes": frontier["passes"],
        "rate52_policy11_e2e_wins_at_least_policy8": (
            p11["e2e_win_count"] >= p8["e2e_win_count"]),
        "rate52_policy11_e2e_median_beats_policy8": (
            p11["e2e_delta_median_ms"] < p8["e2e_delta_median_ms"]),
        "rate52_policy11_tpot_wins_at_least_policy8": (
            p11["tpot_win_count"] >= p8["tpot_win_count"]),
        "rate52_policy11_tpot_median_beats_policy8": (
            p11["tpot_delta_median_ms"] < p8["tpot_delta_median_ms"]),
        "rate56_excluded_after_official_lmcache_fatal": (
            frontier["frontier"]["official_lmcache_fatal_offered_rate_per_s"] == 56),
    }
    passes = all(gates.values())
    return {
        "schema": "tempo-pd-final-selection-analysis-289",
        "allocation_id": allocation,
        "controller": "qwen25-7b-tp4x2-warm-regime-controller-12",
        "regimes": {
            "16_32_48_requests_per_s": {
                "policy": "qwen25-7b-tp4x2-warm-affinity-8",
                "action": "cache_affinity_hybrid",
            },
            "52_requests_per_s": {
                "policy": "qwen25-7b-tp4x2-warm-highload-local-11",
                "action": "decoder_local_bypass_all_remote_warm_hits",
            },
            "unvalidated_or_56_requests_per_s": {
                "action": "fail_closed",
            },
        },
        "rate52_comparison": {"policy8": p8, "policy11": p11},
        "frontier_report": str(frontier_path.resolve()),
        "policy8_rate52_report": str(policy8_rate52_path.resolve()),
        "policy11_rate52_report": str(policy11_rate52_path.resolve()),
        "gates": gates,
        "passes": passes,
        "verdict": "freeze_two_regime_actual_vllm_controller" if passes else
                   "controller_selection_not_established",
        "claim_boundary": (
            "Calibrated only for Qwen2.5-7B, TP4 prefill plus TP4 decode, the "
            "validated warm-cache geometries, and offered loads 16/32/48/52 in "
            "allocation 57057488. Regime is frozen before an epoch; this does not "
            "validate an online load estimator or Mooncake."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--policy8-rate52", type=Path, required=True)
    parser.add_argument("--policy11-rate52", type=Path, required=True)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite")
    report = analyze(args.frontier, args.policy8_rate52,
                     args.policy11_rate52, args.allocation)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": report["verdict"], "gates": report["gates"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
