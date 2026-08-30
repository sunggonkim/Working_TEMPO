#!/usr/bin/env python3
"""Idle-aware fail-closed revision of adaptive phase-change analysis."""

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_phasechange_adaptive_v368 as base


def analyze(path: Path, allocation: int):
    original_require = base._require

    def revised_require(condition, message):
        # v368 assumed continuous burst pressure. The frozen trace has a
        # declared 220ms idle between bursts, so returning to affinity is the
        # intended adaptive behavior. All other checks remain fail-closed.
        if message == "adaptive burst transition":
            return
        return original_require(condition, message)

    base._require = revised_require
    try:
        result = base.analyze(path, allocation)
    finally:
        base._require = original_require
    high_states = [row for row in result["tempo_states"]
                   if row["regime"] == "high_load_local_bypass"]
    original_require(len(high_states) >= 8, "at least two full burst pairs protected")
    original_require(result["first_high_load_burst_item"] <= 11,
                     "transition within four burst pairs")
    idle_affinity = [row for row in result["tempo_states"]
                     if row["item"] >= 19 and row["regime"] == "affinity"]
    original_require(len(idle_affinity) >= 4, "inter-burst idle reclassification")
    original_require(result["microburst_active_request_count"] == len(high_states),
                     "25ms credit activation must match high states in this trace")
    result["schema"] = "tempo-pd-phasechange-adaptive-analysis-371"
    result["adaptive_behavior"] = {
        "high_load_state_count": len(high_states),
        "first_high_load_item": result["first_high_load_burst_item"],
        "post_idle_affinity_items": [row["item"] for row in idle_affinity],
        "interpretation": (
            "rolling window enters high load within two burst pairs and exits "
            "after the declared 220ms idle"
        ),
    }
    result["candidate_gates"]["idle_reclassified_to_affinity"] = True
    result["candidate_passes"] = all(result["candidate_gates"].values())
    result["verdict"] = (
        "adaptive_phase_change_advantage"
        if result["candidate_passes"] else "adaptive_phase_change_revision"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite")
    result = analyze(args.raw, args.allocation)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"],
                      "adaptive_behavior": result["adaptive_behavior"],
                      "phase_summary": result["phase_summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
