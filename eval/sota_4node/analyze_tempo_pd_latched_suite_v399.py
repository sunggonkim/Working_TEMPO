#!/usr/bin/env python3
"""Aggregate four fail-closed latched-controller workload reports."""

import argparse
import json
from pathlib import Path
import statistics


EXPECTED = ("phasechange", "bursty", "steady", "reverse_phasechange")
POLICY = "tempo-pd-latched-bypass-rolling-credit5-382"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(rows):
    e2e = [row["e2e_delta_ms"] for row in rows]
    tpot = [row["tpot_delta_ms"] for row in rows]
    return {
        "pairs": len(rows),
        "e2e_win_count": sum(value < 0 for value in e2e),
        "e2e_win_fraction": sum(value < 0 for value in e2e) / len(e2e),
        "e2e_delta_median_ms": statistics.median(e2e),
        "e2e_delta_max_ms": max(e2e),
        "tpot_win_count": sum(value < 0 for value in tpot),
        "tpot_win_fraction": sum(value < 0 for value in tpot) / len(tpot),
        "tpot_delta_median_ms": statistics.median(tpot),
        "tpot_delta_p90_ms": sorted(tpot)[max(0, int(0.9 * len(tpot)) - 1)],
        "tpot_delta_max_ms": max(tpot),
    }


def analyze(paths):
    require(len(paths) == 4, "exactly four workload reports required")
    reports = [json.loads(path.resolve().read_text()) for path in paths]
    by_class = {report["workload_class"]: report for report in reports}
    require(set(by_class) == set(EXPECTED), "workload classes differ")
    allocation_ids = {report["allocation_id"] for report in reports}
    require(len(allocation_ids) == 1, "reports must share one allocation")
    rows = []
    workload_summaries = {}
    for workload_class in EXPECTED:
        report = by_class[workload_class]
        require(report.get("measurement_valid") is True, f"{workload_class} invalid")
        require(report.get("candidate_passes") is True, f"{workload_class} failed")
        require(report.get("policy") == POLICY, f"{workload_class} policy differs")
        require(all(report.get("candidate_gates", {}).values()),
                f"{workload_class} gate failed")
        selected = report["pairs"][8:] if workload_class == "phasechange" else report["pairs"]
        require(len(selected) == (16 if workload_class == "phasechange" else 24),
                f"{workload_class} pair count")
        rows.extend({"workload_class": workload_class, **row} for row in selected)
        workload_summaries[workload_class] = report["evaluated_summary"]
    combined = summarize(rows)
    suite_gates = {
        "every_workload_passes": all(report["candidate_passes"] for report in reports),
        "combined_e2e_win_fraction_ge_80pct": combined["e2e_win_fraction"] >= 0.8,
        "combined_e2e_median_improves": combined["e2e_delta_median_ms"] < 0,
        "combined_tpot_win_fraction_ge_80pct": combined["tpot_win_fraction"] >= 0.8,
        "combined_tpot_median_improves": combined["tpot_delta_median_ms"] < 0,
        "combined_tpot_p90_nonregression": combined["tpot_delta_p90_ms"] <= 0,
        "both_phase_directions_pass": (
            by_class["phasechange"]["candidate_passes"]
            and by_class["reverse_phasechange"]["candidate_passes"]),
        "rolling_credit_disengages_after_reverse_transition": by_class[
            "reverse_phasechange"]["reverse_transition_gates"][
                "rolling_credit_disengages_in_steady_tail"],
    }
    passes = all(suite_gates.values())
    return {
        "schema": "tempo-pd-latched-controller-suite-analysis-399",
        "allocation_id": next(iter(allocation_ids)),
        "policy": POLICY,
        "same_allocation_non_independent_screen": True,
        "workload_order": list(EXPECTED),
        "workload_summaries": workload_summaries,
        "combined_summary": combined,
        "suite_gates": suite_gates,
        "suite_passes": passes,
        "verdict": ("freeze_latched_controller_structure"
                    if passes else "revise_latched_controller_structure"),
        "claim_boundary": (
            "same-server real-vLLM paired component screen versus the pinned "
            "LMCache remote-prefill route; not an independent multi-allocation or "
            "cross-system Mooncake result"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite")
    result = analyze(args.report)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"],
                      "summary": result["combined_summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
