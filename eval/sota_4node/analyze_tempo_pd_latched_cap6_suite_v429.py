#!/usr/bin/env python3
"""Aggregate the frozen cap-six controller evidence without hiding reverse noise."""

import argparse
import json
from pathlib import Path
import statistics


CAP6 = "tempo-pd-latched-bypass-rolling-credit6-401"
ONSET = "tempo-pd-latched-onset6-sustained5-411"
LABELS = ("phasechange", "bursty", "steady", "heavybursty",
          "reverse_fixed", "reverse_equivalent")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summary(rows):
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


def analyze(labeled):
    require(set(labeled) == set(LABELS), "exact suite labels required")
    reports = {label: json.loads(path.resolve().read_text())
               for label, path in labeled.items()}
    require(len({report["allocation_id"] for report in reports.values()}) == 1,
            "one allocation required")
    for label in ("phasechange", "bursty", "steady", "heavybursty", "reverse_fixed"):
        require(reports[label]["policy"] == CAP6, f"{label} policy")
        require(reports[label]["measurement_valid"] is True, f"{label} invalid")
    equivalent = reports["reverse_equivalent"]
    require(equivalent["policy"] == ONSET, "equivalent policy")
    require(equivalent["max_onset_microburst_tempo_count"] <= 8,
            "equivalent run entered sustained active cap")
    require(not equivalent["capped_requests"]
            and equivalent["route_counts"]["tempo_remote"] == 0,
            "equivalent reverse actions differ")
    require(reports["reverse_fixed"]["route_counts"] == equivalent["route_counts"],
            "reverse routes differ")

    selected = {
        "phasechange": reports["phasechange"]["pairs"][8:],
        "bursty": reports["bursty"]["pairs"],
        "steady": reports["steady"]["pairs"],
        "heavybursty": reports["heavybursty"]["pairs"],
        "reverse_fixed": reports["reverse_fixed"]["pairs"],
        "reverse_equivalent": equivalent["pairs"],
    }
    rows = [{"label": label, **row} for label, values in selected.items()
            for row in values]
    combined = summary(rows)
    reverse_tail_rows = [row for label in ("reverse_fixed", "reverse_equivalent")
                         for row in selected[label] if row["item"] >= 16]
    reverse_tail = summary(reverse_tail_rows)
    primary_pass_labels = ("phasechange", "bursty", "steady", "heavybursty")
    gates = {
        "primary_workloads_pass": all(reports[label]["candidate_passes"]
                                      for label in primary_pass_labels),
        "heavy_backpressure_exercised": reports["heavybursty"]["candidate_gates"][
            "backpressure_exercised"],
        "combined_e2e_win_fraction_ge_80pct": combined["e2e_win_fraction"] >= 0.8,
        "combined_tpot_win_fraction_ge_80pct": combined["tpot_win_fraction"] >= 0.8,
        "combined_e2e_median_improves": combined["e2e_delta_median_ms"] < 0,
        "combined_tpot_median_improves": combined["tpot_delta_median_ms"] < 0,
        "combined_tpot_p90_nonregression": combined["tpot_delta_p90_ms"] <= 0,
        "every_observed_e2e_regression_le_250ms": combined["e2e_delta_max_ms"] <= 250,
        "reverse_tail_pooled_e2e_wins_ge_12_of_16": reverse_tail["e2e_win_count"] >= 12,
        "reverse_tail_pooled_e2e_median_improves": reverse_tail["e2e_delta_median_ms"] < 0,
        "reverse_tail_pooled_tpot_wins_ge_12_of_16": reverse_tail["tpot_win_count"] >= 12,
        "reverse_tail_credit_disengages": all(
            reports[label]["reverse_transition_gates"][
                "rolling_credit_disengages_in_steady_tail"]
            for label in ("reverse_fixed", "reverse_equivalent")),
    }
    passes = all(gates.values())
    return {
        "schema": "tempo-pd-latched-cap6-suite-analysis-429",
        "allocation_id": reports["phasechange"]["allocation_id"],
        "policy": CAP6,
        "same_allocation_non_independent_screen": True,
        "behaviorally_equivalent_reverse_repeat": {
            "policy": ONSET,
            "why_equivalent": (
                "active microburst count never exceeded eight, so every active "
                "decision used cap six; inactive cap values cannot cap a request; "
                "routes and capped-request sets match the fixed-cap run"
            ),
        },
        "workload_summaries": {label: summary(values)
                               for label, values in selected.items()},
        "reverse_tail_pooled_summary": reverse_tail,
        "combined_summary": combined,
        "suite_gates": gates,
        "suite_passes": passes,
        "verdict": ("freeze_cap6_latched_controller"
                    if passes else "revise_cap6_latched_controller"),
        "claim_boundary": (
            "real-vLLM TP4x2-replica P/D same-allocation paired screen against "
            "the pinned official LMCacheConnectorV1 remote-prefill route; "
            "reverse-tail variation is pooled explicitly; no Mooncake comparison"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True,
                        help="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labeled = {}
    for value in args.report:
        label, separator, path = value.partition("=")
        if not separator or label in labeled:
            parser.error("reports must be unique LABEL=PATH values")
        labeled[label] = Path(path)
    if args.output.exists():
        parser.error("refusing to overwrite")
    result = analyze(labeled)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"],
                      "combined": result["combined_summary"],
                      "reverse_tail": result["reverse_tail_pooled_summary"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
