#!/usr/bin/env python3
"""Combine the three validated pair-local offered-rate screens."""

import argparse
import json
from pathlib import Path
import statistics


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(paths):
    reports = [json.loads(path.resolve().read_text()) for path in paths]
    _require(len(reports) == 3, "exactly three reports required")
    _require({row["offered_rate_per_s"] for row in reports} == {48.0, 50.0, 52.0},
             "rates must be 48, 50, and 52")
    _require(len({row["allocation_id"] for row in reports}) == 1,
             "reports must share one allocation")
    pooled_e2e = []
    pooled_tpot = []
    rate_rows = []
    for row in sorted(reports, key=lambda item: item["offered_rate_per_s"]):
        _require(row["schema"] == "tempo-pd-online-regime-pairlocal-analysis-316",
                 "report schema")
        _require(row["policy"] == "tempo-pd-online-regime-router-fast-311",
                 "policy")
        _require(row["passes"] is True and all(row["gates"].values()), "rate gates")
        _require(len(row["pairs"]) == 24, "pair count")
        raw = json.loads(Path(row["raw"]).resolve().read_text())
        contract = raw.get("mixed_crossover_contract", {})
        _require(contract.get("cache_isolation") ==
                 "vllm_cache_salt_plus_unique_18_token_regions_v305",
                 "collision-free workload contract")
        _require(raw["validation"]["performance_claim_allowed"] is True,
                 "raw validity")
        e2e = [pair["e2e_delta_ms"] for pair in row["pairs"]]
        tpot = [pair["tpot_delta_ms"] for pair in row["pairs"]]
        pooled_e2e.extend(e2e)
        pooled_tpot.extend(tpot)
        rate_rows.append({
            "offered_rate_per_s": row["offered_rate_per_s"],
            "route_counts": row["route_counts"],
            "regime_counts": row["regime_counts"],
            "pair_local_median_gap_ns": row["pair_local_median_gap_ns"],
            "summary": row["summary"],
            "report": str(paths[reports.index(row)].resolve()),
        })
    gates = {
        "all_rate_reports_pass": True,
        "pooled_e2e_median_improves": statistics.median(pooled_e2e) < 0,
        "pooled_e2e_win_fraction_ge_80pct": sum(x < 0 for x in pooled_e2e) >= 58,
        "pooled_tpot_median_improves": statistics.median(pooled_tpot) < 0,
        "pooled_tpot_p90_nonregression": sorted(pooled_tpot)[64] <= 0,
    }
    passes = all(gates.values())
    return {
        "schema": "tempo-pd-online-regime-final-analysis-320",
        "allocation_id": reports[0]["allocation_id"],
        "policy": "tempo-pd-online-regime-router-fast-311",
        "rates": rate_rows,
        "pooled": {
            "paired_requests": len(pooled_e2e),
            "e2e_delta_median_ms": statistics.median(pooled_e2e),
            "e2e_win_count": sum(x < 0 for x in pooled_e2e),
            "tpot_delta_median_ms": statistics.median(pooled_tpot),
            "tpot_win_count": sum(x < 0 for x in pooled_tpot),
            "tpot_p90_delta_ms": sorted(pooled_tpot)[64],
        },
        "gates": gates,
        "passes": passes,
        "verdict": ("freeze_pairlocal_online_controller" if passes else
                    "do_not_freeze_pairlocal_online_controller"),
        "claim_boundary": (
            "Actual vLLM 1P1D same-window component comparison against official "
            "LMCacheConnectorV1 on one four-node allocation, using 24 paired cold "
            "requests per offered rate and collision-isolated equal-token prompts. "
            "This is not an independent-repeat, Mooncake, or universal SOTA claim."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite")
    result = analyze(args.report)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"],
                      "pooled": result["pooled"]}, sort_keys=True))


if __name__ == "__main__":
    main()
