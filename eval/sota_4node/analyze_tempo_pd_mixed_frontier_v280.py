#!/usr/bin/env python3
"""Freeze the four-load same-window P/D frontier and the rate-56 failure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


VALID_RATES = (16, 32, 48, 52)
EXPECTED_ROUTES = {
    "lmcache_remote": 24,
    "tempo_local": 19,
    "tempo_remote": 5,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sign_p(wins: int, trials: int) -> float:
    return sum(math.comb(trials, k) for k in range(wins, trials + 1)) / 2**trials


def analyze(runs: dict[int, Path], failure: Path, allocation: int) -> dict:
    _require(tuple(sorted(runs)) == VALID_RATES, "exact valid rates 16/32/48/52 required")
    pooled_e2e: list[float] = []
    pooled_tpot: list[float] = []
    loads = {}
    for rate in VALID_RATES:
        path = runs[rate].resolve()
        value = json.loads(path.read_text())
        _require(value.get("schema") == "tempo-pd-mixed-request-crossover-analysis-263",
                 f"rate{rate}: schema")
        _require(value.get("allocation_id") == allocation, f"rate{rate}: allocation")
        _require(value.get("passes") is True, f"rate{rate}: gates")
        _require(value.get("route_counts") == EXPECTED_ROUTES, f"rate{rate}: routes")
        pairs = value.get("pairs", [])
        _require(len(pairs) == 24, f"rate{rate}: exact pairs")
        _require([row.get("item") for row in pairs] == list(range(24)),
                 f"rate{rate}: item order")
        e2e = [float(row["e2e_delta_ms"]) for row in pairs]
        tpot = [float(row["tpot_delta_ms"]) for row in pairs]
        pooled_e2e.extend(e2e)
        pooled_tpot.extend(tpot)
        loads[str(rate)] = {
            "paired_requests": 24,
            "e2e_win_count": sum(delta < 0 for delta in e2e),
            "e2e_delta_median_ms": statistics.median(e2e),
            "tpot_win_count": sum(delta < 0 for delta in tpot),
            "tpot_delta_median_ms": statistics.median(tpot),
            "report": str(path),
        }

    failed = json.loads(failure.resolve().read_text())
    _require(failed.get("schema") == "tempo-pd-mixed-lmcache-failure-analysis-272",
             "rate56: failure schema")
    _require(failed.get("allocation_id") == allocation, "rate56: allocation")
    _require(failed.get("request_rate_per_s") == 56.0, "rate56: offered rate")
    _require(failed.get("verdict") == "official_lmcache_concurrent_retrieval_fatal",
             "rate56: failure verdict")

    trials = len(pooled_e2e)
    _require(trials == 96 and len(pooled_tpot) == trials, "pooled pair count")
    e2e_wins = sum(delta < 0 for delta in pooled_e2e)
    tpot_wins = sum(delta < 0 for delta in pooled_tpot)
    gates = {
        "every_valid_load_passes": all(loads[str(rate)]["e2e_win_count"] >= 13
                                        for rate in VALID_RATES),
        "pooled_e2e_win_fraction_ge_80pct": e2e_wins >= math.ceil(0.8 * trials),
        "pooled_e2e_median_improves": statistics.median(pooled_e2e) < 0,
        "pooled_tpot_win_fraction_ge_80pct": tpot_wins >= math.ceil(0.8 * trials),
        "pooled_tpot_median_improves": statistics.median(pooled_tpot) < 0,
        "official_lmcache_rate56_fatal_reproduced": failed["invalid_streams"] > 0,
    }
    passes = all(gates.values())
    return {
        "schema": "tempo-pd-mixed-frontier-analysis-280",
        "allocation_id": allocation,
        "policy": "qwen25-7b-tp4x2-warm-affinity-8",
        "topology": "actual-vllm-qwen25-7b-tp4-prefill-plus-tp4-decode-two-replicas",
        "valid_loads": loads,
        "pooled": {
            "paired_requests": trials,
            "e2e_win_count": e2e_wins,
            "e2e_win_fraction": e2e_wins / trials,
            "e2e_delta_median_ms": statistics.median(pooled_e2e),
            "e2e_one_sided_sign_test_p": _sign_p(e2e_wins, trials),
            "tpot_win_count": tpot_wins,
            "tpot_win_fraction": tpot_wins / trials,
            "tpot_delta_median_ms": statistics.median(pooled_tpot),
            "tpot_one_sided_sign_test_p": _sign_p(tpot_wins, trials),
        },
        "frontier": {
            "highest_valid_offered_rate_per_s": 52,
            "official_lmcache_fatal_offered_rate_per_s": 56,
            "rate56_invalid_streams": failed["invalid_streams"],
            "rate56_failure_report": str(failure.resolve()),
        },
        "gates": gates,
        "passes": passes,
        "verdict": ("actual_vllm_pd_same_window_advantage_through_rate52"
                    if passes else "frontier_advantage_not_established"),
        "claim_boundary": (
            "One allocation and one server lifecycle per offered load. Each valid load "
            "contains 24 geometry-paired Tempo/official-LMCache requests in the same "
            "48-request client window. Actual transport bytes differ by design because "
            "Tempo may choose decoder-local recomputation. Request pairs are dependent "
            "within one allocation, and mixed-window data do not establish standalone "
            "throughput superiority. The rate56 artifact is stability evidence only."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="RATE=REPORT.json")
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for specification in args.run:
        rate, path = specification.split("=", 1)
        runs[int(rate)] = Path(path)
    if args.output.exists():
        parser.error("refusing to overwrite")
    report = analyze(runs, args.failure, args.allocation)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": report["verdict"], "pooled": report["pooled"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
