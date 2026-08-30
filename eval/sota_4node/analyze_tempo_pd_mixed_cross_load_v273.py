#!/usr/bin/env python3
"""Aggregate three same-window mixed P/D load points without micro-throughput claims."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


EXPECTED = (16, 32, 48)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _one_sided_sign_p(wins: int, trials: int) -> float:
    return sum(math.comb(trials, k) for k in range(wins, trials + 1)) / (2 ** trials)


def analyze(runs: dict[int, Path], allocation: int,
            failure: Path | None = None) -> dict:
    _require(tuple(sorted(runs)) == EXPECTED, "exact loads 16/32/48 required")
    reports = {rate: json.loads(path.resolve().read_text())
               for rate, path in runs.items()}
    pooled_e2e = []
    pooled_tpot = []
    per_load = {}
    for rate in EXPECTED:
        report = reports[rate]
        _require(report.get("schema") ==
                 "tempo-pd-mixed-request-crossover-analysis-263",
                 f"rate{rate}: schema")
        _require(report.get("allocation_id") == allocation,
                 f"rate{rate}: allocation")
        _require(report.get("passes") is True, f"rate{rate}: gates")
        _require(report.get("route_counts") == {
            "lmcache_remote": 24, "tempo_local": 19, "tempo_remote": 5},
            f"rate{rate}: routes")
        pairs = report["pairs"]
        _require(len(pairs) == 24, f"rate{rate}: pairs")
        e2e = [float(row["e2e_delta_ms"]) for row in pairs]
        tpot = [float(row["tpot_delta_ms"]) for row in pairs]
        pooled_e2e.extend(e2e)
        pooled_tpot.extend(tpot)
        per_load[str(rate)] = {
            "e2e_win_count": sum(value < 0 for value in e2e),
            "e2e_delta_median_ms": statistics.median(e2e),
            "tpot_win_count": sum(value < 0 for value in tpot),
            "tpot_delta_median_ms": statistics.median(tpot),
            "report": str(runs[rate].resolve()),
        }
    e2e_wins = sum(value < 0 for value in pooled_e2e)
    tpot_wins = sum(value < 0 for value in pooled_tpot)
    gates = {
        "every_load_passes": all(reports[rate]["passes"] for rate in EXPECTED),
        "every_load_e2e_majority": all(
            per_load[str(rate)]["e2e_win_count"] >= 13 for rate in EXPECTED),
        "pooled_e2e_win_fraction_ge_80pct": e2e_wins >= math.ceil(0.8 * 72),
        "pooled_e2e_median_improves": statistics.median(pooled_e2e) < 0,
        "pooled_tpot_win_fraction_ge_80pct": tpot_wins >= math.ceil(0.8 * 72),
        "pooled_tpot_median_improves": statistics.median(pooled_tpot) < 0,
    }
    failure_summary = None
    if failure is not None:
        value = json.loads(failure.resolve().read_text())
        _require(value.get("schema") ==
                 "tempo-pd-mixed-lmcache-failure-analysis-272",
                 "failure schema")
        _require(value.get("allocation_id") == allocation, "failure allocation")
        _require(value.get("verdict") ==
                 "official_lmcache_concurrent_retrieval_fatal",
                 "failure verdict")
        failure_summary = {
            "request_rate_per_s": value["request_rate_per_s"],
            "max_workers": value["max_workers"],
            "invalid_streams": value["invalid_streams"],
            "fatal_signature": value["fatal_signature"],
            "report": str(failure.resolve()),
        }
    passes = all(gates.values())
    return {
        "schema": "tempo-pd-mixed-cross-load-analysis-273",
        "allocation_id": allocation,
        "policy": "qwen25-7b-tp4x2-warm-affinity-8",
        "topology": "actual-vllm-qwen25-7b-tp4-prefill-plus-tp4-decode-two-replicas",
        "loads": per_load,
        "pooled": {
            "paired_requests": 72,
            "e2e_win_count": e2e_wins,
            "e2e_win_fraction": e2e_wins / 72,
            "e2e_delta_median_ms": statistics.median(pooled_e2e),
            "e2e_one_sided_sign_test_p": _one_sided_sign_p(e2e_wins, 72),
            "tpot_win_count": tpot_wins,
            "tpot_win_fraction": tpot_wins / 72,
            "tpot_delta_median_ms": statistics.median(pooled_tpot),
            "tpot_one_sided_sign_test_p": _one_sided_sign_p(tpot_wins, 72),
        },
        "rate56_stability_failure": failure_summary,
        "gates": gates,
        "passes": passes,
        "verdict": ("cross_load_same_window_request_advantage" if passes else
                    "cross_load_advantage_not_established"),
        "claim_boundary": (
            "One allocation, three independent server lifecycles at offered rates "
            "16/32/48, with 24 paired Tempo/official-LMCache requests in each shared "
            "window. Sign tests treat request pairs as observations but do not remove "
            "within-allocation or workload dependence. No standalone throughput claim "
            "is made from a mixed window. The separate rate56 artifact is stability "
            "evidence only because its streams are invalid."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="RATE=REPORT.json")
    parser.add_argument("--failure", type=Path)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for spec in args.run:
        rate, path = spec.split("=", 1)
        runs[int(rate)] = Path(path)
    if args.output.exists():
        parser.error("refusing to overwrite")
    report = analyze(runs, args.allocation, args.failure)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": report["verdict"],
                      "pooled": report["pooled"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
