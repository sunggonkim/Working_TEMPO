#!/usr/bin/env python3
"""Aggregate the frozen policy8 mixed-workload evidence at rates 48 and 56."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def analyze(rate48: Path, rate56: Path) -> dict:
    runs = []
    for rate, path in ((48, rate48), (56, rate56)):
        value = json.loads(path.read_text())
        if value.get("schema") != "tempo-pd-composition-headtohead-analysis-236":
            raise ValueError(f"rate{rate}: schema changed")
        if not value.get("aggregate_primary_passes"):
            raise ValueError(f"rate{rate}: aggregate primary gates failed")
        summary = value["summary"]
        runs.append({
            "rate_per_s": rate,
            "path": str(path),
            "throughput_gain_vs_lmcache_percent": float(summary["throughput_gain_vs_lmcache_percent"]),
            "e2e_p99_reduction_vs_lmcache_percent": float(summary["e2e_p99_reduction_vs_lmcache_percent"]),
            "tpot_p99_reduction_vs_lmcache_percent": float(summary["tpot_p99_reduction_vs_lmcache_percent"]),
            "paired_win_count": int(summary["paired_lmcache_win_count"]),
            "paired_delta_median_ms": float(summary["paired_lmcache_delta_median_ms"]),
            "paired_gate_passes": bool(value["paired_request_gate_passes"]),
        })
    keys = (
        "throughput_gain_vs_lmcache_percent",
        "e2e_p99_reduction_vs_lmcache_percent",
        "tpot_p99_reduction_vs_lmcache_percent",
    )
    medians = {key: median(run[key] for run in runs) for key in keys}
    aggregate = all(run[key] > 0.0 for run in runs for key in keys)
    paired = all(run["paired_gate_passes"] for run in runs)
    return {
        "schema": "tempo-pd-composition-cross-load-analysis-244",
        "policy": "qwen25-7b-tp4x2-warm-affinity-8",
        "verdict": "aggregate_advantage_across_rates_with_paired_noise" if aggregate and not paired else "cross_load_inconclusive",
        "aggregate_primary_passes_both_rates": aggregate,
        "paired_request_gate_passes_both_rates": paired,
        "median_gains": medians,
        "runs": runs,
        "claim_boundary": (
            "Two load points in separate live-server lifecycles of one four-node allocation. "
            "Aggregate primary metrics pass at both rates; request-paired majority does not."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate48", type=Path, required=True)
    parser.add_argument("--rate56", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing overwrite: {args.output}")
    result = analyze(args.rate48, args.rate56)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": result["verdict"],
        "aggregate_primary_passes_both_rates": result["aggregate_primary_passes_both_rates"],
        "paired_request_gate_passes_both_rates": result["paired_request_gate_passes_both_rates"],
        "median_gains": result["median_gains"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
