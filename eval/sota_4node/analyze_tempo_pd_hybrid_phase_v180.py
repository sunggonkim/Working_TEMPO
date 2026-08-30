#!/usr/bin/env python3
"""Join cold-miss and reproduced warm-hit evidence for HybridPDController."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LOCAL = "decoder_local_recompute_or_cache"
REMOTE = "remote_prefill_live_kv"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: object required")
    return value


def _cold_summary(value: dict) -> tuple[dict, dict]:
    if value.get("schema") != "tempo-pd-same-server-balanced-analysis-71":
        raise ValueError("cold report schema mismatch")
    tempo = value["tempo"]
    local = value["fixed_local"]
    lmcache = value["lmcache_remote"]
    tp = tempo["performance"]
    lp = local["performance"]
    mp = lmcache["performance"]
    paired = value["paired_tempo_minus_lmcache"]
    summary = {
        "tempo_routes": tempo["routes"],
        "tempo_reasons": tempo["reasons"],
        "tempo_throughput_per_s": tp["request_throughput_per_s"],
        "local_throughput_per_s": lp["request_throughput_per_s"],
        "lmcache_throughput_per_s": mp["request_throughput_per_s"],
        "throughput_gain_vs_lmcache_percent": 100 * (
            tp["request_throughput_per_s"] / mp["request_throughput_per_s"] - 1),
        "throughput_regression_vs_local_percent": 100 * (
            1 - tp["request_throughput_per_s"] / lp["request_throughput_per_s"]),
        "tempo_e2e_p99_ms": tp["e2e_ms"]["p99"],
        "local_e2e_p99_ms": lp["e2e_ms"]["p99"],
        "lmcache_e2e_p99_ms": mp["e2e_ms"]["p99"],
        "e2e_p99_reduction_vs_lmcache_percent": 100 * (
            1 - tp["e2e_ms"]["p99"] / mp["e2e_ms"]["p99"]),
        "e2e_p99_regression_vs_local_percent": 100 * (
            tp["e2e_ms"]["p99"] / lp["e2e_ms"]["p99"] - 1),
        "tempo_tpot_p99_ms": tp["tpot_ms"]["p99"],
        "local_tpot_p99_ms": lp["tpot_ms"]["p99"],
        "lmcache_tpot_p99_ms": mp["tpot_ms"]["p99"],
        "tpot_p99_reduction_vs_lmcache_percent": 100 * (
            1 - tp["tpot_ms"]["p99"] / mp["tpot_ms"]["p99"]),
        "tpot_p99_regression_vs_local_percent": 100 * (
            tp["tpot_ms"]["p99"] / lp["tpot_ms"]["p99"] - 1),
        "paired_win_count_vs_lmcache": paired["e2e_win_count"],
        "paired_e2e_delta_median_ms_vs_lmcache": paired["e2e_delta_median_ms"],
    }
    gates = {
        "all_cold_misses_choose_local": tempo["routes"] == {LOCAL: 48},
        "lmcache_baseline_is_all_remote": lmcache["routes"] == {REMOTE: 48},
        "cold_throughput_beats_lmcache": (
            tp["request_throughput_per_s"] > mp["request_throughput_per_s"]),
        "cold_e2e_p99_beats_lmcache": tp["e2e_ms"]["p99"] < mp["e2e_ms"]["p99"],
        "cold_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < mp["tpot_ms"]["p99"],
        "cold_paired_majority_beats_lmcache": paired["e2e_win_count"] >= 25,
        "cold_paired_median_beats_lmcache": paired["e2e_delta_median_ms"] < 0,
        "cold_throughput_within_2pct_local_oracle": (
            tp["request_throughput_per_s"] >= .98 * lp["request_throughput_per_s"]),
        "cold_e2e_p99_within_2pct_local_oracle": (
            tp["e2e_ms"]["p99"] <= 1.02 * lp["e2e_ms"]["p99"]),
        "cold_tpot_p99_within_2pct_local_oracle": (
            tp["tpot_ms"]["p99"] <= 1.02 * lp["tpot_ms"]["p99"]),
    }
    return summary, gates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--warm-reproduction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cold, cold_gates = _cold_summary(_load(args.cold.resolve()))
    warm = _load(args.warm_reproduction.resolve())
    if warm.get("schema") != "tempo-pd-production-cross-allocation-reproduction-169":
        raise ValueError("warm reproduction schema mismatch")
    warm_gates = {
        "warm_cross_allocation_reproduction_passes": warm.get("passes") is True,
        "warm_cross_allocation_verdict_exact": (
            warm.get("verdict") == "cross_allocation_production_win"),
    }
    gates = {**cold_gates, **warm_gates}
    result = {
        "schema": "tempo-pd-hybrid-phase-analysis-180",
        "controller": "tempo-pd-hybrid-controller-1",
        "cold_miss": cold,
        "warm_hit_reproduction": warm,
        "gates": gates,
        "passes": all(gates.values()),
        "claim_boundary": (
            "Actual Qwen2.5-7B vLLM TP4+TP4 P/D on four A100 nodes. "
            "Cold misses use the calibrated local branch; warm cache items use stable "
            "cache-affinity placement. Warm performance is reproduced across two allocations."
        ),
    }
    result["verdict"] = (
        "phase_aware_hybrid_pd_validated" if result["passes"]
        else "phase_aware_hybrid_pd_needs_revision")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    args.output.resolve().write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"],
                      "failed": [key for key, passed in gates.items() if not passed]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
