#!/usr/bin/env python3
"""Pool two successful live P/D cache-catalog lifecycles fail-closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


LOCAL = "decoder_local_recompute_or_cache"
REMOTE = "remote_prefill_live_kv"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "tempo-pd-same-server-cache-catalog-analysis-137":
        raise ValueError("unexpected cache-catalog analysis schema")
    if value["tempo"]["routes"] != {LOCAL: 32, REMOTE: 16}:
        raise ValueError("cache-catalog route geometry changed")
    reasons = value["tempo"]["reasons"]
    if reasons != {
        "same_server_tempo_measured:cache_catalog_hit_local": 32,
        "same_server_tempo_measured:cache_catalog_hit_remote": 16,
    }:
        raise ValueError("measured cache hit evidence changed")
    if not all(contract.get("cache_catalog_identity") == "stable-item-index-v136"
               for contract in value["contracts_by_sequence"]):
        raise ValueError("stable cache identity contract missing")
    return value


def _mode(value: dict, name: str) -> dict:
    return value[name]["performance"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != 2 or args.run[0].resolve() == args.run[1].resolve():
        raise ValueError("exactly two distinct lifecycle reports required")
    runs = [_load(path.resolve()) for path in args.run]
    summaries = []
    for path, value in zip(args.run, runs, strict=True):
        tempo, local, lm = (_mode(value, name)
                            for name in ("tempo", "fixed_local", "lmcache_remote"))
        pair = value["paired_tempo_minus_lmcache"]
        summaries.append({
            "path": str(path.resolve()),
            "tempo_throughput_per_s": tempo["request_throughput_per_s"],
            "local_throughput_per_s": local["request_throughput_per_s"],
            "lmcache_throughput_per_s": lm["request_throughput_per_s"],
            "throughput_gain_vs_lmcache_percent": 100 * (
                tempo["request_throughput_per_s"] / lm["request_throughput_per_s"] - 1),
            "tempo_e2e_p99_ms": tempo["e2e_ms"]["p99"],
            "local_e2e_p99_ms": local["e2e_ms"]["p99"],
            "lmcache_e2e_p99_ms": lm["e2e_ms"]["p99"],
            "e2e_p99_reduction_vs_lmcache_percent": 100 * (
                1 - tempo["e2e_ms"]["p99"] / lm["e2e_ms"]["p99"]),
            "tempo_tpot_p99_ms": tempo["tpot_ms"]["p99"],
            "lmcache_tpot_p99_ms": lm["tpot_ms"]["p99"],
            "tpot_p99_reduction_vs_lmcache_percent": 100 * (
                1 - tempo["tpot_ms"]["p99"] / lm["tpot_ms"]["p99"]),
            "paired_win_count": pair["e2e_win_count"],
            "paired_e2e_delta_median_ms": pair["e2e_delta_median_ms"],
        })
    gates = {
        "both_lifecycles_tempo_throughput_beats_lmcache": all(
            row["tempo_throughput_per_s"] > row["lmcache_throughput_per_s"]
            for row in summaries),
        "both_lifecycles_tempo_throughput_beats_local": all(
            row["tempo_throughput_per_s"] > row["local_throughput_per_s"]
            for row in summaries),
        "both_lifecycles_tempo_e2e_p99_beats_lmcache": all(
            row["tempo_e2e_p99_ms"] < row["lmcache_e2e_p99_ms"] for row in summaries),
        "both_lifecycles_tempo_e2e_p99_beats_local": all(
            row["tempo_e2e_p99_ms"] < row["local_e2e_p99_ms"] for row in summaries),
        "both_lifecycles_tempo_tpot_p99_beats_lmcache": all(
            row["tempo_tpot_p99_ms"] < row["lmcache_tpot_p99_ms"] for row in summaries),
        "both_lifecycles_at_least_25_of_48_paired_wins": all(
            row["paired_win_count"] >= 25 for row in summaries),
        "both_lifecycles_paired_median_beats_lmcache": all(
            row["paired_e2e_delta_median_ms"] < 0 for row in summaries),
    }
    output = {
        "schema": "tempo-pd-cache-catalog-reproduction-149",
        "runs": summaries,
        "median_throughput_gain_vs_lmcache_percent": statistics.median(
            row["throughput_gain_vs_lmcache_percent"] for row in summaries),
        "median_e2e_p99_reduction_vs_lmcache_percent": statistics.median(
            row["e2e_p99_reduction_vs_lmcache_percent"] for row in summaries),
        "median_tpot_p99_reduction_vs_lmcache_percent": statistics.median(
            row["tpot_p99_reduction_vs_lmcache_percent"] for row in summaries),
        "gates": gates, "passes": all(gates.values()),
        "claim_boundary": (
            "Two separate server lifecycles in one four-node A100 allocation; "
            "Qwen2.5-7B TP4 prefill + TP4 decode per pair; pinned official LMCache; "
            "arm-isolated warm keys; one warmup and two measured replicates per arm."
        ),
    }
    output["verdict"] = ("reproduced_cache_affinity_win" if output["passes"]
                         else "cache_affinity_win_not_reproduced")
    args.output.resolve().write_text(json.dumps(output, sort_keys=True, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps({"verdict": output["verdict"],
                      "failed": [key for key, passed in gates.items() if not passed]},
                     sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
