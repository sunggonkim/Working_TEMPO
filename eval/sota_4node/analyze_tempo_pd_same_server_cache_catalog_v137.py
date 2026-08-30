#!/usr/bin/env python3
"""Fail-closed analysis for the warm-hit cache-catalog hybrid."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

from eval.sota_4node import analyze_tempo_pd_same_server_warm_reuse_v132 as warm


LOCAL_ROUTE = "decoder_local_recompute_or_cache"
REMOTE_ROUTE = "remote_prefill_live_kv"


def _argument(name: str) -> Path:
    return Path(sys.argv[sys.argv.index(name) + 1]).resolve()


def _route_stats(pairs: list[dict], route: str) -> dict:
    rows = [row for row in pairs if row.get("route") == route]
    deltas = [float(row["e2e_delta_ms"]) for row in rows]
    return {
        "count": len(rows),
        "win_count": sum(value < 0.0 for value in deltas),
        "e2e_delta_median_ms": statistics.median(deltas) if deltas else None,
        "e2e_delta_max_ms": max(deltas) if deltas else None,
    }


def main() -> int:
    status = warm.main()
    output = _argument("--output")
    value = json.loads(output.read_text(encoding="utf-8"))
    tempo = value["tempo"]
    local = value["fixed_local"]
    remote = value["lmcache_remote"]
    tp = tempo["performance"]
    lp = local["performance"]
    rp = remote["performance"]
    pair_local = value["paired_tempo_minus_local"]
    pair_remote = value["paired_tempo_minus_lmcache"]
    local_stats = _route_stats(pair_local["pairs"], LOCAL_ROUTE)
    remote_stats = _route_stats(pair_remote["pairs"], REMOTE_ROUTE)
    reasons = tempo["reasons"]
    reason_local = sum(count for reason, count in reasons.items()
                       if reason.endswith("cache_catalog_hit_local"))
    reason_remote = sum(count for reason, count in reasons.items()
                        if reason.endswith("cache_catalog_hit_remote"))
    contracts = value["contracts_by_sequence"]
    gates = {
        "arm_isolated_warm_reuse_contract": value["gates"][
            "arm_isolated_warm_reuse_contract"],
        "stable_cache_catalog_identity": all(
            contract.get("cache_catalog_identity") == "stable-item-index-v136"
            for contract in contracts),
        "exact_normalized_workload_schedule_outputs": value["gates"][
            "exact_normalized_workload_schedule_outputs"],
        "fixed_local_routes_48_local": local["routes"] == {LOCAL_ROUTE: 48},
        "lmcache_routes_48_remote": remote["routes"] == {REMOTE_ROUTE: 48},
        "tempo_routes_32_local_16_remote": tempo["routes"] == {
            LOCAL_ROUTE: 32, REMOTE_ROUTE: 16},
        "measured_reasons_are_32_local_16_remote_cache_hits": (
            reason_local == 32 and reason_remote == 16
            and sum(reasons.values()) == 48),
        "all_tempo_requests_slo_valid": tp["slo_goodput"]["success_fraction"] == 1.0,
        "tempo_goodput_retains_95pct_local": (
            tp["slo_goodput"]["request_goodput_per_s"] >=
            0.95 * lp["slo_goodput"]["request_goodput_per_s"]),
        "tempo_goodput_beats_lmcache": (
            tp["slo_goodput"]["request_goodput_per_s"] >
            rp["slo_goodput"]["request_goodput_per_s"]),
        "tempo_throughput_beats_lmcache": (
            tp["request_throughput_per_s"] > rp["request_throughput_per_s"]),
        "selected_local_pairs_noninferior": (
            local_stats["count"] == 32 and local_stats["win_count"] >= 16
            and local_stats["e2e_delta_median_ms"] <= 10.0),
        "selected_remote_pairs_noninferior": (
            remote_stats["count"] == 16 and remote_stats["win_count"] >= 8
            and remote_stats["e2e_delta_median_ms"] <= 10.0),
        "tempo_overall_paired_beats_lmcache": (
            pair_remote["e2e_win_count"] >= 25
            and pair_remote["e2e_delta_median_ms"] < 0.0),
        "tempo_e2e_p99_within_5pct_local": (
            tp["e2e_ms"]["p99"] <= 1.05 * lp["e2e_ms"]["p99"]),
        "tempo_e2e_p99_beats_lmcache": tp["e2e_ms"]["p99"] < rp["e2e_ms"]["p99"],
        "tempo_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < rp["tpot_ms"]["p99"],
    }
    value["schema"] = "tempo-pd-same-server-cache-catalog-analysis-137"
    value["route_matched_pairs"] = {"local": local_stats, "remote": remote_stats}
    value["gates"] = gates
    value["passes"] = all(gates.values())
    value["verdict"] = (
        "promising_warm_hit_cache_catalog_hybrid" if value["passes"]
        else "revise_warm_hit_cache_catalog_hybrid")
    value["claim_boundary"] = (
        "One live Qwen2.5-7B TP4+TP4 P/D lifecycle; arm-isolated stable cache "
        "items; one warmup plus two measured replicates; 48 req/s."
    )
    output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"],
                      "failed": [key for key, passed in gates.items() if not passed]},
                     sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
