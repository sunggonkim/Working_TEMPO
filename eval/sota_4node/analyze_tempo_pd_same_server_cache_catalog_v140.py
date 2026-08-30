#!/usr/bin/env python3
"""Analysis for the 36-local/12-remote cache-catalog revision."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from eval.sota_4node import analyze_tempo_pd_same_server_cache_catalog_v137 as prior


def main() -> int:
    status = prior.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    value = json.loads(output.read_text(encoding="utf-8"))
    tempo = value["tempo"]
    local = value["fixed_local"]
    remote = value["lmcache_remote"]
    tp, lp, rp = (row["performance"] for row in (tempo, local, remote))
    pair_remote = value["paired_tempo_minus_lmcache"]
    local_stats = value["route_matched_pairs"]["local"]
    remote_stats = value["route_matched_pairs"]["remote"]
    reasons = tempo["reasons"]
    local_reason = sum(v for k, v in reasons.items() if k.endswith("cache_catalog_hit_local"))
    remote_reason = sum(v for k, v in reasons.items() if k.endswith("cache_catalog_hit_remote"))
    gates = {
        "arm_isolated_stable_cache_catalog": (
            value["gates"]["arm_isolated_warm_reuse_contract"]
            and value["gates"]["stable_cache_catalog_identity"]),
        "exact_normalized_workload_schedule_outputs": value["gates"][
            "exact_normalized_workload_schedule_outputs"],
        "fixed_baselines_exact": (
            local["routes"] == {prior.LOCAL_ROUTE: 48}
            and remote["routes"] == {prior.REMOTE_ROUTE: 48}),
        "tempo_routes_36_local_12_remote": tempo["routes"] == {
            prior.LOCAL_ROUTE: 36, prior.REMOTE_ROUTE: 12},
        "measured_reasons_are_exact_cache_hits": (
            local_reason == 36 and remote_reason == 12 and sum(reasons.values()) == 48),
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
            local_stats["count"] == 36 and local_stats["win_count"] >= 18
            and local_stats["e2e_delta_median_ms"] <= 10.0),
        "selected_remote_sacrifice_bounded": (
            remote_stats["count"] == 12
            and remote_stats["e2e_delta_median_ms"] <= 125.0),
        "tempo_overall_paired_beats_lmcache": (
            pair_remote["e2e_win_count"] >= 25
            and pair_remote["e2e_delta_median_ms"] < 0.0),
        "tempo_e2e_p99_within_5pct_local": (
            tp["e2e_ms"]["p99"] <= 1.05 * lp["e2e_ms"]["p99"]),
        "tempo_e2e_p99_beats_lmcache": tp["e2e_ms"]["p99"] < rp["e2e_ms"]["p99"],
        "tempo_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < rp["tpot_ms"]["p99"],
    }
    value["schema"] = "tempo-pd-same-server-cache-catalog-analysis-140"
    value["gates"] = gates
    value["passes"] = all(gates.values())
    value["verdict"] = ("promising_cache_catalog_36_local_12_remote"
                        if value["passes"] else "revise_cache_catalog_36_local_12_remote")
    output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"],
                      "failed": [k for k, passed in gates.items() if not passed]},
                     sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
