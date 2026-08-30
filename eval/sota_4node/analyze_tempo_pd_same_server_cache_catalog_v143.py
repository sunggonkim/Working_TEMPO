#!/usr/bin/env python3
"""Workload-level analysis for the low-KV 32-local/16-remote catalog."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from eval.sota_4node import analyze_tempo_pd_same_server_cache_catalog_v137 as prior


def main() -> int:
    status = prior.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    value = json.loads(output.read_text(encoding="utf-8"))
    tempo, local, remote = value["tempo"], value["fixed_local"], value["lmcache_remote"]
    tp, lp, rp = (row["performance"] for row in (tempo, local, remote))
    pair = value["paired_tempo_minus_lmcache"]
    matched = value["route_matched_pairs"]
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
        "tempo_routes_32_local_16_remote": tempo["routes"] == {
            prior.LOCAL_ROUTE: 32, prior.REMOTE_ROUTE: 16},
        "measured_reasons_are_exact_cache_hits": (
            local_reason == 32 and remote_reason == 16 and sum(reasons.values()) == 48),
        "all_tempo_requests_slo_valid": tp["slo_goodput"]["success_fraction"] == 1.0,
        "tempo_goodput_retains_95pct_local": (
            tp["slo_goodput"]["request_goodput_per_s"] >=
            0.95 * lp["slo_goodput"]["request_goodput_per_s"]),
        "tempo_goodput_beats_lmcache": (
            tp["slo_goodput"]["request_goodput_per_s"] >
            rp["slo_goodput"]["request_goodput_per_s"]),
        "tempo_throughput_beats_lmcache": tp["request_throughput_per_s"] > rp["request_throughput_per_s"],
        "overall_paired_beats_lmcache": (
            pair["e2e_win_count"] >= 25 and pair["e2e_delta_median_ms"] < 0.0),
        "selected_local_median_noninferior": (
            matched["local"]["count"] == 32
            and matched["local"]["e2e_delta_median_ms"] <= 10.0),
        "selected_remote_sacrifice_under_100ms": (
            matched["remote"]["count"] == 16
            and matched["remote"]["e2e_delta_median_ms"] <= 100.0),
        "tempo_e2e_p99_within_5pct_local": tp["e2e_ms"]["p99"] <= 1.05 * lp["e2e_ms"]["p99"],
        "tempo_e2e_p99_beats_lmcache": tp["e2e_ms"]["p99"] < rp["e2e_ms"]["p99"],
        "tempo_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < rp["tpot_ms"]["p99"],
    }
    value["schema"] = "tempo-pd-same-server-low-kv-cache-catalog-analysis-143"
    value["gates"] = gates
    value["passes"] = all(gates.values())
    value["verdict"] = ("promising_low_kv_cache_catalog"
                        if value["passes"] else "revise_low_kv_cache_catalog")
    output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"],
                      "failed": [k for k, passed in gates.items() if not passed]}, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
