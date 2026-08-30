#!/usr/bin/env python3
"""Corrected metadata validation for the v234 composition head-to-head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_cross_geometry_composition_v219 as old
from tempo.pd_cache_affinity import CacheAffinityCatalog


def _partition(value: dict, count: int, reason: str) -> bool:
    rows = value.get("requests")
    decisions = value.get("router_decisions")
    if not isinstance(rows, list) or len(rows) != count or not isinstance(decisions, list):
        return False
    decision_by_id = {row.get("request_id"): row for row in decisions}
    if len(decision_by_id) != len(decisions):
        return False
    catalog = CacheAffinityCatalog()
    local = remote = 0
    for row in sorted(rows, key=lambda item: item.get("request_index", -1)):
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or "-cache-item-" not in request_id:
            return False
        decision = decision_by_id.get(request_id)
        router = row.get("router")
        if not isinstance(decision, dict) or not isinstance(router, dict):
            return False
        cache_item = "cache-item-" + request_id.rsplit("-cache-item-", 1)[1]
        try:
            placement = catalog.seed(
                cache_item,
                int(decision["prompt_tokens"]),
                int(row["requested_max_tokens"]),
            )
        except (KeyError, TypeError, ValueError):
            return False
        route = placement.route.value
        if router.get("reason") != reason or router.get("route") != route:
            return False
        if route == old.old.base.LOCAL:
            local += 1
        elif route == old.old.base.REMOTE:
            remote += 1
        else:
            return False
    return (local, remote) == (19, 5)


def analyze(root: Path, allocation: int) -> dict:
    original = old._partition
    old._partition = _partition
    try:
        result = old.analyze(root, allocation)
    finally:
        old._partition = original
    result["schema"] = "tempo-pd-composition-headtohead-analysis-236"
    aggregate_keys = (
        "all_tempo_slo_valid",
        "cold_24_valid_and_local",
        "fixed_baselines_exact",
        "seed_partition_19_local_5_remote",
        "hit_partitions_38_local_10_remote",
        "tempo_routes_38_local_10_remote",
        "tempo_throughput_beats_lmcache",
        "tempo_e2e_p99_beats_lmcache",
        "tempo_tpot_p99_beats_lmcache",
        "tempo_throughput_retains_98pct_local",
        "tempo_e2e_p99_within_2pct_local",
        "tempo_paired_local_median_within_50ms",
    )
    result["aggregate_primary_passes"] = all(result["gates"][key] for key in aggregate_keys)
    result["paired_request_gate_passes"] = result["gates"]["tempo_paired_majority_beats_lmcache"]
    result["verdict"] = (
        "aggregate_composition_advantage_with_paired_request_noise"
        if result["aggregate_primary_passes"] and not result["paired_request_gate_passes"]
        else result["verdict"]
    )
    result["claim_boundary"] += (
        " Aggregate throughput/E2E-p99/TPOT-p99 gates pass, while the separately "
        "reported request-paired majority gate does not; do not describe the latter as a win."
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing overwrite: {args.output}")
    result = analyze(args.root, args.allocation)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": result["verdict"],
        "aggregate_primary_passes": result["aggregate_primary_passes"],
        "paired_request_gate_passes": result["paired_request_gate_passes"],
        "failed": [key for key, value in result["gates"].items() if not value],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
