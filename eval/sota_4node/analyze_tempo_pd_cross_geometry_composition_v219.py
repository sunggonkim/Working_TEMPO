#!/usr/bin/env python3
"""Validate the mixed epoch with composition-aware output256 placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_cross_geometry_epoch_v218 as old
from tempo.pd_cache_affinity import CacheAffinityCatalog


def _partition(value: dict, count: int, reason: str) -> bool:
    rows = value.get("requests")
    if not isinstance(rows, list) or len(rows) != count:
        return False
    catalog = CacheAffinityCatalog()
    local = remote = 0
    for row in sorted(rows, key=lambda item: item.get("request_index", -1)):
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or "-cache-item-" not in request_id:
            return False
        cache_item = "cache-item-" + request_id.rsplit("-cache-item-", 1)[1]
        usage = row.get("usage")
        router = row.get("router")
        if not isinstance(usage, dict) or not isinstance(router, dict):
            return False
        try:
            placement = catalog.seed(
                cache_item, int(usage["prompt_tokens"]),
                int(row["requested_max_tokens"]))
        except (KeyError, TypeError, ValueError):
            return False
        route = placement.route.value
        if router.get("reason") != reason or router.get("route") != route:
            return False
        if route == old.base.LOCAL:
            local += 1
        elif route == old.base.REMOTE:
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
    final = old.base._load(root.resolve() / "hybrid_controller_final.json")
    tempo = final["tempo"]
    gates = result["gates"]
    seed = gates.pop("seed_partition_17_local_7_remote")
    hits = gates.pop("hit_partitions_34_local_14_remote")
    gates["seed_partition_19_local_5_remote"] = seed
    gates["hit_partitions_38_local_10_remote"] = hits
    gates.pop("tempo_routes_34_local_14_remote")
    gates["tempo_routes_38_local_10_remote"] = (
        tempo.get("routes") == {old.base.LOCAL: 38, old.base.REMOTE: 10})
    result.update({
        "schema": "tempo-pd-cross-geometry-composition-analysis-219",
        "controller": "tempo-pd-hybrid-controller-2",
        "policy": "qwen25-7b-tp4x2-warm-affinity-8",
        "routes": {"cold_local": 24, "seed_local": 19, "seed_remote": 5,
                   "hit_local": 38, "hit_remote": 10},
        "claim_boundary": (
            "One four-node A100 allocation and one actual Qwen2.5-7B vLLM "
            "TP4+TP4 P/D epoch spanning output16-256 and prompt512-4094. "
            "The output256 remote bucket is suppressed when the bounded recent "
            "warm-seed history contains another output class."),
    })
    result["passes"] = all(gates.values())
    result["verdict"] = (
        "cross_geometry_composition_policy_validated" if result["passes"]
        else "cross_geometry_composition_policy_needs_revision")
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
    args.output.resolve().write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"],
                      "failed": [key for key, passed in result["gates"].items()
                                 if not passed]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
