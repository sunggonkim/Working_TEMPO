#!/usr/bin/env python3
"""Analyze v2 snapshot telemetry with the v1 remote-only A/B semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.sota_4node import analyze_lmcache_nixl_hotpath_ab_v1 as v1
from eval.sota_4node import analyze_tempo_pd_performance_v1 as perf
from lmcache.v1.transfer_channel.tempo_nixl_hotpath_v2 import SCHEMA


def _telemetry(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("node-*/nixl-hotpath-*.json"))
    v1._require(paths, "optimized run produced no hot-path snapshot telemetry")
    rows = [v1._load(path) for path in paths]
    for path, row in zip(paths, rows, strict=True):
        v1._require(row.get("schema") == SCHEMA, f"{path}: schema mismatch")
        stats = row.get("stats")
        v1._require(isinstance(stats, dict), f"{path}: stats missing")
        transfers = stats.get("transfer_count")
        made = stats.get("make_count")
        reused = stats.get("reuse_count")
        released = stats.get("release_count")
        idle = row.get("idle_handles_at_snapshot")
        inflight = row.get("inflight_handles_at_snapshot")
        v1._require(all(type(value) is int and value >= 0 for value in
                       (transfers, made, reused, released, idle, inflight)),
                    f"{path}: invalid counters")
        v1._require(transfers > 0 and made + reused == transfers,
                    f"{path}: handle accounting mismatch")
        v1._require(inflight == 0, f"{path}: snapshot captured an in-flight handle")
        v1._require(released + idle == made,
                    f"{path}: handle is neither cached nor released")
    transfers = sum(row["stats"]["transfer_count"] for row in rows)
    made = sum(row["stats"]["make_count"] for row in rows)
    reused = sum(row["stats"]["reuse_count"] for row in rows)
    elapsed = [value / 1_000_000.0 for row in rows
               for value in row["stats"]["elapsed_ns"]]
    v1._require(len(elapsed) == transfers, "elapsed sample coverage mismatch")
    return {
        "process_count": len(rows),
        "transfer_count": transfers,
        "make_count": made,
        "reuse_count": reused,
        "cache_hit_fraction": reused / transfers,
        "object_count": sum(row["stats"]["object_count"] for row in rows),
        "yield_poll_count": sum(row["stats"]["yield_poll_count"] for row in rows),
        "sleep_poll_count": sum(row["stats"]["sleep_poll_count"] for row in rows),
        "transfer_elapsed_ms": perf._distribution(elapsed),
        "managed_idle_handles_at_snapshot": sum(
            row["idle_handles_at_snapshot"] for row in rows
        ),
        "artifacts": [str(path) for path in paths],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = v1._telemetry
    try:
        v1._telemetry = _telemetry
        result = v1.analyze(v1._load(args.stock), v1._load(args.optimized), args.telemetry_root)
    finally:
        v1._telemetry = original
    result["schema"] = "lmcache-nixl-hotpath-ab-analysis-2"
    result["telemetry_persistence"] = "atomic local snapshot after each transfer"
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
