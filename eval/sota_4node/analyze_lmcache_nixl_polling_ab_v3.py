#!/usr/bin/env python3
"""Analyze cache-free aggressive NIXL progress against stock LMCache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_lmcache_nixl_hotpath_ab_v1 as v1
from eval.sota_4node import analyze_lmcache_nixl_hotpath_ab_v2 as v2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = v1._telemetry
    try:
        v1._telemetry = v2._telemetry
        result = v1.analyze(v1._load(args.stock), v1._load(args.optimized), args.telemetry_root)
    finally:
        v1._telemetry = original
    telemetry = result["hotpath_telemetry"]
    result["schema"] = "lmcache-nixl-polling-ab-analysis-3"
    result["optimization"] = {
        "prepared_handle_cache": "disabled after zero hits in v2",
        "completion_progress": "up to 4096 cooperative yield polls, then 100us async sleep",
    }
    result["gates"].pop("prepared_handle_cache_has_hits", None)
    result["gates"]["cache_disabled_and_every_handle_released"] = (
        telemetry["reuse_count"] == 0
        and telemetry["managed_idle_handles_at_snapshot"] == 0
        and telemetry["make_count"] == telemetry["transfer_count"]
    )
    result["passes_hotpath_continuation_gate"] = all(result["gates"].values())
    result["verdict"] = (
        "continue_polling_hotpath"
        if result["passes_hotpath_continuation_gate"]
        else "revise_or_stop_polling_hotpath"
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
