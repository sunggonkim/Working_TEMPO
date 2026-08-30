#!/usr/bin/env python3
"""Fail-closed stock-LMCache versus TEMPO NIXL hot-path analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any

from eval.sota_4node import analyze_tempo_pd_performance_v1 as base
from lmcache.v1.transfer_channel.tempo_nixl_hotpath import SCHEMA as HOTPATH_SCHEMA


SCHEMA = "lmcache-nixl-hotpath-ab-analysis-1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _percent(candidate: float, baseline: float) -> float:
    _require(math.isfinite(candidate) and math.isfinite(baseline) and baseline > 0,
             "percent comparison requires finite positive baseline")
    return (candidate / baseline - 1.0) * 100.0


def _telemetry(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("node-*/nixl-hotpath-*.json"))
    _require(paths, "optimized run produced no hot-path telemetry")
    rows = [_load(path) for path in paths]
    for path, row in zip(paths, rows, strict=True):
        _require(row.get("schema") == HOTPATH_SCHEMA, f"{path}: schema mismatch")
        _require(row.get("inflight_handles_at_close") == 0,
                 f"{path}: transfer still in flight at close")
        stats = row.get("stats")
        _require(isinstance(stats, dict), f"{path}: stats missing")
        transfers = stats.get("transfer_count")
        made = stats.get("make_count")
        reused = stats.get("reuse_count")
        released = stats.get("release_count")
        _require(all(type(value) is int and value >= 0
                     for value in (transfers, made, reused, released)),
                 f"{path}: invalid counters")
        _require(transfers > 0 and made + reused == transfers,
                 f"{path}: handle accounting mismatch")
        _require(released == made, f"{path}: prepared handles leaked")
    transfers = sum(row["stats"]["transfer_count"] for row in rows)
    made = sum(row["stats"]["make_count"] for row in rows)
    reused = sum(row["stats"]["reuse_count"] for row in rows)
    elapsed = [value / 1_000_000.0 for row in rows
               for value in row["stats"]["elapsed_ns"]]
    _require(len(elapsed) == transfers, "telemetry elapsed sample coverage mismatch")
    return {
        "process_count": len(rows),
        "transfer_count": transfers,
        "make_count": made,
        "reuse_count": reused,
        "cache_hit_fraction": reused / transfers,
        "object_count": sum(row["stats"]["object_count"] for row in rows),
        "yield_poll_count": sum(row["stats"]["yield_poll_count"] for row in rows),
        "sleep_poll_count": sum(row["stats"]["sleep_poll_count"] for row in rows),
        "transfer_elapsed_ms": base._distribution(elapsed),
        "artifacts": [str(path) for path in paths],
    }


def analyze(stock_raw: dict[str, Any], optimized_raw: dict[str, Any], telemetry_root: Path) -> dict[str, Any]:
    stock = base._parse_run(
        "stock_lmcache_remote", stock_raw,
        ttft_slo_ms=1000.0, tpot_slo_ms=100.0, e2e_slo_ms=3000.0,
    )
    optimized = base._parse_run(
        "tempo_nixl_remote", optimized_raw,
        ttft_slo_ms=1000.0, tpot_slo_ms=100.0, e2e_slo_ms=3000.0,
    )
    _require(stock["mode"] == optimized["mode"] == "lmcache_always_remote",
             "both arms must use the exact always-remote route")
    same_model = stock["model_config_sha256"] == optimized["model_config_sha256"]
    same_workload = (
        stock["workload_sha256"] == optimized["workload_sha256"]
        and stock["_contracts"] == optimized["_contracts"]
    )
    same_outputs = stock["_outputs"] == optimized["_outputs"]
    correctness = same_model and same_workload and same_outputs
    paired = base._paired(optimized, stock) if correctness else None
    hotpath = _telemetry(telemetry_root)
    stock_perf = stock["performance"]
    optimized_perf = optimized["performance"]
    gates = {
        "same_model_workload_outputs": correctness,
        "all_requests_remote_in_both_arms": (
            stock["routes"] == optimized["routes"] == {
                "remote_prefill_live_kv": stock["request_count"]
            }
        ),
        "telemetry_exact_and_leak_free": True,
        "prepared_handle_cache_has_hits": hotpath["reuse_count"] > 0,
        "paired_e2e_wins_at_least_two_thirds": (
            paired is not None
            and paired["e2e_win_count"] >= math.ceil(stock["request_count"] * 2 / 3)
        ),
        "paired_e2e_median_improves": (
            paired is not None and paired["e2e_delta_median_ms"] < 0
        ),
        "request_goodput_improves": (
            optimized_perf["slo_goodput"]["request_goodput_per_s"]
            > stock_perf["slo_goodput"]["request_goodput_per_s"]
        ),
        "tpot_p99_not_regressed_over_5_percent": (
            optimized_perf["tpot_ms"]["p99"] <= stock_perf["tpot_ms"]["p99"] * 1.05
        ),
    }
    passes = all(gates.values())
    public_stock = {key: value for key, value in stock.items() if not key.startswith("_")}
    public_optimized = {
        key: value for key, value in optimized.items() if not key.startswith("_")
    }
    return {
        "schema": SCHEMA,
        "correctness_valid": correctness,
        "stock_lmcache": public_stock,
        "tempo_nixl_hotpath": public_optimized,
        "hotpath_telemetry": hotpath,
        "paired": paired,
        "relative_percent": {
            metric: _percent(optimized_perf[metric]["p50"], stock_perf[metric]["p50"])
            for metric in ("ttft_ms", "tpot_ms", "e2e_ms")
        } | {
            "request_goodput": _percent(
                optimized_perf["slo_goodput"]["request_goodput_per_s"],
                stock_perf["slo_goodput"]["request_goodput_per_s"],
            )
        },
        "gates": gates,
        "passes_hotpath_continuation_gate": passes,
        "verdict": "continue_hotpath" if passes else "revise_or_stop_hotpath",
        "claim_boundary": (
            "actual-vLLM remote-only component comparison on the frozen workload; "
            "not a production P/D SOTA claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(_load(args.stock), _load(args.optimized), args.telemetry_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
