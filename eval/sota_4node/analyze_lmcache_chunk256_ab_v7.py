#!/usr/bin/env python3
"""Compare actual-vLLM LMCache UCX chunk sizes 64 and 256."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_performance_v1 as base


def _load(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stock = base._parse_run("chunk64", _load(args.stock), ttft_slo_ms=1000,
                            tpot_slo_ms=100, e2e_slo_ms=3000)
    candidate = base._parse_run("chunk256", _load(args.candidate), ttft_slo_ms=1000,
                                tpot_slo_ms=100, e2e_slo_ms=3000)
    correctness = (
        stock["model_config_sha256"] == candidate["model_config_sha256"]
        and stock["workload_sha256"] == candidate["workload_sha256"]
        and stock["_contracts"] == candidate["_contracts"]
        and stock["_outputs"] == candidate["_outputs"]
    )
    configs = sorted(args.config_root.glob("node-*-lmcache.yaml"))
    config_valid = len(configs) == 4 and all(
        "chunk_size: 256" in path.read_text(encoding="utf-8")
        and "nixl_backends: [UCX]" in path.read_text(encoding="utf-8")
        for path in configs
    )
    paired = base._paired(candidate, stock) if correctness else None
    stock_perf = stock["performance"]
    candidate_perf = candidate["performance"]
    gates = {
        "same_model_workload_outputs": correctness,
        "all_requests_remote": stock["routes"] == candidate["routes"] == {
            "remote_prefill_live_kv": stock["request_count"]
        },
        "chunk256_ucx_config_exact": config_valid,
        "paired_e2e_wins_at_least_two_thirds": (
            paired is not None and paired["e2e_win_count"] >= math.ceil(stock["request_count"] * 2 / 3)
        ),
        "paired_e2e_median_improves": paired is not None and paired["e2e_delta_median_ms"] < 0,
        "request_goodput_improves": (
            candidate_perf["slo_goodput"]["request_goodput_per_s"]
            > stock_perf["slo_goodput"]["request_goodput_per_s"]
        ),
        "tpot_p99_not_regressed_over_5_percent": (
            candidate_perf["tpot_ms"]["p99"] <= stock_perf["tpot_ms"]["p99"] * 1.05
        ),
    }
    def public(row): return {key: value for key, value in row.items() if not key.startswith("_")}
    def percent(value, reference): return (value / reference - 1.0) * 100.0
    result = {
        "schema": "lmcache-chunk256-ab-analysis-7",
        "correctness_valid": correctness,
        "stock_chunk64": public(stock),
        "candidate_chunk256": public(candidate),
        "paired": paired,
        "relative_percent": {
            metric: percent(candidate_perf[metric]["p50"], stock_perf[metric]["p50"])
            for metric in ("ttft_ms", "tpot_ms", "e2e_ms")
        } | {"request_goodput": percent(
            candidate_perf["slo_goodput"]["request_goodput_per_s"],
            stock_perf["slo_goodput"]["request_goodput_per_s"],
        )},
        "gates": gates,
        "passes_chunk_continuation_gate": all(gates.values()),
        "verdict": "continue_chunk256" if all(gates.values()) else "stop_chunk256",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": gates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
