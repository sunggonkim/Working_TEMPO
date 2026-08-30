#!/usr/bin/env python3
"""Analyze a two-arm local versus official-LMCache remote crossover scout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_performance_v1 as base


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def analyze(local_raw: dict, remote_raw: dict) -> dict:
    local = base._parse_run(
        "fixed_local", local_raw, ttft_slo_ms=3000,
        tpot_slo_ms=250, e2e_slo_ms=12000,
    )
    remote = base._parse_run(
        "lmcache_remote", remote_raw, ttft_slo_ms=3000,
        tpot_slo_ms=250, e2e_slo_ms=12000,
    )
    correctness = (
        local["model_config_sha256"] == remote["model_config_sha256"]
        and local["workload_sha256"] == remote["workload_sha256"]
        and local["_contracts"] == remote["_contracts"]
        and local["_outputs"] == remote["_outputs"]
    )
    paired = base._paired(remote, local) if correctness else None
    local_perf = local["performance"]
    remote_perf = remote["performance"]
    request_count = local["request_count"]
    gates = {
        "same_model_workload_schedule_outputs": correctness,
        "local_route_exact": local["routes"] == {
            "decoder_local_recompute_or_cache": request_count
        },
        "remote_route_exact": remote["routes"] == {
            "remote_prefill_live_kv": request_count
        },
        "remote_e2e_wins_at_least_two_thirds": (
            paired is not None
            and paired["e2e_win_count"] >= math.ceil(request_count * 2 / 3)
        ),
        "remote_paired_e2e_median_improves_by_5ms": (
            paired is not None and paired["e2e_delta_median_ms"] <= -5.0
        ),
        "remote_request_goodput_improves": (
            remote_perf["slo_goodput"]["request_goodput_per_s"]
            > local_perf["slo_goodput"]["request_goodput_per_s"]
        ),
        "remote_tpot_p99_within_10_percent": (
            remote_perf["tpot_ms"]["p99"] <= local_perf["tpot_ms"]["p99"] * 1.10
        ),
    }

    def public(row: dict) -> dict:
        return {key: value for key, value in row.items() if not key.startswith("_")}

    return {
        "schema": "tempo-pd-remote-crossover-analysis-9",
        "evidence": "actual_vllm_two_arm_remote_crossover_screen",
        "local": public(local),
        "official_lmcache_remote": public(remote),
        "paired_remote_minus_local": paired,
        "gates": gates,
        "remote_crossover_observed": all(gates.values()),
        "verdict": "promote_remote_regime" if all(gates.values()) else "increase_load_or_stop",
        "claim_boundary": (
            "Two-arm same-allocation screen only; a positive result must be followed "
            "by a frozen three-arm controller validation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--remote", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(_load(args.local), _load(args.remote))
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
