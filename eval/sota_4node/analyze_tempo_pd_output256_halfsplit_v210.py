#!/usr/bin/env python3
"""Validate the stable 20-local/4-remote output256 warm partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_output256_balanced_v208 as prior


base = prior.base


def _partition(value: dict, count: int, reason: str) -> bool:
    rows = value.get("router_decisions")
    if not isinstance(rows, list) or len(rows) != count:
        return False
    for row in rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or "-cache-item-" not in request_id:
            return False
        suffix = request_id.rsplit("-cache-item-", 1)[1]
        if len(suffix) != 2 or not suffix.isdigit():
            return False
        remote = row.get("prompt_tokens") == 2048 and int(suffix) % 2 == 0
        if row.get("reason") != reason:
            return False
        if row.get("route") != (base.REMOTE if remote else base.LOCAL):
            return False
    return (sum(row.get("route") == base.LOCAL for row in rows) == 20
            and sum(row.get("route") == base.REMOTE for row in rows) == 4)


def analyze(root: Path, allocation: int) -> dict:
    root = root.resolve()
    stage = root / "tempo_credit_admission"
    cold = base._load(stage / "hybrid_cold_transition.raw.json")
    seed = base._load(stage / "same_server_balanced_warm/02_tempo_r0.raw.json")
    hit0 = base._load(stage / "same_server_balanced_measured/01_tempo_r0.raw.json")
    hit1 = base._load(stage / "same_server_balanced_measured/04_tempo_r1.raw.json")
    final = base._load(root / "hybrid_controller_final.json")
    tempo, local, lmcache = (final[key] for key in
                             ("tempo", "fixed_local", "lmcache_remote"))
    tp, lp, mp = (row["performance"] for row in (tempo, local, lmcache))
    remote_pair = final["paired_tempo_minus_lmcache"]
    local_pair = final["paired_tempo_minus_local"]
    gates = {
        "cold_24_valid_local": (
            base._valid_requests(cold, 24)
            and base._route_contract(
                cold, 24,
                "same_server_tempo_cold:hybrid_cold:output256_direct_local_fast_path")),
        "seed_partition_20_local_4_remote": (
            base._valid_requests(seed, 24)
            and _partition(seed, 24,
                           "same_server_tempo_warm:cache_affinity_warm_seed")),
        "hit_partitions_40_local_8_remote": (
            base._valid_requests(hit0, 24) and base._valid_requests(hit1, 24)
            and _partition(hit0, 24,
                           "same_server_tempo_measured:cache_affinity_warm_hit")
            and _partition(hit1, 24,
                           "same_server_tempo_measured:cache_affinity_warm_hit")),
        "fixed_baselines_exact": (
            local.get("routes") == {base.LOCAL: 48}
            and lmcache.get("routes") == {base.REMOTE: 48}),
        "tempo_routes_40_local_8_remote": (
            tempo.get("routes") == {base.LOCAL: 40, base.REMOTE: 8}),
        "all_tempo_slo_valid": tp["slo_goodput"]["success_fraction"] == 1.0,
        "tempo_throughput_beats_lmcache": (
            tp["request_throughput_per_s"] > mp["request_throughput_per_s"]),
        "tempo_e2e_p99_beats_lmcache": tp["e2e_ms"]["p99"] < mp["e2e_ms"]["p99"],
        "tempo_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < mp["tpot_ms"]["p99"],
        "tempo_paired_majority_beats_lmcache": (
            remote_pair["e2e_win_count"] >= 25
            and remote_pair["e2e_delta_median_ms"] < 0.0),
        "tempo_throughput_retains_98pct_local": (
            tp["request_throughput_per_s"] >= 0.98 * lp["request_throughput_per_s"]),
        "tempo_e2e_p99_within_2pct_local": (
            tp["e2e_ms"]["p99"] <= 1.02 * lp["e2e_ms"]["p99"]),
        "tempo_tpot_p99_within_10pct_local": (
            tp["tpot_ms"]["p99"] <= 1.10 * lp["tpot_ms"]["p99"]),
        "tempo_paired_local_noninferior": (
            local_pair["e2e_win_count"] >= 24
            and local_pair["e2e_delta_median_ms"] <= 20.0),
    }
    summary = {
        "tempo_throughput_per_s": tp["request_throughput_per_s"],
        "local_throughput_per_s": lp["request_throughput_per_s"],
        "lmcache_throughput_per_s": mp["request_throughput_per_s"],
        "throughput_gain_vs_lmcache_percent": 100.0 * (
            tp["request_throughput_per_s"] / mp["request_throughput_per_s"] - 1.0),
        "tempo_e2e_p99_ms": tp["e2e_ms"]["p99"],
        "local_e2e_p99_ms": lp["e2e_ms"]["p99"],
        "lmcache_e2e_p99_ms": mp["e2e_ms"]["p99"],
        "tempo_tpot_p99_ms": tp["tpot_ms"]["p99"],
        "local_tpot_p99_ms": lp["tpot_ms"]["p99"],
        "lmcache_tpot_p99_ms": mp["tpot_ms"]["p99"],
        "paired_lmcache_win_count": remote_pair["e2e_win_count"],
        "paired_lmcache_delta_median_ms": remote_pair["e2e_delta_median_ms"],
        "paired_local_win_count": local_pair["e2e_win_count"],
        "paired_local_delta_median_ms": local_pair["e2e_delta_median_ms"],
    }
    result = {
        "schema": "tempo-pd-output256-halfsplit-analysis-210",
        "allocation_id": allocation,
        "root": str(root),
        "controller": "tempo-pd-hybrid-controller-2",
        "policy": "qwen25-7b-tp4x2-warm-affinity-4",
        "geometry": {"prompt_tokens": [512, 1230, 2048], "output_tokens": 256},
        "routes": {"cold_local": 24, "seed_local": 20, "seed_remote": 4,
                   "hit_local": 40, "hit_remote": 8},
        "summary": summary,
        "gates": gates,
        "claim_boundary": (
            "One four-node A100 allocation, actual Qwen2.5-7B vLLM TP4+TP4 P/D, "
            "stable half split of the output256/prompt2048 bucket; pinned LMCache "
            "remote and fixed-local baselines."),
    }
    result["passes"] = all(gates.values())
    result["verdict"] = (
        "output256_halfsplit_policy_validated" if result["passes"]
        else "output256_halfsplit_policy_needs_revision")
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
