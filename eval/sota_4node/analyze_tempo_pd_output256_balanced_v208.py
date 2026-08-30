#!/usr/bin/env python3
"""Validate the output256 512/1230-local, 2048-remote warm partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_output256_phase_v206 as base


def _partition(value: dict, count: int, reason: str) -> bool:
    rows = value.get("router_decisions")
    if not isinstance(rows, list) or len(rows) != count:
        return False
    return (all(row.get("reason") == reason for row in rows)
            and sum(row.get("route") == base.LOCAL for row in rows) == 16
            and sum(row.get("route") == base.REMOTE for row in rows) == 8
            and all((row.get("prompt_tokens") == 2048)
                    == (row.get("route") == base.REMOTE) for row in rows))


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
        "seed_partition_16_local_8_remote": (
            base._valid_requests(seed, 24)
            and _partition(seed, 24,
                           "same_server_tempo_warm:cache_affinity_warm_seed")),
        "hit_partitions_32_local_16_remote": (
            base._valid_requests(hit0, 24) and base._valid_requests(hit1, 24)
            and _partition(hit0, 24,
                           "same_server_tempo_measured:cache_affinity_warm_hit")
            and _partition(hit1, 24,
                           "same_server_tempo_measured:cache_affinity_warm_hit")),
        "fixed_baselines_exact": (
            local.get("routes") == {base.LOCAL: 48}
            and lmcache.get("routes") == {base.REMOTE: 48}),
        "tempo_routes_32_local_16_remote": (
            tempo.get("routes") == {base.LOCAL: 32, base.REMOTE: 16}),
        "all_tempo_slo_valid": tp["slo_goodput"]["success_fraction"] == 1.0,
        "tempo_throughput_beats_lmcache": (
            tp["request_throughput_per_s"] > mp["request_throughput_per_s"]),
        "tempo_throughput_beats_local": (
            tp["request_throughput_per_s"] > lp["request_throughput_per_s"]),
        "tempo_e2e_p99_beats_lmcache": tp["e2e_ms"]["p99"] < mp["e2e_ms"]["p99"],
        "tempo_e2e_p99_beats_local": tp["e2e_ms"]["p99"] < lp["e2e_ms"]["p99"],
        "tempo_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < mp["tpot_ms"]["p99"],
        "tempo_tpot_p99_within_5pct_local": (
            tp["tpot_ms"]["p99"] <= 1.05 * lp["tpot_ms"]["p99"]),
        "tempo_paired_majority_beats_lmcache": (
            remote_pair["e2e_win_count"] >= 25
            and remote_pair["e2e_delta_median_ms"] < 0.0),
        "tempo_paired_majority_beats_local": (
            local_pair["e2e_win_count"] >= 25
            and local_pair["e2e_delta_median_ms"] < 0.0),
    }
    def gain(numerator: float, denominator: float) -> float:
        return 100.0 * (numerator / denominator - 1.0)
    summary = {
        "tempo_throughput_per_s": tp["request_throughput_per_s"],
        "local_throughput_per_s": lp["request_throughput_per_s"],
        "lmcache_throughput_per_s": mp["request_throughput_per_s"],
        "throughput_gain_vs_lmcache_percent": gain(
            tp["request_throughput_per_s"], mp["request_throughput_per_s"]),
        "throughput_gain_vs_local_percent": gain(
            tp["request_throughput_per_s"], lp["request_throughput_per_s"]),
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
    repo = Path(__file__).resolve().parents[2]
    result = {
        "schema": "tempo-pd-output256-balanced-analysis-208",
        "allocation_id": allocation,
        "root": str(root),
        "controller": "tempo-pd-hybrid-controller-2",
        "policy": "qwen25-7b-tp4x2-warm-affinity-3",
        "geometry": {"prompt_tokens": [512, 1230, 2048], "output_tokens": 256},
        "routes": {"cold_local": 24, "seed_local": 16, "seed_remote": 8,
                   "hit_local": 32, "hit_remote": 16},
        "summary": summary,
        "gates": gates,
        "source_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in ("tempo/pd_hybrid_controller.py", "tempo/pd_cache_affinity.py",
                         "tempo/pd_workload_policy.py")},
        "claim_boundary": (
            "One four-node A100 allocation, actual Qwen2.5-7B vLLM TP4+TP4 P/D, "
            "output256 warm partition with prompt2048 remote and shorter prompts local; "
            "pinned LMCache remote and fixed-local baselines."),
    }
    result["passes"] = all(gates.values())
    result["verdict"] = (
        "output256_balanced_policy_validated" if result["passes"]
        else "output256_balanced_policy_needs_revision")
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
