#!/usr/bin/env python3
"""Validate the frozen output256 local route across MISS, SEED, and HIT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LOCAL = "decoder_local_recompute_or_cache"
REMOTE = "remote_prefill_live_kv"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: object required")
    return value


def _valid_requests(value: dict, count: int) -> bool:
    rows = value.get("requests")
    return (isinstance(rows, list) and len(rows) == count
            and all(row.get("error") is None
                    and not row.get("contract_violations") for row in rows))


def _route_contract(value: dict, count: int, reason: str) -> bool:
    rows = value.get("router_decisions")
    return (isinstance(rows, list) and len(rows) == count
            and all(row.get("route") == LOCAL
                    and row.get("reason") == reason for row in rows))


def analyze(root: Path, allocation: int) -> dict:
    root = root.resolve()
    stage = root / "tempo_credit_admission"
    cold = _load(stage / "hybrid_cold_transition.raw.json")
    seed = _load(stage / "same_server_balanced_warm/02_tempo_r0.raw.json")
    hit0 = _load(stage / "same_server_balanced_measured/01_tempo_r0.raw.json")
    hit1 = _load(stage / "same_server_balanced_measured/04_tempo_r1.raw.json")
    final = _load(root / "hybrid_controller_final.json")
    tempo = final["tempo"]
    local = final["fixed_local"]
    lmcache = final["lmcache_remote"]
    tp = tempo["performance"]
    lp = local["performance"]
    mp = lmcache["performance"]
    remote_pair = final["paired_tempo_minus_lmcache"]
    local_pair = final["paired_tempo_minus_local"]
    gates = {
        "cold_24_valid_local": (
            _valid_requests(cold, 24)
            and _route_contract(
                cold, 24,
                "same_server_tempo_cold:hybrid_cold:output256_direct_local_fast_path")),
        "seed_24_valid_local": (
            _valid_requests(seed, 24)
            and _route_contract(
                seed, 24,
                "same_server_tempo_warm:cache_affinity_warm_seed")),
        "hits_48_valid_local": (
            _valid_requests(hit0, 24) and _valid_requests(hit1, 24)
            and _route_contract(
                hit0, 24,
                "same_server_tempo_measured:cache_affinity_warm_hit")
            and _route_contract(
                hit1, 24,
                "same_server_tempo_measured:cache_affinity_warm_hit")),
        "fixed_baselines_exact": (
            local.get("routes") == {LOCAL: 48}
            and lmcache.get("routes") == {REMOTE: 48}),
        "tempo_routes_48_local": tempo.get("routes") == {LOCAL: 48},
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
        "tempo_tpot_p99_within_2pct_local": (
            tp["tpot_ms"]["p99"] <= 1.02 * lp["tpot_ms"]["p99"]),
        "tempo_paired_local_noninferior": (
            local_pair["e2e_delta_median_ms"] <= 20.0),
    }
    repo = Path(__file__).resolve().parents[2]
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
        "schema": "tempo-pd-output256-phase-analysis-206",
        "allocation_id": allocation,
        "root": str(root),
        "controller": "tempo-pd-hybrid-controller-2",
        "policy": "qwen25-7b-tp4x2-warm-affinity-2",
        "geometry": {"prompt_tokens": [512, 1230, 2048], "output_tokens": 256},
        "routes": {"cold_local": 24, "seed_local": 24, "hit_local": 48},
        "summary": summary,
        "gates": gates,
        "source_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in ("tempo/pd_hybrid_controller.py", "tempo/pd_cache_affinity.py",
                         "tempo/pd_workload_policy.py")
        },
        "claim_boundary": (
            "One four-node A100 allocation, actual Qwen2.5-7B vLLM TP4+TP4 P/D, "
            "24 cold misses, 24 warm seeds, and 48 measured warm hits with 256 "
            "output tokens; pinned LMCache remote and fixed-local baselines."),
    }
    result["passes"] = all(gates.values())
    result["verdict"] = (
        "output256_phase_policy_validated" if result["passes"]
        else "output256_phase_policy_needs_revision")
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
