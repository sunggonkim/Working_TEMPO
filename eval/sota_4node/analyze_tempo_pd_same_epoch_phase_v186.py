#!/usr/bin/env python3
"""Fail-closed proof of live MISS→WARM_SEED→WARM_HIT behavior and performance."""

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


def _cache_items(rows: list[dict]) -> set[str]:
    items = set()
    for row in rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or "-cache-item-" not in request_id:
            raise ValueError("stable cache item identity missing")
        items.add("cache-item-" + request_id.rsplit("-cache-item-", 1)[1])
    return items


def _requests_valid(value: dict, count: int) -> bool:
    rows = value.get("requests")
    return (isinstance(rows, list) and len(rows) == count
            and all(row.get("error") is None
                    and not row.get("contract_violations") for row in rows))


def _decision_contract(value: dict, *, count: int, reason: str,
                       local: int, remote: int) -> bool:
    rows = value.get("router_decisions")
    if not isinstance(rows, list) or len(rows) != count:
        return False
    return (all(row.get("reason") == reason for row in rows)
            and sum(row.get("route") == LOCAL for row in rows) == local
            and sum(row.get("route") == REMOTE for row in rows) == remote)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    stage = root / "tempo_credit_admission"
    cold = _load(stage / "hybrid_cold_transition.raw.json")
    seed = _load(stage / "same_server_balanced_warm/02_tempo_r0.raw.json")
    hit0 = _load(stage / "same_server_balanced_measured/01_tempo_r0.raw.json")
    hit1 = _load(stage / "same_server_balanced_measured/04_tempo_r1.raw.json")
    final = _load(root / "hybrid_controller_final.json")
    if final.get("schema") != "tempo-pd-production-hybrid-controller-analysis-151":
        raise ValueError("production final schema mismatch")

    cold_decisions = cold.get("router_decisions", [])
    cold_reasons = {row.get("reason") for row in cold_decisions}
    seed_items = _cache_items(seed.get("router_decisions", []))
    hit0_items = _cache_items(hit0.get("router_decisions", []))
    hit1_items = _cache_items(hit1.get("router_decisions", []))
    tp = final["tempo"]["performance"]
    lp = final["fixed_local"]["performance"]
    mp = final["lmcache_remote"]["performance"]
    pair = final["paired_tempo_minus_lmcache"]

    gates = {
        "cold_requests_exact_and_valid": _requests_valid(cold, 24),
        "cold_decisions_exactly_24_local": (
            len(cold_decisions) == 24
            and all(row.get("route") == LOCAL for row in cold_decisions)),
        "cold_decisions_use_miss_policy": (
            bool(cold_reasons)
            and all(isinstance(reason, str) and
                    reason.startswith("same_server_tempo_cold:hybrid_cold:")
                    for reason in cold_reasons)),
        "warm_seed_requests_exact_and_valid": _requests_valid(seed, 24),
        "warm_seed_route_partition_exact": _decision_contract(
            seed, count=24, reason="same_server_tempo_warm:cache_affinity_warm_seed",
            local=16, remote=8),
        "warm_hit_replicates_exact_and_valid": (
            _requests_valid(hit0, 24) and _requests_valid(hit1, 24)),
        "warm_hit_route_partitions_exact": (
            _decision_contract(
                hit0, count=24,
                reason="same_server_tempo_measured:cache_affinity_warm_hit",
                local=16, remote=8)
            and _decision_contract(
                hit1, count=24,
                reason="same_server_tempo_measured:cache_affinity_warm_hit",
                local=16, remote=8)),
        "seed_and_hits_bind_same_24_cache_items": (
            len(seed_items) == 24 and seed_items == hit0_items == hit1_items),
        "production_analysis_all_gates_pass": final.get("passes") is True,
        "tempo_throughput_beats_lmcache": (
            tp["request_throughput_per_s"] > mp["request_throughput_per_s"]),
        "tempo_throughput_beats_local": (
            tp["request_throughput_per_s"] > lp["request_throughput_per_s"]),
        "tempo_e2e_p99_beats_lmcache": tp["e2e_ms"]["p99"] < mp["e2e_ms"]["p99"],
        "tempo_e2e_p99_beats_local": tp["e2e_ms"]["p99"] < lp["e2e_ms"]["p99"],
        "tempo_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < mp["tpot_ms"]["p99"],
        "tempo_paired_majority_beats_lmcache": pair["e2e_win_count"] >= 25,
        "tempo_paired_median_beats_lmcache": pair["e2e_delta_median_ms"] < 0,
    }
    repo = Path(__file__).resolve().parents[2]
    summary = {
        "allocation_id": args.allocation,
        "cold_request_count": len(cold.get("requests", [])),
        "cold_routes": {LOCAL: sum(row.get("route") == LOCAL for row in cold_decisions)},
        "warm_seed_routes": {LOCAL: 16, REMOTE: 8},
        "warm_hit_routes_across_two_replicates": {LOCAL: 32, REMOTE: 16},
        "tempo_throughput_per_s": tp["request_throughput_per_s"],
        "local_throughput_per_s": lp["request_throughput_per_s"],
        "lmcache_throughput_per_s": mp["request_throughput_per_s"],
        "throughput_gain_vs_lmcache_percent": 100 * (
            tp["request_throughput_per_s"] / mp["request_throughput_per_s"] - 1),
        "tempo_e2e_p99_ms": tp["e2e_ms"]["p99"],
        "local_e2e_p99_ms": lp["e2e_ms"]["p99"],
        "lmcache_e2e_p99_ms": mp["e2e_ms"]["p99"],
        "e2e_p99_reduction_vs_lmcache_percent": 100 * (
            1 - tp["e2e_ms"]["p99"] / mp["e2e_ms"]["p99"]),
        "tempo_tpot_p99_ms": tp["tpot_ms"]["p99"],
        "lmcache_tpot_p99_ms": mp["tpot_ms"]["p99"],
        "tpot_p99_reduction_vs_lmcache_percent": 100 * (
            1 - tp["tpot_ms"]["p99"] / mp["tpot_ms"]["p99"]),
        "paired_win_count": pair["e2e_win_count"],
        "paired_e2e_delta_median_ms": pair["e2e_delta_median_ms"],
    }
    result = {
        "schema": "tempo-pd-same-epoch-phase-analysis-186",
        "controller": "tempo-pd-hybrid-controller-1",
        "root": str(root),
        "summary": summary,
        "source_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in ("tempo/pd_hybrid_controller.py", "tempo/pd_cache_affinity.py")
        },
        "gates": gates,
        "passes": all(gates.values()),
        "claim_boundary": (
            "One four-node A100 allocation and one live actual-Qwen2.5-7B vLLM "
            "TP4+TP4 P/D epoch; 24 cold MISS requests, 24 warm seeds, then 48 "
            "measured warm hits. Pinned LMCache is the remote baseline."
        ),
    }
    result["verdict"] = (
        "same_epoch_phase_controller_validated" if result["passes"]
        else "same_epoch_phase_controller_needs_revision")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    args.output.resolve().write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"],
                      "failed": [key for key, passed in gates.items() if not passed]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
