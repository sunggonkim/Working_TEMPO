#!/usr/bin/env python3
"""Fail-closed adaptive-controller analysis for stationary burst/steady traces."""

import argparse
import json
from pathlib import Path
import statistics

from eval.sota_4node.analyze_tempo_pd_online_regime_microburst25_v343 import (
    IDENTITY, PROVENANCE, _require,
)


POLICY = "tempo-pd-adaptive-microburst25-credit5-367"


def analyze(path: Path, allocation: int, workload_class: str):
    value = json.loads(path.resolve().read_text())
    contract = value.get("mixed_crossover_contract", {})
    _require(workload_class in {"bursty", "steady"}, "class")
    if workload_class == "bursty":
        _require(contract.get("arrival_trace") ==
                 "six_bursts_four_pairs_14ms_with_220ms_idle_v322", "trace")
    _require(contract.get("leading_unique_region") ==
             "same_length_first_19_token_prefix_substitution_v372", "prefix")
    _require(contract.get("leading_unique_chunk_count") == 48, "chunks")
    _require(contract.get("paired_prompt_token_geometry_equal") is True, "geometry")
    _require(value["validation"]["performance_claim_allowed"] is True, "raw")
    requests = value.get("requests", [])
    decisions = {row["request_id"]: row for row in value.get("router_decisions", [])}
    _require(len(requests) == 48 and len(decisions) == 48, "counts")
    grouped = {}
    routes = {"tempo_local": 0, "tempo_remote": 0, "lmcache_remote": 0}
    regimes = {"pending": 0, "affinity": 0, "high_load_local_bypass": 0}
    active_count = 0
    capped = []
    medians = []
    for row in requests:
        match = IDENTITY.fullmatch(row["request_id"])
        _require(match is not None and row["valid"] is True, "request")
        arm, variant, item_text = match.groups()
        item = int(item_text)
        decision = decisions[row["request_id"]]
        _require(decision["phase"] == "complete" and decision["error"] is None,
                 "decision")
        route = decision["route"]
        if arm == "remote":
            _require(route == "remote_prefill_live_kv", "baseline")
            routes["lmcache_remote"] += 1
        else:
            _require(decision["profile_id"] == POLICY, "policy")
            parsed = PROVENANCE.fullmatch(decision["reason"])
            _require(parsed is not None, "provenance")
            before_text, active_text, capped_text, regime, obs_text, median_text = parsed.groups()
            before = int(before_text)
            is_active = active_text == "true"
            is_capped = capped_text == "true"
            regimes[regime] += 1
            if median_text == "none":
                _require(regime == "pending" and int(obs_text) <= 4, "pending")
            else:
                median = int(median_text)
                medians.append(median)
                _require((median <= 39_000_000) ==
                         (regime == "high_load_local_bypass"), "regime")
                _require(is_active == (median <= 25_000_000), "active")
            active_count += int(is_active)
            if is_capped:
                _require(is_active and before >= 5 and
                         route == "remote_prefill_live_kv", "cap")
                capped.append({"item": item, "local_inflight_before": before})
            elif regime == "high_load_local_bypass":
                _require(route == "decoder_local_recompute_or_cache", "high route")
            routes["tempo_local" if route == "decoder_local_recompute_or_cache"
                   else "tempo_remote"] += 1
        tokens = row["token_arrival_offsets_ns"]
        key = "tempo" if arm == "tempo" else "lmcache"
        grouped.setdefault(item, {})[key] = {
            "e2e": (tokens[-1] - row["dispatch_offset_ns"]) / 1e6,
            "tpot": (tokens[-1] - tokens[0]) / (len(tokens) - 1) / 1e6,
        }
    _require(set(grouped) == set(range(24)), "items")
    _require(regimes["pending"] == 4 and regimes["high_load_local_bypass"] >= 8,
             "adaptive load evidence")
    pairs = []
    for item, pair in sorted(grouped.items()):
        _require(set(pair) == {"tempo", "lmcache"}, "pair")
        pairs.append({
            "item": item,
            "e2e_delta_ms": pair["tempo"]["e2e"] - pair["lmcache"]["e2e"],
            "tpot_delta_ms": pair["tempo"]["tpot"] - pair["lmcache"]["tpot"],
        })
    e2e = [row["e2e_delta_ms"] for row in pairs]
    tpot = [row["tpot_delta_ms"] for row in pairs]
    gates = {
        "e2e_win_fraction_ge_80pct": sum(x < 0 for x in e2e) >= 20,
        "e2e_median_improves": statistics.median(e2e) < 0,
        "tpot_win_fraction_ge_80pct": sum(x < 0 for x in tpot) >= 20,
        "tpot_median_improves": statistics.median(tpot) < 0,
        "tpot_p90_nonregression": sorted(tpot)[21] <= 0,
    }
    passes = all(gates.values())
    return {
        "schema": "tempo-pd-adaptive-stationary-analysis-375",
        "allocation_id": allocation,
        "policy": POLICY,
        "workload_class": workload_class,
        "measurement_valid": True,
        "raw": str(path.resolve()),
        "route_counts": routes,
        "regime_counts": regimes,
        "median_gap_ns_range": [min(medians), max(medians)],
        "microburst_active_request_count": active_count,
        "capped_requests": capped,
        "pairs": pairs,
        "summary": {
            "paired_requests": 24,
            "e2e_win_count": sum(x < 0 for x in e2e),
            "e2e_delta_median_ms": statistics.median(e2e),
            "tpot_win_count": sum(x < 0 for x in tpot),
            "tpot_delta_median_ms": statistics.median(tpot),
            "tpot_p90_delta_ms": sorted(tpot)[21],
        },
        "candidate_gates": gates,
        "candidate_passes": passes,
        "verdict": ("adaptive_stationary_advantage" if passes else
                    "adaptive_stationary_revision"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--workload-class", required=True, choices=("bursty", "steady"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite")
    result = analyze(args.raw, args.allocation, args.workload_class)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"],
                      "summary": result["summary"],
                      "routes": result["route_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
