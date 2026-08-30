#!/usr/bin/env python3
"""Fail-closed two-replica analysis of frozen phase-change behavior."""

import argparse
import json
from pathlib import Path
import statistics

from eval.sota_4node.analyze_tempo_pd_online_regime_microburst25_v343 import (
    IDENTITY, PROVENANCE, _require,
)


TRACE = "steady8_100ms_then_four_bursts4_14ms_idle220_v353"


def _summary(rows):
    e2e = [row["e2e_delta_ms"] for row in rows]
    tpot = [row["tpot_delta_ms"] for row in rows]
    return {
        "pairs": len(rows),
        "e2e_win_count": sum(value < 0 for value in e2e),
        "e2e_delta_median_ms": statistics.median(e2e),
        "tpot_win_count": sum(value < 0 for value in tpot),
        "tpot_delta_median_ms": statistics.median(tpot),
        "tpot_max_delta_ms": max(tpot),
    }


def analyze(path: Path, allocation: int):
    value = json.loads(path.resolve().read_text())
    contract = value.get("mixed_crossover_contract", {})
    _require(contract.get("arrival_trace") == TRACE, "trace")
    _require(contract.get("leading_unique_region") ==
             "same_length_first_19_token_prefix_substitution_v361", "prefix")
    _require(contract.get("leading_unique_chunk_count") == 48, "unique chunks")
    _require(contract.get("paired_prompt_token_geometry_equal") is True,
             "paired geometry")
    _require(contract.get("frozen_prompt_token_buckets_preserved") ==
             [512, 1230, 2048, 4094], "prompt buckets")
    _require(value["validation"]["performance_claim_allowed"] is True, "raw")
    requests = value.get("requests", [])
    decisions = {row["request_id"]: row for row in value.get("router_decisions", [])}
    _require(len(requests) == 48 and len(decisions) == 48, "counts")
    grouped = {}
    routes = {"tempo_local": 0, "tempo_remote": 0, "lmcache_remote": 0}
    regimes = {"pending": 0, "affinity": 0, "high_load_local_bypass": 0}
    medians = set()
    active_count = 0
    capped = []
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
            parsed = PROVENANCE.fullmatch(decision["reason"])
            _require(parsed is not None, "provenance")
            before, active_text, capped_text, regime, observations, median_text = (
                parsed.groups()
            )
            regimes[regime] += 1
            if median_text != "none":
                medians.add(int(median_text))
            is_active = active_text == "true"
            is_capped = capped_text == "true"
            active_count += int(is_active)
            if is_capped:
                capped.append({"item": item, "local_inflight_before": int(before)})
            _require(route in {"remote_prefill_live_kv",
                               "decoder_local_recompute_or_cache"}, "route")
            routes["tempo_local" if route == "decoder_local_recompute_or_cache"
                   else "tempo_remote"] += 1
        tokens = row["token_arrival_offsets_ns"]
        key = "tempo" if arm == "tempo" else "lmcache"
        _require(key not in grouped.setdefault(item, {}), "duplicate")
        grouped[item][key] = {
            "variant": variant,
            "e2e_ms": (tokens[-1] - row["dispatch_offset_ns"]) / 1e6,
            "tpot_ms": (tokens[-1] - tokens[0]) / (len(tokens) - 1) / 1e6,
        }
    _require(set(grouped) == set(range(24)), "items")
    _require(regimes["pending"] == 4, "pending")
    _require(len(medians) == 2 and min(medians) > 39_000_000,
             "two replicas must freeze affinity from slow lead-in")
    _require(active_count == 0 and capped == [], "credit must remain off")
    pairs = []
    for item, pair in sorted(grouped.items()):
        _require(set(pair) == {"tempo", "lmcache"}, "pair")
        pairs.append({
            "item": item,
            "phase": "slow_lead_in" if item < 8 else "microburst_tail",
            "e2e_delta_ms": pair["tempo"]["e2e_ms"] - pair["lmcache"]["e2e_ms"],
            "tpot_delta_ms": pair["tempo"]["tpot_ms"] - pair["lmcache"]["tpot_ms"],
        })
    phase_summary = {
        "slow_lead_in": _summary(pairs[:8]),
        "microburst_tail": _summary(pairs[8:]),
        "overall": _summary(pairs),
    }
    gates = {
        "microburst_tail_e2e_win_fraction_ge_80pct":
            phase_summary["microburst_tail"]["e2e_win_count"] >= 13,
        "microburst_tail_tpot_win_fraction_ge_80pct":
            phase_summary["microburst_tail"]["tpot_win_count"] >= 13,
        "microburst_tail_e2e_median_improves":
            phase_summary["microburst_tail"]["e2e_delta_median_ms"] < 0,
        "microburst_tail_tpot_median_improves":
            phase_summary["microburst_tail"]["tpot_delta_median_ms"] < 0,
    }
    passes = all(gates.values())
    return {
        "schema": "tempo-pd-phasechange-frozen-analysis-364",
        "allocation_id": allocation,
        "policy": "tempo-pd-online-regime-microburst25-credit5-342",
        "measurement_valid": True,
        "raw": str(path.resolve()),
        "pair_local_frozen_median_gap_ns": sorted(medians),
        "route_counts": routes,
        "regime_counts": regimes,
        "microburst_active_request_count": active_count,
        "capped_requests": capped,
        "pairs": pairs,
        "phase_summary": phase_summary,
        "candidate_gates": gates,
        "candidate_passes": passes,
        "verdict": ("frozen_handles_phase_change" if passes else
                    "frozen_classifier_falsified_by_phase_change"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite")
    result = analyze(args.raw, args.allocation)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"],
                      "phase_summary": result["phase_summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
