#!/usr/bin/env python3
"""Fail-closed analysis for latched bypass plus rolling microburst credit."""

import argparse
import json
from pathlib import Path
import re
import statistics

from eval.sota_4node.analyze_tempo_pd_online_regime_microburst25_v343 import (
    IDENTITY, _require,
)


POLICY = "tempo-pd-latched-bypass-rolling-credit5-382"
PROVENANCE = re.compile(
    r".*:local_inflight_before=([0-9]+):local_cap=5:"
    r"microburst_threshold_ns=25000000:microburst_credit_active=(true|false):"
    r"local_capped=(true|false):online_regime="
    r"(pending|affinity|high_load_local_bypass):observations=([0-9]+):"
    r"median_gap_ns=(none|[0-9]+):threshold_ns=39000000:raw_regime="
    r"(pending|affinity|high_load_local_bypass):high_load_latched=(true|false)$"
)


def _summary(rows):
    e2e = [row["e2e_delta_ms"] for row in rows]
    tpot = [row["tpot_delta_ms"] for row in rows]
    return {
        "pairs": len(rows),
        "e2e_win_count": sum(value < 0 for value in e2e),
        "e2e_delta_median_ms": statistics.median(e2e),
        "tpot_win_count": sum(value < 0 for value in tpot),
        "tpot_delta_median_ms": statistics.median(tpot),
        "tpot_p90_delta_ms": sorted(tpot)[max(0, int(0.9 * len(tpot)) - 1)],
        "tpot_max_delta_ms": max(tpot),
    }


def analyze(path: Path, allocation: int, workload_class: str):
    _require(workload_class in {"bursty", "steady", "phasechange"}, "class")
    value = json.loads(path.resolve().read_text())
    contract = value.get("mixed_crossover_contract", {})
    expected_prefix = (
        "same_length_first_19_token_prefix_substitution_v361"
        if workload_class == "phasechange"
        else "same_length_first_19_token_prefix_substitution_v372"
    )
    _require(contract.get("leading_unique_region") == expected_prefix, "prefix")
    _require(contract.get("leading_unique_chunk_count") == 48, "chunks")
    _require(contract.get("paired_prompt_token_geometry_equal") is True, "geometry")
    if workload_class in {"bursty", "phasechange"}:
        expected_trace = (
            "six_bursts_four_pairs_14ms_with_220ms_idle_v322"
            if workload_class == "bursty" else
            "steady8_100ms_then_four_bursts4_14ms_idle220_v353"
        )
        _require(contract.get("arrival_trace") == expected_trace, "trace")
    _require(value["validation"]["performance_claim_allowed"] is True, "raw")
    requests = value.get("requests", [])
    decisions = {row["request_id"]: row for row in value.get("router_decisions", [])}
    _require(len(requests) == 48 and len(decisions) == 48, "counts")
    grouped = {}
    routes = {"tempo_local": 0, "tempo_remote": 0, "lmcache_remote": 0}
    regimes = {"pending": 0, "affinity": 0, "high_load_local_bypass": 0}
    raw_regimes = {"pending": 0, "affinity": 0, "high_load_local_bypass": 0}
    active_count = 0
    capped = []
    states = []
    for row in requests:
        match = IDENTITY.fullmatch(row["request_id"])
        _require(match is not None and row["valid"] is True, "request")
        arm, variant, item_text = match.groups(); item = int(item_text)
        decision = decisions[row["request_id"]]
        _require(decision["phase"] == "complete" and decision["error"] is None,
                 "decision")
        route = decision["route"]
        if arm == "remote":
            _require(route == "remote_prefill_live_kv", "baseline")
            routes["lmcache_remote"] += 1
        else:
            _require(decision["profile_id"] == POLICY and
                     decision["manifest_id"] == POLICY, "policy")
            parsed = PROVENANCE.fullmatch(decision["reason"])
            _require(parsed is not None, "provenance")
            (before_text, active_text, capped_text, regime, observations_text,
             median_text, raw_regime, latched_text) = parsed.groups()
            before = int(before_text); observations = int(observations_text)
            active = active_text == "true"; is_capped = capped_text == "true"
            latched = latched_text == "true"
            regimes[regime] += 1; raw_regimes[raw_regime] += 1
            if median_text == "none":
                median = None
                _require(raw_regime == "pending" and regime == "pending" and
                         not latched and not active and observations <= 4, "pending")
            else:
                median = int(median_text)
                _require((median <= 39_000_000) ==
                         (raw_regime == "high_load_local_bypass"), "raw regime")
                _require(active == (median <= 25_000_000), "rolling credit")
                if raw_regime == "high_load_local_bypass":
                    _require(latched and regime == "high_load_local_bypass", "latch")
                elif latched:
                    _require(regime == "high_load_local_bypass", "held latch")
                else:
                    _require(regime == "affinity", "unlatched affinity")
            active_count += int(active)
            if is_capped:
                _require(active and before >= 5 and
                         route == "remote_prefill_live_kv", "cap")
                capped.append({"item": item, "local_inflight_before": before})
            elif regime == "high_load_local_bypass":
                _require(route == "decoder_local_recompute_or_cache", "bypass route")
            routes["tempo_local" if route == "decoder_local_recompute_or_cache"
                   else "tempo_remote"] += 1
            states.append({"item": item, "regime": regime,
                           "raw_regime": raw_regime, "median_gap_ns": median,
                           "high_load_latched": latched,
                           "microburst_credit_active": active,
                           "local_capped": is_capped})
        tokens = row["token_arrival_offsets_ns"]
        key = "tempo" if arm == "tempo" else "lmcache"
        grouped.setdefault(item, {})[key] = {
            "e2e": (tokens[-1] - row["dispatch_offset_ns"]) / 1e6,
            "tpot": (tokens[-1] - tokens[0]) / (len(tokens) - 1) / 1e6,
        }
    _require(set(grouped) == set(range(24)), "items")
    latched_states = [row for row in states if row["high_load_latched"]]
    _require(len(latched_states) >= 12, "latched coverage")
    first_latched = min(row["item"] for row in latched_states)
    if workload_class == "phasechange":
        _require(first_latched <= 11 and
                 any(row["regime"] == "affinity" and row["item"] < 8
                     for row in states), "phase transition")
        _require(all(row["high_load_latched"] for row in states
                     if row["item"] >= first_latched), "monotone latch")
    pairs = []
    for item, pair in sorted(grouped.items()):
        _require(set(pair) == {"tempo", "lmcache"}, "pair")
        pairs.append({"item": item,
                      "e2e_delta_ms": pair["tempo"]["e2e"] - pair["lmcache"]["e2e"],
                      "tpot_delta_ms": pair["tempo"]["tpot"] - pair["lmcache"]["tpot"]})
    measured = pairs[8:] if workload_class == "phasechange" else pairs
    summary = _summary(measured)
    gates = {
        "e2e_win_fraction_ge_80pct": summary["e2e_win_count"] >=
            (13 if workload_class == "phasechange" else 20),
        "e2e_median_improves": summary["e2e_delta_median_ms"] < 0,
        "tpot_win_fraction_ge_80pct": summary["tpot_win_count"] >=
            (13 if workload_class == "phasechange" else 20),
        "tpot_median_improves": summary["tpot_delta_median_ms"] < 0,
        "tpot_p90_nonregression": summary["tpot_p90_delta_ms"] <= 0,
    }
    passes = all(gates.values())
    return {
        "schema": "tempo-pd-latched-controller-analysis-383",
        "allocation_id": allocation, "policy": POLICY,
        "workload_class": workload_class, "measurement_valid": True,
        "raw": str(path.resolve()), "route_counts": routes,
        "regime_counts": regimes, "raw_regime_counts": raw_regimes,
        "first_latched_item": first_latched,
        "microburst_active_request_count": active_count,
        "capped_requests": capped, "tempo_states": sorted(states, key=lambda x: x["item"]),
        "pairs": pairs, "evaluated_summary": summary,
        "candidate_gates": gates, "candidate_passes": passes,
        "verdict": "latched_controller_advantage" if passes else "latched_controller_revision",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--workload-class", required=True,
                        choices=("bursty", "steady", "phasechange"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("refusing to overwrite")
    result = analyze(args.raw, args.allocation, args.workload_class)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"],
                      "summary": result["evaluated_summary"],
                      "routes": result["route_counts"],
                      "first_latched": result["first_latched_item"]}, sort_keys=True))


if __name__ == "__main__": main()
