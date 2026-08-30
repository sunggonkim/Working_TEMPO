#!/usr/bin/env python3
"""Fail-closed analysis of the adaptive controller on a phase change."""

import argparse
import json
from pathlib import Path
import statistics

from eval.sota_4node.analyze_tempo_pd_online_regime_microburst25_v343 import (
    IDENTITY, PROVENANCE, _require,
)


TRACE = "steady8_100ms_then_four_bursts4_14ms_idle220_v353"
POLICY = "tempo-pd-adaptive-microburst25-credit5-367"


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


def analyze(path: Path, allocation: int):
    value = json.loads(path.resolve().read_text())
    contract = value.get("mixed_crossover_contract", {})
    _require(contract.get("arrival_trace") == TRACE, "trace")
    _require(contract.get("leading_unique_region") ==
             "same_length_first_19_token_prefix_substitution_v361", "prefix")
    _require(contract.get("leading_unique_chunk_count") == 48, "unique chunks")
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
    tempo_states = []
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
            _require(decision["profile_id"] == POLICY and
                     decision["manifest_id"] == POLICY, "policy identity")
            parsed = PROVENANCE.fullmatch(decision["reason"])
            _require(parsed is not None, "provenance")
            before_text, active_text, capped_text, regime, obs_text, median_text = (
                parsed.groups()
            )
            before = int(before_text)
            observations = int(obs_text)
            is_active = active_text == "true"
            is_capped = capped_text == "true"
            regimes[regime] += 1
            if median_text == "none":
                _require(regime == "pending" and not is_active and
                         observations <= 4, "pending")
                median = None
            else:
                median = int(median_text)
                _require((median <= 39_000_000) ==
                         (regime == "high_load_local_bypass"), "regime")
                _require(is_active == (median <= 25_000_000), "microburst")
            if is_active:
                active_count += 1
            if is_capped:
                _require(is_active and before >= 5 and
                         route == "remote_prefill_live_kv", "cap")
                capped.append({"item": item, "local_inflight_before": before})
            elif regime == "high_load_local_bypass":
                _require(route == "decoder_local_recompute_or_cache", "high route")
            _require(route in {"remote_prefill_live_kv",
                               "decoder_local_recompute_or_cache"}, "route")
            routes["tempo_local" if route == "decoder_local_recompute_or_cache"
                   else "tempo_remote"] += 1
            tempo_states.append({
                "item": item, "regime": regime, "observations": observations,
                "median_gap_ns": median, "microburst_credit_active": is_active,
                "local_capped": is_capped, "route": route,
            })
        tokens = row["token_arrival_offsets_ns"]
        key = "tempo" if arm == "tempo" else "lmcache"
        _require(key not in grouped.setdefault(item, {}), "duplicate")
        grouped[item][key] = {
            "e2e_ms": (tokens[-1] - row["dispatch_offset_ns"]) / 1e6,
            "tpot_ms": (tokens[-1] - tokens[0]) / (len(tokens) - 1) / 1e6,
        }
    _require(set(grouped) == set(range(24)), "items")
    slow_states = [row for row in tempo_states if row["item"] < 8]
    burst_states = [row for row in tempo_states if row["item"] >= 8]
    _require(any(row["regime"] == "affinity" for row in slow_states),
             "slow affinity evidence")
    high_burst = [row for row in burst_states
                  if row["regime"] == "high_load_local_bypass"]
    _require(len(high_burst) >= 10, "adaptive burst transition")
    first_high_item = min(row["item"] for row in high_burst)
    _require(first_high_item <= 11, "adaptation must occur within four burst pairs")
    pairs = []
    for item, pair in sorted(grouped.items()):
        _require(set(pair) == {"tempo", "lmcache"}, "pair")
        pairs.append({
            "item": item,
            "phase": "slow_lead_in" if item < 8 else "microburst_tail",
            "e2e_delta_ms": pair["tempo"]["e2e_ms"] - pair["lmcache"]["e2e_ms"],
            "tpot_delta_ms": pair["tempo"]["tpot_ms"] - pair["lmcache"]["tpot_ms"],
        })
    summaries = {
        "slow_lead_in": _summary(pairs[:8]),
        "microburst_tail": _summary(pairs[8:]),
        "overall": _summary(pairs),
    }
    gates = {
        "adaptive_transition_within_four_pairs": first_high_item <= 11,
        "microburst_tail_e2e_win_fraction_ge_80pct":
            summaries["microburst_tail"]["e2e_win_count"] >= 13,
        "microburst_tail_tpot_win_fraction_ge_80pct":
            summaries["microburst_tail"]["tpot_win_count"] >= 13,
        "microburst_tail_e2e_median_improves":
            summaries["microburst_tail"]["e2e_delta_median_ms"] < 0,
        "microburst_tail_tpot_median_improves":
            summaries["microburst_tail"]["tpot_delta_median_ms"] < 0,
        "microburst_tail_tpot_p90_nonregression":
            summaries["microburst_tail"]["tpot_p90_delta_ms"] <= 0,
    }
    passes = all(gates.values())
    return {
        "schema": "tempo-pd-phasechange-adaptive-analysis-368",
        "allocation_id": allocation,
        "policy": POLICY,
        "measurement_valid": True,
        "raw": str(path.resolve()),
        "route_counts": routes,
        "regime_counts": regimes,
        "first_high_load_burst_item": first_high_item,
        "microburst_active_request_count": active_count,
        "capped_requests": capped,
        "tempo_states": sorted(tempo_states, key=lambda row: row["item"]),
        "pairs": pairs,
        "phase_summary": summaries,
        "candidate_gates": gates,
        "candidate_passes": passes,
        "verdict": ("adaptive_phase_change_advantage" if passes else
                    "adaptive_phase_change_revision"),
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
                      "first_high_item": result["first_high_load_burst_item"],
                      "phase_summary": result["phase_summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
