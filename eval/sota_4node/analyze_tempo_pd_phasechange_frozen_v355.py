#!/usr/bin/env python3
"""Fail-closed measurement of the frozen controller on a phase change."""

import argparse
import json
from pathlib import Path
import statistics

from eval.sota_4node.analyze_tempo_pd_online_regime_microburst25_v343 import (
    IDENTITY, PROVENANCE, _require,
)


TRACE = "steady8_100ms_then_four_bursts4_14ms_idle220_v353"


def _phase_summary(pairs):
    e2e = [row["e2e_delta_ms"] for row in pairs]
    tpot = [row["tpot_delta_ms"] for row in pairs]
    return {
        "pairs": len(pairs),
        "e2e_win_count": sum(value < 0 for value in e2e),
        "e2e_delta_median_ms": statistics.median(e2e),
        "tpot_win_count": sum(value < 0 for value in tpot),
        "tpot_delta_median_ms": statistics.median(tpot),
        "tpot_max_delta_ms": max(tpot),
    }


def analyze(path: Path, allocation: int):
    value = json.loads(path.resolve().read_text())
    contract = value.get("mixed_crossover_contract", {})
    _require(contract.get("arrival_trace") == TRACE, "phase-change trace")
    _require(contract.get("cache_isolation") ==
             "vllm_cache_salt_plus_unique_18_token_regions_v305", "isolation")
    _require(value["validation"]["performance_claim_allowed"] is True, "raw")
    requests = value.get("requests", [])
    decisions = {row["request_id"]: row for row in value.get("router_decisions", [])}
    _require(len(requests) == 48 and len(decisions) == 48, "counts")
    grouped = {}
    routes = {"tempo_local": 0, "tempo_remote": 0, "lmcache_remote": 0}
    regimes = {"pending": 0, "affinity": 0, "high_load_local_bypass": 0}
    frozen_medians = set()
    active = 0
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
            before, active_text, capped_text, regime, obs, median_text = parsed.groups()
            regimes[regime] += 1
            is_active = active_text == "true"
            is_capped = capped_text == "true"
            if median_text != "none":
                frozen_medians.add(int(median_text))
            if is_active:
                active += 1
            if is_capped:
                capped.append({"item": item, "local_inflight_before": int(before)})
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
    _require(regimes["pending"] == 4, "pending geometry")
    _require(len(frozen_medians) == 1, "classifier must remain frozen")
    frozen_median = next(iter(frozen_medians))
    _require(frozen_median > 39_000_000, "slow lead-in must freeze affinity")
    _require(active == 0 and not capped, "frozen microburst credit must stay off")
    pairs = []
    for item, pair in sorted(grouped.items()):
        _require(set(pair) == {"tempo", "lmcache"}, "pair")
        pairs.append({
            "item": item,
            "phase": "slow_lead_in" if item < 8 else "microburst_tail",
            "e2e_delta_ms": pair["tempo"]["e2e_ms"] - pair["lmcache"]["e2e_ms"],
            "tpot_delta_ms": pair["tempo"]["tpot_ms"] - pair["lmcache"]["tpot_ms"],
        })
    phases = {
        "slow_lead_in": _phase_summary(pairs[:8]),
        "microburst_tail": _phase_summary(pairs[8:]),
        "overall": _phase_summary(pairs),
    }
    candidate_gates = {
        "microburst_tail_e2e_win_fraction_ge_80pct":
            phases["microburst_tail"]["e2e_win_count"] >= 13,
        "microburst_tail_tpot_win_fraction_ge_80pct":
            phases["microburst_tail"]["tpot_win_count"] >= 13,
        "microburst_tail_e2e_median_improves":
            phases["microburst_tail"]["e2e_delta_median_ms"] < 0,
        "microburst_tail_tpot_median_improves":
            phases["microburst_tail"]["tpot_delta_median_ms"] < 0,
    }
    return {
        "schema": "tempo-pd-phasechange-frozen-analysis-355",
        "allocation_id": allocation,
        "policy": "tempo-pd-online-regime-microburst25-credit5-342",
        "measurement_valid": True,
        "raw": str(path.resolve()),
        "frozen_median_gap_ns": frozen_median,
        "route_counts": routes,
        "regime_counts": regimes,
        "microburst_active_request_count": active,
        "capped_requests": capped,
        "pairs": pairs,
        "phase_summary": phases,
        "candidate_gates": candidate_gates,
        "candidate_passes": all(candidate_gates.values()),
        "verdict": ("frozen_handles_phase_change" if all(candidate_gates.values())
                    else "frozen_classifier_falsified_by_phase_change"),
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
