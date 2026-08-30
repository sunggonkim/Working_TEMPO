#!/usr/bin/env python3
"""Fail-closed paired analysis for the local decoder credit revision."""

import argparse
import json
from pathlib import Path
import re
import statistics

from eval.sota_4node.analyze_tempo_pd_policy10_mixed_v276 import IDENTITY, _require


PROVENANCE = re.compile(
    r".*:local_inflight_before=([0-9]+):local_cap=5:local_capped=(true|false):"
    r"online_regime=(pending|affinity|high_load_local_bypass):"
    r"observations=([0-9]+):median_gap_ns=(none|[0-9]+):threshold_ns=39000000$"
)


def analyze(path: Path, allocation: int):
    value = json.loads(path.resolve().read_text())
    contract = value.get("mixed_crossover_contract", {})
    _require(contract.get("arrival_trace") ==
             "six_bursts_four_pairs_14ms_with_220ms_idle_v322", "trace")
    _require(contract.get("cache_isolation") ==
             "vllm_cache_salt_plus_unique_18_token_regions_v305", "isolation")
    _require(value["validation"]["performance_claim_allowed"] is True, "raw invalid")
    requests = value.get("requests", [])
    decisions = {row["request_id"]: row for row in value.get("router_decisions", [])}
    _require(len(requests) == 48 and len(decisions) == 48, "counts")
    grouped = {}
    routes = {"tempo_local": 0, "tempo_remote": 0, "lmcache_remote": 0}
    regimes = {"pending": 0, "affinity": 0, "high_load_local_bypass": 0}
    frozen_medians = set()
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
            _require(route == "remote_prefill_live_kv", "baseline route")
            routes["lmcache_remote"] += 1
        else:
            parsed = PROVENANCE.fullmatch(decision["reason"])
            _require(parsed is not None, "credit provenance")
            before_text, capped_text, regime, obs_text, median_text = parsed.groups()
            before = int(before_text)
            is_capped = capped_text == "true"
            observations = int(obs_text)
            regimes[regime] += 1
            if is_capped:
                _require(before >= 5 and route == "remote_prefill_live_kv",
                         "capped route")
                capped.append({"item": item, "local_inflight_before": before})
            else:
                _require(before < 5, "uncapped depth")
                if regime == "high_load_local_bypass":
                    _require(route == "decoder_local_recompute_or_cache",
                             "uncapped high-load route")
            if regime == "pending":
                _require(median_text == "none" and 1 <= observations <= 4,
                         "pending provenance")
            else:
                _require(observations == 5 and median_text != "none",
                         "frozen provenance")
                median = int(median_text)
                frozen_medians.add(median)
                _require((median <= 39_000_000) ==
                         (regime == "high_load_local_bypass"), "threshold")
            if route == "remote_prefill_live_kv":
                routes["tempo_remote"] += 1
            else:
                _require(route == "decoder_local_recompute_or_cache", "Tempo route")
                routes["tempo_local"] += 1
        tokens = row["token_arrival_offsets_ns"]
        key = "tempo" if arm == "tempo" else "lmcache"
        _require(key not in grouped.setdefault(item, {}), "duplicate")
        grouped[item][key] = {
            "variant": variant,
            "e2e_ms": (tokens[-1] - row["dispatch_offset_ns"]) / 1e6,
            "ttft_ms": (tokens[0] - row["dispatch_offset_ns"]) / 1e6,
            "tpot_ms": (tokens[-1] - tokens[0]) / (len(tokens) - 1) / 1e6,
        }
    _require(set(grouped) == set(range(24)), "items")
    _require(routes["lmcache_remote"] == 24, "baseline count")
    _require(routes["tempo_local"] + routes["tempo_remote"] == 24, "Tempo count")
    _require(regimes["pending"] == 4 and
             regimes["affinity"] + regimes["high_load_local_bypass"] == 20,
             "regime counts")
    _require(len(frozen_medians) == 2, "pair medians")
    _require(len(capped) > 0, "credit never exercised")
    pairs = []
    for item, pair in sorted(grouped.items()):
        _require(set(pair) == {"tempo", "lmcache"}, "pair")
        pairs.append({
            "item": item,
            "e2e_delta_ms": pair["tempo"]["e2e_ms"] - pair["lmcache"]["e2e_ms"],
            "ttft_delta_ms": pair["tempo"]["ttft_ms"] - pair["lmcache"]["ttft_ms"],
            "tpot_delta_ms": pair["tempo"]["tpot_ms"] - pair["lmcache"]["tpot_ms"],
        })
    e2e = [pair["e2e_delta_ms"] for pair in pairs]
    tpot = [pair["tpot_delta_ms"] for pair in pairs]
    gates = {
        "classifier_and_credit_exact": True,
        "paired_e2e_median_improves": statistics.median(e2e) < 0,
        "paired_e2e_win_fraction_ge_80pct": sum(x < 0 for x in e2e) >= 20,
        "paired_tpot_median_improves": statistics.median(tpot) < 0,
        "paired_tpot_p90_nonregression": sorted(tpot)[21] <= 0,
    }
    passes = all(gates.values())
    return {
        "schema": "tempo-pd-online-regime-local-credit-analysis-329",
        "allocation_id": allocation,
        "policy": "tempo-pd-online-regime-local-credit-326",
        "raw": str(path.resolve()),
        "route_counts": routes,
        "regime_counts": regimes,
        "pair_local_median_gap_ns": sorted(frozen_medians),
        "capped_requests": capped,
        "pairs": pairs,
        "summary": {
            "paired_requests": 24,
            "e2e_delta_median_ms": statistics.median(e2e),
            "e2e_win_count": sum(x < 0 for x in e2e),
            "tpot_delta_median_ms": statistics.median(tpot),
            "tpot_win_count": sum(x < 0 for x in tpot),
            "tpot_p90_delta_ms": sorted(tpot)[21],
        },
        "gates": gates,
        "passes": passes,
        "verdict": ("local_credit_bursty_advantage" if passes else
                    "local_credit_needs_revision"),
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
                      "summary": result["summary"],
                      "capped": result["capped_requests"]}, sort_keys=True))


if __name__ == "__main__":
    main()
