#!/usr/bin/env python3
"""Fail-closed same-window analysis for the online regime classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics

from eval.sota_4node.analyze_tempo_pd_policy10_mixed_v276 import IDENTITY, _require


REGIME = re.compile(
    r".*:online_regime=(pending|affinity|high_load_local_bypass):"
    r"observations=([0-9]+):median_gap_ns=(none|[0-9]+):threshold_ns=39000000$"
)
EXPECTED = {
    48.0: ({"tempo_local": 19, "tempo_remote": 5, "lmcache_remote": 24},
           {"pending": 6, "affinity": 18, "high_load_local_bypass": 0}),
    52.0: ({"tempo_local": 23, "tempo_remote": 1, "lmcache_remote": 24},
           {"pending": 6, "affinity": 0, "high_load_local_bypass": 18}),
}


def analyze(path: Path, allocation: int) -> dict:
    value = json.loads(path.resolve().read_text())
    contract = value.get("mixed_crossover_contract", {})
    _require(contract.get("schema") == "tempo-pd-mixed-request-crossover-260",
             "contract schema")
    _require(contract.get("phase") == "measured", "phase")
    _require(contract.get("variant_assignment_counterbalanced_by_item_parity") is True,
             "counterbalance")
    _require(value["validation"]["performance_claim_allowed"] is True, "raw invalid")
    rate = float(value["workload"]["request_rate_per_s"])
    _require(rate in EXPECTED, "offered rate outside online classifier contract")
    expected_routes, expected_regimes = EXPECTED[rate]
    requests = value.get("requests", [])
    decisions = {row["request_id"]: row for row in value.get("router_decisions", [])}
    _require(len(requests) == 48 and len(decisions) == 48, "counts")
    grouped = {}
    routes = {"tempo_local": 0, "tempo_remote": 0, "lmcache_remote": 0}
    regimes = {"pending": 0, "affinity": 0, "high_load_local_bypass": 0}
    medians = {"affinity": set(), "high_load_local_bypass": set()}
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
            parsed = REGIME.fullmatch(decision["reason"])
            _require(parsed is not None, "online regime provenance")
            regime, observations_text, median_text = parsed.groups()
            regimes[regime] += 1
            observations = int(observations_text)
            _require(1 <= observations <= 7, "classifier observation count")
            if regime == "pending":
                _require(median_text == "none" and observations <= 6,
                         "pending provenance")
            else:
                _require(observations == 7 and median_text != "none",
                         "frozen provenance")
                medians[regime].add(int(median_text))
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
    _require(routes == expected_routes, "online route partition")
    _require(regimes == expected_regimes, "online regime partition")
    frozen_name = "high_load_local_bypass" if rate == 52.0 else "affinity"
    frozen_medians = medians[frozen_name]
    _require(len(frozen_medians) == 2, "exact two pair-local classifier medians")
    _require(all((gap <= 39_000_000) == (rate == 52.0)
                 for gap in frozen_medians), "classifier threshold direction")
    pairs = []
    for item, pair in sorted(grouped.items()):
        _require(set(pair) == {"tempo", "lmcache"}, "pair")
        pairs.append({
            "item": item,
            "e2e_delta_ms": pair["tempo"]["e2e_ms"] - pair["lmcache"]["e2e_ms"],
            "ttft_delta_ms": pair["tempo"]["ttft_ms"] - pair["lmcache"]["ttft_ms"],
            "tpot_delta_ms": pair["tempo"]["tpot_ms"] - pair["lmcache"]["tpot_ms"],
        })
    e2e = [row["e2e_delta_ms"] for row in pairs]
    tpot = [row["tpot_delta_ms"] for row in pairs]
    gates = {
        "classifier_exact": True,
        "paired_e2e_median_improves": statistics.median(e2e) < 0,
        "paired_e2e_win_fraction_ge_80pct": sum(x < 0 for x in e2e) >= 20,
        "paired_tpot_median_improves": statistics.median(tpot) < 0,
        "paired_tpot_p90_nonregression": sorted(tpot)[21] <= 0,
    }
    passes = all(gates.values())
    return {
        "schema": "tempo-pd-online-regime-mixed-analysis-292",
        "allocation_id": allocation,
        "policy": "tempo-pd-online-regime-router-291",
        "offered_rate_per_s": rate,
        "raw": str(path.resolve()),
        "route_counts": routes,
        "regime_counts": regimes,
        "pair_local_median_gap_ns": sorted(frozen_medians),
        "pairs": pairs,
        "summary": {
            "paired_requests": 24,
            "e2e_delta_median_ms": statistics.median(e2e),
            "e2e_win_count": sum(x < 0 for x in e2e),
            "tpot_delta_median_ms": statistics.median(tpot),
            "tpot_win_count": sum(x < 0 for x in tpot),
        },
        "gates": gates,
        "passes": passes,
        "verdict": "online_regime_same_window_advantage" if passes else
                   "online_regime_needs_revision",
        "claim_boundary": (
            "One actual-vLLM same-window lifecycle. Each pair-local router freezes "
            "its regime after six local-clock arrival gaps; classification includes "
            "both counterbalanced arms and uses no cross-host timestamp subtraction."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite")
    report = analyze(args.raw, args.allocation)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": report["verdict"], "summary": report["summary"],
                      "medians": report["pair_local_median_gap_ns"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
