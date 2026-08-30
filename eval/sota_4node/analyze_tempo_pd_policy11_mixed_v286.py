#!/usr/bin/env python3
"""Analyze the rate52 all-local high-load branch in one mixed window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from eval.sota_4node.analyze_tempo_pd_policy10_mixed_v276 import IDENTITY, _require


def analyze(path: Path, allocation: int) -> dict:
    value = json.loads(path.resolve().read_text())
    contract = value.get("mixed_crossover_contract", {})
    _require(contract.get("schema") == "tempo-pd-mixed-request-crossover-260",
             "contract schema")
    _require(contract.get("phase") == "measured", "phase")
    _require(contract.get("variant_assignment_counterbalanced_by_item_parity") is True,
             "counterbalance")
    _require(value["validation"]["performance_claim_allowed"] is True, "raw invalid")
    requests = value.get("requests", [])
    decisions = {row["request_id"]: row for row in value.get("router_decisions", [])}
    _require(len(requests) == 48 and len(decisions) == 48, "counts")
    grouped = {}
    routes = {"tempo_local": 0, "tempo_remote": 0, "lmcache_remote": 0}
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
            _require(route == "decoder_local_recompute_or_cache", "policy11 route")
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
    _require(routes == {"tempo_local": 24, "tempo_remote": 0,
                        "lmcache_remote": 24}, "routes")
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
        "paired_e2e_median_improves": statistics.median(e2e) < 0,
        "paired_e2e_win_fraction_ge_80pct": sum(x < 0 for x in e2e) >= 20,
        "paired_tpot_median_improves": statistics.median(tpot) < 0,
        "paired_tpot_p90_nonregression": sorted(tpot)[21] <= 0,
    }
    passes = all(gates.values())
    return {
        "schema": "tempo-pd-policy11-highload-mixed-analysis-286",
        "allocation_id": allocation,
        "policy": "qwen25-7b-tp4x2-warm-highload-local-11",
        "raw": str(path.resolve()),
        "route_counts": routes,
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
        "verdict": "policy11_highload_advantage" if passes else "policy11_rejected",
        "claim_boundary": (
            "One rate52 same-window lifecycle. Policy11 is a calibrated high-load "
            "bypass branch: 24 Tempo requests recompute locally while the paired "
            "official-LMCache requests use actual remote P/D."
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
    print(json.dumps({"verdict": report["verdict"], "summary": report["summary"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
