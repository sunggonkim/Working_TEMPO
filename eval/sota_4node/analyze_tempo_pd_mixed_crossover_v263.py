#!/usr/bin/env python3
"""Analyze paired Tempo/LMCache requests from one shared client window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics


IDENTITY = re.compile(
    r"^ssb-(tempo|remote)-r0-measured-mix([AB])-cache-item-([0-9]{2})$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def analyze(path: Path, allocation: int) -> dict:
    value = json.loads(path.resolve().read_text())
    contract = value.get("mixed_crossover_contract", {})
    _require(contract.get("schema") == "tempo-pd-mixed-request-crossover-260",
             "contract schema mismatch")
    _require(contract.get("phase") == "measured", "not measured phase")
    _require(value["validation"]["performance_claim_allowed"] is True,
             "raw metrics invalid")
    requests = value.get("requests", [])
    decisions = {row["request_id"]: row for row in value["router_decisions"]}
    _require(len(requests) == 48 and len(decisions) == 48,
             "request/decision count mismatch")
    grouped: dict[int, dict[str, dict]] = {}
    route_counts = {"tempo_local": 0, "tempo_remote": 0, "lmcache_remote": 0}
    for row in requests:
        match = IDENTITY.fullmatch(row["request_id"])
        _require(match is not None, "request identity mismatch")
        arm, variant, item_text = match.groups()
        item = int(item_text)
        _require(row["valid"] is True, "invalid request")
        decision = decisions[row["request_id"]]
        _require(decision["phase"] == "complete" and decision["error"] is None,
                 "router completion mismatch")
        if arm == "remote":
            _require(decision["route"] == "remote_prefill_live_kv",
                     "LMCache route mismatch")
            route_counts["lmcache_remote"] += 1
        elif decision["route"] == "remote_prefill_live_kv":
            route_counts["tempo_remote"] += 1
        else:
            _require(decision["route"] == "decoder_local_recompute_or_cache",
                     "Tempo route mismatch")
            route_counts["tempo_local"] += 1
        key = "tempo" if arm == "tempo" else "lmcache"
        _require(key not in grouped.setdefault(item, {}), "duplicate item arm")
        completion = row["token_arrival_offsets_ns"][-1]
        dispatch = row["dispatch_offset_ns"]
        tokens = row["token_arrival_offsets_ns"]
        grouped[item][key] = {
            "variant": variant,
            "e2e_ms": (completion - dispatch) / 1e6,
            "ttft_ms": (tokens[0] - dispatch) / 1e6,
            "tpot_ms": ((tokens[-1] - tokens[0]) / (len(tokens) - 1) / 1e6),
            "route": decision["route"],
        }
    _require(set(grouped) == set(range(24)), "base item coverage mismatch")
    _require(all(set(pair) == {"tempo", "lmcache"} for pair in grouped.values()),
             "pair coverage mismatch")
    _require(route_counts == {"tempo_local": 19, "tempo_remote": 5,
                              "lmcache_remote": 24}, "route partition mismatch")
    pairs = []
    for item, pair in sorted(grouped.items()):
        pairs.append({
            "item": item,
            "tempo_route": pair["tempo"]["route"],
            "e2e_delta_ms": pair["tempo"]["e2e_ms"] - pair["lmcache"]["e2e_ms"],
            "ttft_delta_ms": pair["tempo"]["ttft_ms"] - pair["lmcache"]["ttft_ms"],
            "tpot_delta_ms": pair["tempo"]["tpot_ms"] - pair["lmcache"]["tpot_ms"],
        })
    e2e = [row["e2e_delta_ms"] for row in pairs]
    tpot = [row["tpot_delta_ms"] for row in pairs]
    gates = {
        "paired_e2e_median_improves": statistics.median(e2e) < 0.0,
        "paired_e2e_majority_improves": sum(x < 0 for x in e2e) >= 13,
        "paired_tpot_median_improves": statistics.median(tpot) < 0.0,
        "paired_tpot_p90_nonregression": sorted(tpot)[21] <= 0.0,
    }
    passes = all(gates.values())
    return {
        "schema": "tempo-pd-mixed-request-crossover-analysis-263",
        "allocation_id": allocation,
        "raw": str(path.resolve()),
        "route_counts": route_counts,
        "shared_client_window": True,
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
        "verdict": ("same_window_request_level_advantage" if passes else
                    "same_window_policy_needs_revision"),
        "claim_boundary": (
            "Twenty-four geometry-paired comparisons inside one 48-request "
            "client window. This removes arm-level time drift but does not by "
            "itself establish standalone throughput superiority."
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
    print(json.dumps({"verdict": report["verdict"],
                      "summary": report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
