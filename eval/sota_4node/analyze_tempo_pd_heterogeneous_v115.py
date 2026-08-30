#!/usr/bin/env python3
"""Unified production-policy gates for a heterogeneous P/D workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from eval.sota_4node import analyze_tempo_pd_performance_v1 as metrics
from eval.sota_4node.analyze_tempo_pd_interleaved_v100 import _arm_raw


def _pairs(left: dict, right: dict, marker: str | None = None) -> dict:
    lrows = {row["request_id"]: row for row in left["request_metrics"]
             if marker is None or marker in row["request_id"]}
    rrows = {row["request_id"]: row for row in right["request_metrics"]
             if marker is None or marker in row["request_id"]}
    expected = 48 if marker is None else 12
    if lrows.keys() != rrows.keys() or len(lrows) != expected:
        raise ValueError(f"paired geometry mismatch for {marker}")
    deltas = [lrows[key]["e2e_ms"] - rrows[key]["e2e_ms"] for key in sorted(lrows)]
    return {"count": expected, "wins": sum(value < 0 for value in deltas),
            "median_delta_ms": statistics.median(deltas), "raw_delta_ms": deltas}


def analyze(raw: dict) -> dict:
    contract = raw.get("same_server_interleaved_contract")
    if not isinstance(contract, dict) or contract.get("arm_counts") != {
            "local": 48, "tempo": 48, "remote": 48}:
        raise ValueError("exact interleaved contract required")
    parsed = {arm: metrics._parse_run(
        arm, _arm_raw(raw, arm), ttft_slo_ms=3000,
        tpot_slo_ms=250, e2e_slo_ms=16000)
        for arm in ("local", "tempo", "remote")}
    local, tempo, remote = parsed["local"], parsed["tempo"], parsed["remote"]
    expected_tokens = sorted([value for value in (16, 32, 64, 128) for _ in range(12)])
    for arm in parsed.values():
        if sorted(row["completion_tokens"] for row in arm["request_metrics"]) != expected_tokens:
            raise ValueError("heterogeneous completion-token geometry mismatch")
    overall_local, overall_remote = _pairs(tempo, local), _pairs(tempo, remote)
    per_output = {}
    gates = {
        "exact_output_equivalence": tempo["_outputs"] == local["_outputs"] == remote["_outputs"],
        "tempo_routes_exactly_48": sum(tempo["routes"].values()) == 48,
        "tempo_reasons_exactly_48": sum(tempo["reasons"].values()) == 48,
        "tempo_overall_local_noninferior": (
            overall_local["wins"] >= 24 and overall_local["median_delta_ms"] <= 10.0),
        "tempo_overall_beats_lmcache": (
            overall_remote["wins"] >= 27 and overall_remote["median_delta_ms"] < 0.0),
    }
    for output_tokens in (16, 32, 64, 128):
        marker = f"-o{output_tokens}-"
        tl, tr = _pairs(tempo, local, marker), _pairs(tempo, remote, marker)
        per_output[str(output_tokens)] = {"tempo_minus_local": tl,
                                          "tempo_minus_lmcache": tr}
        gates[f"output{output_tokens}_local_noninferior"] = (
            tl["wins"] >= 6 and tl["median_delta_ms"] <= 25.0)
        gates[f"output{output_tokens}_beats_lmcache"] = (
            tr["wins"] >= 7 and tr["median_delta_ms"] < 0.0)
    gates["output16_direct_reason_12"] = sum(
        count for reason, count in tempo["reasons"].items()
        if reason.endswith("output16_direct_local_fast_path")) == 12
    gates["output128_direct_reason_12"] = sum(
        count for reason, count in tempo["reasons"].items()
        if reason.endswith("output128_direct_local_fast_path")) == 12
    tp, lp, rp = (row["performance"] for row in (tempo, local, remote))
    gates.update({
        "tempo_e2e_p50_beats_lmcache": tp["e2e_ms"]["p50"] < rp["e2e_ms"]["p50"],
        "tempo_e2e_p99_beats_lmcache": tp["e2e_ms"]["p99"] < rp["e2e_ms"]["p99"],
        "tempo_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < rp["tpot_ms"]["p99"],
        "tempo_e2e_p99_within_5pct_local": tp["e2e_ms"]["p99"] <= 1.05 * lp["e2e_ms"]["p99"],
    })
    public = lambda row: {k: v for k, v in row.items() if not k.startswith("_")}
    return {
        "schema": "tempo-pd-heterogeneous-production-analysis-115",
        "contract": contract, "fixed_local": public(local), "tempo": public(tempo),
        "lmcache_remote": public(remote),
        "paired_overall": {"tempo_minus_local": overall_local,
                            "tempo_minus_lmcache": overall_remote},
        "paired_by_output_tokens": per_output,
        "gates": gates, "passes": all(gates.values()),
        "verdict": ("promising_unified_production_policy" if all(gates.values())
                    else "reject_unified_production_policy"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
