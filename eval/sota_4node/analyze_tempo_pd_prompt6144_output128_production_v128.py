#!/usr/bin/env python3
"""Production gates for output128 at the prompt6144 boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from eval.sota_4node import analyze_tempo_pd_performance_v1 as metrics
from eval.sota_4node.analyze_tempo_pd_interleaved_v100 import _arm_raw


def _pairs(left: dict, right: dict) -> dict:
    lrows = {row["request_id"]: row for row in left["request_metrics"]}
    rrows = {row["request_id"]: row for row in right["request_metrics"]}
    if lrows.keys() != rrows.keys() or len(lrows) != 48:
        raise ValueError("exact 48 paired rows required")
    deltas = [lrows[key]["e2e_ms"] - rrows[key]["e2e_ms"] for key in sorted(lrows)]
    return {"count": 48, "wins": sum(value < 0 for value in deltas),
            "median_delta_ms": statistics.median(deltas), "raw_delta_ms": deltas}


def analyze(raw: dict) -> dict:
    contract = raw.get("same_server_interleaved_contract")
    if not isinstance(contract, dict) or contract.get("arm_counts") != {
            "local": 48, "tempo": 48, "remote": 48}:
        raise ValueError("exact interleaved contract required")
    parsed = {arm: metrics._parse_run(
        arm, _arm_raw(raw, arm), ttft_slo_ms=8000,
        tpot_slo_ms=300, e2e_slo_ms=30000)
        for arm in ("local", "tempo", "remote")}
    local, tempo, remote = parsed["local"], parsed["tempo"], parsed["remote"]
    for arm in parsed.values():
        if len(arm["request_metrics"]) != 48 or any(
                row["completion_tokens"] != 128 for row in arm["request_metrics"]):
            raise ValueError("exact 48 output128 rows per arm required")
    tl, tr = _pairs(tempo, local), _pairs(tempo, remote)
    tp, lp, rp = (row["performance"] for row in (tempo, local, remote))
    gates = {
        "exact_output_equivalence": tempo["_outputs"] == local["_outputs"] == remote["_outputs"],
        "tempo_routes_48_local": tempo["routes"] == {
            "decoder_local_recompute_or_cache": 48},
        "tempo_reason_48_output128_direct_local": sum(
            count for reason, count in tempo["reasons"].items()
            if reason.endswith("output128_direct_local_fast_path")) == 48,
        "tempo_local_noninferior": tl["wins"] >= 24 and tl["median_delta_ms"] <= 25.0,
        "tempo_beats_lmcache": tr["wins"] >= 28 and tr["median_delta_ms"] < 0.0,
        "tempo_e2e_p50_beats_lmcache": tp["e2e_ms"]["p50"] < rp["e2e_ms"]["p50"],
        "tempo_e2e_p99_beats_lmcache": tp["e2e_ms"]["p99"] < rp["e2e_ms"]["p99"],
        "tempo_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < rp["tpot_ms"]["p99"],
        "tempo_e2e_p99_within_5pct_local": tp["e2e_ms"]["p99"] <= 1.05 * lp["e2e_ms"]["p99"],
    }
    public = lambda row: {k: v for k, v in row.items() if not k.startswith("_")}
    return {
        "schema": "tempo-pd-prompt6144-output128-production-analysis-128",
        "contract": contract, "fixed_local": public(local), "tempo": public(tempo),
        "lmcache_remote": public(remote), "tempo_minus_local": tl,
        "tempo_minus_lmcache": tr, "gates": gates, "passes": all(gates.values()),
        "verdict": ("promising_prompt6144_output128_production" if all(gates.values())
                    else "reject_prompt6144_output128_production"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
