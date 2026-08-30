#!/usr/bin/env python3
"""Focused prompt512/output128 direct-local validation with 48 paired rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from eval.sota_4node import analyze_tempo_pd_performance_v1 as base
from eval.sota_4node.analyze_tempo_pd_interleaved_v100 import _arm_raw


def _pairs(left: dict, right: dict) -> dict:
    left_rows = {row["request_id"]: row for row in left["request_metrics"]}
    right_rows = {row["request_id"]: row for row in right["request_metrics"]}
    if left_rows.keys() != right_rows.keys() or len(left_rows) != 48:
        raise ValueError("exact 48 paired rows required")
    deltas = [left_rows[key]["e2e_ms"] - right_rows[key]["e2e_ms"]
              for key in sorted(left_rows)]
    return {"count": 48, "wins": sum(value < 0 for value in deltas),
            "median_delta_ms": statistics.median(deltas),
            "raw_delta_ms": deltas}


def analyze(raw: dict) -> dict:
    contract = raw.get("same_server_interleaved_contract")
    if not isinstance(contract, dict) or contract.get("arm_counts") != {
            "local": 48, "tempo": 48, "remote": 48}:
        raise ValueError("exact interleaved contract required")
    parsed = {arm: base._parse_run(
        arm, _arm_raw(raw, arm), ttft_slo_ms=3000,
        tpot_slo_ms=250, e2e_slo_ms=16000)
        for arm in ("local", "tempo", "remote")}
    local, tempo, remote = parsed["local"], parsed["tempo"], parsed["remote"]
    if any(row["completion_tokens"] != 128 for arm in parsed.values()
           for row in arm["request_metrics"]):
        raise ValueError("every measured request must generate exactly 128 tokens")
    tl, tr = _pairs(tempo, local), _pairs(tempo, remote)
    tp, lp, rp = (row["performance"] for row in (tempo, local, remote))
    gates = {
        "exact_output_equivalence": (
            tempo["_outputs"] == local["_outputs"] == remote["_outputs"]),
        "tempo_routes_48_local": tempo["routes"] == {
            "decoder_local_recompute_or_cache": 48},
        "tempo_reason_48_output128_diagnostic": sum(
            count for reason, count in tempo["reasons"].items()
            if reason.endswith("output128_local_diagnostic")) == 48,
        "tempo_local_noninferior_paired": (
            tl["wins"] >= 24 and tl["median_delta_ms"] <= 10.0),
        "tempo_beats_lmcache_paired": (
            tr["wins"] >= 27 and tr["median_delta_ms"] < 0.0),
        "tempo_e2e_p99_within_5pct_local": (
            tp["e2e_ms"]["p99"] <= 1.05 * lp["e2e_ms"]["p99"]),
        "tempo_e2e_p50_beats_lmcache": tp["e2e_ms"]["p50"] < rp["e2e_ms"]["p50"],
        "tempo_e2e_p99_beats_lmcache": tp["e2e_ms"]["p99"] < rp["e2e_ms"]["p99"],
        "tempo_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < rp["tpot_ms"]["p99"],
    }
    public = lambda row: {k: v for k, v in row.items() if not k.startswith("_")}
    return {
        "schema": "tempo-pd-output128-prompt512-analysis-110",
        "contract": contract, "fixed_local": public(local), "tempo": public(tempo),
        "lmcache_remote": public(remote),
        "paired": {"tempo_minus_local": tl, "tempo_minus_lmcache": tr},
        "gates": gates, "passes": all(gates.values()),
        "verdict": ("promising_output128_prompt512_local_candidate"
                    if all(gates.values()) else
                    "reject_output128_prompt512_local_candidate"),
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
