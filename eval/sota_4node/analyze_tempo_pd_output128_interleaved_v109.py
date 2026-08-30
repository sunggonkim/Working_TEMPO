#!/usr/bin/env python3
"""Predeclared gates for request-interleaved output128 local routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_performance_v1 as base
from eval.sota_4node.analyze_tempo_pd_interleaved_v100 import _arm_raw
from eval.sota_4node.finalize_tempo_pd_output16_mixed_v98 import _pairs


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
    if any(metric["completion_tokens"] != 128
           for row in parsed.values() for metric in row["request_metrics"]):
        raise ValueError("every measured request must generate exactly 128 tokens")
    gates = {
        "exact_output_equivalence": (
            tempo["_outputs"] == local["_outputs"] == remote["_outputs"]),
        "tempo_routes_48_local": tempo["routes"] == {
            "decoder_local_recompute_or_cache": 48},
        "tempo_reason_48_output128_diagnostic": sum(
            count for reason, count in tempo["reasons"].items()
            if reason.endswith("output128_local_diagnostic")) == 48,
    }
    buckets = {}
    for bucket in ("512", "1230", "2048"):
        tl = _pairs(tempo, local, f"mix{bucket}-")
        tr = _pairs(tempo, remote, f"mix{bucket}-")
        buckets[bucket] = {"tempo_minus_local": tl, "tempo_minus_lmcache": tr}
        gates[f"bucket_{bucket}_local_noninferior"] = (
            tl["wins"] >= 8 and tl["median_delta_ms"] <= 25.0)
        gates[f"bucket_{bucket}_beats_lmcache"] = (
            tr["wins"] >= 9 and tr["median_delta_ms"] < 0.0)
    tp, lp, rp = (row["performance"] for row in (tempo, local, remote))
    gates.update({
        "tempo_e2e_p50_beats_lmcache": tp["e2e_ms"]["p50"] < rp["e2e_ms"]["p50"],
        "tempo_e2e_p99_beats_lmcache": tp["e2e_ms"]["p99"] < rp["e2e_ms"]["p99"],
        "tempo_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < rp["tpot_ms"]["p99"],
        "tempo_e2e_p99_within_5pct_local": (
            tp["e2e_ms"]["p99"] <= 1.05 * lp["e2e_ms"]["p99"]),
    })
    public = lambda row: {k: v for k, v in row.items() if not k.startswith("_")}
    return {
        "schema": "tempo-pd-output128-interleaved-analysis-109",
        "contract": contract, "fixed_local": public(local), "tempo": public(tempo),
        "lmcache_remote": public(remote), "prompt_buckets": buckets,
        "gates": gates, "passes": all(gates.values()),
        "verdict": ("promising_output128_local_candidate" if all(gates.values())
                    else "reject_output128_local_candidate"),
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
