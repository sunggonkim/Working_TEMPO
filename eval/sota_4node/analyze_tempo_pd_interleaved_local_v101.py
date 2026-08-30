#!/usr/bin/env python3
"""Analyze the request-interleaved output16 all-local candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_performance_v1 as base
from eval.sota_4node.analyze_tempo_pd_interleaved_v100 import _arm_raw
from eval.sota_4node.finalize_tempo_pd_output16_mixed_v98 import _pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    parsed = {arm: base._parse_run(
        arm, _arm_raw(raw, arm), ttft_slo_ms=3000,
        tpot_slo_ms=250, e2e_slo_ms=8000)
        for arm in ("local", "tempo", "remote")}
    local, tempo, remote = parsed["local"], parsed["tempo"], parsed["remote"]
    exact = tempo["_outputs"] == local["_outputs"] == remote["_outputs"]
    buckets = {}
    gates = {"exact_output_equivalence": exact}
    for bucket in ("512", "1230", "2048"):
        tl = _pairs(tempo, local, f"mix{bucket}-")
        tr = _pairs(tempo, remote, f"mix{bucket}-")
        buckets[bucket] = {"tempo_minus_local": tl, "tempo_minus_lmcache": tr}
        gates[f"bucket_{bucket}_local_noninferior"] = (
            tl["wins"] >= 8 and tl["median_delta_ms"] <= 10.0)
        gates[f"bucket_{bucket}_beats_lmcache"] = (
            tr["wins"] >= 9 and tr["median_delta_ms"] < 0.0)
    gates["routes_48_local"] = tempo["routes"] == {
        "decoder_local_recompute_or_cache": 48}
    reasons = tempo["reasons"]
    gates["workload_guard_reason_48"] = sum(
        count for reason, count in reasons.items()
        if reason.startswith("same_server_tempo_measured:workload_guard_local:")
    ) == 48
    tp, lp, rp = (row["performance"] for row in (tempo, local, remote))
    gates["goodput_retains_98pct_local"] = (
        tp["slo_goodput"]["request_goodput_per_s"] >=
        0.98 * lp["slo_goodput"]["request_goodput_per_s"])
    gates["goodput_beats_lmcache"] = (
        tp["slo_goodput"]["request_goodput_per_s"] >
        rp["slo_goodput"]["request_goodput_per_s"])
    public = lambda row: {k: v for k, v in row.items() if not k.startswith("_")}
    result = {
        "schema": "tempo-pd-request-interleaved-output16-local-analysis-101",
        "contract": raw["same_server_interleaved_contract"],
        "fixed_local": public(local), "tempo": public(tempo),
        "lmcache_remote": public(remote), "prompt_buckets": buckets,
        "gates": gates, "passes": all(gates.values()),
        "verdict": ("promising_request_interleaved_output16_local"
                    if all(gates.values()) else "reject_request_interleaved_output16_local"),
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "buckets": buckets,
                      "failed": [k for k, v in gates.items() if not v]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
