#!/usr/bin/env python3
"""Request-interleaved gates for the production output32/output64 controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_performance_v1 as metrics
from eval.sota_4node.analyze_tempo_pd_interleaved_v100 import _arm_raw
from eval.sota_4node.finalize_tempo_pd_output16_mixed_v98 import _pairs


def analyze(raw: dict, output_tokens: int) -> dict:
    if output_tokens not in (32, 64):
        raise ValueError("only frozen output32/output64 policies are supported")
    contract = raw.get("same_server_interleaved_contract")
    if not isinstance(contract, dict) or contract.get("arm_counts") != {
            "local": 48, "tempo": 48, "remote": 48}:
        raise ValueError("exact interleaved contract required")
    parsed = {arm: metrics._parse_run(
        arm, _arm_raw(raw, arm), ttft_slo_ms=3000,
        tpot_slo_ms=250, e2e_slo_ms=12000)
        for arm in ("local", "tempo", "remote")}
    local, tempo, remote = parsed["local"], parsed["tempo"], parsed["remote"]
    if any(row["completion_tokens"] != output_tokens for arm in parsed.values()
           for row in arm["request_metrics"]):
        raise ValueError("completion-token contract mismatch")
    routes = tempo["routes"]
    reasons = tempo["reasons"]
    gates = {
        "exact_output_equivalence": tempo["_outputs"] == local["_outputs"] == remote["_outputs"],
        "tempo_routes_exactly_48": sum(routes.values()) == 48 and set(routes) <= {
            "decoder_local_recompute_or_cache", "remote_prefill_live_kv"},
        "tempo_reasons_exactly_48": sum(
            count for reason, count in reasons.items()
            if reason.startswith("same_server_tempo_measured:")) == 48,
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
        "tempo_e2e_p99_within_5pct_local": tp["e2e_ms"]["p99"] <= 1.05 * lp["e2e_ms"]["p99"],
    })
    public = lambda row: {k: v for k, v in row.items() if not k.startswith("_")}
    return {
        "schema": "tempo-pd-production-controller-interleaved-analysis-114",
        "output_tokens": output_tokens, "contract": contract,
        "fixed_local": public(local), "tempo": public(tempo),
        "lmcache_remote": public(remote), "prompt_buckets": buckets,
        "gates": gates, "passes": all(gates.values()),
        "verdict": (f"promising_output{output_tokens}_production_controller"
                    if all(gates.values()) else
                    f"reject_output{output_tokens}_production_controller"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    result = analyze(json.loads(args.input.read_text(encoding="utf-8")), args.output_tokens)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
