#!/usr/bin/env python3
"""Analyze the mixed-prompt output16 split policy per prompt bucket."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node.finalize_tempo_pd_output16_mixed_v98 import _pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    value = json.loads(args.input.read_text(encoding="utf-8"))
    if value.get("schema") != "tempo-pd-same-server-balanced-analysis-71":
        raise ValueError("balanced v71 input required")
    tempo, local, remote = value["tempo"], value["fixed_local"], value["lmcache_remote"]
    if any(row["completion_tokens"] != 16 for arm in (tempo, local, remote)
           for row in arm["request_metrics"]):
        raise ValueError("every request must generate exactly 16 tokens")
    gates = value["gates"]
    for name in (
        "tempo_routes_32_local_16_remote", "tempo_goodput_beats_local",
        "tempo_paired_majority_beats_local", "tempo_paired_median_beats_local",
    ):
        del gates[name]
    buckets = {}
    for bucket in ("512", "1230", "2048"):
        tl = _pairs(tempo, local, f"mix{bucket}-")
        tr = _pairs(tempo, remote, f"mix{bucket}-")
        buckets[bucket] = {"tempo_minus_local": tl, "tempo_minus_lmcache": tr}
    for bucket in ("512", "1230"):
        chosen = buckets[bucket]["tempo_minus_lmcache"]
        gates[f"bucket_{bucket}_remote_noninferior"] = (
            chosen["wins"] >= 8 and chosen["median_delta_ms"] <= 10.0)
    alt = buckets["1230"]["tempo_minus_local"]
    gates["bucket_1230_remote_beats_local"] = (
        alt["wins"] >= 9 and alt["median_delta_ms"] < 0.0)
    chosen = buckets["2048"]["tempo_minus_local"]
    gates["bucket_2048_local_noninferior"] = (
        chosen["wins"] >= 8 and chosen["median_delta_ms"] <= 10.0)
    alt = buckets["2048"]["tempo_minus_lmcache"]
    gates["bucket_2048_local_beats_lmcache"] = (
        alt["wins"] >= 9 and alt["median_delta_ms"] < 0.0)
    gates["output16_split_routes_16_local_32_remote"] = tempo["routes"] == {
        "decoder_local_recompute_or_cache": 16, "remote_prefill_live_kv": 32}
    reasons = tempo["reasons"]
    gates["output16_split_reason_counts"] = (
        sum(v for k, v in reasons.items() if "output16_prompt_le_1536_remote" in k) == 32
        and sum(v for k, v in reasons.items() if "output16_prompt_gt_1536_local" in k) == 16)
    tp, lp = tempo["performance"], local["performance"]
    gates["output16_split_goodput_retains_98pct_local"] = (
        tp["slo_goodput"]["request_goodput_per_s"] >=
        0.98 * lp["slo_goodput"]["request_goodput_per_s"])
    value["schema"] = "tempo-pd-mixed-prompt-output16-split-analysis-99"
    value["prompt_buckets"] = buckets
    value["passes"] = all(gates.values())
    value["verdict"] = (
        "promising_output16_prompt_split" if value["passes"]
        else "reject_output16_prompt_split")
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"], "buckets": buckets,
                      "failed": [k for k, v in gates.items() if not v]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
