#!/usr/bin/env python3
"""Apply per-prompt-bucket gates to the mixed output16 diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def _pairs(tempo: dict, baseline: dict, prefix: str) -> dict:
    left = {row["request_id"]: row for row in tempo["request_metrics"]
            if prefix in row["request_id"]}
    right = {row["request_id"]: row for row in baseline["request_metrics"]
             if prefix in row["request_id"]}
    if left.keys() != right.keys() or len(left) != 16:
        raise ValueError(f"{prefix}: exact 16 paired rows required")
    deltas = [left[key]["e2e_ms"] - right[key]["e2e_ms"] for key in sorted(left)]
    return {"count": len(deltas), "wins": sum(value < 0 for value in deltas),
            "median_delta_ms": statistics.median(deltas), "raw_delta_ms": deltas}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    value = json.loads(args.input.read_text(encoding="utf-8"))
    if value.get("schema") != "tempo-pd-output128-diagnostic-analysis-96":
        raise ValueError("v96-shaped analysis required")
    tempo, local, remote = value["tempo"], value["fixed_local"], value["lmcache_remote"]
    if any(row["completion_tokens"] != 16 for arm in (tempo, local, remote)
           for row in arm["request_metrics"]):
        raise ValueError("every measured request must generate exactly 16 tokens")
    buckets = {}
    gates = value["gates"]
    for name in tuple(gates):
        if name.startswith("output128_"):
            del gates[name]
    for bucket in ("512", "1230", "2048"):
        local_pair = _pairs(tempo, local, f"mix{bucket}-")
        remote_pair = _pairs(tempo, remote, f"mix{bucket}-")
        buckets[bucket] = {"tempo_minus_local": local_pair,
                           "tempo_minus_lmcache": remote_pair}
        gates[f"bucket_{bucket}_local_noninferior"] = (
            local_pair["wins"] >= 8 and local_pair["median_delta_ms"] <= 10.0)
        gates[f"bucket_{bucket}_beats_lmcache"] = (
            remote_pair["wins"] >= 9 and remote_pair["median_delta_ms"] < 0.0)
    reasons = tempo["reasons"]
    gates["output16_routes_48_local"] = tempo["routes"] == {
        "decoder_local_recompute_or_cache": 48}
    gates["output16_guard_reason_48"] = sum(
        count for reason, count in reasons.items()
        if reason.startswith("same_server_tempo_measured:workload_guard_local:")
    ) == 48
    value["schema"] = "tempo-pd-mixed-prompt-output16-diagnostic-analysis-98"
    value["prompt_buckets"] = buckets
    value["passes"] = all(gates.values())
    value["verdict"] = (
        "promising_mixed_prompt_output16_guard" if value["passes"]
        else "reject_mixed_prompt_output16_guard")
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"], "buckets": buckets,
                      "failed": [key for key, passed in gates.items() if not passed]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
