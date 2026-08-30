#!/usr/bin/env python3
"""Saturated throughput gates for production output128 direct-local routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    gates = value["gates"]
    for name in (
        "tempo_routes_32_local_16_remote", "tempo_goodput_beats_local",
        "tempo_paired_majority_beats_local", "tempo_paired_median_beats_local",
    ):
        del gates[name]
    tempo, local, remote = value["tempo"], value["fixed_local"], value["lmcache_remote"]
    tp, lp, rp = (row["performance"] for row in (tempo, local, remote))
    gates.update({
        "output128_direct_routes_48_local": tempo["routes"] == {
            "decoder_local_recompute_or_cache": 48},
        "output128_direct_reason_48": sum(
            count for reason, count in tempo["reasons"].items()
            if reason.endswith("output128_direct_local_fast_path")) == 48,
        "output128_goodput_retains_98pct_local": (
            tp["slo_goodput"]["request_goodput_per_s"] >=
            0.98 * lp["slo_goodput"]["request_goodput_per_s"]),
        "output128_goodput_beats_lmcache": (
            tp["slo_goodput"]["request_goodput_per_s"] >
            rp["slo_goodput"]["request_goodput_per_s"]),
        "output128_request_throughput_beats_lmcache": (
            tp["request_throughput_per_s"] > rp["request_throughput_per_s"]),
        "output128_e2e_p99_beats_lmcache": (
            tp["e2e_ms"]["p99"] < rp["e2e_ms"]["p99"]),
        "output128_tpot_p99_beats_lmcache": (
            tp["tpot_ms"]["p99"] < rp["tpot_ms"]["p99"]),
    })
    value["schema"] = "tempo-pd-output128-saturated-analysis-112"
    value["passes"] = all(gates.values())
    value["verdict"] = (
        "promising_output128_saturated_direct_local" if value["passes"]
        else "reject_output128_saturated_direct_local")
    value["claim_boundary"] = (
        "One live server lifecycle with order-balanced, cold-key-disjoint, "
        "single-arm saturated blocks at the frozen offered load.")
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"],
                      "failed": [k for k, passed in gates.items() if not passed]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
