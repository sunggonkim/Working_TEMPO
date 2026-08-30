#!/usr/bin/env python3
"""Saturated mixed-output gates for the multiepoch production controller."""

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
    routes, reasons = tempo["routes"], tempo["reasons"]
    gates.update({
        "tempo_routes_exactly_48_mixed": (
            sum(routes.values()) == 48
            and routes.get("decoder_local_recompute_or_cache", 0) > 0
            and routes.get("remote_prefill_live_kv", 0) > 0),
        "output16_direct_reason_12": sum(
            count for reason, count in reasons.items()
            if reason.endswith("output16_direct_local_fast_path")) == 12,
        "output128_direct_reason_12": sum(
            count for reason, count in reasons.items()
            if reason.endswith("output128_direct_local_fast_path")) == 12,
        "tempo_goodput_retains_98pct_local": (
            tp["slo_goodput"]["request_goodput_per_s"] >=
            0.98 * lp["slo_goodput"]["request_goodput_per_s"]),
        "tempo_request_throughput_beats_lmcache": (
            tp["request_throughput_per_s"] > rp["request_throughput_per_s"]),
        "tempo_e2e_p99_within_5pct_local": (
            tp["e2e_ms"]["p99"] <= 1.05 * lp["e2e_ms"]["p99"]),
    })
    value["schema"] = "tempo-pd-heterogeneous-saturated-analysis-121"
    value["passes"] = all(gates.values())
    value["verdict"] = (
        "promising_multiepoch_saturated_controller" if value["passes"]
        else "reject_multiepoch_saturated_controller")
    value["claim_boundary"] = (
        "One live server lifecycle, six order-balanced cold-key-disjoint blocks, "
        "mixed output16/32/64/128 at a predeclared 64 request/s offered load.")
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"],
                      "failed": [k for k, passed in gates.items() if not passed]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
