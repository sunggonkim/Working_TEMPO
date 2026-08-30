#!/usr/bin/env python3
"""Fail-closed analysis for the short-prompt/output128 local diagnostic."""

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
        "tempo_routes_32_local_16_remote",
        "tempo_goodput_beats_local",
        "tempo_paired_majority_beats_local",
        "tempo_paired_median_beats_local",
    ):
        if name not in gates:
            raise ValueError(f"missing replaced gate: {name}")
        del gates[name]
    tempo, local = value["tempo"], value["fixed_local"]
    tp, lp = tempo["performance"], local["performance"]
    pair = value["paired_tempo_minus_local"]
    reasons = tempo["reasons"]
    gates.update({
        "output128_routes_48_local": tempo["routes"] == {
            "decoder_local_recompute_or_cache": 48},
        "output128_workload_guard_reason_48": sum(
            count for reason, count in reasons.items()
            if reason.startswith("same_server_tempo_measured:workload_guard_local:")
        ) == 48,
        "output128_goodput_retains_98pct_local": (
            tp["slo_goodput"]["request_goodput_per_s"] >=
            0.98 * lp["slo_goodput"]["request_goodput_per_s"]),
        "output128_pair_win_count_at_least_half": pair["e2e_win_count"] >= 24,
        "output128_paired_median_within_10ms_local": (
            pair["e2e_delta_median_ms"] <= 10.0),
        "output128_e2e_p99_within_5pct_local": (
            tp["e2e_ms"]["p99"] <= 1.05 * lp["e2e_ms"]["p99"]),
    })
    value["schema"] = "tempo-pd-output128-diagnostic-analysis-96"
    value["controller_variant"] = {
        "output_tokens": 128, "prompt_tokens_max": 512,
        "policy": "diagnostic_workload_guard_local",
    }
    value["passes"] = all(gates.values())
    value["verdict"] = (
        "promising_output128_local_guard" if value["passes"]
        else "reject_output128_local_guard")
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"], "gates": gates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
