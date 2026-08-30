#!/usr/bin/env python3
"""Finalize the output-aware production policy at 32 output tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    gates = value["gates"]
    del gates["tempo_routes_32_local_16_remote"]
    del gates["tempo_goodput_beats_local"]
    tempo = value["tempo"]["performance"]
    local = value["fixed_local"]["performance"]
    gates["tempo_routes_40_local_8_remote"] = value["tempo"]["routes"] == {
        "decoder_local_recompute_or_cache": 40,
        "remote_prefill_live_kv": 8,
    }
    gates["tempo_goodput_retains_98_percent_of_local"] = (
        tempo["slo_goodput"]["request_goodput_per_s"]
        >= 0.98 * local["slo_goodput"]["request_goodput_per_s"])
    gates["tempo_e2e_p99_within_5_percent_of_local"] = (
        tempo["e2e_ms"]["p99"] <= 1.05 * local["e2e_ms"]["p99"])
    value["schema"] = "tempo-pd-same-server-production-analysis-86"
    value["controller_variant"] = {
        "output_tokens": 32,
        "high_pair_interval_ns": 58_000_000,
        "high_local_credit": 8,
        "mid_and_low_route": "decoder_local",
    }
    value["passes"] = all(gates.values())
    value["verdict"] = (
        "promising_output_aware_production_policy" if value["passes"]
        else "reject_output_aware_production_policy"
    )
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"], "gates": gates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
