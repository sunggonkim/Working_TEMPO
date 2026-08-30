#!/usr/bin/env python3
"""Finalize mid-load local-routing non-inferiority and LMCache comparison."""

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
    for key in (
        "tempo_goodput_beats_local",
        "tempo_paired_majority_beats_local",
        "tempo_paired_median_beats_local",
    ):
        del gates[key]
    tempo = value["tempo"]["performance"]
    local = value["fixed_local"]["performance"]
    paired = value["paired_tempo_minus_local"]
    gates["tempo_mid_routes_48_local"] = value["tempo"]["routes"] == {
        "decoder_local_recompute_or_cache": 48,
    }
    gates["tempo_mid_local_goodput_retains_98_percent"] = (
        tempo["slo_goodput"]["request_goodput_per_s"]
        >= 0.98 * local["slo_goodput"]["request_goodput_per_s"]
    )
    gates["tempo_mid_local_paired_median_within_25ms"] = abs(
        paired["e2e_delta_median_ms"]) <= 25.0
    gates["tempo_mid_local_p99_within_5_percent"] = (
        tempo["e2e_ms"]["p99"] <= 1.05 * local["e2e_ms"]["p99"]
    )
    value["schema"] = "tempo-pd-same-server-balanced-midlocal-analysis-79"
    value["controller_variant"] = {
        "request_rate_per_s": 24, "arrival_regime": "mid", "route": "decoder_local"
    }
    value["passes"] = all(gates.values())
    value["verdict"] = (
        "promising_order_balanced_midlocal" if value["passes"]
        else "reject_order_balanced_midlocal"
    )
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"], "gates": gates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
