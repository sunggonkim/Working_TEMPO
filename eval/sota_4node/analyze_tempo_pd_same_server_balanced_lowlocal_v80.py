#!/usr/bin/env python3
"""Finalize low-load local-routing non-inferiority and LMCache comparison."""

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
    del gates["fixed_local_routes_48_local"]
    del gates["lmcache_routes_48_remote"]
    del gates["tempo_routes_32_local_16_remote"]
    for key in (
        "tempo_goodput_beats_local",
        "tempo_paired_majority_beats_local",
        "tempo_paired_median_beats_local",
        "tempo_paired_majority_beats_lmcache",
    ):
        del gates[key]
    tempo = value["tempo"]["performance"]
    local = value["fixed_local"]["performance"]
    pair_local = value["paired_tempo_minus_local"]
    pair_remote = value["paired_tempo_minus_lmcache"]
    gates["fixed_local_routes_18_local"] = value["fixed_local"]["routes"] == {
        "decoder_local_recompute_or_cache": 18}
    gates["lmcache_routes_18_remote"] = value["lmcache_remote"]["routes"] == {
        "remote_prefill_live_kv": 18}
    gates["tempo_low_routes_18_local"] = value["tempo"]["routes"] == {
        "decoder_local_recompute_or_cache": 18}
    gates["tempo_low_local_goodput_retains_98_percent"] = (
        tempo["slo_goodput"]["request_goodput_per_s"]
        >= 0.98 * local["slo_goodput"]["request_goodput_per_s"])
    gates["tempo_low_local_paired_median_within_25ms"] = abs(
        pair_local["e2e_delta_median_ms"]) <= 25.0
    gates["tempo_low_local_p99_within_5_percent"] = (
        tempo["e2e_ms"]["p99"] <= 1.05 * local["e2e_ms"]["p99"])
    gates["tempo_low_majority_beats_lmcache"] = pair_remote["e2e_win_count"] >= 10
    value["schema"] = "tempo-pd-same-server-balanced-lowlocal-analysis-80"
    value["controller_variant"] = {
        "request_rate_per_s": 16, "arrival_regime": "low", "route": "decoder_local"
    }
    value["passes"] = all(gates.values())
    value["verdict"] = (
        "promising_order_balanced_lowlocal" if value["passes"]
        else "reject_order_balanced_lowlocal"
    )
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"], "gates": gates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
