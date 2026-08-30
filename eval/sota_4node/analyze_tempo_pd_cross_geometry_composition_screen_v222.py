#!/usr/bin/env python3
"""Judge the mixed composition candidate against fixed-local at rate48."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LOCAL = "decoder_local_recompute_or_cache"
REMOTE = "remote_prefill_live_kv"


def analyze(value: dict) -> dict:
    if value.get("schema") != "tempo-pd-hybrid-saturation-analysis-192":
        raise ValueError("saturation result schema mismatch")
    tempo = value["tempo"]
    local = value["fixed_local_primary"]
    tp = tempo["performance"]
    lp = local["performance"]
    pair = value["paired_tempo_minus_local"]
    gates = {
        "tempo_routes_38_local_10_remote": (
            tempo.get("routes") == {LOCAL: 38, REMOTE: 10}),
        "fixed_local_routes_48_local": local.get("routes") == {LOCAL: 48},
        "all_tempo_slo_valid": tp["slo_goodput"]["success_fraction"] == 1.0,
        "tempo_throughput_beats_local": (
            tp["request_throughput_per_s"] > lp["request_throughput_per_s"]),
        "tempo_e2e_p99_beats_local": tp["e2e_ms"]["p99"] < lp["e2e_ms"]["p99"],
        "tempo_tpot_p99_within_5pct_local": (
            tp["tpot_ms"]["p99"] <= 1.05 * lp["tpot_ms"]["p99"]),
        "tempo_paired_majority_beats_local": (
            pair["e2e_win_count"] >= 25
            and pair["e2e_delta_median_ms"] < 0.0),
    }
    summary = {
        "tempo_throughput_per_s": tp["request_throughput_per_s"],
        "local_throughput_per_s": lp["request_throughput_per_s"],
        "throughput_gain_vs_local_percent": 100.0 * (
            tp["request_throughput_per_s"] / lp["request_throughput_per_s"] - 1.0),
        "tempo_e2e_p99_ms": tp["e2e_ms"]["p99"],
        "local_e2e_p99_ms": lp["e2e_ms"]["p99"],
        "tempo_tpot_p99_ms": tp["tpot_ms"]["p99"],
        "local_tpot_p99_ms": lp["tpot_ms"]["p99"],
        "paired_local_win_count": pair["e2e_win_count"],
        "paired_local_delta_median_ms": pair["e2e_delta_median_ms"],
    }
    result = {
        "schema": "tempo-pd-cross-geometry-composition-screen-222",
        "policy": "qwen25-7b-tp4x2-warm-affinity-8",
        "rate": 48,
        "summary": summary,
        "gates": gates,
        "claim_boundary": (
            "One actual Qwen2.5-7B TP4+TP4 Tempo/fixed-local screen at rate48. "
            "The pinned LMCache arm is intentionally absent after its completed "
            "HTTP responses repeatedly failed to flush the result artifact."),
    }
    result["passes"] = all(gates.values())
    result["verdict"] = (
        "composition_guard_beats_local" if result["passes"]
        else "composition_guard_needs_revision")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing overwrite: {args.output}")
    value = json.loads(args.input.resolve().read_text(encoding="utf-8"))
    result = analyze(value)
    args.output.resolve().write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"],
                      "failed": [key for key, passed in result["gates"].items()
                                 if not passed]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
