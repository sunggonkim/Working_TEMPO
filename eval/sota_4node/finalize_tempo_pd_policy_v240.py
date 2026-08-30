#!/usr/bin/env python3
"""Select the frozen mixed-workload policy from direct head-to-head epochs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("schema") != "tempo-pd-production-hybrid-controller-analysis-151":
        raise ValueError(f"{path}: unexpected schema")
    return value


def _metrics(value: dict) -> dict:
    result = {}
    for name in ("tempo", "lmcache_remote", "fixed_local"):
        arm = value[name]
        perf = arm["performance"]
        result[name] = {
            "routes": arm.get("routes"),
            "throughput_per_s": float(perf["request_throughput_per_s"]),
            "e2e_p99_ms": float(perf["e2e_ms"]["p99"]),
            "tpot_p99_ms": float(perf["tpot_ms"]["p99"]),
            "slo_success_fraction": float(perf["slo_goodput"]["success_fraction"]),
        }
    paired = value["paired_tempo_minus_lmcache"]
    result["paired"] = {
        "win_count": int(paired["e2e_win_count"]),
        "median_delta_ms": float(paired["e2e_delta_median_ms"]),
    }
    return result


def finalize(policy8_path: Path, policy9_path: Path) -> dict:
    p8 = _metrics(_load(policy8_path))
    p9 = _metrics(_load(policy9_path))
    if p8["tempo"]["routes"] != {
        "decoder_local_recompute_or_cache": 38,
        "remote_prefill_live_kv": 10,
    }:
        raise ValueError("policy8 route contract changed")
    if p9["tempo"]["routes"] != {
        "decoder_local_recompute_or_cache": 44,
        "remote_prefill_live_kv": 4,
    }:
        raise ValueError("policy9 route contract changed")

    def primary_pass(metrics: dict) -> bool:
        return (
            metrics["tempo"]["slo_success_fraction"] == 1.0
            and metrics["tempo"]["throughput_per_s"] > metrics["lmcache_remote"]["throughput_per_s"]
            and metrics["tempo"]["e2e_p99_ms"] < metrics["lmcache_remote"]["e2e_p99_ms"]
            and metrics["tempo"]["tpot_p99_ms"] < metrics["lmcache_remote"]["tpot_p99_ms"]
        )

    policy8_dominates_policy9 = (
        p8["tempo"]["throughput_per_s"] > p9["tempo"]["throughput_per_s"]
        and p8["tempo"]["e2e_p99_ms"] < p9["tempo"]["e2e_p99_ms"]
        and p8["tempo"]["tpot_p99_ms"] < p9["tempo"]["tpot_p99_ms"]
        and p8["paired"]["win_count"] > p9["paired"]["win_count"]
        and p8["paired"]["median_delta_ms"] < p9["paired"]["median_delta_ms"]
    )
    selected = primary_pass(p8) and not primary_pass(p9) and policy8_dominates_policy9
    return {
        "schema": "tempo-pd-policy-selection-240",
        "selected_policy": "qwen25-7b-tp4x2-warm-affinity-8" if selected else None,
        "verdict": "freeze_policy8" if selected else "selection_inconclusive",
        "policy8_primary_pass": primary_pass(p8),
        "policy9_primary_pass": primary_pass(p9),
        "policy8_dominates_policy9": policy8_dominates_policy9,
        "policy8": p8,
        "policy9": p9,
        "claim_boundary": (
            "One measured lifecycle per candidate in the same four-node allocation. "
            "Policy9 was a one-factor mixed-composition falsification and is rejected."
        ),
        "paths": {"policy8": str(policy8_path), "policy9": str(policy9_path)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy8", type=Path, required=True)
    parser.add_argument("--policy9", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing overwrite: {args.output}")
    result = finalize(args.policy8, args.policy9)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "verdict", "selected_policy", "policy8_primary_pass",
        "policy9_primary_pass", "policy8_dominates_policy9")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
