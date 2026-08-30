#!/usr/bin/env python3
"""Summarize the validated load envelope and the rate-64 Pareto boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LOCAL = "decoder_local_recompute_or_cache"
REMOTE = "remote_prefill_live_kv"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: object required")
    return value


def _standard(value: dict, rate: int) -> tuple[dict, dict]:
    if value.get("schema") != "tempo-pd-production-hybrid-controller-analysis-151":
        raise ValueError(f"rate{rate}: production schema mismatch")
    t = value["tempo"]["performance"]
    l = value["fixed_local"]["performance"]
    m = value["lmcache_remote"]["performance"]
    pair = value["paired_tempo_minus_lmcache"]
    summary = {
        "rate": rate,
        "tempo_throughput_per_s": t["request_throughput_per_s"],
        "local_throughput_per_s": l["request_throughput_per_s"],
        "lmcache_throughput_per_s": m["request_throughput_per_s"],
        "throughput_gain_vs_lmcache_percent": 100 * (
            t["request_throughput_per_s"] / m["request_throughput_per_s"] - 1),
        "tempo_e2e_p99_ms": t["e2e_ms"]["p99"],
        "local_e2e_p99_ms": l["e2e_ms"]["p99"],
        "lmcache_e2e_p99_ms": m["e2e_ms"]["p99"],
        "e2e_p99_reduction_vs_lmcache_percent": 100 * (
            1 - t["e2e_ms"]["p99"] / m["e2e_ms"]["p99"]),
        "tempo_tpot_p99_ms": t["tpot_ms"]["p99"],
        "local_tpot_p99_ms": l["tpot_ms"]["p99"],
        "lmcache_tpot_p99_ms": m["tpot_ms"]["p99"],
        "tpot_p99_reduction_vs_lmcache_percent": 100 * (
            1 - t["tpot_ms"]["p99"] / m["tpot_ms"]["p99"]),
        "paired_win_count": pair["e2e_win_count"],
        "paired_e2e_delta_median_ms": pair["e2e_delta_median_ms"],
    }
    gates = {
        f"rate{rate}_throughput_beats_lmcache": (
            t["request_throughput_per_s"] > m["request_throughput_per_s"]),
        f"rate{rate}_e2e_p99_beats_lmcache": (
            t["e2e_ms"]["p99"] < m["e2e_ms"]["p99"]),
        f"rate{rate}_tpot_p99_beats_lmcache": (
            t["tpot_ms"]["p99"] < m["tpot_ms"]["p99"]),
        f"rate{rate}_paired_majority_beats_lmcache": pair["e2e_win_count"] >= 25,
        f"rate{rate}_paired_median_beats_lmcache": pair["e2e_delta_median_ms"] < 0,
        f"rate{rate}_e2e_p99_within_half_percent_local": (
            t["e2e_ms"]["p99"] <= 1.005 * l["e2e_ms"]["p99"]),
    }
    return summary, gates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate40", type=Path, required=True)
    parser.add_argument("--rate48-reproduction", type=Path, required=True)
    parser.add_argument("--rate56", type=Path, required=True)
    parser.add_argument("--rate64-base", type=Path, required=True)
    parser.add_argument("--rate64-tail", type=Path, required=True)
    parser.add_argument("--rate64-lmcache-failure-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rate40, gates40 = _standard(_load(args.rate40.resolve()), 40)
    rate56, gates56 = _standard(_load(args.rate56.resolve()), 56)
    rate48 = _load(args.rate48_reproduction.resolve())
    if rate48.get("schema") != "tempo-pd-production-cross-allocation-reproduction-169":
        raise ValueError("rate48 reproduction schema mismatch")
    base64 = _load(args.rate64_base.resolve())
    tail64 = _load(args.rate64_tail.resolve())
    if base64.get("schema") != "tempo-pd-hybrid-saturation-analysis-192":
        raise ValueError("rate64 base schema mismatch")
    if tail64.get("schema") != "tempo-pd-hybrid-tailaware-analysis-197":
        raise ValueError("rate64 tail schema mismatch")
    bt = base64["tempo"]["performance"]
    bl = base64["fixed_local_primary"]["performance"]
    tt = tail64["tempo"]["performance"]
    tl = tail64["fixed_local_primary"]["performance"]
    bp = base64["paired_tempo_minus_local"]
    failure = args.rate64_lmcache_failure_root.resolve()
    cold = _load(failure / "tempo_credit_admission/hybrid_cold_transition.raw.json")
    failed_final_absent = not (failure / "hybrid_controller_final.json").exists()
    failed_warm_raw_absent = not any(
        (failure / "tempo_credit_admission/same_server_balanced_warm").glob("*.raw.json"))

    gates = {
        **gates40,
        "rate48_cross_allocation_reproduction_passes": rate48.get("passes") is True,
        **gates56,
        "rate64_base_all_correctness_gates_except_tpot_pass": all(
            passed for key, passed in base64["gates"].items()
            if key != "tempo_tpot_p99_within_10pct_local"),
        "rate64_base_throughput_beats_local": (
            bt["request_throughput_per_s"] > bl["request_throughput_per_s"]),
        "rate64_base_e2e_p99_beats_local": bt["e2e_ms"]["p99"] < bl["e2e_ms"]["p99"],
        "rate64_base_paired_median_beats_local": bp["e2e_delta_median_ms"] < 0,
        "rate64_tail_tpot_within_10pct_local": (
            tt["tpot_ms"]["p99"] <= 1.10 * tl["tpot_ms"]["p99"]),
        "rate64_tail_e2e_p99_beats_local": tt["e2e_ms"]["p99"] < tl["e2e_ms"]["p99"],
        "rate64_lmcache_attempt_cold_phase_completed": (
            len(cold.get("requests", [])) == 24
            and all(row.get("error") is None for row in cold["requests"])),
        "rate64_lmcache_warm_arm_did_not_finish": (
            failed_final_absent and failed_warm_raw_absent),
    }
    result = {
        "schema": "tempo-pd-load-envelope-analysis-204",
        "rate40": rate40,
        "rate48_reproduction": rate48,
        "rate56": rate56,
        "rate64": {
            "throughput_e2e_policy": base64,
            "tail_policy": tail64,
            "lmcache_incomplete_attempt_root": str(failure),
            "lmcache_result_available": False,
            "interpretation": (
                "Base placement is the throughput/E2E point; tail-aware placement is "
                "the TPOT point. Neither dominates the other. The pinned LMCache "
                "always-remote warm arm did not produce a completed result at rate64."
            ),
        },
        "gates": gates,
        "passes": all(gates.values()),
        "claim_boundary": (
            "Actual Qwen2.5-7B vLLM TP4+TP4 P/D, four A100 nodes. Complete "
            "LMCache comparisons at rates40/48/56; rate64 is an availability/Pareto "
            "screen because the LMCache arm did not finish."
        ),
    }
    result["verdict"] = (
        "load_envelope_validated_with_rate64_boundary" if result["passes"]
        else "load_envelope_needs_revision")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    args.output.resolve().write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"],
                      "failed": [key for key, passed in gates.items() if not passed]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
