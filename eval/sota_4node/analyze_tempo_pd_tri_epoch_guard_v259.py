#!/usr/bin/env python3
"""Fail-closed report for the three-arm live epoch guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node.run_tempo_pd_same_server_tri_epoch_guard_client_v256 import select_mode
from eval.sota_4node.tempo_pd_same_server_tri_epoch_guard_router_v255 import MODE_SCHEMA


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def analyze(root: Path, allocation: int) -> dict:
    root = root.resolve()
    final = json.loads((root / "hybrid_controller_final.json").read_text())
    stage = root / "tempo_credit_admission"
    mode = json.loads((stage / "epoch_mode.json").read_text())
    _require(mode.get("schema") == MODE_SCHEMA, "mode schema mismatch")
    _require(mode.get("calibration_replicates_per_candidate") == 3,
             "calibration replicate mismatch")
    recomputed = select_mode(mode["arms"])
    selected = mode.get("selected_mode")
    _require(selected == recomputed["selected_mode"], "mode selection mismatch")
    _require(final.get("schema") == "tempo-pd-production-hybrid-controller-analysis-151",
             "hybrid final schema mismatch")
    arms = {name: final[name] for name in
            ("fixed_local", "tempo", "lmcache_remote")}
    for name, value in arms.items():
        _require(value.get("request_count") == 48, f"{name}: request count")
        _require(len(value.get("request_metrics", [])) == 48,
                 f"{name}: request metric count")
        _require(all(row.get("slo_pass") is True
                     for row in value["request_metrics"]), f"{name}: SLO")
    _require(arms["fixed_local"]["routes"] == {
        "decoder_local_recompute_or_cache": 48}, "fixed route mismatch")
    _require(arms["lmcache_remote"]["routes"] == {
        "remote_prefill_live_kv": 48}, "remote route mismatch")
    expected_routes = {
        "policy8": {"decoder_local_recompute_or_cache": 38,
                    "remote_prefill_live_kv": 10},
        "fixed_local": {"decoder_local_recompute_or_cache": 48},
        "lmcache_remote": {"remote_prefill_live_kv": 48},
    }[selected]
    _require(arms["tempo"]["routes"] == expected_routes,
             "selected tempo route partition mismatch")
    expected_reason = {
        "fixed_local": "same_server_tempo_measured:tri_epoch_fixed_local",
        "lmcache_remote": "same_server_tempo_measured:tri_epoch_lmcache_remote",
    }.get(selected)
    if expected_reason:
        _require(arms["tempo"]["reasons"] == {expected_reason: 48},
                 "selected tempo reason mismatch")
    _require(final["gates"].get("exact_workload") is True,
             "workload validation failed")
    _require(final["gates"].get("fixed_baselines_exact") is True,
             "fixed baseline validation failed")
    _require(final["gates"].get("arm_isolated_warm_reuse_contract") is True,
             "warm reuse contract failed")
    _require(final["gates"].get("stable_cache_catalog_identity") is True,
             "cache catalog identity failed")

    tempo = arms["tempo"]["performance"]
    remote = arms["lmcache_remote"]["performance"]
    local = arms["fixed_local"]["performance"]
    tput = tempo["request_throughput_per_s"]
    remote_tput = remote["request_throughput_per_s"]
    e2e = tempo["e2e_ms"]["p99"]
    remote_e2e = remote["e2e_ms"]["p99"]
    tpot = tempo["tpot_ms"]["p99"]
    remote_tpot = remote["tpot_ms"]["p99"]
    paired = final["paired_tempo_minus_lmcache"]
    strict = {
        "throughput_beats_lmcache": tput > remote_tput,
        "e2e_p99_beats_lmcache": e2e < remote_e2e,
        "tpot_p99_beats_lmcache": tpot < remote_tpot,
        "paired_majority_beats_lmcache": paired["e2e_win_count"] >= 25,
        "paired_median_beats_lmcache": paired["e2e_delta_median_ms"] < 0.0,
    }
    fallback_safe = (
        tput >= 0.98 * remote_tput
        and e2e <= 1.02 * remote_e2e
        and tpot <= 1.10 * remote_tpot
    )
    all_strict = all(strict.values())
    primary = selected != "lmcache_remote" and all_strict
    if primary:
        verdict = "tri_epoch_selected_policy_beats_lmcache"
    elif selected == "lmcache_remote" and fallback_safe:
        verdict = "tri_epoch_safe_lmcache_fallback_no_advantage_claim"
    else:
        verdict = "tri_epoch_selection_or_performance_failed"
    return {
        "schema": "tempo-pd-tri-epoch-analysis-259",
        "allocation_id": allocation,
        "root": str(root),
        "selected_mode": selected,
        "selection": mode,
        "calibration_recomputed": recomputed,
        "routes": arms["tempo"]["routes"],
        "summary": {
            "tempo_throughput_per_s": tput,
            "lmcache_throughput_per_s": remote_tput,
            "throughput_gain_percent": 100.0 * (tput / remote_tput - 1.0),
            "tempo_e2e_p99_ms": e2e,
            "lmcache_e2e_p99_ms": remote_e2e,
            "e2e_p99_reduction_percent": 100.0 * (1.0 - e2e / remote_e2e),
            "tempo_tpot_p99_ms": tpot,
            "lmcache_tpot_p99_ms": remote_tpot,
            "tpot_p99_reduction_percent": 100.0 * (1.0 - tpot / remote_tpot),
            "local_throughput_per_s": local["request_throughput_per_s"],
            "paired_e2e_delta_median_ms": paired["e2e_delta_median_ms"],
            "paired_e2e_win_count": paired["e2e_win_count"],
        },
        "strict_lmcache_gates": strict,
        "fallback_safe": fallback_safe,
        "primary_advantage_passes": primary,
        "verdict": verdict,
        "claim_boundary": (
            "One actual Qwen2.5-7B vLLM TP4+TP4 P/D lifecycle. Calibration "
            "is unmeasured; performance is from the unchanged measured six-block "
            "crossover. LMCache fallback is safety, not an advantage claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    report = analyze(args.root, args.allocation)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected_mode": report["selected_mode"],
                      "verdict": report["verdict"],
                      "summary": report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
