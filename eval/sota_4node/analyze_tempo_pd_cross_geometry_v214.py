#!/usr/bin/env python3
"""Synthesize the frozen phase-aware policy across rates and KV geometries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tempo.pd_admission import PDRoute
from tempo.pd_cache_affinity import POLICY_ID, calibrated_route


def _load(path: Path, schema: str) -> dict:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{path}: expected {schema}")
    return value


def _only_failed(value: dict, allowed: set[str]) -> bool:
    gates = value.get("gates")
    return (isinstance(gates, dict)
            and {key for key, passed in gates.items() if passed is False} == allowed
            and all(type(passed) is bool for passed in gates.values()))


def _policy_contract() -> bool:
    expected_remote = {
        (512, 32), (512, 64), (512, 128), (2048, 64), (2048, 256)}
    for prompt in (512, 1230, 2048):
        for output in (16, 32, 64, 128, 256):
            expected = (PDRoute.REMOTE_PREFILL
                        if (prompt, output) in expected_remote
                        else PDRoute.DECODER_LOCAL)
            if calibrated_route(prompt, output) is not expected:
                return False
    for prompt in (4094, 4096):
        for output in (16, 128):
            if calibrated_route(prompt, output) is not PDRoute.DECODER_LOCAL:
                return False
        for output in (32, 64, 256):
            try:
                calibrated_route(prompt, output)
            except ValueError:
                continue
            return False
    return POLICY_ID == "qwen25-7b-tp4x2-warm-affinity-7"


def analyze(core: dict, load: dict, output256: dict, prompt4094: dict) -> dict:
    out = output256["summary"]
    long = prompt4094["summary"]
    output_allowed = {
        "tempo_throughput_beats_local", "tempo_e2e_p99_beats_local",
        "tempo_tpot_p99_within_5pct_local",
        "tempo_paired_majority_beats_local"}
    prompt_allowed = {"tempo_paired_local_noninferior"}
    gates = {
        "same_epoch_miss_seed_hit_core_validated": core.get("passes") is True,
        "rate40_48_56_load_envelope_validated": load.get("passes") is True,
        "output256_only_expected_local_tradeoff_gates_fail": (
            _only_failed(output256, output_allowed)),
        "output256_beats_lmcache_throughput": (
            out["tempo_throughput_per_s"] > out["lmcache_throughput_per_s"]),
        "output256_beats_lmcache_e2e_p99": (
            out["tempo_e2e_p99_ms"] < out["lmcache_e2e_p99_ms"]),
        "output256_beats_lmcache_tpot_p99": (
            out["tempo_tpot_p99_ms"] < out["lmcache_tpot_p99_ms"]),
        "output256_paired_majority_beats_lmcache": (
            out["paired_lmcache_win_count"] >= 25
            and out["paired_lmcache_delta_median_ms"] < 0.0),
        "output256_throughput_within_2pct_local": (
            out["tempo_throughput_per_s"] >= 0.98 * out["local_throughput_per_s"]),
        "output256_e2e_p99_within_2pct_local": (
            out["tempo_e2e_p99_ms"] <= 1.02 * out["local_e2e_p99_ms"]),
        "prompt4094_only_expected_paired_local_gate_fails": (
            _only_failed(prompt4094, prompt_allowed)),
        "prompt4094_beats_lmcache_e2e_p99": (
            long["tempo_e2e_p99_ms"] < long["lmcache_e2e_p99_ms"]),
        "prompt4094_beats_lmcache_tpot_p99": (
            long["tempo_tpot_p99_ms"] < long["lmcache_tpot_p99_ms"]),
        "prompt4094_throughput_beats_lmcache": (
            long["tempo_throughput_per_s"] > long["lmcache_throughput_per_s"]),
        "prompt4094_paired_majority_beats_lmcache": (
            long["paired_lmcache_win_count"] >= 25
            and long["paired_lmcache_delta_median_ms"] < 0.0),
        "prompt4094_aggregate_local_noninferior": (
            long["tempo_throughput_per_s"] >= 0.98 * long["local_throughput_per_s"]
            and long["tempo_e2e_p99_ms"] <= 1.02 * long["local_e2e_p99_ms"]
            and long["tempo_tpot_p99_ms"] <= 1.02 * long["local_tpot_p99_ms"]
            and long["paired_local_delta_median_ms"] <= 20.0),
        "current_policy_exact_and_fail_closed": _policy_contract(),
    }
    rate_rows = {
        "40": load["rate40"],
        "48": {
            "throughput_gain_vs_lmcache_percent": load["rate48_reproduction"][
                "median_throughput_gain_vs_lmcache_percent"],
            "e2e_p99_reduction_vs_lmcache_percent": load["rate48_reproduction"][
                "median_e2e_p99_reduction_vs_lmcache_percent"],
            "tpot_p99_reduction_vs_lmcache_percent": load["rate48_reproduction"][
                "median_tpot_p99_reduction_vs_lmcache_percent"],
            "allocations": 2},
        "56": load["rate56"],
    }
    repo = Path(__file__).resolve().parents[2]
    result = {
        "schema": "tempo-pd-cross-geometry-analysis-214",
        "policy": POLICY_ID,
        "controller": "tempo-pd-hybrid-controller-2",
        "evidence": {
            "core_same_epoch": core,
            "load_envelope": load,
            "output256": output256,
            "prompt4094": prompt4094,
        },
        "headline": {
            "validated_rate_comparisons": rate_rows,
            "output256": out,
            "prompt4094": long,
        },
        "gates": gates,
        "source_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in ("tempo/pd_hybrid_controller.py", "tempo/pd_cache_affinity.py",
                         "tempo/pd_workload_policy.py")},
        "claim_boundary": {
            "validated": (
                "Actual Qwen2.5-7B vLLM TP4+TP4 P/D on four A100 nodes; pinned "
                "LMCache remote and fixed-local baselines; output16-256, actual "
                "prompt lengths 512/1230/2048/4094; complete LMCache comparisons "
                "at rates 40/48/56."),
            "rate64": (
                "Tempo/local Pareto and availability screen only; the LMCache arm "
                "did not finish, so no rate64 LMCache performance win is claimed."),
            "fixed_local": (
                "An oracle/control, not the primary distributed-KV baseline. "
                "Output256 accepts a measured TPOT tradeoff while remaining within "
                "2% of local throughput and E2E p99."),
            "mooncake": (
                "No same-harness actual-vLLM Mooncake P/D result exists in this "
                "environment; no direct Mooncake superiority claim is made."),
        },
    }
    result["passes"] = all(gates.values())
    result["verdict"] = (
        "cross_geometry_lmcache_advantage_validated" if result["passes"]
        else "cross_geometry_policy_needs_revision")
    return result


def _markdown(value: dict) -> str:
    h = value["headline"]
    out, long = h["output256"], h["prompt4094"]
    lines = [
        "# TEMPO phase-aware P/D controller: cross-geometry evidence",
        "",
        f"Verdict: `{value['verdict']}`.",
        "",
        "## Main results",
        "",
        "| Workload | Throughput vs LMCache | E2E p99 vs LMCache | TPOT p99 vs LMCache | Paired result |",
        "|---|---:|---:|---:|---:|",
    ]
    for rate in ("40", "48", "56"):
        row = h["validated_rate_comparisons"][rate]
        lines.append(
            f"| Mixed output16-128, rate {rate} | "
            f"+{row['throughput_gain_vs_lmcache_percent']:.2f}% | "
            f"-{row['e2e_p99_reduction_vs_lmcache_percent']:.2f}% | "
            f"-{row['tpot_p99_reduction_vs_lmcache_percent']:.2f}% | see artifact |")
    lines.extend([
        f"| Output256, prompts512/1230/2048 | "
        f"+{100*(out['tempo_throughput_per_s']/out['lmcache_throughput_per_s']-1):.2f}% | "
        f"-{100*(1-out['tempo_e2e_p99_ms']/out['lmcache_e2e_p99_ms']):.2f}% | "
        f"-{100*(1-out['tempo_tpot_p99_ms']/out['lmcache_tpot_p99_ms']):.2f}% | "
        f"{out['paired_lmcache_win_count']}/48, median {out['paired_lmcache_delta_median_ms']:.1f} ms |",
        f"| Prompt4094, output16/128 | "
        f"+{100*(long['tempo_throughput_per_s']/long['lmcache_throughput_per_s']-1):.2f}% | "
        f"-{100*(1-long['tempo_e2e_p99_ms']/long['lmcache_e2e_p99_ms']):.2f}% | "
        f"-{100*(1-long['tempo_tpot_p99_ms']/long['lmcache_tpot_p99_ms']):.2f}% | "
        f"{long['paired_lmcache_win_count']}/48, median {long['paired_lmcache_delta_median_ms']:.1f} ms |",
        "",
        "## Frozen policy",
        "",
        "Cold misses use the validated local fast path. Warm cache items retain stable placement. "
        "Remote buckets are `(512,32)`, `(512,64)`, `(512,128)`, `(2048,64)`, and `(2048,256)`; "
        "other validated buckets are local. Prompt4094/4096 is accepted only for output16/128 and is local.",
        "",
        "## Boundaries",
        "",
        f"- Rate64: {value['claim_boundary']['rate64']}",
        f"- Local control: {value['claim_boundary']['fixed_local']}",
        f"- Mooncake: {value['claim_boundary']['mooncake']}",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--load", type=Path, required=True)
    parser.add_argument("--output256", type=Path, required=True)
    parser.add_argument("--prompt4094", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.report):
        if path.exists():
            raise ValueError(f"refusing overwrite: {path}")
    result = analyze(
        _load(args.core, "tempo-pd-same-epoch-phase-analysis-186"),
        _load(args.load, "tempo-pd-load-envelope-analysis-204"),
        _load(args.output256, "tempo-pd-output256-balanced-analysis-208"),
        _load(args.prompt4094, "tempo-pd-prompt4096-phase-analysis-212"))
    args.output.resolve().write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    args.report.resolve().write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"],
                      "failed": [key for key, passed in result["gates"].items()
                                 if not passed]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
