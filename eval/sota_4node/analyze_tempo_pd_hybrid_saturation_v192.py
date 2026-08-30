#!/usr/bin/env python3
"""Analyze a live rate-saturation TEMPO/fixed-local crossover."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_performance_v1 as metrics
from eval.sota_4node import analyze_tempo_pd_same_server_balanced_v71 as balanced


LOCAL = "decoder_local_recompute_or_cache"
REMOTE = "remote_prefill_live_kv"
ORDER = (
    "fixed_local", "tempo", "fixed_local",
    "fixed_local", "tempo", "fixed_local",
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: object required")
    return value


def _parse(stage_root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    root = stage_root / "same_server_balanced_measured"
    local_rows = []
    tempo_rows = []
    all_rows = []
    counts = {"fixed_local": 0, "tempo": 0}
    for index, arm in enumerate(ORDER):
        replicate = counts[arm]
        counts[arm] += 1
        key = f"{index:02d}_{arm}_r{replicate}"
        normalized, contract = balanced._normalize(_load(root / f"{key}.raw.json"), arm)
        if contract.get("sequence_index") != index or contract.get("replicate") != replicate:
            raise ValueError(f"{key}: sequence contract mismatch")
        parsed = metrics._parse_run(
            key, normalized, ttft_slo_ms=3000, tpot_slo_ms=250,
            e2e_slo_ms=12000)
        all_rows.append(parsed)
        (local_rows if arm == "fixed_local" else tempo_rows).append(parsed)
    return local_rows, tempo_rows, all_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage_root = args.stage_root.resolve()
    local_rows, tempo_rows, all_rows = _parse(stage_root)
    first = all_rows[0]
    exact = all(
        row["model_config_sha256"] == first["model_config_sha256"]
        and row["workload_sha256"] == first["workload_sha256"]
        and row["_contracts"] == first["_contracts"]
        and row["_outputs"] == first["_outputs"] for row in all_rows)
    local = balanced._combine(local_rows[:2], "fixed_local")
    tempo = balanced._combine(tempo_rows, "tempo")
    pair = metrics._paired(tempo, local) if exact else None
    lp = local["performance"]
    tp = tempo["performance"]
    extra_local_valid = all(
        row["routes"] == {LOCAL: 24}
        and row["performance"]["slo_goodput"]["success_fraction"] == 1.0
        for row in local_rows[2:])

    warm_root = stage_root / "same_server_balanced_warm"
    seed0 = _load(warm_root / "01_tempo_r0.raw.json")
    seed1 = _load(warm_root / "02_tempo_r1.raw.json")
    seed_valid = all(
        len(value.get("requests", [])) == 24
        and all(row.get("error") is None and not row.get("contract_violations")
                for row in value["requests"])
        and sum(row.get("route") == LOCAL for row in value["router_decisions"]) == 16
        and sum(row.get("route") == REMOTE for row in value["router_decisions"]) == 8
        and all(row.get("reason") == "same_server_tempo_warm:cache_affinity_warm_seed"
                for row in value["router_decisions"])
        for value in (seed0, seed1))

    gates = {
        "one_live_server_saturation_order_exact": len(all_rows) == 6,
        "all_six_blocks_exact_outputs": exact,
        "both_tempo_replicates_complete": len(tempo_rows) == 2,
        "four_local_replicates_complete": len(local_rows) == 4 and extra_local_valid,
        "warm_seed_replicates_exact": seed_valid,
        "tempo_routes_32_local_16_remote": tempo["routes"] == {LOCAL: 32, REMOTE: 16},
        "tempo_all_requests_slo_valid": (
            tp["slo_goodput"]["success_fraction"] == 1.0),
        "tempo_throughput_beats_local": (
            tp["request_throughput_per_s"] > lp["request_throughput_per_s"]),
        "tempo_e2e_p99_within_2pct_local": (
            tp["e2e_ms"]["p99"] <= 1.02 * lp["e2e_ms"]["p99"]),
        "tempo_tpot_p99_within_10pct_local": (
            tp["tpot_ms"]["p99"] <= 1.10 * lp["tpot_ms"]["p99"]),
        "tempo_paired_majority_beats_local": (
            pair is not None and pair["e2e_win_count"] >= 25),
        "tempo_paired_median_beats_local": (
            pair is not None and pair["e2e_delta_median_ms"] < 0),
    }
    public = lambda row: {key: value for key, value in row.items()
                          if not key.startswith("_")}
    result = {
        "schema": "tempo-pd-hybrid-saturation-analysis-192",
        "arm_order": list(ORDER),
        "fixed_local_primary": public(local),
        "tempo": public(tempo),
        "paired_tempo_minus_local": pair,
        "gates": gates,
        "passes": all(gates.values()),
        "claim_boundary": (
            "Rate-64 actual vLLM P/D availability screen after the pinned LMCache "
            "always-remote arm failed to finish. This artifact compares only TEMPO "
            "and fixed-local in a fresh one-live-server epoch."
        ),
    }
    result["verdict"] = (
        "tempo_saturation_available" if result["passes"]
        else "tempo_saturation_needs_revision")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"],
                      "failed": [key for key, passed in gates.items() if not passed]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
