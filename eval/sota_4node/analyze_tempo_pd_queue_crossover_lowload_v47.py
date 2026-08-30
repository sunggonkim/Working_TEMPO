#!/usr/bin/env python3
"""Validate that the frozen threshold-eight controller bypasses remote P/D at low load."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from eval.sota_4node import analyze_tempo_pd_performance_v1 as base


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parse = lambda label, path: base._parse_run(
        label, _load(path), ttft_slo_ms=3000, tpot_slo_ms=250, e2e_slo_ms=12000)
    local = parse("local", args.reference / "crossover_local/raw.json")
    remote = parse("remote", args.reference / "crossover_remote/raw.json")
    tempo = parse("tempo", args.candidate_root / "tempo_credit_admission/raw.json")
    exact = (local["_contracts"] == remote["_contracts"] == tempo["_contracts"]
             and local["_outputs"] == remote["_outputs"] == tempo["_outputs"]
             and local["workload_sha256"] == remote["workload_sha256"] == tempo["workload_sha256"])
    tl = base._paired(tempo, local) if exact else None
    tr = base._paired(tempo, remote) if exact else None
    lp, rp, tp = local["performance"], remote["performance"], tempo["performance"]
    gates = {
        "exact_outputs_and_contracts": exact,
        "route_mix_is_9_local_0_remote": tempo["routes"] == {
            "decoder_local_recompute_or_cache": 9},
        "all_requests_slo_valid": tp["slo_goodput"]["success_fraction"] == 1.0,
        "e2e_p50_within_5_percent_of_fixed_local": tp["e2e_ms"]["p50"] <= 1.05 * lp["e2e_ms"]["p50"],
        "e2e_p99_within_5_percent_of_fixed_local": tp["e2e_ms"]["p99"] <= 1.05 * lp["e2e_ms"]["p99"],
        "goodput_at_least_95_percent_of_fixed_local": tp["slo_goodput"]["request_goodput_per_s"] >= 0.95 * lp["slo_goodput"]["request_goodput_per_s"],
        "e2e_p50_beats_all_remote": tp["e2e_ms"]["p50"] < rp["e2e_ms"]["p50"],
        "goodput_beats_all_remote": tp["slo_goodput"]["request_goodput_per_s"] > rp["slo_goodput"]["request_goodput_per_s"],
    }
    public = lambda value: {key: item for key, item in value.items() if not key.startswith("_")}
    result = {"schema":"tempo-pd-queue-crossover-lowload-analysis-47", "threshold":8,
              "local":public(local), "remote":public(remote), "tempo":public(tempo),
              "paired_tempo_minus_local":tl, "paired_tempo_minus_remote":tr,
              "gates":gates, "passes":all(gates.values()),
              "verdict":"promising_lowload_local_bypass" if all(gates.values()) else "reject_lowload_local_bypass"}
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"verdict":result["verdict"],"gates":gates},sort_keys=True))
    return 0 if result["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
