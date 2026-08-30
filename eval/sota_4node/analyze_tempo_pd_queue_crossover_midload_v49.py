#!/usr/bin/env python3
"""Evaluate the frozen threshold-eight controller at an intermediate arrival rate."""

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
    local_routes = tempo["routes"].get("decoder_local_recompute_or_cache", 0)
    remote_routes = tempo["routes"].get("remote_prefill_live_kv", 0)
    gates = {
        "exact_outputs_and_contracts": exact,
        "route_mix_is_dynamic_and_complete": local_routes + remote_routes == 24 and local_routes > 0 and remote_routes > 0,
        "all_requests_slo_valid": tp["slo_goodput"]["success_fraction"] == 1.0,
        "goodput_beats_local": tp["slo_goodput"]["request_goodput_per_s"] > lp["slo_goodput"]["request_goodput_per_s"],
        "goodput_beats_remote": tp["slo_goodput"]["request_goodput_per_s"] > rp["slo_goodput"]["request_goodput_per_s"],
        "e2e_p50_beats_local": tp["e2e_ms"]["p50"] < lp["e2e_ms"]["p50"],
        "e2e_p99_beats_local": tp["e2e_ms"]["p99"] < lp["e2e_ms"]["p99"],
        "tpot_p99_below_remote": tp["tpot_ms"]["p99"] < rp["tpot_ms"]["p99"],
    }
    public = lambda value: {key: item for key, item in value.items() if not key.startswith("_")}
    result = {"schema":"tempo-pd-queue-crossover-midload-analysis-49", "threshold":8,
              "arrival_rate_requests_per_s":24, "local":public(local), "remote":public(remote),
              "tempo":public(tempo), "paired_tempo_minus_local":tl,
              "paired_tempo_minus_remote":tr, "gates":gates, "passes":all(gates.values()),
              "verdict":"promising_midload_crossover" if all(gates.values()) else "reject_midload_crossover"}
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"verdict":result["verdict"],"routes":tempo["routes"],"gates":gates},sort_keys=True))
    return 0 if result["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
