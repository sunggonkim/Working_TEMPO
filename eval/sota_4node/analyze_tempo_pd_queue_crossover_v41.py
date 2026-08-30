#!/usr/bin/env python3
"""Compare the mixed queue-crossover policy to both fixed baselines."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from eval.sota_4node import analyze_tempo_pd_performance_v1 as base


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    parse = lambda label, path: base._parse_run(
        label, load(path), ttft_slo_ms=3000, tpot_slo_ms=250, e2e_slo_ms=12000)
    local = parse("local", a.reference / "crossover_local/raw.json")
    remote = parse("remote", a.reference / "crossover_remote/raw.json")
    tempo = parse("tempo", a.candidate_root / "tempo_credit_admission/raw.json")
    exact = (local["_contracts"] == remote["_contracts"] == tempo["_contracts"]
             and local["_outputs"] == remote["_outputs"] == tempo["_outputs"]
             and local["workload_sha256"] == remote["workload_sha256"] == tempo["workload_sha256"])
    tl = base._paired(tempo, local) if exact else None
    tr = base._paired(tempo, remote) if exact else None
    lp, rp, tp = local["performance"], remote["performance"], tempo["performance"]
    gates = {
        "exact_outputs_and_contracts": exact,
        "route_mix_is_16_local_8_remote": tempo["routes"] == {
            "decoder_local_recompute_or_cache": 16, "remote_prefill_live_kv": 8},
        "all_requests_slo_valid": tp["slo_goodput"]["success_fraction"] == 1.0,
        "goodput_beats_local": tp["slo_goodput"]["request_goodput_per_s"] > lp["slo_goodput"]["request_goodput_per_s"],
        "goodput_beats_remote": tp["slo_goodput"]["request_goodput_per_s"] > rp["slo_goodput"]["request_goodput_per_s"],
        "e2e_p50_beats_local": tp["e2e_ms"]["p50"] < lp["e2e_ms"]["p50"],
        "e2e_p99_beats_local": tp["e2e_ms"]["p99"] < lp["e2e_ms"]["p99"],
        "tpot_p99_below_remote": tp["tpot_ms"]["p99"] < rp["tpot_ms"]["p99"],
    }
    public = lambda x: {k:v for k,v in x.items() if not k.startswith("_")}
    result = {"schema":"tempo-pd-queue-crossover-analysis-41",
              "local":public(local),"remote":public(remote),"tempo":public(tempo),
              "paired_tempo_minus_local":tl,"paired_tempo_minus_remote":tr,
              "gates":gates,"passes":all(gates.values()),
              "verdict":"promising_queue_crossover" if all(gates.values()) else "revise_queue_crossover"}
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"verdict":result["verdict"],"gates":gates},sort_keys=True))
    return 0 if result["passes"] else 2


if __name__ == "__main__": raise SystemExit(main())
