#!/usr/bin/env python3
"""Analyze the same-server 18-local/6-remote credit-nine diagnostic."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from eval.sota_4node import analyze_tempo_pd_performance_v1 as base
from eval.sota_4node import analyze_tempo_pd_same_server_v63 as shared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.stage_root / "same_server_measured"
    normalized, contracts = {}, {}
    for arm in ("fixed_local", "tempo", "lmcache_remote"):
        normalized[arm], contracts[arm] = shared._normalize(
            shared._load(root / f"{arm}.raw.json"), arm)
    parsed = {arm: base._parse_run(
        arm, raw, ttft_slo_ms=3000, tpot_slo_ms=250, e2e_slo_ms=12000)
        for arm, raw in normalized.items()}
    local, tempo, remote = parsed["fixed_local"], parsed["tempo"], parsed["lmcache_remote"]
    lp, tp, rp = (row["performance"] for row in (local, tempo, remote))
    exact = (local["model_config_sha256"] == tempo["model_config_sha256"] == remote["model_config_sha256"]
             and local["workload_sha256"] == tempo["workload_sha256"] == remote["workload_sha256"]
             and local["_contracts"] == tempo["_contracts"] == remote["_contracts"]
             and local["_outputs"] == tempo["_outputs"] == remote["_outputs"])
    same_server = (len({row["server_epoch_root"] for row in contracts.values()}) == 1
                   and [contracts[arm]["sequence_index"] for arm in ("fixed_local", "tempo", "lmcache_remote")] == [0, 1, 2])
    gates = {
        "same_live_server_epoch_contract": same_server,
        "exact_normalized_workload_schedule_outputs": exact,
        "tempo_routes_18_local_6_remote": tempo["routes"] == {
            "decoder_local_recompute_or_cache": 18, "remote_prefill_live_kv": 6},
        "all_tempo_requests_slo_valid": tp["slo_goodput"]["success_fraction"] == 1.0,
        "tempo_goodput_beats_local": tp["slo_goodput"]["request_goodput_per_s"] > lp["slo_goodput"]["request_goodput_per_s"],
        "tempo_goodput_beats_lmcache": tp["slo_goodput"]["request_goodput_per_s"] > rp["slo_goodput"]["request_goodput_per_s"],
        "tempo_e2e_p50_beats_both": tp["e2e_ms"]["p50"] < min(lp["e2e_ms"]["p50"], rp["e2e_ms"]["p50"]),
        "tempo_e2e_p99_beats_both": tp["e2e_ms"]["p99"] < min(lp["e2e_ms"]["p99"], rp["e2e_ms"]["p99"]),
        "tempo_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < rp["tpot_ms"]["p99"],
    }
    public = lambda row: {key:value for key,value in row.items() if not key.startswith("_")}
    result = {"schema":"tempo-pd-same-server-credit9-analysis-67",
              "fixed_local":public(local), "tempo":public(tempo), "lmcache_remote":public(remote),
              "paired_tempo_minus_local":base._paired(tempo,local) if exact else None,
              "paired_tempo_minus_lmcache":base._paired(tempo,remote) if exact else None,
              "gates":gates,"passes":all(gates.values()),
              "verdict":"promising_same_server_credit9" if all(gates.values()) else "reject_same_server_credit9"}
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"verdict":result["verdict"],"gates":gates},sort_keys=True))
    return 0 if result["passes"] else 2


if __name__ == "__main__": raise SystemExit(main())
