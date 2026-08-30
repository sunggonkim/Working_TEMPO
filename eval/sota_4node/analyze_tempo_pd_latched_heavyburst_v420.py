#!/usr/bin/env python3
"""Fail-closed heavy-burst analyzer for cap-five or cap-six."""

import argparse
import copy
import json
from pathlib import Path
import tempfile

from eval.sota_4node import analyze_tempo_pd_latched_controller_v383 as base
from eval.sota_4node.analyze_tempo_pd_online_regime_microburst25_v343 import _require


POLICIES = {
    "cap5": (base.POLICY, 5),
    "cap6": ("tempo-pd-latched-bypass-rolling-credit6-401", 6),
}
TRACE = "six_bursts_four_pairs_8ms_with_100ms_idle_v419"


def analyze(path: Path, allocation: int, variant: str):
    policy, cap = POLICIES[variant]
    path = path.resolve()
    raw = json.loads(path.read_text())
    _require(raw.get("mixed_crossover_contract", {}).get("arrival_trace") == TRACE,
             "heavy trace")
    tempo = [row for row in raw.get("router_decisions", [])
             if "-tempo-" in row.get("request_id", "")]
    _require(len(tempo) == 24, "tempo decisions")
    for row in tempo:
        _require(row.get("profile_id") == policy and row.get("manifest_id") == policy,
                 "heavy policy")
        _require(f":local_cap={cap}:" in row.get("reason", ""), "heavy cap")
    adapted = copy.deepcopy(raw)
    adapted["mixed_crossover_contract"]["arrival_trace"] = (
        "six_bursts_four_pairs_14ms_with_220ms_idle_v322")
    if variant == "cap6":
        for row in adapted["router_decisions"]:
            if "-tempo-" not in row.get("request_id", ""):
                continue
            row["profile_id"] = base.POLICY
            row["manifest_id"] = base.POLICY
            row["reason"] = row["reason"].replace(":local_cap=6:", ":local_cap=5:")
    with tempfile.TemporaryDirectory(prefix="tempo-heavy-analysis-", dir="/tmp") as tmp:
        adapted_path = Path(tmp) / "raw.json"
        adapted_path.write_text(json.dumps(adapted))
        result = base.analyze(adapted_path, allocation, "bursty")
    max_e2e = max(row["e2e_delta_ms"] for row in result["pairs"])
    result["schema"] = "tempo-pd-latched-heavyburst-analysis-420"
    result["policy"] = policy
    result["workload_class"] = "heavybursty"
    result["raw"] = str(path)
    result["cap_variant"] = variant
    result["evaluated_summary"]["e2e_max_delta_ms"] = max_e2e
    result["candidate_gates"]["backpressure_exercised"] = len(result["capped_requests"]) > 0
    result["candidate_gates"]["e2e_worst_regression_le_250ms"] = max_e2e <= 250.0
    result["candidate_passes"] = all(result["candidate_gates"].values())
    result["verdict"] = (f"{variant}_heavyburst_advantage"
                         if result["candidate_passes"] else f"{variant}_heavyburst_revision")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--variant", choices=tuple(POLICIES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite")
    result = analyze(args.raw, args.allocation, args.variant)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"],
                      "summary": result["evaluated_summary"],
                      "capped": result["capped_requests"]}, sort_keys=True))


if __name__ == "__main__":
    main()
