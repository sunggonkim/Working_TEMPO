#!/usr/bin/env python3
"""Fail-closed phase-change analysis for the one-factor cap-six variant."""

import argparse
import copy
import json
from pathlib import Path
import tempfile

from eval.sota_4node import analyze_tempo_pd_latched_controller_v383 as base
from eval.sota_4node.analyze_tempo_pd_online_regime_microburst25_v343 import _require


POLICY = "tempo-pd-latched-bypass-rolling-credit6-401"


def analyze(path: Path, allocation: int):
    path = path.resolve()
    raw = json.loads(path.read_text())
    decisions = raw.get("router_decisions", [])
    tempo = [row for row in decisions if "-tempo-" in row.get("request_id", "")]
    _require(len(tempo) == 24, "tempo decisions")
    for row in tempo:
        _require(row.get("profile_id") == POLICY and row.get("manifest_id") == POLICY,
                 "cap-six policy")
        _require(":local_cap=6:" in row.get("reason", ""), "cap-six provenance")
    adapted = copy.deepcopy(raw)
    for row in adapted["router_decisions"]:
        if "-tempo-" not in row.get("request_id", ""):
            continue
        row["profile_id"] = base.POLICY
        row["manifest_id"] = base.POLICY
        row["reason"] = row["reason"].replace(":local_cap=6:", ":local_cap=5:")
    with tempfile.TemporaryDirectory(prefix="tempo-cap6-analysis-", dir="/tmp") as tmp:
        adapted_path = Path(tmp) / "raw.json"
        adapted_path.write_text(json.dumps(adapted))
        result = base.analyze(adapted_path, allocation, "phasechange")
    measured = result["pairs"][8:]
    max_e2e = max(row["e2e_delta_ms"] for row in measured)
    result["schema"] = "tempo-pd-latched-cap6-analysis-402"
    result["policy"] = POLICY
    result["raw"] = str(path)
    result["evaluated_summary"]["e2e_max_delta_ms"] = max_e2e
    result["candidate_gates"]["e2e_worst_regression_le_250ms"] = max_e2e <= 250.0
    result["candidate_passes"] = all(result["candidate_gates"].values())
    result["verdict"] = ("cap6_phasechange_advantage"
                         if result["candidate_passes"] else "cap6_revision")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite")
    result = analyze(args.raw, args.allocation)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"],
                      "summary": result["evaluated_summary"],
                      "routes": result["route_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
