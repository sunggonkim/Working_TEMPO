#!/usr/bin/env python3
"""Fail-closed cap-six extreme-burst overload analysis."""

import argparse
import copy
import json
from pathlib import Path
import tempfile

from eval.sota_4node import analyze_tempo_pd_latched_heavyburst_v420 as heavy
from eval.sota_4node.analyze_tempo_pd_online_regime_microburst25_v343 import _require


TRACE = "six_bursts_four_pairs_4ms_with_50ms_idle_v433"


def analyze(path: Path, allocation: int):
    path = path.resolve()
    raw = json.loads(path.read_text())
    _require(raw.get("mixed_crossover_contract", {}).get("arrival_trace") == TRACE,
             "extreme trace")
    adapted = copy.deepcopy(raw)
    adapted["mixed_crossover_contract"]["arrival_trace"] = heavy.TRACE
    with tempfile.TemporaryDirectory(prefix="tempo-extreme-analysis-", dir="/tmp") as tmp:
        adapted_path = Path(tmp) / "raw.json"
        adapted_path.write_text(json.dumps(adapted))
        result = heavy.analyze(adapted_path, allocation, "cap6")
    result["schema"] = "tempo-pd-latched-extreme-analysis-434"
    result["workload_class"] = "extremebursty"
    result["raw"] = str(path)
    result["candidate_gates"]["at_least_two_overflow_requests_capped"] = (
        len(result["capped_requests"]) >= 2)
    result["candidate_passes"] = all(result["candidate_gates"].values())
    result["verdict"] = ("cap6_extremeburst_advantage"
                         if result["candidate_passes"] else "cap6_extremeburst_limit")
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
                      "capped": result["capped_requests"]}, sort_keys=True))


if __name__ == "__main__":
    main()
