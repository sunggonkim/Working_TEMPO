#!/usr/bin/env python3
"""Fail-closed reverse phase-change analysis for the latched controller."""

import argparse
import copy
import json
from pathlib import Path
import tempfile

from eval.sota_4node import analyze_tempo_pd_latched_controller_v383 as base
from eval.sota_4node.analyze_tempo_pd_online_regime_microburst25_v343 import _require


TRACE = "four_bursts4_14ms_idle220_then_steady8_100ms_v394"
PREFIX = "same_length_first_19_token_prefix_substitution_reverse_v395"


def analyze(path: Path, allocation: int):
    path = path.resolve()
    raw = json.loads(path.read_text())
    contract = raw.get("mixed_crossover_contract", {})
    _require(contract.get("arrival_trace") == TRACE, "reverse trace")
    _require(contract.get("leading_unique_region") == PREFIX, "reverse prefix")
    _require(contract.get("leading_unique_chunk_count") == 48, "chunks")
    _require(contract.get("paired_prompt_token_geometry_equal") is True, "geometry")

    adapted = copy.deepcopy(raw)
    adapted_contract = adapted["mixed_crossover_contract"]
    adapted_contract["arrival_trace"] = "six_bursts_four_pairs_14ms_with_220ms_idle_v322"
    adapted_contract["leading_unique_region"] = (
        "same_length_first_19_token_prefix_substitution_v372")
    with tempfile.TemporaryDirectory(prefix="tempo-reverse-analysis-", dir="/tmp") as tmp:
        adapted_path = Path(tmp) / "raw.json"
        adapted_path.write_text(json.dumps(adapted))
        result = base.analyze(adapted_path, allocation, "bursty")

    states = result["tempo_states"]
    _require(result["first_latched_item"] <= 6, "early burst latch")
    _require(all(row["high_load_latched"] for row in states
                 if row["item"] >= result["first_latched_item"]), "held latch")
    steady_tail = [row for row in states if row["item"] >= 20]
    _require(len(steady_tail) == 4, "steady tail states")
    credit_disengaged = all(
        row["raw_regime"] == "affinity"
        and row["high_load_latched"]
        and not row["microburst_credit_active"]
        and not row["local_capped"]
        for row in steady_tail)
    tail_pairs = [row for row in result["pairs"] if row["item"] >= 16]
    tail_summary = base._summary(tail_pairs)
    reverse_gates = {
        "rolling_credit_disengages_in_steady_tail": credit_disengaged,
        "steady_tail_e2e_wins_ge_6_of_8": tail_summary["e2e_win_count"] >= 6,
        "steady_tail_e2e_median_improves": tail_summary["e2e_delta_median_ms"] < 0,
        "steady_tail_tpot_wins_ge_6_of_8": tail_summary["tpot_win_count"] >= 6,
        "steady_tail_tpot_median_improves": tail_summary["tpot_delta_median_ms"] < 0,
        "steady_tail_tpot_p90_nonregression": tail_summary["tpot_p90_delta_ms"] <= 0,
    }
    result["schema"] = "tempo-pd-latched-reverse-analysis-396"
    result["workload_class"] = "reverse_phasechange"
    result["raw"] = str(path)
    result["steady_tail_summary"] = tail_summary
    result["reverse_transition_gates"] = reverse_gates
    result["candidate_gates"].update(reverse_gates)
    result["candidate_passes"] = all(result["candidate_gates"].values())
    result["verdict"] = ("latched_reverse_transition_advantage"
                         if result["candidate_passes"]
                         else "latched_reverse_transition_revision")
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
                      "steady_tail": result["steady_tail_summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
