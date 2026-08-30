#!/usr/bin/env python3
"""Fail-closed analyzer for onset-six, sustained-five latched credit."""

import argparse
import copy
import json
from pathlib import Path
import re
import tempfile

from eval.sota_4node import analyze_tempo_pd_latched_controller_v383 as frozen
from eval.sota_4node import analyze_tempo_pd_latched_reverse_v396 as reverse
from eval.sota_4node.analyze_tempo_pd_online_regime_microburst25_v343 import _require


POLICY = "tempo-pd-latched-onset6-sustained5-411"
COUNT = re.compile(r":local_cap=(5|6):onset_microburst_tempo_count=([0-9]+):")


def analyze(path: Path, allocation: int, workload_class: str):
    path = path.resolve()
    raw = json.loads(path.read_text())
    tempo = [row for row in raw.get("router_decisions", [])
             if "-tempo-" in row.get("request_id", "")]
    _require(len(tempo) == 24, "tempo decisions")
    effective_caps = {5: 0, 6: 0}
    onset_counts = []
    adapted = copy.deepcopy(raw)
    for original, row in zip(raw["router_decisions"], adapted["router_decisions"]):
        if "-tempo-" not in original.get("request_id", ""):
            continue
        _require(original.get("profile_id") == POLICY
                 and original.get("manifest_id") == POLICY, "onset policy")
        match = COUNT.search(original.get("reason", ""))
        _require(match is not None, "onset provenance")
        cap, count = int(match.group(1)), int(match.group(2))
        active = ":microburst_credit_active=true:" in original["reason"]
        _require((count > 0) == active, "onset count activity")
        _require(cap == (6 if active and count <= 8 else 5), "effective cap")
        effective_caps[cap] += 1
        onset_counts.append(count)
        row["profile_id"] = frozen.POLICY
        row["manifest_id"] = frozen.POLICY
        row["reason"] = COUNT.sub(":local_cap=5:", row["reason"])
    with tempfile.TemporaryDirectory(prefix="tempo-onset-analysis-", dir="/tmp") as tmp:
        adapted_path = Path(tmp) / "raw.json"
        adapted_path.write_text(json.dumps(adapted))
        if workload_class == "reverse_phasechange":
            result = reverse.analyze(adapted_path, allocation)
        else:
            result = frozen.analyze(adapted_path, allocation, workload_class)
    selected = result["pairs"][8:] if workload_class == "phasechange" else result["pairs"]
    max_e2e = max(row["e2e_delta_ms"] for row in selected)
    result["schema"] = "tempo-pd-latched-onset-analysis-412"
    result["policy"] = POLICY
    result["raw"] = str(path)
    result["effective_cap_decision_counts"] = effective_caps
    result["max_onset_microburst_tempo_count"] = max(onset_counts)
    result["evaluated_summary"]["e2e_max_delta_ms"] = max_e2e
    result["candidate_gates"]["both_effective_caps_observed"] = (
        effective_caps[5] > 0 and effective_caps[6] > 0)
    result["candidate_gates"]["e2e_worst_regression_le_250ms"] = max_e2e <= 250.0
    result["candidate_passes"] = all(result["candidate_gates"].values())
    result["verdict"] = ("onset_sustained_controller_advantage"
                         if result["candidate_passes"] else "onset_sustained_revision")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--workload-class", required=True,
                        choices=("phasechange", "bursty", "reverse_phasechange"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite")
    result = analyze(args.raw, args.allocation, args.workload_class)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"],
                      "summary": result["evaluated_summary"],
                      "caps": result["effective_cap_decision_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
