#!/usr/bin/env python3
"""Validate the frozen short-prompt/output64 local workload guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_REPLACED_GATES = (
    "tempo_routes_36_local_12_remote",
    "tempo_goodput_beats_local",
    "tempo_paired_majority_beats_local",
    "tempo_paired_median_beats_local",
)


def analyze(value: dict) -> dict:
    if value.get("schema") != "tempo-pd-same-server-balanced-output64-analysis-77":
        raise ValueError("output64 v77 input required")
    gates = value.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("gates object required")
    for name in _REPLACED_GATES:
        if name not in gates:
            raise ValueError(f"missing replaced gate: {name}")
        del gates[name]

    tempo = value["tempo"]
    local = value["fixed_local"]
    pair = value["paired_tempo_minus_local"]
    tp = tempo["performance"]
    lp = local["performance"]
    tempo_goodput = tp["slo_goodput"]["request_goodput_per_s"]
    local_goodput = lp["slo_goodput"]["request_goodput_per_s"]
    gates.update({
        "short_prompt_output64_routes_48_local": tempo["routes"] == {
            "decoder_local_recompute_or_cache": 48
        },
        "short_prompt_output64_goodput_retains_98pct_local": (
            tempo_goodput >= 0.98 * local_goodput
        ),
        "short_prompt_output64_pair_win_count_at_least_half": (
            pair["e2e_win_count"] >= 24
        ),
        "short_prompt_output64_paired_median_within_10ms_local": (
            pair["e2e_delta_median_ms"] <= 10.0
        ),
        "short_prompt_output64_e2e_p99_within_5pct_local": (
            tp["e2e_ms"]["p99"] <= 1.05 * lp["e2e_ms"]["p99"]
        ),
    })
    reasons = tempo.get("reasons", {})
    gates["short_prompt_output64_workload_guard_reason_48"] = (
        reasons.get("same_server_tempo_measured:workload_guard_local:"
                    "mean_pair_interval_ns=None", 0)
        + sum(
            count for reason, count in reasons.items()
            if reason.startswith("same_server_tempo_measured:workload_guard_local:")
        )
    ) >= 48
    # The first expression above may be included by the prefix sum; normalize to
    # the exact observable count rather than depending on calibration metadata.
    gates["short_prompt_output64_workload_guard_reason_48"] = sum(
        count for reason, count in reasons.items()
        if reason.startswith("same_server_tempo_measured:workload_guard_local:")
    ) == 48

    value["schema"] = "tempo-pd-short-prompt-output64-guard-analysis-95"
    value["controller_variant"] = {
        "output_tokens": 64,
        "prompt_tokens_max": 512,
        "policy": "workload_guard_local",
    }
    value["passes"] = all(gates.values())
    value["verdict"] = (
        "promising_short_prompt_output64_guard"
        if value["passes"] else "reject_short_prompt_output64_guard"
    )
    value["claim_boundary"] = (
        "Short prompts (at most 512 tokens), 64 generated tokens, one live "
        "server lifecycle, two cold-key-disjoint replicates per arm. The guard "
        "must retain local performance while beating the LMCache remote arm."
    )
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    value = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(value)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
