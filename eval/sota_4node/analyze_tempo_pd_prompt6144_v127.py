#!/usr/bin/env python3
"""Predeclared request-interleaved gates for prompt6144 local routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_prompt4096_v123 as base


def analyze(raw: dict) -> dict:
    result = base.analyze(raw)
    result["schema"] = "tempo-pd-prompt6144-interleaved-analysis-127"
    gates = result["gates"]
    reasons = result["tempo"]["reasons"]
    for output_tokens in (16, 128):
        gates[f"output{output_tokens}_reason_24"] = sum(
            count for reason, count in reasons.items()
            if reason.endswith(f"output{output_tokens}_prompt6144_local_diagnostic")
        ) == 24
    result["passes"] = all(gates.values())
    result["verdict"] = (
        "promising_prompt6144_direct_local" if result["passes"]
        else "reject_prompt6144_direct_local"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
