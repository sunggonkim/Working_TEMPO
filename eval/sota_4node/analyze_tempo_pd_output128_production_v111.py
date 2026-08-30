#!/usr/bin/env python3
"""Production-policy validation for request-interleaved output128 routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_output128_interleaved_v109 as diagnostic


def analyze(raw: dict) -> dict:
    result = diagnostic.analyze(raw)
    result["schema"] = "tempo-pd-output128-production-analysis-111"
    gates = result["gates"]
    del gates["tempo_reason_48_output128_diagnostic"]
    gates["tempo_reason_48_output128_direct_local"] = sum(
        count for reason, count in result["tempo"]["reasons"].items()
        if reason.endswith("output128_direct_local_fast_path")) == 48
    result["passes"] = all(gates.values())
    result["verdict"] = (
        "promising_output128_production_policy" if result["passes"]
        else "reject_output128_production_policy")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
