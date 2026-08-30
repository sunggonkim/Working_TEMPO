#!/usr/bin/env python3
"""Adapt the saturation analyzer to the one-bucket tail-aware route change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from eval.sota_4node import analyze_tempo_pd_hybrid_saturation_v192 as base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    temporary = args.output.resolve().with_name(args.output.name + ".v192.tmp")
    if args.output.exists() or temporary.exists():
        raise ValueError("refusing stale tail-aware analysis output")
    original_argv = sys.argv
    sys.argv = [original_argv[0], "--stage-root", str(args.stage_root),
                "--output", str(temporary)]
    try:
        base.main()
    finally:
        sys.argv = original_argv
    value = json.loads(temporary.read_text(encoding="utf-8"))
    temporary.unlink()
    routes = value["tempo"]["routes"]
    value["gates"].pop("tempo_routes_32_local_16_remote")
    value["gates"]["tempo_routes_36_local_12_remote"] = (
        routes == {base.LOCAL: 36, base.REMOTE: 12})
    value["schema"] = "tempo-pd-hybrid-tailaware-analysis-197"
    value["passes"] = all(value["gates"].values())
    value["verdict"] = (
        "tailaware_saturation_validated" if value["passes"]
        else "tailaware_saturation_needs_revision")
    value["claim_boundary"] = (
        "Rate-64 actual vLLM P/D TEMPO/fixed-local screen. Relative to v192, "
        "only the observed 2048-token/output64 warm bucket moves from remote to local."
    )
    args.output.resolve().write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"],
                      "failed": [key for key, passed in value["gates"].items()
                                 if not passed]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
