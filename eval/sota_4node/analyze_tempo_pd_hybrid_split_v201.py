#!/usr/bin/env python3
"""Adapt the saturation analyzer to the stable one-item split."""

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
        raise ValueError("refusing stale split analysis output")
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
    value["gates"]["tempo_routes_34_local_14_remote"] = (
        routes == {base.LOCAL: 34, base.REMOTE: 14})
    value["schema"] = "tempo-pd-hybrid-split-analysis-201"
    value["passes"] = all(value["gates"].values())
    value["verdict"] = (
        "split_saturation_validated" if value["passes"]
        else "split_saturation_needs_revision")
    value["claim_boundary"] = (
        "Rate-64 actual vLLM P/D TEMPO/fixed-local screen. Relative to v192, "
        "only one stable cache item in the 2048/output64 bucket moves local."
    )
    args.output.resolve().write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"],
                      "failed": [key for key, passed in value["gates"].items()
                                 if not passed]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
