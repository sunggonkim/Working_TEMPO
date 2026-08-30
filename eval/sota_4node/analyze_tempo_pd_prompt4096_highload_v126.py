#!/usr/bin/env python3
"""High-load throughput gates for production prompt4096 routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def analyze(value: dict) -> dict:
    if value.get("schema") != "tempo-pd-prompt4096-production-analysis-125":
        raise ValueError("production prompt4096 v125 analysis required")
    tempo = value["tempo"]["performance"]
    local = value["fixed_local"]["performance"]
    remote = value["lmcache_remote"]["performance"]
    gates = value["gates"]
    gates.update({
        "highload_tempo_request_throughput_beats_lmcache": (
            tempo["request_throughput_per_s"] > remote["request_throughput_per_s"]
        ),
        "highload_tempo_goodput_beats_lmcache": (
            tempo["slo_goodput"]["request_goodput_per_s"] >
            remote["slo_goodput"]["request_goodput_per_s"]
        ),
        "highload_tempo_goodput_retains_95pct_local": (
            tempo["slo_goodput"]["request_goodput_per_s"] >=
            0.95 * local["slo_goodput"]["request_goodput_per_s"]
        ),
    })
    value["schema"] = "tempo-pd-prompt4096-highload-analysis-126"
    value["passes"] = all(gates.values())
    value["verdict"] = (
        "promising_prompt4096_highload_production" if value["passes"]
        else "reject_prompt4096_highload_production"
    )
    value["claim_boundary"] = (
        "One request-interleaved high-load lifecycle; production routing and "
        "identical exact-4094-token workload across fixed local, TEMPO, and LMCache."
    )
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    value = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "verdict": value["verdict"],
        "failed": [name for name, passed in value["gates"].items() if not passed],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
