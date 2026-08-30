#!/usr/bin/env python3
"""Promote a passing output128 diagnostic analysis to production provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    value = json.loads(args.input.read_text(encoding="utf-8"))
    if value.get("schema") != "tempo-pd-output128-diagnostic-analysis-96":
        raise ValueError("output128 v96 analysis required")
    if value.get("passes") is not True:
        raise ValueError("refusing to promote a failing analysis")
    if value["tempo"]["routes"] != {"decoder_local_recompute_or_cache": 48}:
        raise ValueError("production output128 guard route mismatch")
    value["schema"] = "tempo-pd-short-prompt-output128-production-analysis-97"
    value["controller_variant"] = {
        "output_tokens": 128, "prompt_tokens_max": 512,
        "policy": "production_workload_guard_local",
    }
    value["verdict"] = "promising_short_prompt_output128_production_guard"
    value["claim_boundary"] = (
        "Production router, prompts at most 512 tokens, 128 generated tokens, "
        "one live server lifecycle and two cold-key-disjoint replicates per arm."
    )
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"], "passes": value["passes"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
