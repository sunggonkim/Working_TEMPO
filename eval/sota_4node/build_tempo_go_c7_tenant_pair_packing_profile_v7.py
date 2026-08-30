#!/usr/bin/env python3
"""Derive a C7 profile with auditable low-priority decoder-pair packing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--background-pair-spread-limit", type=int, default=1,
        help="maximum decoder pairs occupied by the background class per busy epoch",
    )
    args = parser.parse_args()
    base = args.base.resolve()
    output = args.output.resolve()
    if not base.is_file() or output.exists():
        raise ValueError("profile input/output boundary is invalid")
    if args.background_pair_spread_limit <= 0:
        raise ValueError("background pair spread limit must be positive")

    raw = json.loads(base.read_text(encoding="utf-8"))
    pair_count = int(raw["topology"]["pair_count"])
    if args.background_pair_spread_limit > pair_count:
        raise ValueError("background pair spread limit exceeds topology")
    raw["profile_id"] = "tempo-go-qwen25-perlmutter-c7-tenant-pair-packing-v1"
    tenants = raw.get("tenants")
    if not isinstance(tenants, list):
        raise ValueError("profile tenants are missing")
    by_id = {item.get("tenant_id"): item for item in tenants}
    if set(by_id) != {"latency", "interactive", "batch", "background"}:
        raise ValueError("tenant-packing inventory differs")

    # Only the explicitly low-priority background class is consolidated.
    # Higher-priority classes remain free to use every feasible pair; the
    # orchestrator activates/prefers a pair outside the packed scope whenever
    # one is available.  This is a business placement contract, not a phase or
    # hot-decoder label supplied by the workload.
    by_id["background"]["pair_spread_limit"] = (
        args.background_pair_spread_limit)

    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = load_global_profile(output)
    background = next(
        item for item in loaded.tenants if item.tenant_id == "background")
    if background.pair_spread_limit != args.background_pair_spread_limit:
        raise RuntimeError("tenant pair packing profile did not round-trip")
    print(output)
    print("fingerprint", loaded.fingerprint_sha256)
    print("background_pair_spread_limit", background.pair_spread_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
