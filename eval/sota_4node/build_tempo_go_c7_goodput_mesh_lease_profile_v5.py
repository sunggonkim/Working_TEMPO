#!/usr/bin/env python3
"""Derive the C7 v5 profile with an explicit interactive stale-grace policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interactive-stale-grace-ns", type=int, default=5_000_000_000)
    args = parser.parse_args()
    base = args.base.resolve()
    output = args.output.resolve()
    if not base.is_file() or output.exists():
        raise ValueError("profile input/output boundary is invalid")
    if args.interactive_stale_grace_ns < 0:
        raise ValueError("interactive stale grace must be non-negative")
    raw = json.loads(base.read_text(encoding="utf-8"))
    raw["profile_id"] = "tempo-go-qwen25-perlmutter-c7-goodput-mesh-lease-v2"
    tenants = raw.get("tenants")
    if not isinstance(tenants, list):
        raise ValueError("profile tenants are missing")
    interactive = [item for item in tenants if item.get("tenant_id") == "interactive"]
    if len(interactive) != 1:
        raise ValueError("interactive tenant inventory is not unique")
    interactive[0]["telemetry_stale_grace_ns"] = args.interactive_stale_grace_ns
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = load_global_profile(output)
    if loaded.fingerprint_sha256 != raw["fingerprint_sha256"]:
        raise RuntimeError("derived profile did not round-trip")
    print(output)
    print("fingerprint", loaded.fingerprint_sha256)
    print("interactive_stale_grace_ns", args.interactive_stale_grace_ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
