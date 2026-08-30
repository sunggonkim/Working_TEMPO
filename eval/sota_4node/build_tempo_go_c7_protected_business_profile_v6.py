#!/usr/bin/env python3
"""Derive a C7 profile with explicit protected business capacity lanes."""

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
        "--protected-capacity-fraction", type=float, default=0.20,
        help="per-resource capacity kept for higher-priority tenants",
    )
    args = parser.parse_args()
    base = args.base.resolve()
    output = args.output.resolve()
    if not base.is_file() or output.exists():
        raise ValueError("profile input/output boundary is invalid")
    if not 0.0 <= args.protected_capacity_fraction < 1.0:
        raise ValueError("protected capacity fraction must be in [0, 1)")

    raw = json.loads(base.read_text(encoding="utf-8"))
    raw["profile_id"] = "tempo-go-qwen25-perlmutter-c7-protected-business-v1"
    tenants = raw.get("tenants")
    if not isinstance(tenants, list):
        raise ValueError("profile tenants are missing")
    by_id = {item.get("tenant_id"): item for item in tenants}
    expected = {"latency", "interactive", "batch", "background"}
    if set(by_id) != expected:
        raise ValueError("protected-business tenant inventory differs")

    # Priority is an explicit business contract; weight remains the long-run
    # fairness debt.  The reserve is consumed only by lower-priority classes.
    priorities = {"latency": 1000, "interactive": 800,
                  "batch": 400, "background": 0}
    for tenant_id, priority in priorities.items():
        by_id[tenant_id]["admission_priority"] = priority
        by_id[tenant_id]["protected_capacity_fraction"] = (
            args.protected_capacity_fraction
            if tenant_id in {"latency", "interactive"}
            else 0.0
        )

    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = load_global_profile(output)
    if loaded.fingerprint_sha256 != raw["fingerprint_sha256"]:
        raise RuntimeError("derived protected-business profile did not round-trip")
    print(output)
    print("fingerprint", loaded.fingerprint_sha256)
    print("protected_capacity_fraction", args.protected_capacity_fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
