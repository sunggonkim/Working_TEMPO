#!/usr/bin/env python3
"""Derive the v106 fabric-aware discovery profile from frozen v105."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tempo.pd_global_profile import (
    global_profile_fingerprint,
    load_global_profile,
)


RESERVATIONS = {
    "latency": 8,
    "interactive": 8,
    "batch": 4,
    "background": 2,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile-id",
        default="tempo-go-qwen25-perlmutter-v106-cxi-completion-credit",
    )
    parser.add_argument(
        "--endpoint-queue-capacity", type=int, default=32,
    )
    args = parser.parse_args()

    base = args.base_profile.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    load_global_profile(base)
    value = json.loads(base.read_text(encoding="utf-8"))
    value["profile_id"] = args.profile_id
    controller = value["controller"]
    controller["endpoint_queue_debt_mode"] = (
        "completion_credit_endpoint_queue_v3"
    )
    controller["endpoint_queue_capacity"] = args.endpoint_queue_capacity
    controller["telemetry_stale_grace_ns"] = 1_000_000_000
    for tenant in value["tenants"]:
        tenant["queue_reservation_slots"] = RESERVATIONS[tenant["tenant_id"]]
    value["fingerprint_sha256"] = global_profile_fingerprint(value)

    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = load_global_profile(output)
    if loaded.fingerprint_sha256 != value["fingerprint_sha256"]:
        raise RuntimeError("v106 profile failed its round-trip fingerprint")
    print(json.dumps({
        "profile": str(output),
        "profile_id": loaded.profile_id,
        "fingerprint_sha256": loaded.fingerprint_sha256,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
