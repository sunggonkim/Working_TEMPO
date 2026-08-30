#!/usr/bin/env python3
"""Create a fingerprinted discovery-policy variant for burst admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-id", default="tempo-go-real-trace-fixed8064-burst100-global-screen-v2")
    parser.add_argument("--latency-ttft-slo-ms", type=float, default=1000.0)
    parser.add_argument("--latency-e2e-slo-ms", type=float, default=4000.0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite policy variant")
    raw = json.loads(args.source.read_text(encoding="utf-8"))
    raw["profile_id"] = args.profile_id
    found = False
    for tenant in raw["tenants"]:
        if tenant["tenant_id"] == "latency":
            # A latency request may wait for a bounded completion credit.  The
            # previous false value converted transient pair occupancy into a
            # 503 even though local capacity remained healthy.
            tenant["queue_lease_on_timeout"] = True
            tenant["ttft_slo_ms"] = args.latency_ttft_slo_ms
            tenant["e2e_slo_ms"] = args.latency_e2e_slo_ms
            found = True
    if not found:
        raise ValueError("latency tenant is missing")
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    profile = load_global_profile(args.output)
    print(json.dumps({
        "profile_id": profile.profile_id,
        "fingerprint_sha256": profile.fingerprint_sha256,
        "latency_queue_lease_on_timeout": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
