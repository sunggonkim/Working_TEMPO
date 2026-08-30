#!/usr/bin/env python3
"""Derive a discovery-only queue-stress profile without changing pair count."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-wait-s", type=float, default=60.0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite derived profile: {args.output}")
    if args.queue_wait_s <= 0:
        raise ValueError("queue wait must be positive")
    raw = copy.deepcopy(json.loads(args.input.read_text(encoding="utf-8")))
    wait_ns = int(args.queue_wait_s * 1_000_000_000)
    controller = raw["controller"]
    controller["maximum_queue_wait_ns"] = wait_ns
    controller["completion_liveness_shared_probe_mode"] = "headroom_shared_v1"
    controller["endpoint_queue_headroom_admission_mode"] = "completion_progress_v1"
    controller["endpoint_queue_capacity"] = max(64, int(controller["endpoint_queue_capacity"]))
    controller["queue_capacity"] = max(256, int(controller["queue_capacity"]))
    # max_num_seqs is fixed at 16 by the Perlmutter runner.  Preserve the
    # validated 8-slot priority lane so decoder admission retains residual
    # active-sequence capacity for ordinary traffic.
    controller["priority_service_lane_capacity"] = min(
        8, int(controller["priority_service_lane_capacity"])
    )
    controller["shared_remote_requests_capacity"] = max(
        64, int(controller["shared_remote_requests_capacity"])
    )
    for tenant in raw["tenants"]:
        tenant["maximum_queue_wait_ns"] = wait_ns
        # This is a separate drain experiment.  It must not turn a lower
        # priority tenant into an immediate 503 merely because its original
        # business profile disallows queue leasing.
        tenant["queue_lease_on_timeout"] = True
    raw["profile_id"] = "tempo-go-real-trace-queue-stress-discovery-v1"
    raw["deployment_scope"] = "discovery"
    raw.pop("fingerprint_sha256", None)
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    load_global_profile(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    loaded = load_global_profile(args.output)
    print(json.dumps({
        "profile_id": loaded.profile_id,
        "fingerprint_sha256": loaded.fingerprint_sha256,
        "queue_wait_ns": wait_ns,
        "queue_capacity": raw["controller"]["queue_capacity"],
        "endpoint_queue_capacity": raw["controller"]["endpoint_queue_capacity"],
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "performance_claim_allowed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
