#!/usr/bin/env python3
"""Freeze the C7 goodput-aware mesh queue-debt discovery profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maximum-queue-wait-ns", type=int, default=6_000_000_000)
    parser.add_argument(
        "--endpoint-queue-admission-mode",
        choices=("after_timeout", "headroom_first_v1"),
        default="after_timeout",
    )
    parser.add_argument(
        "--route-benefit-margin-ms", type=float, default=0.0)
    args = parser.parse_args()

    base_path = args.base.resolve()
    output = args.output.resolve()
    _require(base_path.is_file(), "base global profile is missing")
    _require(not output.exists(), "refusing to overwrite C7 profile")
    base = load_global_profile(base_path)
    _require(base.topology.pair_count == 2, "C7 requires two P/D pairs")
    _require(base.topology.prewarmed_pair_count == 2,
             "C7 requires two prewarmed pairs")
    _require(args.route_benefit_margin_ms >= 0.0,
             "route-benefit margin must be non-negative")
    raw = json.loads(base_path.read_text(encoding="utf-8"))
    _require(raw.get("fingerprint_sha256") == base.fingerprint_sha256,
             "base profile fingerprint differs")

    raw["profile_id"] = "tempo-go-qwen25-perlmutter-c7-goodput-mesh-lease-v1"
    controller = raw["controller"]
    controller.update({
        "mesh_control_mode": "receiver_credit_pxd_v1",
        "endpoint_queue_debt_mode": (
            "completion_credit_mesh_endpoint_queue_v1"),
        "endpoint_queue_admission_mode": args.endpoint_queue_admission_mode,
        "endpoint_queue_capacity": 32,
        "overload_action": "endpoint_queue_lease",
        "maximum_queue_wait_ns": args.maximum_queue_wait_ns,
        "proactive_scale_up_route_benefit_margin_ms": (
            args.route_benefit_margin_ms),
        "mesh_cool_remote_route_pressure_fraction": 0.5,
    })
    for tenant in raw["tenants"]:
        if tenant["tenant_id"] == "interactive":
            tenant["maximum_queue_wait_ns"] = args.maximum_queue_wait_ns
            tenant["queue_lease_on_timeout"] = True
        else:
            tenant["queue_lease_on_timeout"] = False

    raw.pop("fingerprint_sha256", None)
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = load_global_profile(output)
    _require(loaded.fingerprint_sha256 == raw["fingerprint_sha256"],
             "C7 profile did not round-trip")
    config = loaded.orchestrator_config()
    _require(
        config.endpoint_queue_debt_mode
        == "completion_credit_mesh_endpoint_queue_v1",
        "C7 mesh queue-debt mode was not enabled",
    )
    _require(config.mesh_control_mode == "receiver_credit_pxd_v1",
             "C7 mesh controller was not enabled")
    print(output)
    print(loaded.fingerprint_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
