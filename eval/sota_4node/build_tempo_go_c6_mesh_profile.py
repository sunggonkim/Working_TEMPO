#!/usr/bin/env python3
"""Freeze the C6 receiver-credit P-by-D profile from a bound C5 profile."""

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
    args = parser.parse_args()
    base_path = args.base.resolve()
    output = args.output.resolve()
    _require(base_path.is_file(), "base global profile is missing")
    _require(not output.exists(), "refusing to overwrite C6 mesh profile")
    base = load_global_profile(base_path)
    _require(base.topology.pair_count == 2, "C6 requires two P/D pairs")
    _require(base.topology.prewarmed_pair_count == 2,
             "C6 requires two prewarmed pairs")
    raw = json.loads(base_path.read_text(encoding="utf-8"))
    _require(raw.get("fingerprint_sha256") == base.fingerprint_sha256,
             "base profile fingerprint differs")
    raw["profile_id"] = "tempo-go-qwen25-perlmutter-c6-receiver-credit-pxd-v1"
    controller = raw["controller"]
    controller.update({
        "mesh_control_mode": "receiver_credit_pxd_v1",
        "mesh_receiver_stagger_max_us": 2000,
        "mesh_edge_service_ewma_alpha": 0.5,
        "overload_action": "reject_new_request",
        "endpoint_queue_debt_mode": "disabled",
        "endpoint_queue_admission_mode": "after_timeout",
        "maximum_queue_wait_ns": 2_000_000_000,
    })
    # C6 owns source, edge, receiver, and decoder credits before dispatch.
    # A downstream queue lease would bypass that transaction, so no tenant is
    # permitted to opt into it in this profile.
    for tenant in raw["tenants"]:
        tenant["queue_lease_on_timeout"] = False
    raw.pop("fingerprint_sha256", None)
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    loaded = load_global_profile(output)
    _require(loaded.fingerprint_sha256 == raw["fingerprint_sha256"],
             "C6 mesh profile did not round-trip")
    _require(
        loaded.orchestrator_config().mesh_control_mode
        == "receiver_credit_pxd_v1",
        "C6 mesh controller was not enabled",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
