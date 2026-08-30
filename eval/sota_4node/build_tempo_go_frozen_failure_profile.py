#!/usr/bin/env python3
"""Create a quarantine-enabled CPU-replay profile from a frozen primary profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("frozen global profile is not an object")
    if raw.get("deployment_scope") != "frozen_validation":
        raise ValueError("failure profile source must be frozen_validation")
    controller = raw.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("frozen global controller is missing")
    if not isinstance(controller.get("frozen_service_proxy_policy"), dict):
        raise ValueError("failure profile must retain frozen service proxy policy")
    value = dict(raw)
    value["profile_id"] = args.profile_id
    value["controller"] = dict(controller)
    value["controller"]["route_failure_quarantine_mode"] = "deny_until_probe"
    value.pop("fingerprint_sha256", None)
    value["fingerprint_sha256"] = global_profile_fingerprint(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    loaded = load_global_profile(output)
    print(json.dumps({
        "output": str(output),
        "fingerprint_sha256": loaded.fingerprint_sha256,
        "profile_id": loaded.profile_id,
        "route_failure_quarantine_mode": (
            loaded.controller["route_failure_quarantine_mode"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
