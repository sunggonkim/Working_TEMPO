#!/usr/bin/env python3
"""Rebind frozen numeric TEMPO profiles to a new immutable workload manifest.

This utility is intentionally identity-only: endpoint/global numeric rows and
controller values are copied byte-for-byte from an existing screen/discovery
profile.  The new workload manifest is a separately generated held-out slice,
so its SHA is made explicit in both profile identities and a provenance
receipt.  It never changes measured latency/capacity numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tempo.pd_endpoint_profile import (
    endpoint_service_profile_fingerprint,
    load_endpoint_service_profile,
)
from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


def sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-endpoint", type=Path, required=True)
    parser.add_argument("--base-global", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--endpoint-output", type=Path, required=True)
    parser.add_argument("--global-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    args = parser.parse_args()

    base_endpoint_path = args.base_endpoint.resolve()
    base_global_path = args.base_global.resolve()
    manifest_path = args.workload_manifest.resolve()
    base_endpoint = json.loads(base_endpoint_path.read_text(encoding="utf-8"))
    base_global = json.loads(base_global_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if base_endpoint.get("schema") != "tempo-pd-endpoint-service-profile-v2":
        raise ValueError("base endpoint must be the v2 profile")
    if base_global.get("schema") != "tempo-go-profile-v1":
        raise ValueError("base global profile schema differs")
    if manifest.get("schema") != "tempo-go-contention-manifest-v1":
        raise ValueError("short workload manifest schema differs")
    workload = manifest.get("validation_workload")
    if not isinstance(workload, dict):
        raise ValueError("short workload manifest lacks validation_workload")
    if Path(str(workload.get("path"))).resolve().parent != manifest_path.parent / "workloads":
        raise ValueError("workload must remain below the manifest workload directory")
    manifest_sha = sha256(manifest_path)
    base_endpoint_rows = base_endpoint["rows"]
    base_global_numeric = {
        "capacities": base_global["capacities"],
        "tenants": base_global["tenants"],
        "controller": base_global["controller"],
        "telemetry": base_global["telemetry"],
    }

    endpoint = dict(base_endpoint)
    endpoint["profile_id"] = "tempo-pd-endpoint-qwen25-c4-short-slice-calibration-v1"
    endpoint["workload_manifest_sha256"] = manifest_sha
    endpoint.pop("fingerprint_sha256", None)
    endpoint["fingerprint_sha256"] = endpoint_service_profile_fingerprint(endpoint)

    global_profile = dict(base_global)
    identity = dict(global_profile["identity"])
    identity["endpoint_profile_id"] = endpoint["profile_id"]
    identity["endpoint_profile_fingerprint_sha256"] = endpoint["fingerprint_sha256"]
    identity["workload_manifest_sha256"] = manifest_sha
    global_profile["identity"] = identity
    global_profile["profile_id"] = "tempo-go-qwen25-perlmutter-short-slice-v1"
    global_profile.pop("fingerprint_sha256", None)
    global_profile["fingerprint_sha256"] = global_profile_fingerprint(global_profile)

    write_json(args.endpoint_output.resolve(), endpoint)
    write_json(args.global_output.resolve(), global_profile)
    load_endpoint_service_profile(args.endpoint_output.resolve())
    load_global_profile(args.global_output.resolve())
    provenance = {
        "schema": "tempo-go-workload-profile-rebind-v1",
        "identity_only": True,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "base_endpoint": str(base_endpoint_path),
        "base_endpoint_sha256": sha256(base_endpoint_path),
        "base_endpoint_numeric_rows_sha256": hashlib.sha256(
            json.dumps(base_endpoint_rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "endpoint_output": str(args.endpoint_output.resolve()),
        "endpoint_output_sha256": sha256(args.endpoint_output),
        "base_global": str(base_global_path),
        "base_global_sha256": sha256(base_global_path),
        "base_global_numeric_state_sha256": hashlib.sha256(
            json.dumps(base_global_numeric, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "global_output": str(args.global_output.resolve()),
        "global_output_sha256": sha256(args.global_output),
        "numeric_measurements_unchanged": True,
        "performance_claim_allowed": False,
    }
    write_json(args.provenance_output.resolve(), provenance)
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
