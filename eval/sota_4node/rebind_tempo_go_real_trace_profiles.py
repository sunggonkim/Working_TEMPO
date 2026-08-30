#!/usr/bin/env python3
"""Rebind discovery numeric profiles to a new immutable workload manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from tempo.pd_endpoint_profile import endpoint_service_profile_fingerprint, load_endpoint_service_profile
from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--global", dest="global_profile", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-endpoint", type=Path, required=True)
    parser.add_argument("--output-global", type=Path, required=True)
    args = parser.parse_args()
    if args.output_endpoint.exists() or args.output_global.exists():
        raise ValueError("refusing to overwrite derived profile")
    manifest_sha = sha(args.manifest)
    endpoint = copy.deepcopy(json.loads(args.endpoint.read_text(encoding="utf-8")))
    endpoint["profile_id"] = "tempo-go-real-trace-load25-endpoint-discovery-v1"
    endpoint["workload_manifest_sha256"] = manifest_sha
    endpoint["deployment_scope"] = "calibration_only"
    endpoint.pop("fingerprint_sha256", None)
    endpoint["fingerprint_sha256"] = endpoint_service_profile_fingerprint(endpoint)
    load_endpoint_service_profile_data = load_endpoint_service_profile
    args.output_endpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output_endpoint.write_text(json.dumps(endpoint, indent=2, sort_keys=True) + "\n")
    endpoint_loaded = load_endpoint_service_profile_data(args.output_endpoint)
    global_profile = copy.deepcopy(json.loads(args.global_profile.read_text(encoding="utf-8")))
    global_profile["profile_id"] = "tempo-go-real-trace-load25-global-discovery-v1"
    global_profile["deployment_scope"] = "discovery"
    global_profile["identity"].update({
        "endpoint_profile_id": endpoint_loaded.profile_id,
        "endpoint_profile_fingerprint_sha256": endpoint_loaded.fingerprint_sha256,
        "endpoint_profile_deployment_scope": endpoint_loaded.deployment_scope,
        "workload_manifest_sha256": manifest_sha,
    })
    global_profile.pop("fingerprint_sha256", None)
    global_profile["fingerprint_sha256"] = global_profile_fingerprint(global_profile)
    args.output_global.write_text(json.dumps(global_profile, indent=2, sort_keys=True) + "\n")
    global_loaded = load_global_profile(args.output_global)
    print(json.dumps({
        "endpoint_fingerprint": endpoint_loaded.fingerprint_sha256,
        "global_fingerprint": global_loaded.fingerprint_sha256,
        "manifest_sha256": manifest_sha,
        "performance_claim_allowed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
