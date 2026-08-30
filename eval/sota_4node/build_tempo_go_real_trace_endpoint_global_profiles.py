#!/usr/bin/env python3
"""Bind the exact real-trace geometry to screen-only endpoint/global profiles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from tempo.pd_endpoint_profile import (
    endpoint_service_profile_fingerprint,
    load_endpoint_service_profile,
)
from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ttft_ms(row: dict) -> float:
    offsets = row.get("token_arrival_offsets_ns") or []
    if not offsets:
        raise ValueError("request has no first-token receipt")
    return (int(offsets[0]) - int(row["dispatch_offset_ns"])) / 1_000_000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-raw", type=Path, required=True)
    parser.add_argument("--remote-raw", type=Path, required=True)
    parser.add_argument("--elastic", type=Path, required=True)
    parser.add_argument("--source-endpoint", type=Path, required=True)
    parser.add_argument("--source-global", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--kv-bytes-per-token", type=int, required=True)
    parser.add_argument("--output-endpoint", type=Path, required=True)
    parser.add_argument("--output-global", type=Path, required=True)
    args = parser.parse_args()
    if args.output_endpoint.exists() or args.output_global.exists():
        raise ValueError("refusing to overwrite derived profile")

    local = _read(args.local_raw)["requests"]
    remote = _read(args.remote_raw)["requests"]
    local = [x for x in local if x["prompt_token_count"] == 8064 and x["requested_max_tokens"] == 128]
    remote = [x for x in remote if x["prompt_token_count"] == 8064 and x["requested_max_tokens"] == 128]
    if len(local) != 6 or len(remote) != 6:
        raise ValueError(f"expected six measured rows, got local={len(local)} remote={len(remote)}")
    if {x["semantic_request_id"] for x in local} != {x["semantic_request_id"] for x in remote}:
        raise ValueError("local and remote source identities differ")
    elastic = _read(args.elastic)
    elastic_row = next(
        x for x in elastic["rows"]
        if x["prompt_tokens"] == 8064 and x["output_tokens"] == 128
    )
    local_ttft = max(_ttft_ms(x) for x in local)
    remote_ttft = max(_ttft_ms(x) for x in remote)
    endpoint = copy.deepcopy(_read(args.source_endpoint))
    endpoint["profile_id"] = "tempo-go-real-trace-fixed8064-burst100-endpoint-screen-v1"
    endpoint["elastic_profile_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(elastic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    endpoint["workload_manifest_sha256"] = hashlib.sha256(
        args.workload_manifest.read_bytes()
    ).hexdigest()
    endpoint["deployment_scope"] = "calibration_only"
    endpoint["default_e2e_deadline_ms"] = 16000
    endpoint["rows"] = [{
        "prompt_tokens": 8064,
        "output_tokens": 128,
        # The local arm is a confirmed miss and the remote arm is a natural
        # LMCache hit.  This row is therefore an explicitly conservative
        # prefill-only proxy for the discovery screen, never final evidence.
        "cache_residency": "prefill_only",
        "local_ttft_prior_ms": local_ttft,
        "remote_ttft_prior_ms": remote_ttft,
        "local_token_ms": max(1, math.ceil(8064 * local_ttft)),
        "remote_prefill_token_ms": max(1, math.ceil(8064 * remote_ttft)),
        "samples_local": 6,
        "samples_remote": 6,
        "outputs_equivalent": True,
        "evidence_valid": True,
    }]
    endpoint["controller"] = {
        **endpoint["controller"],
        "local_token_ms_window": max(1, math.ceil(8064 * local_ttft * 6)),
        "remote_prefill_token_ms_window": max(1, math.ceil(8064 * remote_ttft * 2)),
        "remote_kv_bytes_window": 8064 * args.kv_bytes_per_token * 2,
        "remote_semantic_ops_window": 8,
        "minimum_feedback": 2,
    }
    endpoint["fingerprint_sha256"] = endpoint_service_profile_fingerprint(endpoint)
    args.output_endpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output_endpoint.write_text(json.dumps(endpoint, indent=2, sort_keys=True) + "\n")
    endpoint_loaded = load_endpoint_service_profile(args.output_endpoint)

    global_profile = copy.deepcopy(_read(args.source_global))
    global_profile["profile_id"] = "tempo-go-real-trace-fixed8064-burst100-global-screen-v1"
    global_profile["deployment_scope"] = "discovery"
    global_profile["identity"].update({
        "elastic_profile_fingerprint_sha256": endpoint["elastic_profile_fingerprint_sha256"],
        "endpoint_profile_id": endpoint_loaded.profile_id,
        "endpoint_profile_fingerprint_sha256": endpoint_loaded.fingerprint_sha256,
        "endpoint_profile_deployment_scope": endpoint_loaded.deployment_scope,
        "workload_manifest_sha256": endpoint["workload_manifest_sha256"],
    })
    for capacity in global_profile["capacities"]:
        capacity["local_prefill_token_ms"] = endpoint["controller"]["local_token_ms_window"]
        capacity["remote_prefill_token_ms"] = endpoint["controller"]["remote_prefill_token_ms_window"]
        capacity["remote_kv_bytes"] = endpoint["controller"]["remote_kv_bytes_window"]
        capacity["remote_semantic_ops"] = endpoint["controller"]["remote_semantic_ops_window"]
    global_profile["fingerprint_sha256"] = global_profile_fingerprint(global_profile)
    args.output_global.write_text(json.dumps(global_profile, indent=2, sort_keys=True) + "\n")
    global_loaded = load_global_profile(args.output_global)
    print(json.dumps({
        "endpoint_profile_id": endpoint_loaded.profile_id,
        "endpoint_fingerprint": endpoint_loaded.fingerprint_sha256,
        "global_profile_id": global_loaded.profile_id,
        "global_fingerprint": global_loaded.fingerprint_sha256,
        "local_ttft_prior_ms": local_ttft,
        "remote_ttft_prior_ms": remote_ttft,
        "screen_only": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
