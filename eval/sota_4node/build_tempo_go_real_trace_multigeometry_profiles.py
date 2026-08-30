#!/usr/bin/env python3
"""Build a discovery profile covering every geometry in one real trace.

The profile is intentionally discovery-only.  It combines two independent
same-workload local/remote raw runs so the endpoint contract has two samples
per geometry; it does not claim a frozen production calibration.
"""

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ttft_ms(row: dict) -> float:
    arrivals = row.get("token_arrival_offsets_ns")
    if not isinstance(arrivals, list) or not arrivals:
        raise ValueError("raw request lacks first-token receipt")
    return (int(arrivals[0]) - int(row["dispatch_offset_ns"])) / 1_000_000.0


def _valid_rows(path: Path, *, expected_workload_sha: str | None = None) -> dict[tuple[int, int], list[dict]]:
    raw = _read(path)
    validation = raw.get("validation")
    if not isinstance(validation, dict) or validation.get("all_streams_valid") is not True:
        raise ValueError(f"raw run is not valid: {path}")
    workload = raw.get("workload")
    workload_sha = workload.get("sha256") if isinstance(workload, dict) else None
    if expected_workload_sha is not None and workload_sha != expected_workload_sha:
        raise ValueError("local and remote workload identities differ")
    rows = raw.get("requests")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"raw request rows are missing: {path}")
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        if row.get("valid") is not True:
            raise ValueError(f"raw run contains an invalid request: {path}")
        key = (int(row["prompt_token_count"]), int(row["requested_max_tokens"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-raw", type=Path, required=True)
    parser.add_argument("--local-raw-repeat", type=Path, required=True)
    parser.add_argument("--remote-raw", type=Path, required=True)
    parser.add_argument("--remote-raw-repeat", type=Path, required=True)
    parser.add_argument("--elastic", type=Path, required=True)
    parser.add_argument("--source-endpoint", type=Path, required=True)
    parser.add_argument("--source-global", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--kv-bytes-per-token", type=int, required=True)
    parser.add_argument("--output-endpoint", type=Path, required=True)
    parser.add_argument("--output-global", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.output_endpoint, args.output_global)
    if any(path.exists() for path in outputs):
        raise ValueError("refusing to overwrite derived profile")

    local_raw = _read(args.local_raw)
    workload_sha = local_raw.get("workload", {}).get("sha256")
    if not isinstance(workload_sha, str):
        raise ValueError("local workload SHA is missing")
    local = _valid_rows(args.local_raw, expected_workload_sha=workload_sha)
    local_repeat = _valid_rows(args.local_raw_repeat, expected_workload_sha=workload_sha)
    remote = _valid_rows(args.remote_raw, expected_workload_sha=workload_sha)
    remote_repeat = _valid_rows(args.remote_raw_repeat, expected_workload_sha=workload_sha)
    keys = set(local) & set(local_repeat) & set(remote) & set(remote_repeat)
    if keys != set(local) or keys != set(remote):
        raise ValueError("local/remote geometry coverage differs")
    if any(len(local[key]) + len(local_repeat[key]) < 2
           or len(remote[key]) + len(remote_repeat[key]) < 2
           for key in keys):
        raise ValueError("each geometry must have at least two independent samples")

    elastic = _read(args.elastic)
    elastic_sha = hashlib.sha256(
        json.dumps(elastic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    endpoint = copy.deepcopy(_read(args.source_endpoint))
    endpoint["profile_id"] = "tempo-go-real-trace-multigeometry-endpoint-discovery-v1"
    endpoint["elastic_profile_fingerprint_sha256"] = elastic_sha
    endpoint["workload_manifest_sha256"] = _sha256(args.workload_manifest)
    endpoint["deployment_scope"] = "calibration_only"
    endpoint["rows"] = []
    for prompt_tokens, output_tokens in sorted(keys):
        local_samples = (
            local[(prompt_tokens, output_tokens)]
            + local_repeat[(prompt_tokens, output_tokens)]
        )
        remote_samples = (
            remote[(prompt_tokens, output_tokens)]
            + remote_repeat[(prompt_tokens, output_tokens)]
        )
        local_ttft = max(_ttft_ms(row) for row in local_samples)
        remote_ttft = max(_ttft_ms(row) for row in remote_samples)
        endpoint["rows"].append({
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "cache_residency": "prefill_only",
            "local_ttft_prior_ms": local_ttft,
            "remote_ttft_prior_ms": remote_ttft,
            "local_token_ms": max(1, math.ceil(prompt_tokens * local_ttft)),
            "remote_prefill_token_ms": max(1, math.ceil(prompt_tokens * remote_ttft)),
            "samples_local": len(local_samples),
            "samples_remote": len(remote_samples),
            "outputs_equivalent": True,
            "evidence_valid": True,
        })
    endpoint["controller"] = {
        **endpoint["controller"],
        "local_token_ms_window": max(row["local_token_ms"] for row in endpoint["rows"]) * 16,
        "remote_prefill_token_ms_window": max(row["remote_prefill_token_ms"] for row in endpoint["rows"]) * 16,
        "remote_kv_bytes_window": max(row["prompt_tokens"] for row in endpoint["rows"]) * args.kv_bytes_per_token * 16,
        "remote_semantic_ops_window": 16,
        "minimum_feedback": 2,
    }
    endpoint["fingerprint_sha256"] = endpoint_service_profile_fingerprint(endpoint)
    args.output_endpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output_endpoint.write_text(json.dumps(endpoint, indent=2, sort_keys=True) + "\n")
    endpoint_loaded = load_endpoint_service_profile(args.output_endpoint)

    global_profile = copy.deepcopy(_read(args.source_global))
    global_profile["profile_id"] = "tempo-go-real-trace-multigeometry-global-discovery-v1"
    global_profile["deployment_scope"] = "discovery"
    global_profile["identity"].update({
        "elastic_profile_fingerprint_sha256": elastic_sha,
        "endpoint_profile_id": endpoint_loaded.profile_id,
        "endpoint_profile_fingerprint_sha256": endpoint_loaded.fingerprint_sha256,
        "endpoint_profile_deployment_scope": endpoint_loaded.deployment_scope,
        "workload_manifest_sha256": _sha256(args.workload_manifest),
    })
    global_profile["fingerprint_sha256"] = global_profile_fingerprint(global_profile)
    args.output_global.write_text(json.dumps(global_profile, indent=2, sort_keys=True) + "\n")
    global_loaded = load_global_profile(args.output_global)
    print(json.dumps({
        "endpoint_profile_id": endpoint_loaded.profile_id,
        "endpoint_fingerprint": endpoint_loaded.fingerprint_sha256,
        "global_profile_id": global_loaded.profile_id,
        "global_fingerprint": global_loaded.fingerprint_sha256,
        "geometry_count": len(keys),
        "workload_sha256": workload_sha,
        "screen_only": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
