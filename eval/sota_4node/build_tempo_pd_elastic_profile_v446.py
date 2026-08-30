#!/usr/bin/env python3
"""Correct Elastic profile builder for the official proxy head-token usage.

The pinned LMCache proxy reports the one-token prefill head in its subsequent
prompt usage.  We normalize exactly one token only when the request is on the
official remote route and carries the corresponding stream proof.  No other
geometry correction is permitted.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile

from eval.sota_4node import build_tempo_pd_elastic_profile_v445 as prior
from tempo.pd_elastic_profile_v444 import load_elastic_profile


def normalize_artifact(artifact: dict) -> tuple[dict, int]:
    value = copy.deepcopy(artifact)
    normalized = 0
    for row in value.get("requests", []):
        router = row.get("router")
        if not isinstance(router, dict):
            continue
        if router.get("route") not in prior._REMOTE_ROUTES:
            continue
        proofs = row.get("output_token_proofs")
        prior._require(
            isinstance(proofs, list)
            and proofs.count("official_lmcache_proxy_single_prefill_token") == 1,
            "remote prompt normalization requires exact proxy head-token proof",
        )
        usage = row.get("usage")
        prior._require(isinstance(usage, dict), "remote usage missing")
        prompt_tokens = usage.get("prompt_tokens")
        total_tokens = usage.get("total_tokens")
        prior._require(type(prompt_tokens) is int and prompt_tokens >= 2,
                       "remote prompt usage cannot normalize")
        usage["prompt_tokens"] = prompt_tokens - 1
        if type(total_tokens) is int:
            usage["total_tokens"] = total_tokens - 1
        normalized += 1
    return value, normalized


def build_profile(raw_paths: list[Path], **kwargs):
    with tempfile.TemporaryDirectory(prefix="tempo-elastic-profile-") as temporary:
        root = Path(temporary)
        normalized_paths = []
        normalized_count = 0
        for index, raw_path in enumerate(raw_paths):
            artifact = json.loads(raw_path.resolve().read_text())
            normalized, count = normalize_artifact(artifact)
            normalized_count += count
            destination = root / f"normalized-{index}.json"
            destination.write_text(json.dumps(normalized))
            normalized_paths.append(destination)
        prior._require(normalized_count > 0,
                       "no official proxy usage rows were normalized")
        return prior.build_profile(normalized_paths, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--remote-backend", required=True)
    parser.add_argument("--classifier-version", required=True)
    parser.add_argument("--kv-bytes-per-token", type=int, required=True)
    parser.add_argument("--local-capacity-equivalent", type=int, default=6)
    parser.add_argument("--remote-capacity-equivalent", type=int, default=1)
    parser.add_argument("--latency-estimator", choices=("max", "median"),
                        default="max")
    parser.add_argument("--spill-regression-budget-ms", type=float,
                        default=5.0)
    args = parser.parse_args()
    prior._require(not args.output.exists(), "refusing to overwrite profile")
    payload = build_profile(
        args.raw, profile_id=args.profile_id, model_id=args.model_id,
        model_revision=args.model_revision, topology_id=args.topology_id,
        remote_backend=args.remote_backend, classifier_version=args.classifier_version,
        kv_bytes_per_token=args.kv_bytes_per_token,
        local_capacity_equivalent=args.local_capacity_equivalent,
        remote_capacity_equivalent=args.remote_capacity_equivalent,
        latency_estimator=args.latency_estimator,
        spill_regression_budget_ms=args.spill_regression_budget_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    loaded = load_elastic_profile(args.output)
    print(json.dumps({"profile_id": loaded.profile_id,
                      "fingerprint_sha256": loaded.fingerprint_sha256,
                      "rows": len(loaded.rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
