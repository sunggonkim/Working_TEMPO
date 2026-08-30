#!/usr/bin/env python3
"""Build a screen-only Elastic profile from native real-trace fixed arms.

The Mooncake population uses semantic IDs rather than the historical
``item-N`` calibration IDs.  This adapter preserves every measured row while
giving the strict profile builder a deterministic paired identity.  It never
turns a screen profile into a replicated/final-validation profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import build_tempo_pd_elastic_profile_v445 as builder
from tempo.pd_elastic_profile import load_elastic_profile


def _load(path: Path, *, arm: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("validation", {}).get("performance_claim_allowed") is not True:
        raise ValueError(f"raw artifact is not performance-valid: {path}")
    rows = value.get("requests")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"raw artifact has no requests: {path}")
    for index, row in enumerate(rows):
        if row.get("valid") is not True:
            raise ValueError(f"invalid request {index} in {path}")
        semantic = str(row.get("semantic_request_id", ""))
        if not semantic:
            raise ValueError(f"request {index} lacks semantic ID: {path}")
        # The numeric semantic suffix is the immutable source item identity.
        suffix = semantic.rsplit("-", 1)[-1]
        if not suffix.isdigit():
            raise ValueError(f"semantic ID lacks numeric suffix: {semantic}")
        row["request_id"] = f"epd-{arm}-r0-measured-item-{int(suffix):06d}"
        if arm == "remote":
            # The official proxy's usage prompt count includes its one-token
            # prefill head.  Profile geometry is the source prompt geometry,
            # which is carried by the client-side prompt_token_count receipt.
            row["usage"]["prompt_tokens"] = row["prompt_token_count"]
    value["run"]["run_id"] = f"real-trace-{arm}"
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--remote", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--remote-backend", required=True)
    parser.add_argument("--classifier-version", required=True)
    parser.add_argument("--kv-bytes-per-token", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite profile: {args.output}")
    artifacts = [_load(args.local, arm="local"), _load(args.remote, arm="remote")]
    # v445 uses the canonical route names and verifies output equivalence for
    # every paired source item.  One native sample per geometry is valid for a
    # discovery screen; replicated deployment remains forbidden.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="tempo-real-trace-profile-") as tmp:
        paths = []
        for index, artifact in enumerate(artifacts):
            path = Path(tmp) / f"{index}.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            paths.append(path)
        payload = builder.build_profile(
            paths,
            profile_id=args.profile_id,
            model_id=args.model_id,
            model_revision=args.model_revision,
            topology_id=args.topology_id,
            remote_backend=args.remote_backend,
            classifier_version=args.classifier_version,
            kv_bytes_per_token=args.kv_bytes_per_token,
            local_capacity_equivalent=6,
            remote_capacity_equivalent=1,
            latency_estimator="median",
            spill_regression_budget_ms=95.0,
        )
    payload["deployment_scope"] = "screen_only"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    loaded = load_elastic_profile(args.output)
    print(json.dumps({
        "profile_id": loaded.profile_id,
        "deployment_scope": loaded.deployment_scope,
        "fingerprint_sha256": loaded.fingerprint_sha256,
        "rows": len(loaded.rows),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
