#!/usr/bin/env python3
"""Derive a verified, source-bound fixed-geometry Mooncake replay.

The source trace is immutable.  This adapter selects rows with one exact
input/output geometry and optionally compresses only their recorded arrival
offsets to expose a controlled burst.  Token IDs, request IDs, and source
metadata remain unchanged; the manifest records the derived replay contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from tempo.mooncake_fast25_workload import (
    _canonical_json_bytes,
    _quantiles,
    verify_population,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-workload", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--arrival-compression", type=float, default=1.0)
    args = parser.parse_args()
    if args.arrival_compression <= 0:
        raise ValueError("arrival compression must be positive")
    if args.output_workload.exists() or args.output_manifest.exists():
        raise ValueError("refusing to overwrite derived replay")

    source_bytes = args.workload.read_bytes()
    source_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verify_population(source_bytes, source_manifest)
    source_rows = [json.loads(line) for line in source_bytes.decode().splitlines()]
    selected = [
        row for row in source_rows
        if len(row["prompt"]) == 8064 and row["max_tokens"] == 128
    ]
    if len(selected) != 6:
        raise ValueError(f"expected six exact source rows, found {len(selected)}")
    rows = copy.deepcopy(selected)
    for row in rows:
        row["arrival_offset_ms"] = round(
            float(row["arrival_offset_ms"]) / args.arrival_compression, 6,
        )
    rows[0]["arrival_offset_ms"] = 0.0
    workload_bytes = b"".join(
        _canonical_json_bytes(row) + b"\n" for row in rows
    )

    source_index = source_manifest["request_index"]
    request_index = {
        row["request_id"]: copy.deepcopy(source_index[row["request_id"]])
        for row in rows
    }
    semantic_rows = [
        {"request_id": row["request_id"], **request_index[row["request_id"]]}
        for row in rows
    ]
    manifest = copy.deepcopy(source_manifest)
    manifest["selection"] = {
        "contiguous_source_window": False,
        "derived_from_request_ids": [row["request_id"] for row in rows],
        "request_count": len(rows),
        "source_geometry": {"input_tokens": 8064, "output_tokens": 128},
    }
    duration_ms = float(rows[-1]["arrival_offset_ms"])
    manifest["arrival"] = {
        "semantics": "source_order_offsets_divided_by_arrival_compression",
        "source_order_preserved": True,
        "load_multiplier": args.arrival_compression,
        "duration_ms": duration_ms,
        "effective_offered_rate_per_s": (
            (len(rows) - 1) * 1000.0 / duration_ms
            if duration_ms > 0 else None
        ),
    }
    manifest["context"] = {
        **manifest["context"],
        "clipped_requests": 0,
        "clipped_input_tokens": 0,
    }
    manifest["output"] = {
        **manifest["output"],
        "floor_adjusted_requests": 0,
        "floor_added_tokens": 0,
        "cap_adjusted_requests": 0,
        "cap_removed_tokens": 0,
    }
    block_ids = {
        tuple(row["prompt"][start:start + 512])
        for row in rows
        for start in range(0, len(row["prompt"]), 512)
    }
    manifest["token_materialization"] = {
        **manifest["token_materialization"],
        "materialized_prompt_tokens": sum(len(row["prompt"]) for row in rows),
        "unique_hash_blocks": len(block_ids),
    }
    manifest["reuse"] = {
        **manifest["reuse"],
        "requests_with_prior_reusable_prefix": 0,
        "prior_reusable_prefix_tokens": _quantiles([0] * len(rows)),
    }
    manifest["distribution"] = {
        "original_input_tokens": _quantiles([8064] * len(rows)),
        "effective_input_tokens": _quantiles([8064] * len(rows)),
        "original_output_tokens": _quantiles([128] * len(rows)),
        "effective_output_tokens": _quantiles([128] * len(rows)),
    }
    manifest["population_semantic_sha256"] = _sha(
        _canonical_json_bytes(semantic_rows),
    )
    manifest["request_index"] = request_index
    manifest["workload_sha256"] = _sha(workload_bytes)
    manifest["performance_claim_allowed"] = False
    manifest["derived_replay"] = {
        "schema": "tempo-go-source-bound-fixed-geometry-replay-v1",
        "source_workload_sha256": source_manifest["workload_sha256"],
        "arrival_compression": args.arrival_compression,
        "geometry_exact": True,
        "synthetic_tokens": False,
    }
    verify_population(workload_bytes, manifest)
    args.output_workload.parent.mkdir(parents=True, exist_ok=True)
    args.output_workload.write_bytes(workload_bytes)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "request_count": len(rows),
        "arrival_compression": args.arrival_compression,
        "offered_rate_per_s": manifest["arrival"]["effective_offered_rate_per_s"],
        "workload_sha256": manifest["workload_sha256"],
        "verified": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
