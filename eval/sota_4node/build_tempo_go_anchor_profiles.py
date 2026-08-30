#!/usr/bin/env python3
"""Derive the output=2 TEMPO-GO replay priors from frozen anchor raw traces.

The C1/C2 actual-inference traces contain the exact short decoder geometry
used by the anchor rates, while the older elastic profile intentionally
starts at output=16.  This utility joins measured anchor rows without inventing a
latency model, applies the already-pinned official-proxy head-token
normalization, and emits new screen/calibration-only profiles.  The strict
profiles contain no extra metadata; an adjacent provenance receipt records
the source hashes and pairing checks.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from tempo.pd_endpoint_profile import (
    endpoint_service_profile_fingerprint,
    load_endpoint_service_profile,
)
from tempo.pd_elastic_profile import load_elastic_profile


LOCAL_ROUTE = "decoder_local_chunked_prefill"
REMOTE_ROUTE = "official_lmcache_remote_prefill"
PROMPT_TOKENS = 4094
OUTPUT_TOKENS = 2
P_ONLY = "prefill_only"
REMOTE_HEAD_TOKEN_PROOF = "official_lmcache_proxy_single_prefill_token"
ELASTIC_SCHEMA = "tempo-elastic-pd-profile-444"
PROVENANCE_SCHEMA = "tempo-go-anchor-profile-provenance-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _semantic_key(request_id: object) -> tuple[str, int]:
    _require(isinstance(request_id, str), "anchor request ID is missing")
    try:
        tail = request_id.rsplit("-measured-", 1)[1]
        ordinal = int(tail.rsplit("-", 1)[1])
        phase = tail.rsplit("-", 1)[0].rsplit("-", 1)[0]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"anchor request ID lacks a semantic key: {request_id}") from exc
    _require(phase, "anchor semantic phase is empty")
    return phase, ordinal


def _load_records(path: Path) -> list[dict[str, Any]]:
    artifact = json.loads(path.resolve().read_text(encoding="utf-8"))
    validation = artifact.get("validation")
    _require(
        isinstance(validation, dict)
        and validation.get("all_streams_valid") is True
        and validation.get("router_decisions_exact") is True
        and validation.get("performance_claim_allowed") is True,
        f"anchor raw is not a valid calibration artifact: {path}",
    )
    requests = artifact.get("requests")
    decisions = artifact.get("router_decisions")
    _require(
        isinstance(requests, list) and isinstance(decisions, list),
        f"anchor request/decision lists are missing: {path}",
    )
    decision_index = {item.get("request_id"): item for item in decisions}
    _require(
        len(decision_index) == len(decisions)
        and all(request.get("request_id") in decision_index for request in requests),
        f"anchor request/decision IDs are not exact: {path}",
    )
    records: list[dict[str, Any]] = []
    for request in requests:
        decision = decision_index[request.get("request_id")]
        if request.get("requested_max_tokens") != OUTPUT_TOKENS:
            continue
        route = decision.get("route")
        if route not in {LOCAL_ROUTE, REMOTE_ROUTE}:
            continue
        _require(request.get("valid") is True, "selected anchor request is invalid")
        _require(decision.get("prompt_tokens") == PROMPT_TOKENS,
                 "selected anchor router geometry is not 4094 tokens")
        _require(decision.get("output_tokens") == OUTPUT_TOKENS,
                 "selected anchor router output geometry is not 2 tokens")
        usage = request.get("usage")
        _require(isinstance(usage, dict), "selected anchor request has no usage")
        observed_prompt = usage.get("prompt_tokens")
        _require(observed_prompt in {PROMPT_TOKENS, PROMPT_TOKENS + 1},
                 "anchor usage geometry is outside the pinned one-token correction")
        if observed_prompt == PROMPT_TOKENS + 1:
            proofs = request.get("output_token_proofs")
            _require(
                isinstance(proofs, list)
                and proofs.count(REMOTE_HEAD_TOKEN_PROOF) == 1,
                "anchor row lacks the exact head-token proof",
            )
        arrivals = request.get("token_arrival_offsets_ns")
        dispatch = request.get("dispatch_offset_ns")
        end = request.get("stream_end_offset_ns")
        _require(
            isinstance(arrivals, list)
            and arrivals
            and type(dispatch) is int
            and type(end) is int
            and arrivals[0] > dispatch
            and end > dispatch,
            "selected anchor timestamps are incomplete",
        )
        output_hash = request.get("output_text_sha256")
        _require(isinstance(output_hash, str) and len(output_hash) == 64,
                 "selected anchor output hash is missing")
        records.append({
            "key": _semantic_key(request.get("request_id")),
            "route": route,
            "cache_residency": decision.get("cache_residency"),
            "ttft_ms": (arrivals[0] - dispatch) / 1_000_000.0,
            "e2e_ms": (end - dispatch) / 1_000_000.0,
            "output_hash": output_hash,
        })
    _require(records, f"anchor raw has no selected output=2 rows: {path}")
    return records


def _cross_route_output_gate(
    records: list[dict[str, Any]], *, residency: str | None = None,
) -> dict[str, Any]:
    by_route: dict[str, set[str]] = {
        LOCAL_ROUTE: set(),
        REMOTE_ROUTE: set(),
    }
    for record in records:
        if residency is not None and record["cache_residency"] != residency:
            continue
        by_route[record["route"]].add(record["output_hash"])
    _require(by_route[LOCAL_ROUTE] and by_route[REMOTE_ROUTE],
             "anchor output hash evidence lacks one route")
    _require(by_route[LOCAL_ROUTE] == by_route[REMOTE_ROUTE],
             "anchor cross-route output hashes differ")
    return {
        "local_output_hashes": sorted(by_route[LOCAL_ROUTE]),
        "remote_output_hashes": sorted(by_route[REMOTE_ROUTE]),
        "local_samples": sum(
            1 for record in records
            if record["route"] == LOCAL_ROUTE
            and (residency is None or record["cache_residency"] == residency)),
        "remote_samples": sum(
            1 for record in records
            if record["route"] == REMOTE_ROUTE
            and (residency is None or record["cache_residency"] == residency)),
        "same_semantic_request_pairing": False,
    }


def _mad(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def _build_anchor_elastic(
    base_path: Path, records: list[dict[str, Any]], *, profile_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = json.loads(base_path.resolve().read_text(encoding="utf-8"))
    _require(base.get("schema") == ELASTIC_SCHEMA, "base elastic schema differs")
    _require(base.get("deployment_scope") in {"screen_only", "replicated"},
             "base elastic deployment scope differs")
    rows = list(base["rows"])
    _require(not any(
        row["prompt_tokens"] == PROMPT_TOKENS
        and row["output_tokens"] == OUTPUT_TOKENS
        for row in rows
    ), "base elastic profile already contains output=2")
    local = [row for row in records if row["route"] == LOCAL_ROUTE]
    remote = [row for row in records if row["route"] == REMOTE_ROUTE]
    output_gate = _cross_route_output_gate(records)
    _require(len(local) >= 2 and len(remote) >= 2,
             "output=2 elastic row needs both measured routes")
    _require(
        output_gate["local_samples"] >= 2
        and output_gate["remote_samples"] >= 2,
        "output=2 elastic row lacks cross-route output evidence",
    )
    local_latencies = [row["e2e_ms"] for row in local]
    remote_latencies = [row["e2e_ms"] for row in remote]
    row = {
        "prompt_tokens": PROMPT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "local_upper_bound_ms": max(local_latencies),
        "remote_upper_bound_ms": max(remote_latencies),
        "uncertainty_ms": max(1.0, 3.0 * _mad(local_latencies + remote_latencies)),
        "local_tbt_safe": True,
        "remote_evidence_valid": True,
        "local_compute_cost_us": math.ceil(max(row["ttft_ms"] for row in local) * 1000.0),
        "remote_kv_bytes": PROMPT_TOKENS * int(base["identity"]["kv_bytes_per_token"]),
        "samples_local": len(local),
        "samples_remote": len(remote),
        "outputs_equivalent": True,
        "remote_transfer_failures": 0,
    }
    rows.append(row)
    payload = dict(base)
    payload["profile_id"] = profile_id
    payload["deployment_scope"] = "screen_only"
    payload["rows"] = sorted(rows, key=lambda value: (
        value["prompt_tokens"], value["output_tokens"]))
    # Round-trip validation is done by the caller after the file is written.
    receipt = {
        "elastic_anchor_row": {
            "geometry": [PROMPT_TOKENS, OUTPUT_TOKENS],
            "cross_route_output_gate": output_gate,
            "local_samples": len(local),
            "remote_samples": len(remote),
            "local_e2e_upper_bound_ms": row["local_upper_bound_ms"],
            "remote_e2e_upper_bound_ms": row["remote_upper_bound_ms"],
            "local_compute_cost_us": row["local_compute_cost_us"],
        },
    }
    return payload, receipt


def _build_anchor_endpoint(
    base_path: Path, elastic_path: Path,
    *, profile_id: str, workload_manifest: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(base_path.resolve().read_text(encoding="utf-8"))
    _require(payload.get("schema") == "tempo-pd-endpoint-service-profile-v2",
             "base endpoint profile must be semantic v2")
    rows = list(payload["rows"])
    _require(
        any(
            row["prompt_tokens"] == PROMPT_TOKENS
            and row["output_tokens"] >= OUTPUT_TOKENS
            and row["cache_residency"] == P_ONLY
            for row in rows
        ),
        "base endpoint profile lacks a P_ONLY geometry ceiling",
    )
    payload = dict(payload)
    payload["profile_id"] = profile_id
    payload["elastic_profile_fingerprint_sha256"] = (
        load_elastic_profile(elastic_path.resolve()).fingerprint_sha256)
    payload["workload_manifest_sha256"] = _sha256(workload_manifest.resolve())
    payload["rows"] = sorted(rows, key=lambda value: (
        value["prompt_tokens"], value["output_tokens"], value["cache_residency"]))
    payload.pop("fingerprint_sha256", None)
    payload["fingerprint_sha256"] = endpoint_service_profile_fingerprint(payload)
    return payload, {
        "endpoint_anchor_row": {
            "geometry": [PROMPT_TOKENS, OUTPUT_TOKENS],
            "cache_residency": P_ONLY,
            "mode": "same_residency_geometry_ceiling_proxy",
            "exact_row_present": False,
            "source_geometries": [
                [PROMPT_TOKENS, 16, P_ONLY],
                [PROMPT_TOKENS, 128, P_ONLY],
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-elastic", type=Path, required=True)
    parser.add_argument("--base-endpoint", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--elastic-output", type=Path, required=True)
    parser.add_argument("--endpoint-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    args = parser.parse_args()
    _require(len(args.raw) == 4, "exactly four anchor raw artifacts are required")
    for output in (args.elastic_output, args.endpoint_output, args.provenance_output):
        _require(not output.exists(), f"refusing to overwrite {output}")
    records = []
    source_receipts = []
    for path in args.raw:
        records.extend(_load_records(path))
        source_receipts.append({"path": str(path.resolve()), "sha256": _sha256(path.resolve())})
    elastic_payload, elastic_receipt = _build_anchor_elastic(
        args.base_elastic, records,
        profile_id="tempo-pd-qwen25-c4-anchor-output2-screen-v1",
    )
    args.elastic_output.parent.mkdir(parents=True, exist_ok=True)
    args.elastic_output.write_text(
        json.dumps(elastic_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    elastic = load_elastic_profile(args.elastic_output.resolve())
    endpoint_payload, endpoint_receipt = _build_anchor_endpoint(
        args.base_endpoint,
        args.elastic_output,
        profile_id="tempo-pd-endpoint-qwen25-c4-anchor-output2-calibration-v1",
        workload_manifest=args.workload_manifest,
    )
    args.endpoint_output.parent.mkdir(parents=True, exist_ok=True)
    args.endpoint_output.write_text(
        json.dumps(endpoint_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    endpoint = load_endpoint_service_profile(args.endpoint_output.resolve())
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "deployment_scope": {
            "elastic": "screen_only",
            "endpoint": "calibration_only",
            "performance_claim_allowed": False,
        },
        "geometry": {
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "endpoint_cache_residency": P_ONLY,
        },
        "base_elastic": {
            "path": str(args.base_elastic.resolve()),
            "sha256": _sha256(args.base_elastic.resolve()),
        },
        "base_endpoint": {
            "path": str(args.base_endpoint.resolve()),
            "sha256": _sha256(args.base_endpoint.resolve()),
        },
        "workload_manifest": {
            "path": str(args.workload_manifest.resolve()),
            "sha256": _sha256(args.workload_manifest.resolve()),
        },
        "anchor_raw": source_receipts,
        "elastic_profile": {
            "path": str(args.elastic_output.resolve()),
            "fingerprint_sha256": elastic.fingerprint_sha256,
        },
        "endpoint_profile": {
            "path": str(args.endpoint_output.resolve()),
            "fingerprint_sha256": endpoint.fingerprint_sha256,
        },
        "checks": {
            "remote_head_token_normalization": REMOTE_HEAD_TOKEN_PROOF,
            "route_contract": [LOCAL_ROUTE, REMOTE_ROUTE],
            **elastic_receipt,
            **endpoint_receipt,
        },
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "elastic_profile": str(args.elastic_output.resolve()),
        "elastic_fingerprint_sha256": elastic.fingerprint_sha256,
        "endpoint_profile": str(args.endpoint_output.resolve()),
        "endpoint_fingerprint_sha256": endpoint.fingerprint_sha256,
        "provenance": str(args.provenance_output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
