#!/usr/bin/env python3
"""Validate a TEMPO-GO explicit-arrival manifest before GPU execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from eval.sota_4node.run_vllm_stream_metrics import load_workload
from tempo.pd_global_workload import MANIFEST_SCHEMA, WORKLOAD_SCHEMA


_TENANT = re.compile(r"^epd-tempo-(latency|interactive|batch|background)-")
_CACHE_CONTRACT = re.compile(r"-cache-(p-only|miss)-measured-")


def validate_manifest(manifest_path: Path, workload_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("TEMPO-GO manifest schema mismatch")
    if manifest.get("workload_schema") != WORKLOAD_SCHEMA:
        raise ValueError("TEMPO-GO workload schema mismatch")
    if manifest.get("performance_claim_allowed") is not False:
        raise ValueError("manifest must remain discovery-only")
    if manifest.get("workload_fields") != [
        "request_id", "prompt", "max_tokens", "arrival_offset_ms",
    ]:
        raise ValueError("workload field contract differs")
    items, workload_sha256 = load_workload(
        workload_path, default_max_tokens=64, request_rate=None)
    if not items:
        raise ValueError("workload is empty")
    offsets = [item.arrival_offset_ns for item in items]
    if offsets != sorted(offsets):
        raise ValueError("arrival offsets are not monotonic")
    tenants = []
    cache_contracts = []
    for item in items:
        match = _TENANT.match(item.request_id)
        if match is None:
            raise ValueError(f"request ID lacks a canonical tenant: {item.request_id}")
        tenants.append(match.group(1))
        matches = _CACHE_CONTRACT.findall(item.request_id)
        if len(matches) != 1:
            raise ValueError(
                "request ID must contain exactly one measured cache contract "
                f"(p-only or miss): {item.request_id}")
        cache_contracts.append(matches[0])

    manifest_cache_contracts = manifest.get("cache_contracts")
    if not isinstance(manifest_cache_contracts, dict):
        raise ValueError("manifest cache_contracts contract is missing")
    if manifest_cache_contracts.get("encoded_in_request_id") is not True:
        raise ValueError("manifest must encode cache contracts in request IDs")
    if manifest_cache_contracts.get("miss_prompt_namespace") != (
        "token_preserving_unique_first_chunk_v1"
    ):
        raise ValueError("manifest MISS namespace contract is missing")
    miss_prompts = [
        item.prompt for item, state in zip(items, cache_contracts)
        if state == "miss"
    ]
    if len(set(miss_prompts)) != len(miss_prompts):
        raise ValueError(
            "explicit MISS prompt namespace is reused; workload is not cold"
        )
    declared_unique = manifest_cache_contracts.get("miss_unique_prompt_count")
    if declared_unique != len(miss_prompts):
        raise ValueError(
            "manifest MISS unique prompt count differs from workload: "
            f"declared={declared_unique!r} observed={len(miss_prompts)}"
        )
    declared_counts = manifest_cache_contracts.get("counts")
    observed_counts = {
        state: cache_contracts.count(state) for state in ("miss", "p-only")
    }
    if declared_counts != observed_counts:
        raise ValueError(
            "manifest cache contract counts differ from workload: "
            f"declared={declared_counts!r} observed={observed_counts!r}")

    phases = manifest.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("manifest phases are missing")
    expected_start = 0
    phase_reports = []
    for phase in phases:
        if not isinstance(phase, dict):
            raise ValueError("manifest phase is not an object")
        start = phase.get("row_start")
        end = phase.get("row_end")
        if type(start) is not int or type(end) is not int or start != expected_start:
            raise ValueError("phase row ranges are not contiguous")
        if not 0 <= start < end <= len(items):
            raise ValueError("phase row range is outside workload")
        phase_offsets = offsets[start:end]
        phase_start_ms = float(phase["start_offset_ms"])
        phase_end_ms = phase_start_ms + float(phase["duration_ms"])
        if phase_offsets[0] < round(phase_start_ms * 1_000_000):
            raise ValueError("phase request precedes its start")
        if phase_offsets[-1] >= round(phase_end_ms * 1_000_000):
            raise ValueError("phase request reaches cooldown boundary")
        expected_start = end
        phase_reports.append({
            "replicate": phase.get("replicate"),
            "name": phase.get("name"),
            "request_count": end - start,
            "first_offset_ms": phase_offsets[0] / 1_000_000,
            "last_offset_ms": phase_offsets[-1] / 1_000_000,
        })
    if expected_start != len(items):
        raise ValueError("phase ranges do not cover the workload")
    return {
        "schema": "tempo-go-contention-manifest-validation-v1",
        "manifest": str(manifest_path.resolve()),
        "workload": str(workload_path.resolve()),
        "workload_sha256": workload_sha256,
        "request_count": len(items),
        "tenant_counts": {
            tenant: tenants.count(tenant)
            for tenant in sorted(set(tenants))
        },
        "cache_contract_counts": observed_counts,
        "phase_reports": phase_reports,
        "arrival_offsets_monotonic": True,
        "native_client_fields_only": True,
        "performance_claim_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_manifest(args.manifest.resolve(), args.workload.resolve())
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        if args.output.exists():
            raise ValueError(f"refusing to overwrite validation report: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
