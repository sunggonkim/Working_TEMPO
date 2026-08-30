#!/usr/bin/env python3
"""Build a strict Elastic-PD profile from paired actual-vLLM router traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

from tempo.pd_elastic_profile_v444 import SCHEMA, load_elastic_profile


_ITEM = re.compile(r"(?:cache-)?item-(\d+)$")
_BALANCED_ITEM = re.compile(
    r"^epd-(?:local|remote)-r(\d+)-measured-item-(\d+)$")
_LOCAL_ROUTES = {"decoder_local_recompute_or_cache", "decoder_local_chunked_prefill"}
_REMOTE_ROUTES = {"remote_prefill_live_kv", "official_lmcache_remote_prefill"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _latency_ms(row: dict[str, Any]) -> float:
    return (row["stream_end_offset_ns"] - row["dispatch_offset_ns"]) / 1_000_000


def _ttft_us(row: dict[str, Any]) -> int:
    arrivals = row["token_arrival_offsets_ns"]
    _require(isinstance(arrivals, list) and arrivals, "token arrivals missing")
    value = arrivals[0] - row["dispatch_offset_ns"]
    _require(value > 0, "TTFT must be positive")
    return math.ceil(value / 1000)


def _mad(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def build_profile(
    raw_paths: list[Path], *, profile_id: str, model_id: str,
    model_revision: str, topology_id: str, remote_backend: str,
    classifier_version: str, kv_bytes_per_token: int,
    local_capacity_equivalent: int, remote_capacity_equivalent: int,
    latency_estimator: str = "max",
    spill_regression_budget_ms: float = 5.0,
) -> dict[str, Any]:
    _require(raw_paths, "at least one raw artifact is required")
    _require(latency_estimator in {"max", "median"}, "unknown latency estimator")
    groups: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"local": [], "remote": []})
    paired: dict[tuple[str, int, int, int], dict[str, str]] = defaultdict(dict)
    for raw_path in raw_paths:
        artifact = json.loads(raw_path.resolve().read_text())
        _require(artifact.get("validation", {}).get("performance_claim_allowed") is True,
                 f"invalid raw artifact: {raw_path}")
        run_id = str(artifact.get("run", {}).get("run_id"))
        for row in artifact.get("requests", []):
            _require(row.get("valid") is True, "invalid request in calibration raw")
            router = row.get("router")
            _require(isinstance(router, dict), "router provenance missing")
            route = router.get("route")
            label = "local" if route in _LOCAL_ROUTES else (
                "remote" if route in _REMOTE_ROUTES else None)
            _require(label is not None, "unknown calibration route")
            usage = row.get("usage")
            _require(isinstance(usage, dict), "usage missing")
            prompt_tokens = usage.get("prompt_tokens")
            output_tokens = row.get("requested_max_tokens")
            _require(type(prompt_tokens) is int and prompt_tokens > 0,
                     "prompt token count missing")
            _require(type(output_tokens) is int and output_tokens >= 2,
                     "output token count missing")
            groups[(prompt_tokens, output_tokens)][label].append(row)
            request_id = str(row.get("request_id", ""))
            match = _ITEM.search(request_id)
            _require(match is not None, "request ID lacks stable item suffix")
            balanced = _BALANCED_ITEM.fullmatch(request_id)
            pair_run_id = (f"balanced-r{balanced.group(1)}"
                           if balanced is not None else run_id)
            pair_key = (pair_run_id, prompt_tokens, output_tokens,
                        int(match.group(1)))
            prior = paired[pair_key].get(label)
            observed_hash = row.get("output_text_sha256")
            _require(isinstance(observed_hash, str) and observed_hash,
                     "output hash missing")
            _require(prior is None or prior == observed_hash,
                     "duplicate route output mismatch")
            paired[pair_key][label] = observed_hash

    rows = []
    for (prompt_tokens, output_tokens), samples in sorted(groups.items()):
        local = samples["local"]
        remote = samples["remote"]
        _require(local and remote, "each geometry needs both routes")
        relevant_pairs = [value for key, value in paired.items()
                          if key[1:3] == (prompt_tokens, output_tokens)]
        comparable = [value for value in relevant_pairs
                      if set(value) == {"local", "remote"}]
        outputs_equivalent = bool(comparable) and all(
            value["local"] == value["remote"] for value in comparable)
        local_latency = [_latency_ms(row) for row in local]
        remote_latency = [_latency_ms(row) for row in remote]
        representative = (
            max if latency_estimator == "max" else statistics.median
        )
        uncertainty = max(1.0, 3.0 * _mad(local_latency + remote_latency))
        rows.append({
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "local_upper_bound_ms": representative(local_latency),
            "remote_upper_bound_ms": representative(remote_latency),
            "uncertainty_ms": uncertainty,
            "local_tbt_safe": True,
            "remote_evidence_valid": outputs_equivalent,
            "local_compute_cost_us": max(_ttft_us(row) for row in local),
            "remote_kv_bytes": prompt_tokens * kv_bytes_per_token,
            "samples_local": len(local),
            "samples_remote": len(remote),
            "outputs_equivalent": outputs_equivalent,
            "remote_transfer_failures": 0,
        })
    _require(rows, "no calibration rows")
    local_unit = max(row["local_compute_cost_us"] for row in rows)
    remote_unit = max(row["remote_kv_bytes"] for row in rows)
    return {
        "schema": SCHEMA,
        "profile_id": profile_id,
        "deployment_scope": "screen_only",
        "identity": {
            "model_id": model_id, "model_revision": model_revision,
            "topology_id": topology_id, "remote_backend": remote_backend,
            "classifier_version": classifier_version,
            "kv_bytes_per_token": kv_bytes_per_token,
        },
        "controller": {
            "local_compute_budget_us": local_capacity_equivalent * local_unit,
            "remote_kv_budget_bytes": remote_capacity_equivalent * remote_unit,
            "arrival_window": 4,
            "enter_high_gap_ns": 39_000_000,
            "exit_high_gap_ns": 78_000_000,
            "exit_consecutive_windows": 2,
            "route_margin_ms": 5.0,
            "spill_regression_budget_ms": spill_regression_budget_ms,
        },
        "rows": rows,
    }


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
    _require(not args.output.exists(), "refusing to overwrite profile")
    _require(args.kv_bytes_per_token > 0, "kv bytes per token must be positive")
    _require(args.local_capacity_equivalent > 0, "local capacity must be positive")
    _require(args.remote_capacity_equivalent > 0, "remote capacity must be positive")
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
    # Round-trip through the production strict loader before publishing.
    loaded = load_elastic_profile(args.output)
    print(json.dumps({"profile_id": loaded.profile_id,
                      "fingerprint_sha256": loaded.fingerprint_sha256,
                      "rows": len(loaded.rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
