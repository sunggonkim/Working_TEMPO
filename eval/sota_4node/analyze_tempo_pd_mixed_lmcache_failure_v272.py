#!/usr/bin/env python3
"""Validate an official-LMCache EngineCore failure in a mixed P/D window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def analyze(raw_path: Path, decoder_log: Path, allocation: int,
            request_rate: float, workers: int) -> dict:
    raw = json.loads(raw_path.resolve().read_text())
    requests = raw.get("requests", [])
    decisions = raw.get("router_decisions", [])
    _require(len(requests) == 48 and len(decisions) == 48,
             "mixed request/decision geometry changed")
    _require(raw["validation"]["performance_claim_allowed"] is False,
             "artifact unexpectedly performance-valid")
    invalid = [row for row in requests if not row.get("valid")]
    _require(invalid, "no failed streams")
    log = decoder_log.resolve().read_text(errors="replace")
    markers = {
        "lmcache_missing_local_key_assertion": (
            "AssertionError: Key CacheEngineKey" in log
            and "not found in local data" in log),
        "vllm_engine_core_fatal": "EngineCore encountered a fatal error" in log,
        "lmcache_retrieve_path": (
            "lmcache/v1/storage_backend/pd_backend_async.py" in log
            and "get_blocking" in log),
    }
    _require(all(markers.values()), "fatal LMCache signature incomplete")
    route_counts: dict[str, int] = {}
    for row in decisions:
        route_counts[row["route"]] = route_counts.get(row["route"], 0) + 1
    return {
        "schema": "tempo-pd-mixed-lmcache-failure-analysis-272",
        "allocation_id": allocation,
        "request_rate_per_s": request_rate,
        "max_workers": workers,
        "raw": str(raw_path.resolve()),
        "decoder_log": str(decoder_log.resolve()),
        "requests": 48,
        "valid_streams": 48 - len(invalid),
        "invalid_streams": len(invalid),
        "invalid_request_ids": [row["request_id"] for row in invalid],
        "router_decisions_complete": len(decisions) == 48,
        "route_counts": route_counts,
        "fatal_signature": markers,
        "verdict": "official_lmcache_concurrent_retrieval_fatal",
        "performance_claim_allowed": False,
        "claim_boundary": (
            "Correctness/stability failure under one same-window mixed actual-vLLM "
            "P/D run. The invalid artifact must not be used for latency or throughput "
            "claims; it is evidence only of the reproduced fatal retrieval path."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--decoder-log", type=Path, required=True)
    parser.add_argument("--allocation", type=int, required=True)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite")
    report = analyze(args.raw, args.decoder_log, args.allocation,
                     args.request_rate, args.workers)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": report["verdict"],
                      "invalid_streams": report["invalid_streams"],
                      "fatal_signature": report["fatal_signature"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
