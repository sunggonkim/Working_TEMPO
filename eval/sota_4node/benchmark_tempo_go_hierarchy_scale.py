#!/usr/bin/env python3
"""Measure TEMPO-GO hierarchy fan-in against a full candidate population.

This is a control-plane scale receipt only.  It does not model GPU, NCCL,
LMCache, Slingshot, latency, or goodput and must never be used as a native
performance claim.  Both paths use the same request/candidate/telemetry
population; the bounded path only limits what reaches the shard/global layer.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from tempo.pd_global_hierarchy import (
    HierarchicalCandidateReducer,
    HierarchicalRequestHeader,
)
from tempo.pd_global_orchestrator import (
    GlobalRequest,
    GlobalRoute,
    PairTelemetry,
    ResourceVector,
    RouteCandidate,
)


PROFILE = "a" * 64
EPOCH = "hierarchy-scale-allocation"
NOW_NS = 10
SHARD_COUNT = 64
MAX_PAIRS_PER_SHARD = 2


def _telemetry(pair: int) -> PairTelemetry:
    return PairTelemetry(
        pair_index=pair,
        sequence=1,
        sampled_ns=NOW_NS,
        collected_ns=NOW_NS + 1,
        agent_epoch=EPOCH,
        profile_fingerprint_sha256=PROFILE,
        controller_generation=0,
        observed_total=ResourceVector(),
    )


def _candidate(pair: int, route: GlobalRoute) -> RouteCandidate:
    if route is GlobalRoute.LOCAL:
        work = ResourceVector(
            decode_tokens=40,
            active_sequences=1,
            endpoint_requests=1,
            local_prefill_token_ms=40,
        )
    else:
        work = ResourceVector(
            decode_tokens=40,
            active_sequences=1,
            endpoint_requests=1,
            remote_prefill_token_ms=30,
            remote_kv_bytes=400,
            remote_semantic_ops=1,
        )
    # Keep the population deterministic while making pair/routing ranking
    # non-identical, as a real global decision would be.
    e2e = 10.0 + (pair % 17) * 0.25 + (0.5 if route is GlobalRoute.REMOTE else 0.0)
    return RouteCandidate(
        pair_index=pair,
        route=route,
        work=work,
        predicted_e2e_ms=e2e,
        predicted_ttft_ms=e2e / 2.0,
        uncertainty_ms=1.0,
    )


def _request(pair_count: int) -> GlobalRequest:
    return GlobalRequest(
        request_id=f"hierarchy-scale-{pair_count}",
        tenant_id="latency",
        arrival_ns=NOW_NS,
        deadline_ns=1_000_000_000,
        candidates=tuple(
            _candidate(pair, route)
            for pair in range(pair_count)
            for route in (GlobalRoute.LOCAL, GlobalRoute.REMOTE)
        ),
    )


def _candidate_dict(candidate: RouteCandidate) -> dict[str, object]:
    return {
        "pair_index": candidate.pair_index,
        "route": candidate.route.value,
        "work": candidate.work.as_dict(),
        "predicted_e2e_ms": candidate.predicted_e2e_ms,
        "predicted_ttft_ms": candidate.predicted_ttft_ms,
        "uncertainty_ms": candidate.uncertainty_ms,
        "cache_affinity": candidate.cache_affinity,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": statistics.median(values),
        "p99_ms": _percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def measure(pair_count: int, repeats: int) -> dict[str, object]:
    request = _request(pair_count)
    telemetry = tuple(_telemetry(pair) for pair in range(pair_count))
    values = {item.pair_index: item for item in telemetry}
    # Pair agents own their local candidate population before the global
    # reduction begins.  Do this partitioning once outside the measured
    # frontier-build interval; filtering the entire request once per pair
    # would add an artificial O(pair_count * candidate_count) cost that no
    # distributed agent pays and would make the hierarchy look worse than its
    # actual local ranking path.
    candidates_by_pair = {
        pair: tuple(
            item for item in request.candidates if item.pair_index == pair
        )
        for pair in range(pair_count)
    }
    full = HierarchicalCandidateReducer(
        shard_count=1,
        max_pairs_per_shard=pair_count,
        max_routes_per_pair=2,
    )
    bounded = HierarchicalCandidateReducer(
        shard_count=SHARD_COUNT,
        max_pairs_per_shard=MAX_PAIRS_PER_SHARD,
        max_routes_per_pair=2,
    )
    header = HierarchicalRequestHeader.from_request(request)

    full_ms: list[float] = []
    pair_agent_ms: list[float] = []
    bounded_global_ms: list[float] = []
    bounded_total_ms: list[float] = []
    full_result = None
    bounded_result = None
    frontiers = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        full_result = full.reduce(request, telemetry=values, now_ns=NOW_NS)
        full_ms.append((time.perf_counter_ns() - start) / 1e6)

        start = time.perf_counter_ns()
        frontiers = tuple(
            bounded.build_pair_frontier(
                pair_index=pair,
                candidates=candidates_by_pair[pair],
                telemetry=telemetry[pair],
            )
            for pair in range(pair_count)
        )
        pair_agent_ms.append((time.perf_counter_ns() - start) / 1e6)

        start = time.perf_counter_ns()
        bounded_result = bounded.reduce_frontiers(
            header,
            frontiers=frontiers,
            telemetry=values,
            now_ns=NOW_NS,
        )
        bounded_global_ms.append((time.perf_counter_ns() - start) / 1e6)
        bounded_total_ms.append(pair_agent_ms[-1] + bounded_global_ms[-1])

    assert full_result is not None and bounded_result is not None
    assert frontiers is not None
    full_payload = json.dumps(
        [_candidate_dict(item) for item in request.candidates],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    bounded_payload = json.dumps(
        [_candidate_dict(item) for item in bounded_result.request.candidates],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    full_selected = {
        (item.pair_index, item.route.value)
        for item in full_result.request.candidates
    }
    bounded_selected = {
        (item.pair_index, item.route.value)
        for item in bounded_result.request.candidates
    }
    return {
        "pair_count": pair_count,
        "raw_candidate_count": full_result.receipt.raw_candidate_count,
        "full_forwarded_candidate_count": full_result.receipt.forwarded_candidate_count,
        "bounded_forwarded_candidate_count": bounded_result.receipt.forwarded_candidate_count,
        "bounded_omitted_pair_count": bounded_result.receipt.omitted_pair_count,
        "full_payload_bytes": len(full_payload),
        "bounded_global_payload_bytes": len(bounded_payload),
        "payload_reduction_fraction": 1.0 - len(bounded_payload) / len(full_payload),
        "full_selected_identity_count": len(full_selected),
        "bounded_selected_identity_count": len(bounded_selected),
        "full_reduction": _summary(full_ms),
        "pair_agent_frontier_build": _summary(pair_agent_ms),
        "bounded_global_reduction": _summary(bounded_global_ms),
        "bounded_total_control_path": _summary(bounded_total_ms),
        "receipt_schema": bounded_result.receipt.schema,
        "bounded_fingerprint_sha256": bounded_result.fingerprint,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 3:
        raise SystemExit("--repeats must be >= 3")
    sizes = (2, 8, 32, 128, 512, 1024)
    result = {
        "schema": "tempo-go-hierarchy-scale-receipt-v1",
        "scope": "CPU control-plane only",
        "claim_boundary": (
            "No GPU, NCCL, LMCache, Slingshot, latency, goodput, or production claim"
        ),
        "same_population": True,
        "pair_agent_input_prepartitioned": True,
        "pair_agent_timing_scope": (
            "local frontier ranking and receipt construction after ownership "
            "partition; input partition is not charged to each agent"
        ),
        "full_global_scan": "all pair candidates reach one reducer",
        "bounded_path": {
            "shard_count": SHARD_COUNT,
            "max_pairs_per_shard": MAX_PAIRS_PER_SHARD,
            "max_routes_per_pair": 2,
        },
        "repeats": args.repeats,
        "scales": [measure(size, args.repeats) for size in sizes],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "schema": result["schema"],
        "scales": len(result["scales"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
