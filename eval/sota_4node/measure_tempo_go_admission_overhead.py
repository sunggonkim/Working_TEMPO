#!/usr/bin/env python3
"""Measure TEMPO-GO control-plane CPU overhead on two frozen profiles.

This benchmark deliberately does not model GPU, network, vLLM, or LMCache
latency.  It runs the same causal telemetry/admission/first-response/EOF
lifecycle against a baseline profile and one controller candidate, then emits
only control-plane timing evidence.  It cannot authorize a performance claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from tempo.pd_global_orchestrator import (
    GlobalOrchestrator,
    GlobalRequest,
    GlobalRoute,
    PairTelemetry,
    PathHealth,
    ResourceVector,
    RouteCandidate,
)
from tempo.pd_global_profile import load_global_profile


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        raise ValueError("cannot summarize an empty timing series")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _summary(values: list[int]) -> dict[str, Any]:
    return {
        "samples": len(values),
        "min_ns": min(values),
        "p50_ns": _percentile(values, 0.50),
        "p95_ns": _percentile(values, 0.95),
        "p99_ns": _percentile(values, 0.99),
        "max_ns": max(values),
        "p50_us": _percentile(values, 0.50) / 1_000.0,
        "p99_us": _percentile(values, 0.99) / 1_000.0,
    }


def _telemetry(profile: Any, sequence: int, sampled_ns: int) -> tuple[PairTelemetry, ...]:
    values = []
    for pair in range(profile.topology.pair_count):
        values.append(PairTelemetry(
            pair_index=pair,
            sequence=sequence,
            sampled_ns=sampled_ns,
            collected_ns=sampled_ns + 1,
            agent_epoch="tempo-go-overhead-benchmark-epoch",
            profile_fingerprint_sha256=profile.fingerprint_sha256,
            controller_generation=profile.telemetry.controller_generation,
            observed_total=ResourceVector(),
            local_health=PathHealth.GOOD,
            remote_health=PathHealth.GOOD,
            local_service_multiplier=1.0,
            remote_service_multiplier=1.0,
        ))
    return tuple(values)


def _request(profile: Any, request_id: str, tenant_id: str, arrival_ns: int) -> GlobalRequest:
    local_work = ResourceVector(
        decode_tokens=8,
        active_sequences=1,
        endpoint_requests=1,
        local_prefill_token_ms=8,
    )
    candidates = tuple(RouteCandidate(
        pair_index=pair,
        route=GlobalRoute.LOCAL,
        work=local_work,
        predicted_e2e_ms=10.0 + pair,
        predicted_ttft_ms=5.0 + pair,
        uncertainty_ms=1.0,
    ) for pair in range(profile.topology.pair_count))
    return GlobalRequest(
        request_id=request_id,
        tenant_id=tenant_id,
        arrival_ns=arrival_ns,
        deadline_ns=arrival_ns + 10_000_000_000,
        candidates=candidates,
    )


def _measure(profile_path: Path, *, warmup: int, samples: int) -> dict[str, Any]:
    profile = load_global_profile(profile_path)
    tenant_id = profile.tenants[0].tenant_id
    orchestrator = GlobalOrchestrator(profile.orchestrator_config())
    sequence = 1
    clock_ns = 1_000_000

    for _ in range(warmup):
        orchestrator.update_telemetry_batch(_telemetry(profile, sequence, clock_ns))
        sequence += 1
        request = _request(profile, f"warmup-{sequence}", tenant_id, clock_ns)
        decision = orchestrator.submit(request, now_ns=clock_ns)
        if decision.kind.value != "admit":
            raise RuntimeError(f"warmup request was not admitted: {decision.reason}")
        orchestrator.mark_first_response(request.request_id, now_ns=clock_ns + 1)
        orchestrator.complete(request.request_id, now_ns=clock_ns + 2)
        clock_ns += 10_000

    telemetry_ns: list[int] = []
    admission_ns: list[int] = []
    lifecycle_ns: list[int] = []
    total_ns: list[int] = []
    for index in range(samples):
        request = _request(profile, f"sample-{index}", tenant_id, clock_ns)
        start = time.perf_counter_ns()
        telemetry_start = time.perf_counter_ns()
        orchestrator.update_telemetry_batch(_telemetry(profile, sequence, clock_ns))
        telemetry_ns.append(time.perf_counter_ns() - telemetry_start)
        sequence += 1

        admission_start = time.perf_counter_ns()
        decision = orchestrator.submit(request, now_ns=clock_ns)
        admission_ns.append(time.perf_counter_ns() - admission_start)
        if decision.kind.value != "admit":
            raise RuntimeError(f"sample request was not admitted: {decision.reason}")

        lifecycle_start = time.perf_counter_ns()
        orchestrator.mark_first_response(request.request_id, now_ns=clock_ns + 1)
        orchestrator.complete(request.request_id, now_ns=clock_ns + 2)
        lifecycle_ns.append(time.perf_counter_ns() - lifecycle_start)
        total_ns.append(time.perf_counter_ns() - start)
        clock_ns += 10_000

    return {
        "profile_id": profile.profile_id,
        "profile_fingerprint_sha256": profile.fingerprint_sha256,
        "transport": profile.transport,
        "warmup": warmup,
        "samples": samples,
        "all_requests_admitted": True,
        "telemetry_refresh": _summary(telemetry_ns),
        "admission_submit": _summary(admission_ns),
        "lifecycle_first_response_eof": _summary(lifecycle_ns),
        "control_plane_total": _summary(total_ns),
        "performance_claim_allowed": False,
        "measurement_scope": "CPU control-plane only; no GPU/network/LMCache latency",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples <= 0:
        raise ValueError("warmup must be non-negative and samples must be positive")
    result = {
        "schema": "tempo-go-control-plane-overhead-v1",
        "baseline": _measure(args.baseline_profile.resolve(), warmup=args.warmup, samples=args.samples),
        "candidate": _measure(args.candidate_profile.resolve(), warmup=args.warmup, samples=args.samples),
        "performance_claim_allowed": False,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
