#!/usr/bin/env python3
"""Build the signed same-allocation adaptive hybrid pulse retry."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

from eval.sota_4node import compile_lmcache_active_pulse_plan as pilot
from tempo.inference_service import ServiceQuantum
from tempo.inference_service_active import (
    ActiveServicePlan,
    ActiveServiceProfile,
    load_active_service_artifact,
    make_active_service_artifact,
    validate_active_service_plan,
)


SCHEMA = "tempo-lmcache-active-pulse-hybrid-plan-1"
DERIVATION = "same_allocation_adaptive_fixed_hybrid_replay_not_independent"
PULSE_TOKENS = (4, 5, 7, 10, 12, 15, 17, 18, 20, 23, 25, 26, 28)
GROUPED_PULSE_TOKENS = (7, 12, 20)
RUNTIME_WIDTH_BY_TOKEN = tuple(
    8 if token in GROUPED_PULSE_TOKENS else 4 if token in PULSE_TOKENS else 0
    for token in range(64)
)
LOGICAL_WIDTH_BY_TOKEN = tuple(
    4 if token in PULSE_TOKENS else 0 for token in range(64)
)
LOGICAL_QUANTA = 52
ONE_MIB_BYTES = 1_048_576
TWO_MIB_BYTES = 2_097_152
ONE_MIB_SERVICE_NS = 5_253_812
TWO_MIB_SERVICE_NS = 8_892_306
EXPECTED_COMPLETION_NS = 89_977_642
EXPECTED_MAX_LAG_NS = 2_165_292
EXPECTED_TOTAL_PENALTY_NS = 22_030_380
EXPECTED_PEAK_PENALTY_NS = 815_940


def group_bytes() -> tuple[int, ...]:
    return tuple(
        TWO_MIB_BYTES if token in GROUPED_PULSE_TOKENS else ONE_MIB_BYTES
        for token in PULSE_TOKENS
    )


def group_service_ns() -> tuple[int, ...]:
    return tuple(
        TWO_MIB_SERVICE_NS
        if token in GROUPED_PULSE_TOKENS
        else ONE_MIB_SERVICE_NS
        for token in PULSE_TOKENS
    )


def frozen_profile() -> ActiveServiceProfile:
    quanta = tuple(
        ServiceQuantum(lane=lane, bytes=payload, service_ns=service)
        for payload, service in zip(group_bytes(), group_service_ns(), strict=True)
        for lane in range(4)
    )
    return ActiveServiceProfile(
        token_base_times_ns=pilot.TOKEN_BASE_TIMES_NS,
        active_lane_penalties_ns=pilot.ACTIVE_LANE_PENALTIES_NS,
        deadline_ns=pilot.DEADLINE_NS,
        start_lag_cap_ns=pilot.START_LAG_CAP_NS,
        max_issue_width=4,
        protect_prefix_tokens=4,
        protect_prefix_max_width=0,
        quanta=quanta,
    )


def replay_hybrid(profile: ActiveServiceProfile) -> ActiveServicePlan:
    widths = LOGICAL_WIDTH_BY_TOKEN
    ready = {lane: 0 for lane in range(4)}
    assignments: list[tuple[int, ...]] = []
    cursor = 0
    total_penalty = 0
    peak_penalty = 0
    completion = 0
    max_lag = 0
    for token, (base_ns, width) in enumerate(
        zip(profile.token_base_times_ns, widths, strict=True)
    ):
        indices = tuple(range(cursor, cursor + width))
        assignments.append(indices)
        issue_ns = base_ns + total_penalty
        for index in indices:
            quantum = profile.quanta[index]
            start_ns = max(issue_ns, ready[quantum.lane])
            lag_ns = start_ns - issue_ns
            finish_ns = start_ns + quantum.service_ns
            if lag_ns > profile.start_lag_cap_ns or finish_ns > profile.deadline_ns:
                raise ValueError(f"hybrid replay infeasible at token {token}")
            ready[quantum.lane] = finish_ns
            max_lag = max(max_lag, lag_ns)
            completion = max(completion, finish_ns)
        cursor += width
        active_lanes = sum(value > issue_ns for value in ready.values())
        penalty = profile.active_lane_penalties_ns[active_lanes]
        total_penalty += penalty
        peak_penalty = max(peak_penalty, penalty)
    if cursor != LOGICAL_QUANTA:
        raise ValueError("hybrid replay did not consume all logical quanta")
    unsigned = ActiveServicePlan(
        feasible=True,
        reason="compiled",
        width_by_token=widths,
        quantum_indices_by_token=tuple(assignments),
        predicted_completion_ns=completion,
        predicted_max_start_lag_ns=max_lag,
        total_predicted_penalty_ns=total_penalty,
        peak_predicted_penalty_ns=peak_penalty,
        signature="",
    )
    plan = replace(
        unsigned, signature=pilot._active_plan_signature(profile, unsigned)
    )
    validate_active_service_plan(profile, plan)
    observed = (completion, max_lag, total_penalty, peak_penalty)
    expected = (
        EXPECTED_COMPLETION_NS,
        EXPECTED_MAX_LAG_NS,
        EXPECTED_TOTAL_PENALTY_NS,
        EXPECTED_PEAK_PENALTY_NS,
    )
    if observed != expected:
        raise ValueError(f"hybrid replay changed: {observed} != {expected}")
    return plan


def provenance() -> dict[str, Any]:
    return {
        "selection": {
            "status": "same_allocation_adaptive_pilot",
            "source_job_id": 56929977,
            "source_result": "results/lmcache_active_pulse_job_56929977",
            "observed_lag_violations": [
                {"scheduled_token": 8},
                {"scheduled_token": 13},
                {"scheduled_token": 21},
            ],
            "action": "merge_only_each_violation_with_its_preceding_pulse",
            "independent_validation": False,
            "promotion_evidence": False,
        },
        "service_measurements": {
            "job_id": 56929977,
            "one_mib_source": "results/lmcache_active_pulse_job_56929977",
            "two_mib_source": "results/lmcache_active_pulse_group2_job_56929977",
            "one_mib_ms": {
                "p50": 4.367285,
                "p99": 5.253812,
                "max": 5.365608,
            },
            "two_mib_ms": {
                "p50": 7.618090,
                "p99": 8.892306,
                "max": 8.999303,
            },
            "model_statistic": "p99",
            "measured": True,
        },
        "model": {
            "absolute_deadline_ns": pilot.DEADLINE_NS,
            "start_lag_cap_ns": pilot.START_LAG_CAP_NS,
            "active_lane_penalties_ns": list(pilot.ACTIVE_LANE_PENALTIES_NS),
            "active_penalty_caveat": "lanes_1_to_3_unmeasured;_actions_use_0_or_4_only",
            "pulse_tokens": list(PULSE_TOKENS),
            "grouped_pulse_tokens": list(GROUPED_PULSE_TOKENS),
            "runtime_width_by_token": list(RUNTIME_WIDTH_BY_TOKEN),
        },
        "claim_scope": {
            "derivation": DERIVATION,
            "service_execution_valid": False,
            "lag_model_validated": False,
            "performance_screen_valid": False,
            "promotion_valid": False,
        },
    }


def make_artifact() -> dict[str, Any]:
    profile = frozen_profile()
    plan = replay_hybrid(profile)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "derivation": DERIVATION,
        "pulse_tokens": list(PULSE_TOKENS),
        "grouped_pulse_tokens": list(GROUPED_PULSE_TOKENS),
        "runtime_width_by_token": list(RUNTIME_WIDTH_BY_TOKEN),
        "provenance": provenance(),
        "active_service_artifact": make_active_service_artifact(profile, plan),
    }
    payload["artifact_signature_sha256"] = pilot._envelope_signature(payload)
    return payload


def load_artifact(
    payload: Mapping[str, Any],
) -> tuple[ActiveServiceProfile, ActiveServicePlan]:
    expected = {
        "schema_version", "derivation", "pulse_tokens", "grouped_pulse_tokens",
        "runtime_width_by_token", "provenance", "active_service_artifact",
        "artifact_signature_sha256",
    }
    if set(payload) != expected:
        raise ValueError("hybrid artifact fields are not exact")
    if payload["schema_version"] != SCHEMA or payload["derivation"] != DERIVATION:
        raise ValueError("hybrid artifact identity changed")
    if payload["pulse_tokens"] != list(PULSE_TOKENS):
        raise ValueError("hybrid pulse tokens changed")
    if payload["grouped_pulse_tokens"] != list(GROUPED_PULSE_TOKENS):
        raise ValueError("hybrid grouped tokens changed")
    if payload["runtime_width_by_token"] != list(RUNTIME_WIDTH_BY_TOKEN):
        raise ValueError("hybrid runtime widths changed")
    if payload["provenance"] != provenance():
        raise ValueError("hybrid provenance changed")
    if payload["artifact_signature_sha256"] != pilot._envelope_signature(payload):
        raise ValueError("hybrid envelope signature mismatch")
    active = payload["active_service_artifact"]
    if not isinstance(active, dict):
        raise ValueError("active_service_artifact must be an object")
    profile, plan = load_active_service_artifact(active)
    if profile != frozen_profile() or plan.width_by_token != LOGICAL_WIDTH_BY_TOKEN:
        raise ValueError("hybrid active-service payload changed")
    return profile, plan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[2]
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    if output == root or root not in output.parents:
        raise SystemExit("output must resolve below the repository root")
    artifact = make_artifact()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "completion_ns": EXPECTED_COMPLETION_NS,
                      "max_lag_ns": EXPECTED_MAX_LAG_NS}, sort_keys=True))


if __name__ == "__main__":
    main()
