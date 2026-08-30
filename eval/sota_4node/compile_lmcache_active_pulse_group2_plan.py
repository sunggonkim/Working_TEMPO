#!/usr/bin/env python3
"""Build the signed group-two retry for the 8 MiB active-pulse screen.

This artifact is an exact replay of a fixed ``{0, 4}`` *logical* calendar,
not an unconstrained compiler result.  Each logical quantum represents one
2 MiB source call: two adjacent 512 KiB chunks for each of two requests.  The
runtime adapter expands one logical width-four pulse to canonical width eight.
"""

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


EXPERIMENT_ARTIFACT_SCHEMA = "tempo-lmcache-active-pulse-group2-plan-1"
DERIVATION = "fixed_group2_width_0_or_4_exact_active_service_replay"
ALLOWED_LOGICAL_WIDTHS = (0, 4)
ALLOWED_RUNTIME_WIDTHS = (0, 8)
LOGICAL_QUANTA = 32
LOGICAL_QUANTUM_BYTES = 2_097_152
SERVICE_NS = 8_733_599
EXPECTED_PULSE_TOKENS = (4, 7, 10, 13, 17, 20, 23, 26)
EXPECTED_LOGICAL_WIDTH_BY_TOKEN = tuple(
    4 if token in EXPECTED_PULSE_TOKENS else 0 for token in range(64)
)
EXPECTED_RUNTIME_WIDTH_BY_TOKEN = tuple(
    8 if token in EXPECTED_PULSE_TOKENS else 0 for token in range(64)
)
EXPECTED_COMPLETION_NS = 85_222_385
EXPECTED_MAX_START_LAG_NS = 0
EXPECTED_TOTAL_PENALTY_NS = 19_582_560
EXPECTED_PEAK_PENALTY_NS = 815_940
PILOT_RESULT_DIR = "results/lmcache_active_pulse_job_56929977"
PILOT_ONE_MIB_RECORDS = 256
PILOT_ONE_MIB_MEDIAN_SERVICE_MS = 4.3667995


def frozen_profile() -> ActiveServiceProfile:
    return ActiveServiceProfile(
        token_base_times_ns=pilot.TOKEN_BASE_TIMES_NS,
        active_lane_penalties_ns=pilot.ACTIVE_LANE_PENALTIES_NS,
        deadline_ns=pilot.DEADLINE_NS,
        start_lag_cap_ns=pilot.START_LAG_CAP_NS,
        max_issue_width=4,
        protect_prefix_tokens=pilot.PROTECT_PREFIX_TOKENS,
        protect_prefix_max_width=pilot.PROTECT_PREFIX_MAX_WIDTH,
        quanta=tuple(
            ServiceQuantum(
                lane=quantum % 4,
                bytes=LOGICAL_QUANTUM_BYTES,
                service_ns=SERVICE_NS,
            )
            for quantum in range(LOGICAL_QUANTA)
        ),
    )


def replay_fixed_group2_calendar(
    profile: ActiveServiceProfile,
    widths: tuple[int, ...] = EXPECTED_LOGICAL_WIDTH_BY_TOKEN,
) -> ActiveServicePlan:
    if len(widths) != len(profile.token_base_times_ns):
        raise ValueError("group2 calendar token count differs from profile")
    if any(width not in ALLOWED_LOGICAL_WIDTHS for width in widths):
        raise ValueError("group2 logical calendar uses an action outside {0, 4}")
    if any(widths[token] for token in range(profile.protect_prefix_tokens)):
        raise ValueError("group2 calendar violates the protected prefix")
    if sum(widths) != len(profile.quanta):
        raise ValueError("group2 calendar does not assign all logical quanta")

    ready = {lane: 0 for lane in range(4)}
    assignments: list[tuple[int, ...]] = []
    cursor = 0
    total_penalty_ns = 0
    peak_penalty_ns = 0
    max_start_lag_ns = 0
    completion_ns = 0
    for token, (base_time_ns, width) in enumerate(
        zip(profile.token_base_times_ns, widths, strict=True)
    ):
        indices = tuple(range(cursor, cursor + width))
        assignments.append(indices)
        issue_ns = base_time_ns + total_penalty_ns
        for quantum_index in indices:
            quantum = profile.quanta[quantum_index]
            start_ns = max(issue_ns, ready[quantum.lane])
            lag_ns = start_ns - issue_ns
            finish_ns = start_ns + quantum.service_ns
            if lag_ns > profile.start_lag_cap_ns:
                raise ValueError(f"token {token} exceeds the start-lag cap")
            if finish_ns > profile.deadline_ns:
                raise ValueError(f"token {token} completes after the deadline")
            ready[quantum.lane] = finish_ns
            max_start_lag_ns = max(max_start_lag_ns, lag_ns)
            completion_ns = max(completion_ns, finish_ns)
        cursor += width
        active_lanes = sum(value > issue_ns for value in ready.values())
        penalty_ns = profile.active_lane_penalties_ns[active_lanes]
        total_penalty_ns += penalty_ns
        peak_penalty_ns = max(peak_penalty_ns, penalty_ns)

    unsigned = ActiveServicePlan(
        feasible=True,
        reason="compiled",
        width_by_token=widths,
        quantum_indices_by_token=tuple(assignments),
        predicted_completion_ns=completion_ns,
        predicted_max_start_lag_ns=max_start_lag_ns,
        total_predicted_penalty_ns=total_penalty_ns,
        peak_predicted_penalty_ns=peak_penalty_ns,
        signature="",
    )
    plan = replace(
        unsigned, signature=pilot._active_plan_signature(profile, unsigned)
    )
    validate_active_service_plan(profile, plan)
    observed = (
        plan.predicted_completion_ns,
        plan.predicted_max_start_lag_ns,
        plan.total_predicted_penalty_ns,
        plan.peak_predicted_penalty_ns,
    )
    expected = (
        EXPECTED_COMPLETION_NS,
        EXPECTED_MAX_START_LAG_NS,
        EXPECTED_TOTAL_PENALTY_NS,
        EXPECTED_PEAK_PENALTY_NS,
    )
    if observed != expected:
        raise ValueError(f"group2 fixed replay changed: {observed} != {expected}")
    return plan


def calibration_provenance() -> dict[str, Any]:
    return {
        "retry_scope": {
            "name": "group_two_coalesced_active_pulse_retry",
            "claim": "fixed_calendar_replay_not_unconstrained_compiler_search",
            "logical_action_widths": list(ALLOWED_LOGICAL_WIDTHS),
            "runtime_action_widths": list(ALLOWED_RUNTIME_WIDTHS),
            "pulse_tokens": list(EXPECTED_PULSE_TOKENS),
        },
        "workload": {
            "requests": 2,
            "kv_kib": 8192,
            "chunk_kib": 512,
            "tokens": 64,
            "layers": 8,
            "logical_quanta": LOGICAL_QUANTA,
            "logical_quantum_bytes": LOGICAL_QUANTUM_BYTES,
            "runtime_mapping": (
                "one_logical_lane_quantum_maps_two_adjacent_canonical_chunks;_"
                "logical_width4_expands_to_runtime_width8"
            ),
        },
        "frozen_token_clock_ns": list(pilot.TOKEN_BASE_TIMES_NS),
        "absolute_deadline_ns": pilot.DEADLINE_NS,
        "start_lag_cap_ns": pilot.START_LAG_CAP_NS,
        "active_lane_penalties_ns": list(pilot.ACTIVE_LANE_PENALTIES_NS),
        "active_penalty_caveat": (
            "lanes_1_to_3_are_unmeasured_saturation_assumptions;_only_actions_"
            "0_and_4_are_used"
        ),
        "service_estimate": {
            "status": "pilot_derived_linear_estimate_not_2mib_measurement",
            "source_result_directory": PILOT_RESULT_DIR,
            "source_mode": "tempo_epoch",
            "source_ranks": [0, 1, 2, 3],
            "source_one_mib_transfer_records": PILOT_ONE_MIB_RECORDS,
            "source_one_mib_median_service_ms": PILOT_ONE_MIB_MEDIAN_SERVICE_MS,
            "formula": "round(2_times_4.3667995ms_times_1e6)",
            "estimated_two_mib_service_ns": SERVICE_NS,
            "measured_at_two_mib": False,
        },
        "initial_validation_state": {
            "service_execution_valid": False,
            "lag_model_validated": False,
            "promotion_valid": False,
            "reason": "requires_live_group2_execution",
        },
    }


def make_group2_experiment_artifact() -> dict[str, Any]:
    profile = frozen_profile()
    plan = replay_fixed_group2_calendar(profile)
    payload: dict[str, Any] = {
        "schema_version": EXPERIMENT_ARTIFACT_SCHEMA,
        "derivation": DERIVATION,
        "allowed_logical_widths": list(ALLOWED_LOGICAL_WIDTHS),
        "allowed_runtime_widths": list(ALLOWED_RUNTIME_WIDTHS),
        "expected_width4_pulse_tokens": list(EXPECTED_PULSE_TOKENS),
        "calibration_provenance": calibration_provenance(),
        "active_service_artifact": make_active_service_artifact(profile, plan),
    }
    payload["artifact_signature_sha256"] = pilot._envelope_signature(payload)
    return payload


def load_group2_experiment_artifact(
    payload: Mapping[str, Any],
) -> tuple[ActiveServiceProfile, ActiveServicePlan]:
    expected_fields = {
        "schema_version",
        "derivation",
        "allowed_logical_widths",
        "allowed_runtime_widths",
        "expected_width4_pulse_tokens",
        "calibration_provenance",
        "active_service_artifact",
        "artifact_signature_sha256",
    }
    if set(payload) != expected_fields:
        raise ValueError("group2 artifact fields are not exact")
    if payload["schema_version"] != EXPERIMENT_ARTIFACT_SCHEMA:
        raise ValueError("unsupported group2 artifact schema")
    if payload["derivation"] != DERIVATION:
        raise ValueError("group2 derivation changed")
    if payload["allowed_logical_widths"] != list(ALLOWED_LOGICAL_WIDTHS):
        raise ValueError("group2 logical action set changed")
    if payload["allowed_runtime_widths"] != list(ALLOWED_RUNTIME_WIDTHS):
        raise ValueError("group2 runtime action set changed")
    if payload["expected_width4_pulse_tokens"] != list(EXPECTED_PULSE_TOKENS):
        raise ValueError("group2 pulse tokens changed")
    if payload["calibration_provenance"] != calibration_provenance():
        raise ValueError("group2 calibration provenance changed")
    if payload["artifact_signature_sha256"] != pilot._envelope_signature(payload):
        raise ValueError("group2 envelope signature mismatch")
    active_payload = payload["active_service_artifact"]
    if not isinstance(active_payload, dict):
        raise ValueError("active_service_artifact must be an object")
    profile, plan = load_active_service_artifact(active_payload)
    if profile != frozen_profile():
        raise ValueError("group2 profile differs from frozen retry inputs")
    if plan.width_by_token != EXPECTED_LOGICAL_WIDTH_BY_TOKEN:
        raise ValueError("group2 logical calendar changed")
    if any(width not in ALLOWED_LOGICAL_WIDTHS for width in plan.width_by_token):
        raise ValueError("group2 plan uses an unsupported logical action")
    return profile, plan


def _resolve_output(path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = path if path.is_absolute() else repo_root / path
    resolved = candidate.resolve()
    if resolved == repo_root or repo_root not in resolved.parents:
        raise ValueError("output must resolve below the repository root")
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        output = _resolve_output(args.output)
        artifact = make_group2_experiment_artifact()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _, plan = load_group2_experiment_artifact(artifact)
    print(
        json.dumps(
            {
                "output": str(output),
                "derivation": DERIVATION,
                "pulse_tokens": list(EXPECTED_PULSE_TOKENS),
                "predicted_completion_ns": plan.predicted_completion_ns,
                "predicted_max_start_lag_ns": plan.predicted_max_start_lag_ns,
                "service_estimate_ns": SERVICE_NS,
                "artifact_signature_sha256": artifact[
                    "artifact_signature_sha256"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
