#!/usr/bin/env python3
"""Materialize the frozen 8 MiB LMCache active-pulse replay artifact.

This is deliberately a fixed-calendar replay, not an unconstrained compiler
search.  The active-service compiler currently admits every width from zero
through four, whereas the calibration supports only actions ``{0, 4}``.
Accordingly, this adapter replays the preselected width-four pulse calendar
with :mod:`tempo.inference_service_active`, signs the exact active-service
profile and plan, and records the measurement/assumption boundary explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tempo.inference_service import ServiceQuantum
from tempo.inference_service_active import (
    ActiveServicePlan,
    ActiveServiceProfile,
    load_active_service_artifact,
    make_active_service_artifact,
    validate_active_service_plan,
)


ACTIVE_ARTIFACT_SCHEMA = "tempo-inference-active-service-plan-1"
EXPERIMENT_ARTIFACT_SCHEMA = "tempo-lmcache-active-pulse-plan-1"
DERIVATION = "fixed_width_0_or_4_calendar_exact_active_service_replay"
ALLOWED_ACTION_WIDTHS = (0, 4)
TOKEN_BASE_TIMES_NS = (
    0,
    2_341_995,
    4_620_967,
    6_901_733,
    9_189_764,
    11_490_419,
    13_803_378,
    16_082_611,
    18_391_312,
    20_663_542,
    22_930_352,
    25_203_514,
    27_465_675,
    29_724_670,
    31_981_079,
    34_255_784,
    36_519_718,
    38_805_293,
    41_094_887,
    43_463_573,
    45_733_018,
    47_993_326,
    50_279_283,
    52_535_282,
    54_815_718,
    57_081_466,
    59_354_046,
    61_636_106,
    63_905_259,
    66_175_185,
    68_429_751,
    70_693_566,
    73_069_365,
    75_331_205,
    77_600_739,
    79_861_627,
    82_116_163,
    84_396_669,
    86_661_585,
    88_991_967,
    91_257_744,
    93_526_437,
    95_791_523,
    98_124_220,
    100_387_993,
    102_647_237,
    104_972_590,
    107_234_041,
    109_485_759,
    111_866_781,
    114_134_041,
    116_390_722,
    118_636_491,
    120_964_228,
    123_232_099,
    125_481_655,
    127_731_852,
    130_001_487,
    132_261_854,
    134_540_767,
    136_909_184,
    139_177_396,
    141_444_606,
    143_715_223,
)
ACTIVE_LANE_PENALTIES_NS = (0, 815_940, 815_940, 815_940, 815_940)
DEADLINE_NS = 91_257_744
START_LAG_CAP_NS = 2_272_580
SERVICE_NS = 4_902_303
QUANTUM_BYTES = 1_048_576
TOTAL_QUANTA = 64
PROTECT_PREFIX_TOKENS = 4
PROTECT_PREFIX_MAX_WIDTH = 0
EXPECTED_PULSE_TOKENS = (
    4,
    5,
    7,
    8,
    10,
    12,
    13,
    15,
    17,
    18,
    20,
    21,
    23,
    25,
    26,
    28,
)
EXPECTED_WIDTH_BY_TOKEN = tuple(
    4 if token in EXPECTED_PULSE_TOKENS else 0 for token in range(64)
)
EXPECTED_COMPLETION_NS = 88_923_115
EXPECTED_MAX_START_LAG_NS = 2_255_116
EXPECTED_TOTAL_PENALTY_NS = 21_214_440
EXPECTED_PEAK_PENALTY_NS = 815_940

TOKEN_CLOCK_SOURCE_DIRS = (
    "results/lmcache_microburst_deadline_job_56928504",
    "results/lmcache_microburst_deadline_job_56928504_rep2",
    "results/lmcache_microburst_kv8_job_56928504",
    "results/lmcache_microburst_kv8_job_56928504_rep2",
)
PENALTY_SOURCE_DIRS = (
    "results/lmcache_microburst_kv8_job_56928504",
    "results/lmcache_microburst_kv8_job_56928504_rep2",
)


def frozen_profile() -> ActiveServiceProfile:
    """Return the exact profile used by the fixed active-pulse replay."""

    return ActiveServiceProfile(
        token_base_times_ns=TOKEN_BASE_TIMES_NS,
        active_lane_penalties_ns=ACTIVE_LANE_PENALTIES_NS,
        deadline_ns=DEADLINE_NS,
        start_lag_cap_ns=START_LAG_CAP_NS,
        max_issue_width=4,
        protect_prefix_tokens=PROTECT_PREFIX_TOKENS,
        protect_prefix_max_width=PROTECT_PREFIX_MAX_WIDTH,
        quanta=tuple(
            ServiceQuantum(
                lane=quantum % 4,
                bytes=QUANTUM_BYTES,
                service_ns=SERVICE_NS,
            )
            for quantum in range(TOTAL_QUANTA)
        ),
    )


def _active_plan_signature(
    profile: ActiveServiceProfile, plan: ActiveServicePlan
) -> str:
    """Mirror the v1 public artifact signature payload for a fixed replay."""

    plan_payload = plan.to_dict()
    plan_payload.pop("signature")
    encoded = json.dumps(
        {
            "schema_version": ACTIVE_ARTIFACT_SCHEMA,
            "profile": profile.to_dict(),
            "plan": plan_payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def replay_fixed_calendar(
    profile: ActiveServiceProfile,
    widths: tuple[int, ...] = EXPECTED_WIDTH_BY_TOKEN,
) -> ActiveServicePlan:
    """Exactly replay a calendar whose only calibrated actions are 0 and 4."""

    if len(widths) != len(profile.token_base_times_ns):
        raise ValueError("fixed calendar token count differs from the profile")
    if any(width not in ALLOWED_ACTION_WIDTHS for width in widths):
        raise ValueError("fixed calendar uses an action outside {0, 4}")
    if any(widths[token] for token in range(profile.protect_prefix_tokens)):
        raise ValueError("fixed calendar violates the protected prefix")
    if sum(widths) != len(profile.quanta):
        raise ValueError("fixed calendar does not assign every quantum")

    lanes = tuple(sorted({quantum.lane for quantum in profile.quanta}))
    lane_ready_ns = {lane: 0 for lane in lanes}
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
            start_ns = max(issue_ns, lane_ready_ns[quantum.lane])
            start_lag_ns = start_ns - issue_ns
            finish_ns = start_ns + quantum.service_ns
            if start_lag_ns > profile.start_lag_cap_ns:
                raise ValueError(f"token {token} exceeds the start-lag cap")
            if finish_ns > profile.deadline_ns:
                raise ValueError(f"token {token} completes after the deadline")
            lane_ready_ns[quantum.lane] = finish_ns
            max_start_lag_ns = max(max_start_lag_ns, start_lag_ns)
            completion_ns = max(completion_ns, finish_ns)
        cursor += width
        active_lanes = sum(ready > issue_ns for ready in lane_ready_ns.values())
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
    plan = replace(unsigned, signature=_active_plan_signature(profile, unsigned))
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
        raise ValueError(f"fixed replay changed: observed={observed}, expected={expected}")
    return plan


def calibration_provenance() -> dict[str, Any]:
    return {
        "frozen_8mib_workload": {
            "requests": 2,
            "kv_kib": 8192,
            "chunk_kib": 512,
            "tokens": 64,
            "layers": 8,
            "quanta": 64,
            "quantum_bytes": QUANTUM_BYTES,
            "lane_assignment": "quantum_index_mod_4",
        },
        "token_clock": {
            "source_directories": list(TOKEN_CLOCK_SOURCE_DIRS),
            "samples_per_token": 128,
            "statistic": "sorted_upper_median_index_64",
            "conversion": "round(duration_ms_times_1e6)_then_cumulative_sum",
            "deadline_boundary_token_exclusive": 40,
            "deadline_ns": DEADLINE_NS,
        },
        "service": {
            "service_ns_per_quantum": SERVICE_NS,
            "start_lag_cap_ns": START_LAG_CAP_NS,
        },
        "active_lane_penalty": {
            "source_directories": list(PENALTY_SOURCE_DIRS),
            "measured_width4_derivation": (
                "round((greedy_mean_3.2673994635416617ms_minus_"
                "fg_mean_2.4514598416666606ms)_times_1e6)"
            ),
            "entries": [
                {
                    "active_lanes": 0,
                    "penalty_ns": 0,
                    "status": "defined_no_active_service",
                },
                {
                    "active_lanes": 1,
                    "penalty_ns": 815_940,
                    "status": "unmeasured_saturation_assumption",
                },
                {
                    "active_lanes": 2,
                    "penalty_ns": 815_940,
                    "status": "unmeasured_saturation_assumption",
                },
                {
                    "active_lanes": 3,
                    "penalty_ns": 815_940,
                    "status": "unmeasured_saturation_assumption",
                },
                {
                    "active_lanes": 4,
                    "penalty_ns": 815_940,
                    "status": "measured_8mib_width4_greedy_minus_fg_mean",
                },
            ],
        },
        "search_scope": {
            "allowed_action_widths": list(ALLOWED_ACTION_WIDTHS),
            "claim": "fixed_calendar_replay_not_unconstrained_compiler_search",
        },
    }


def _envelope_signature(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("artifact_signature_sha256", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def make_active_pulse_experiment_artifact() -> dict[str, Any]:
    profile = frozen_profile()
    plan = replay_fixed_calendar(profile)
    active_artifact = make_active_service_artifact(profile, plan)
    payload: dict[str, Any] = {
        "schema_version": EXPERIMENT_ARTIFACT_SCHEMA,
        "derivation": DERIVATION,
        "allowed_action_widths": list(ALLOWED_ACTION_WIDTHS),
        "expected_width4_pulse_tokens": list(EXPECTED_PULSE_TOKENS),
        "calibration_provenance": calibration_provenance(),
        "active_service_artifact": active_artifact,
    }
    payload["artifact_signature_sha256"] = _envelope_signature(payload)
    return payload


def load_active_pulse_experiment_artifact(
    payload: Mapping[str, Any],
) -> tuple[ActiveServiceProfile, ActiveServicePlan]:
    expected_fields = {
        "schema_version",
        "derivation",
        "allowed_action_widths",
        "expected_width4_pulse_tokens",
        "calibration_provenance",
        "active_service_artifact",
        "artifact_signature_sha256",
    }
    if set(payload) != expected_fields:
        raise ValueError("active-pulse artifact fields are not exact")
    if payload["schema_version"] != EXPERIMENT_ARTIFACT_SCHEMA:
        raise ValueError("unsupported active-pulse artifact schema")
    if payload["derivation"] != DERIVATION:
        raise ValueError("active-pulse derivation changed")
    if payload["allowed_action_widths"] != list(ALLOWED_ACTION_WIDTHS):
        raise ValueError("active-pulse action set must be exactly {0, 4}")
    if payload["expected_width4_pulse_tokens"] != list(EXPECTED_PULSE_TOKENS):
        raise ValueError("active-pulse token list changed")
    if payload["calibration_provenance"] != calibration_provenance():
        raise ValueError("active-pulse calibration provenance changed")
    if payload["artifact_signature_sha256"] != _envelope_signature(payload):
        raise ValueError("active-pulse envelope signature mismatch")
    active_payload = payload["active_service_artifact"]
    if not isinstance(active_payload, dict):
        raise ValueError("active_service_artifact must be an object")
    profile, plan = load_active_service_artifact(active_payload)
    if profile != frozen_profile():
        raise ValueError("active-pulse profile differs from the frozen calibration")
    if plan.width_by_token != EXPECTED_WIDTH_BY_TOKEN:
        raise ValueError("active-pulse plan differs from the fixed calendar")
    if tuple(
        token for token, width in enumerate(plan.width_by_token) if width == 4
    ) != EXPECTED_PULSE_TOKENS:
        raise ValueError("active-pulse plan has the wrong pulse tokens")
    if any(width not in ALLOWED_ACTION_WIDTHS for width in plan.width_by_token):
        raise ValueError("active-pulse plan uses an unsupported action")
    return profile, plan


def _resolve_output(path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = path if path.is_absolute() else repo_root / path
    resolved = candidate.resolve()
    if resolved == repo_root or repo_root not in resolved.parents:
        raise ValueError("output must resolve to a file below the repository root")
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        output = _resolve_output(args.output)
        artifact = make_active_pulse_experiment_artifact()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _, plan = load_active_pulse_experiment_artifact(artifact)
    print(
        json.dumps(
            {
                "output": str(output),
                "derivation": DERIVATION,
                "allowed_action_widths": list(ALLOWED_ACTION_WIDTHS),
                "width4_pulse_tokens": list(EXPECTED_PULSE_TOKENS),
                "predicted_completion_ns": plan.predicted_completion_ns,
                "predicted_max_start_lag_ns": plan.predicted_max_start_lag_ns,
                "active_service_signature": plan.signature,
                "artifact_signature_sha256": artifact[
                    "artifact_signature_sha256"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
