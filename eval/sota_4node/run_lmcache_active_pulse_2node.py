#!/usr/bin/env python3
"""Run the signed 8 MiB active-pulse calendar on official LMCache/NIXL.

The runtime adapts the active-service assignments to the canonical ``EpochPlan``
lookup used by the LMCache runner.  Service validity is evaluated against the
frozen absolute wall-clock boundary from block start (91.257744 ms), never a
token boundary whose wall-clock position can move with candidate interference.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

from eval.sota_4node import compile_lmcache_active_pulse_plan as compiled
from eval.sota_4node import run_lmcache_epoch_2node as base
from eval.sota_4node import run_lmcache_microburst_2node as microburst
from tempo.inference_epoch import EpochPlan, EpochProfile, WidthPoint
from tempo.inference_service_active import ActiveServicePlan, ActiveServiceProfile


ACTIVE_PLAN_ENV = "TEMPO_ACTIVE_SERVICE_PLAN"
ABSOLUTE_SERVICE_DEADLINE_MS = compiled.DEADLINE_NS / 1_000_000.0
ABSOLUTE_START_LAG_CAP_MS = compiled.START_LAG_CAP_NS / 1_000_000.0
ISSUE_COMPLETION_TOKEN_EXCLUSIVE = max(compiled.EXPECTED_PULSE_TOKENS) + 1

_ORIGINAL_RUN_BLOCK = base._run_block
_ORIGINAL_AGGREGATE = base.aggregate_rank_records
_ACTIVE_PROFILE: ActiveServiceProfile | None = None
_ACTIVE_PLAN: ActiveServicePlan | None = None


def _parse_args() -> argparse.Namespace:
    args = microburst._parse_args()
    exact = {
        "requests": 2,
        "kv_mib": 8192,
        "chunk_mib": 512,
        "tokens": 64,
        "layers": 8,
    }
    observed = {name: getattr(args, name) for name in exact}
    if observed != exact:
        raise SystemExit(
            "active-pulse workload must be exactly requests=2, kv-kib=8192, "
            "chunk-kib=512, tokens=64, layers=8"
        )
    return args


def install_active_pulse_geometry() -> None:
    """Install 16 x 512 KiB chunks/request and the exact workload parser."""

    microburst.install_microburst_geometry()
    base._parse_args = _parse_args


def _resolve_active_plan_path() -> Path:
    raw_path = os.environ.get(ACTIVE_PLAN_ENV)
    if not raw_path:
        raise SystemExit(f"{ACTIVE_PLAN_ENV} must name a signed artifact")
    repo_root = Path(__file__).resolve().parents[2]
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve()
    if resolved == repo_root or repo_root not in resolved.parents:
        raise SystemExit(f"{ACTIVE_PLAN_ENV} must resolve inside the repository")
    return resolved


def _adapt_active_plan(
    profile: ActiveServiceProfile, plan: ActiveServicePlan
) -> tuple[EpochProfile, EpochPlan]:
    if profile != compiled.frozen_profile():
        raise ValueError("active-service profile is not the frozen 8 MiB profile")
    if plan.width_by_token != compiled.EXPECTED_WIDTH_BY_TOKEN:
        raise ValueError("active-service widths are not the fixed pulse calendar")
    pulses = tuple(
        token for token, width in enumerate(plan.width_by_token) if width == 4
    )
    if pulses != compiled.EXPECTED_PULSE_TOKENS:
        raise ValueError("active-service pulse tokens changed")
    if any(
        width not in compiled.ALLOWED_ACTION_WIDTHS for width in plan.width_by_token
    ):
        raise ValueError("active-service plan uses an action outside {0, 4}")

    runtime_profile = EpochProfile(
        total_quanta=compiled.TOTAL_QUANTA,
        deadline_tokens=len(compiled.TOKEN_BASE_TIMES_NS),
        token_slack_ns=(compiled.EXPECTED_PEAK_PENALTY_NS,)
        * len(compiled.TOKEN_BASE_TIMES_NS),
        width_points=(
            WidthPoint(0, 0),
            WidthPoint(4, compiled.EXPECTED_PEAK_PENALTY_NS),
        ),
        max_width=4,
        protect_prefix_tokens=compiled.PROTECT_PREFIX_TOKENS,
        protect_prefix_max_width=compiled.PROTECT_PREFIX_MAX_WIDTH,
    )
    runtime_plan = EpochPlan(
        feasible=True,
        reason="active_service_fixed_replay_adapter",
        width_by_token=plan.width_by_token,
        quantum_indices_by_token=plan.quantum_indices_by_token,
        completion_token_exclusive=ISSUE_COMPLETION_TOKEN_EXCLUSIVE,
        total_predicted_penalty_ns=plan.total_predicted_penalty_ns,
        peak_predicted_penalty_ns=plan.peak_predicted_penalty_ns,
        signature=plan.signature,
    )
    return runtime_profile, runtime_plan


def _load_active_pulse_plan() -> tuple[EpochProfile, EpochPlan, dict[str, Any], str]:
    global _ACTIVE_PROFILE, _ACTIVE_PLAN
    resolved = _resolve_active_plan_path()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("artifact must contain an object")
        active_profile, active_plan = (
            compiled.load_active_pulse_experiment_artifact(payload)
        )
        runtime_profile, runtime_plan = _adapt_active_plan(
            active_profile, active_plan
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid {ACTIVE_PLAN_ENV}: {exc}") from exc
    _ACTIVE_PROFILE = active_profile
    _ACTIVE_PLAN = active_plan
    return (
        runtime_profile,
        runtime_plan,
        payload,
        str(resolved.relative_to(repo_root)),
    )


def _run_active_pulse_block(
    *args: Any,
    plan: EpochPlan,
    **kwargs: Any,
) -> dict[str, Any]:
    if _ACTIVE_PROFILE is None or _ACTIVE_PLAN is None:
        raise RuntimeError("active-pulse artifact was not loaded")
    if plan.width_by_token != compiled.EXPECTED_WIDTH_BY_TOKEN:
        raise RuntimeError("runtime active-pulse plan changed")
    tokens = kwargs.get("tokens")
    if tokens != len(compiled.TOKEN_BASE_TIMES_NS):
        raise RuntimeError("runtime token horizon differs from the frozen profile")

    # The base runner needs an in-range token only to annotate its legacy
    # candidate-relative deadline field.  We overwrite that field below with
    # the absolute block-start deadline used by this experiment.
    permissive_legacy_view = replace(plan, completion_token_exclusive=tokens)
    result = _ORIGINAL_RUN_BLOCK(
        *args, plan=permissive_legacy_view, **kwargs
    )
    if result["mode"] != "tempo_epoch":
        return result

    finish_from_start_ns = round(
        float(result["background_finish_from_block_start_ms"]) * 1_000_000
    )
    records = result["transfer_records"]
    if records:
        last_finished_ns = max(int(record["finished_ns"]) for record in records)
        block_start_ns = last_finished_ns - finish_from_start_ns
        absolute_deadline_ns = block_start_ns + compiled.DEADLINE_NS
        for record in records:
            record["finished_by_plan_deadline"] = (
                int(record["finished_ns"]) <= absolute_deadline_ns
            )
            record["deadline_semantics"] = "absolute_from_block_start"

    absolute_deadline_met = (
        finish_from_start_ns <= compiled.DEADLINE_NS
        and all(bool(record["finished_by_plan_deadline"]) for record in records)
    )
    start_lag_cap_met = (
        round(float(result["max_descriptor_start_lag_ms"]) * 1_000_000)
        <= compiled.START_LAG_CAP_NS
    )
    no_post_foreground_drain = (
        float(result["post_foreground_drain_ms"]) == 0.0
    )
    result.update(
        {
            "execution": "fixed_active_service_width4_pulse_absolute_deadline",
            "active_service_plan_signature": _ACTIVE_PLAN.signature,
            "active_service_derivation": compiled.DERIVATION,
            "allowed_action_widths": list(compiled.ALLOWED_ACTION_WIDTHS),
            "width4_pulse_tokens": list(compiled.EXPECTED_PULSE_TOKENS),
            "plan_last_issue_token_exclusive": ISSUE_COMPLETION_TOKEN_EXCLUSIVE,
            "absolute_service_deadline_origin": "block_start_perf_counter_ns",
            "absolute_service_deadline_ns": compiled.DEADLINE_NS,
            "absolute_service_deadline_from_block_start_ms": (
                ABSOLUTE_SERVICE_DEADLINE_MS
            ),
            "absolute_background_finish_from_block_start_ns": (
                finish_from_start_ns
            ),
            "absolute_service_deadline_met": absolute_deadline_met,
            "candidate_relative_token_deadline_used": False,
            "actual_start_lag_cap_ns": compiled.START_LAG_CAP_NS,
            "actual_start_lag_cap_met": start_lag_cap_met,
            "no_post_foreground_drain_met": no_post_foreground_drain,
            "plan_deadline_met": absolute_deadline_met,
        }
    )
    return result


def _aggregate_active_pulse_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    result = _ORIGINAL_AGGREGATE(records)
    candidate_blocks = [
        block for block in result["blocks"] if block["mode"] == "tempo_epoch"
    ]
    # Added rank-local execution fields are not copied by the base aggregator;
    # recover the gates from the source ranks for every candidate block.
    source_candidate_blocks = [
        rank_record["blocks"][block_index]
        for rank_record in sorted(records, key=lambda item: item["rank"])[
            : base.RANKS_PER_NODE
        ]
        for block_index, mode in enumerate(base.BLOCK_MODES)
        if mode == "tempo_epoch"
    ]
    absolute_deadline_met = all(
        block.get("absolute_service_deadline_met") is True
        for block in source_candidate_blocks
    )
    start_lag_cap_met = all(
        block.get("actual_start_lag_cap_met") is True
        for block in source_candidate_blocks
    )
    no_drain_met = all(
        block.get("no_post_foreground_drain_met") is True
        for block in source_candidate_blocks
    )
    schedule_adherence_met = all(
        bool(block["schedule_start_adherence_met"])
        for block in source_candidate_blocks
    )
    candidate_correct = all(
        bool(block["correctness_met"])
        for block in candidate_blocks
    )
    execution_valid = (
        bool(result["overall_correctness_met"])
        and candidate_correct
        and schedule_adherence_met
        and absolute_deadline_met
        and start_lag_cap_met
        and no_drain_met
    )

    result["modes"]["tempo_epoch"].update(
        {
            "absolute_service_deadline_ns": compiled.DEADLINE_NS,
            "absolute_service_deadline_from_block_start_ms": (
                ABSOLUTE_SERVICE_DEADLINE_MS
            ),
            "absolute_service_deadline_met": absolute_deadline_met,
            "actual_start_lag_cap_ns": compiled.START_LAG_CAP_NS,
            "actual_start_lag_cap_met": start_lag_cap_met,
            "no_post_foreground_drain_met": no_drain_met,
            "candidate_relative_token_deadline_used": False,
        }
    )
    result["scheduler_semantics"].update(
        {
            "name": "TEMPO fixed active-service width-four pulse calendar",
            "derivation": compiled.DERIVATION,
            "allowed_action_widths": list(compiled.ALLOWED_ACTION_WIDTHS),
            "absolute_service_deadline_origin": "block_start_perf_counter_ns",
            "candidate_relative_token_deadline": False,
        }
    )
    result["active_pulse_execution_valid"] = execution_valid
    result["tempo_epoch_execution_valid"] = execution_valid
    if not result["overall_correctness_met"] or not candidate_correct:
        result["screen_outcome"] = "invalid_correctness"
    elif not schedule_adherence_met:
        result["screen_outcome"] = "kill_descriptor_calendar_service_mismatch"
    elif not start_lag_cap_met:
        result["screen_outcome"] = "kill_absolute_start_lag_cap_miss"
    elif not absolute_deadline_met:
        result["screen_outcome"] = "kill_absolute_service_deadline_miss"
    elif not no_drain_met:
        result["screen_outcome"] = "kill_post_foreground_drain"
    else:
        result["screen_outcome"] = (
            "valid_measurement_requires_performance_comparison"
        )
    return result


def main() -> None:
    install_active_pulse_geometry()
    base._load_plan = _load_active_pulse_plan
    base._run_block = _run_active_pulse_block
    base.aggregate_rank_records = _aggregate_active_pulse_records
    base.main()


if __name__ == "__main__":
    main()
