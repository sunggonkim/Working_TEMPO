#!/usr/bin/env python3
"""Run the coalesced group-two active-pulse retry on LMCache/NIXL."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

from eval.sota_4node import compile_lmcache_active_pulse_group2_plan as compiled
from eval.sota_4node import run_lmcache_active_pulse_2node as pilot_runner
from eval.sota_4node import run_lmcache_epoch_2node as base
from tempo.inference_epoch import EpochPlan, EpochProfile, WidthPoint
from tempo.inference_service_active import ActiveServicePlan, ActiveServiceProfile


ACTIVE_PLAN_ENV = "TEMPO_ACTIVE_GROUP2_SERVICE_PLAN"
ISSUE_COMPLETION_TOKEN_EXCLUSIVE = max(compiled.EXPECTED_PULSE_TOKENS) + 1
_ORIGINAL_RUN_BLOCK = base._run_block
_ORIGINAL_AGGREGATE = base.aggregate_rank_records
_ACTIVE_PROFILE: ActiveServiceProfile | None = None
_ACTIVE_PLAN: ActiveServicePlan | None = None


def install_group2_geometry() -> None:
    # Reuse the exact 2-request, 8 MiB, 16 x 512 KiB geometry guard.
    pilot_runner.install_active_pulse_geometry()


def _expand_logical_assignments(
    assignments: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    expanded: list[tuple[int, ...]] = []
    for token, logical_indices in enumerate(assignments):
        if not logical_indices:
            expanded.append(())
            continue
        if len(logical_indices) != 4:
            raise ValueError(f"token {token} must issue exactly four logical lanes")
        groups = {index // 4 for index in logical_indices}
        lanes = {index % 4 for index in logical_indices}
        if len(groups) != 1 or lanes != {0, 1, 2, 3}:
            raise ValueError("logical pulse must contain one quantum per lane")
        group = next(iter(groups))
        canonical = tuple(range(group * 8, group * 8 + 8))
        expanded.append(canonical)
    flattened = tuple(index for token in expanded for index in token)
    if flattened != tuple(range(64)):
        raise ValueError("expanded group2 assignments are not canonical 0..63")
    return tuple(expanded)


def _adapt_group2_plan(
    profile: ActiveServiceProfile, plan: ActiveServicePlan
) -> tuple[EpochProfile, EpochPlan]:
    if profile != compiled.frozen_profile():
        raise ValueError("group2 profile is not the frozen retry profile")
    if plan.width_by_token != compiled.EXPECTED_LOGICAL_WIDTH_BY_TOKEN:
        raise ValueError("group2 logical pulse calendar changed")
    expanded = _expand_logical_assignments(plan.quantum_indices_by_token)
    widths = tuple(len(indices) for indices in expanded)
    if widths != compiled.EXPECTED_RUNTIME_WIDTH_BY_TOKEN:
        raise ValueError("group2 runtime width-eight calendar changed")

    runtime_profile = EpochProfile(
        total_quanta=64,
        deadline_tokens=64,
        token_slack_ns=(compiled.EXPECTED_PEAK_PENALTY_NS,) * 64,
        width_points=(
            WidthPoint(0, 0),
            WidthPoint(8, compiled.EXPECTED_PEAK_PENALTY_NS),
        ),
        max_width=8,
        protect_prefix_tokens=4,
        protect_prefix_max_width=0,
    )
    runtime_plan = EpochPlan(
        feasible=True,
        reason="active_service_group2_fixed_replay_adapter",
        width_by_token=widths,
        quantum_indices_by_token=expanded,
        completion_token_exclusive=ISSUE_COMPLETION_TOKEN_EXCLUSIVE,
        total_predicted_penalty_ns=plan.total_predicted_penalty_ns,
        peak_predicted_penalty_ns=plan.peak_predicted_penalty_ns,
        signature=plan.signature,
    )
    return runtime_profile, runtime_plan


def _resolve_plan_path() -> Path:
    raw = os.environ.get(ACTIVE_PLAN_ENV)
    if not raw:
        raise SystemExit(f"{ACTIVE_PLAN_ENV} must name a signed artifact")
    repo_root = Path(__file__).resolve().parents[2]
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve()
    if resolved == repo_root or repo_root not in resolved.parents:
        raise SystemExit(f"{ACTIVE_PLAN_ENV} must resolve inside the repository")
    return resolved


def _load_group2_plan() -> tuple[EpochProfile, EpochPlan, dict[str, Any], str]:
    global _ACTIVE_PROFILE, _ACTIVE_PLAN
    resolved = _resolve_plan_path()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("artifact must contain an object")
        active_profile, active_plan = compiled.load_group2_experiment_artifact(
            payload
        )
        runtime_profile, runtime_plan = _adapt_group2_plan(
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


def _run_group2_block(
    *args: Any,
    plan: EpochPlan,
    **kwargs: Any,
) -> dict[str, Any]:
    if _ACTIVE_PROFILE is None or _ACTIVE_PLAN is None:
        raise RuntimeError("group2 artifact was not loaded")
    if plan.width_by_token != compiled.EXPECTED_RUNTIME_WIDTH_BY_TOKEN:
        raise RuntimeError("runtime group2 pulse calendar changed")
    tokens = kwargs.get("tokens")
    if tokens != 64:
        raise RuntimeError("group2 runtime requires exactly 64 tokens")

    legacy_view = replace(plan, completion_token_exclusive=tokens)
    result = _ORIGINAL_RUN_BLOCK(*args, plan=legacy_view, **kwargs)
    if result["mode"] != "tempo_epoch":
        return result

    finish_from_start_ns = round(
        float(result["background_finish_from_block_start_ms"]) * 1_000_000
    )
    records = result["transfer_records"]
    if records:
        last_finished_ns = max(int(record["finished_ns"]) for record in records)
        block_start_ns = last_finished_ns - finish_from_start_ns
        deadline_ns = block_start_ns + compiled.pilot.DEADLINE_NS
        for record in records:
            record["finished_by_plan_deadline"] = (
                int(record["finished_ns"]) <= deadline_ns
            )
            record["deadline_semantics"] = "absolute_from_block_start"
            record["logical_group_chunks"] = 2

    deadline_met = (
        finish_from_start_ns <= compiled.pilot.DEADLINE_NS
        and all(bool(record["finished_by_plan_deadline"]) for record in records)
    )
    lag_model_validated = (
        round(float(result["max_descriptor_start_lag_ms"]) * 1_000_000)
        <= compiled.pilot.START_LAG_CAP_NS
    )
    no_drain = float(result["post_foreground_drain_ms"]) == 0.0
    service_execution_valid = (
        bool(result["correctness_met"])
        and bool(result["schedule_start_adherence_met"])
        and deadline_met
        and no_drain
    )
    promotion_valid = service_execution_valid and lag_model_validated
    result.update(
        {
            "execution": "fixed_group2_width8_pulse_absolute_deadline",
            "active_service_plan_signature": _ACTIVE_PLAN.signature,
            "active_service_derivation": compiled.DERIVATION,
            "logical_action_widths": list(compiled.ALLOWED_LOGICAL_WIDTHS),
            "runtime_action_widths": list(compiled.ALLOWED_RUNTIME_WIDTHS),
            "width4_logical_pulse_tokens": list(compiled.EXPECTED_PULSE_TOKENS),
            "runtime_width8_pulse_tokens": list(compiled.EXPECTED_PULSE_TOKENS),
            "plan_last_issue_token_exclusive": ISSUE_COMPLETION_TOKEN_EXCLUSIVE,
            "logical_quantum_bytes": compiled.LOGICAL_QUANTUM_BYTES,
            "estimated_service_ns": compiled.SERVICE_NS,
            "service_estimate_status": (
                "pilot_derived_linear_estimate_not_2mib_measurement"
            ),
            "absolute_service_deadline_origin": "block_start_perf_counter_ns",
            "absolute_service_deadline_ns": compiled.pilot.DEADLINE_NS,
            "absolute_background_finish_from_block_start_ns": (
                finish_from_start_ns
            ),
            "absolute_service_deadline_met": deadline_met,
            "candidate_relative_token_deadline_used": False,
            "actual_start_lag_cap_ns": compiled.pilot.START_LAG_CAP_NS,
            "lag_model_validated": lag_model_validated,
            "no_post_foreground_drain_met": no_drain,
            "service_execution_valid": service_execution_valid,
            "promotion_valid": promotion_valid,
            "plan_deadline_met": deadline_met,
        }
    )
    return result


def _aggregate_group2_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = _ORIGINAL_AGGREGATE(records)
    ordered = sorted(records, key=lambda item: item["rank"])
    source_candidate_blocks = [
        rank_record["blocks"][block_index]
        for rank_record in ordered[: base.RANKS_PER_NODE]
        for block_index, mode in enumerate(base.BLOCK_MODES)
        if mode == "tempo_epoch"
    ]
    global_candidate_blocks = [
        block for block in result["blocks"] if block["mode"] == "tempo_epoch"
    ]
    candidate_correct = all(
        bool(block["correctness_met"]) for block in global_candidate_blocks
    )
    adherence = all(
        bool(block["schedule_start_adherence_met"])
        for block in source_candidate_blocks
    )
    deadline = all(
        block.get("absolute_service_deadline_met") is True
        for block in source_candidate_blocks
    )
    no_drain = all(
        block.get("no_post_foreground_drain_met") is True
        for block in source_candidate_blocks
    )
    lag_model_validated = all(
        block.get("lag_model_validated") is True
        for block in source_candidate_blocks
    )
    service_execution_valid = (
        bool(result["overall_correctness_met"])
        and candidate_correct
        and adherence
        and deadline
        and no_drain
    )
    promotion_valid = service_execution_valid and lag_model_validated

    result["modes"]["tempo_epoch"].update(
        {
            "absolute_service_deadline_ns": compiled.pilot.DEADLINE_NS,
            "absolute_service_deadline_met": deadline,
            "actual_start_lag_cap_ns": compiled.pilot.START_LAG_CAP_NS,
            "lag_model_validated": lag_model_validated,
            "no_post_foreground_drain_met": no_drain,
            "service_execution_valid": service_execution_valid,
            "promotion_valid": promotion_valid,
            "candidate_relative_token_deadline_used": False,
        }
    )
    result["scheduler_semantics"].update(
        {
            "name": "TEMPO fixed coalesced group-two active-service calendar",
            "derivation": compiled.DERIVATION,
            "logical_action_widths": list(compiled.ALLOWED_LOGICAL_WIDTHS),
            "runtime_action_widths": list(compiled.ALLOWED_RUNTIME_WIDTHS),
            "service_estimate_status": (
                "pilot_derived_linear_estimate_not_2mib_measurement"
            ),
            "absolute_service_deadline_origin": "block_start_perf_counter_ns",
            "candidate_relative_token_deadline": False,
        }
    )
    result["service_execution_valid"] = service_execution_valid
    result["lag_model_validated"] = lag_model_validated
    result["promotion_valid"] = promotion_valid
    result["tempo_epoch_execution_valid"] = service_execution_valid
    if not result["overall_correctness_met"] or not candidate_correct:
        result["screen_outcome"] = "invalid_correctness"
    elif not adherence:
        result["screen_outcome"] = "kill_descriptor_calendar_service_mismatch"
    elif not deadline:
        result["screen_outcome"] = "kill_absolute_service_deadline_miss"
    elif not no_drain:
        result["screen_outcome"] = "kill_post_foreground_drain"
    elif not lag_model_validated:
        result["screen_outcome"] = (
            "valid_service_execution_but_lag_model_not_validated"
        )
    else:
        result["screen_outcome"] = (
            "valid_measurement_requires_performance_comparison"
        )
    return result


def main() -> None:
    install_group2_geometry()
    base._load_plan = _load_group2_plan
    base._run_block = _run_group2_block
    base.aggregate_rank_records = _aggregate_group2_records
    base.main()


if __name__ == "__main__":
    main()
