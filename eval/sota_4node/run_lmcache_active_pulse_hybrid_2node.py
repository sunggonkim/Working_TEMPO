#!/usr/bin/env python3
"""Run the same-allocation adaptive hybrid active-pulse retry."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

from eval.sota_4node import compile_lmcache_active_pulse_hybrid_plan as compiled
from eval.sota_4node import run_lmcache_active_pulse_2node as pilot_runner
from eval.sota_4node import run_lmcache_epoch_2node as base
from tempo.inference_epoch import EpochPlan, EpochProfile, WidthPoint
from tempo.inference_service_active import ActiveServicePlan, ActiveServiceProfile


PLAN_ENV = "TEMPO_ACTIVE_HYBRID_SERVICE_PLAN"
ISSUE_COMPLETION = max(compiled.PULSE_TOKENS) + 1
_ORIGINAL_RUN_BLOCK = base._run_block
_ORIGINAL_AGGREGATE = base.aggregate_rank_records
_PROFILE: ActiveServiceProfile | None = None
_PLAN: ActiveServicePlan | None = None


def install_geometry() -> None:
    pilot_runner.install_active_pulse_geometry()


def _adapt(
    profile: ActiveServiceProfile, plan: ActiveServicePlan
) -> tuple[EpochProfile, EpochPlan]:
    if profile != compiled.frozen_profile() or plan.width_by_token != compiled.LOGICAL_WIDTH_BY_TOKEN:
        raise ValueError("hybrid logical plan differs from the signed replay")
    expanded: list[tuple[int, ...]] = []
    canonical_cursor = 0
    for token, logical in enumerate(plan.quantum_indices_by_token):
        if not logical:
            expanded.append(())
            continue
        if len(logical) != 4 or {index % 4 for index in logical} != {0, 1, 2, 3}:
            raise ValueError(f"token {token} is not one complete logical lane group")
        payloads = {profile.quanta[index].bytes for index in logical}
        if len(payloads) != 1:
            raise ValueError("one hybrid pulse mixes logical payload sizes")
        payload = next(iter(payloads))
        width = 4 if payload == compiled.ONE_MIB_BYTES else 8 if payload == compiled.TWO_MIB_BYTES else 0
        if not width:
            raise ValueError("hybrid pulse has an unsupported logical payload")
        indices = tuple(range(canonical_cursor, canonical_cursor + width))
        expanded.append(indices)
        canonical_cursor += width
    if canonical_cursor != 64:
        raise ValueError("hybrid runtime mapping did not consume canonical 0..63")
    widths = tuple(len(indices) for indices in expanded)
    if widths != compiled.RUNTIME_WIDTH_BY_TOKEN:
        raise ValueError("hybrid runtime width calendar changed")
    runtime_profile = EpochProfile(
        total_quanta=64,
        deadline_tokens=64,
        token_slack_ns=(compiled.EXPECTED_PEAK_PENALTY_NS,) * 64,
        width_points=(WidthPoint(0, 0), WidthPoint(4, compiled.EXPECTED_PEAK_PENALTY_NS),
                      WidthPoint(8, compiled.EXPECTED_PEAK_PENALTY_NS)),
        max_width=8,
        protect_prefix_tokens=4,
        protect_prefix_max_width=0,
    )
    runtime_plan = EpochPlan(
        feasible=True,
        reason="same_allocation_adaptive_hybrid_adapter",
        width_by_token=widths,
        quantum_indices_by_token=tuple(expanded),
        completion_token_exclusive=ISSUE_COMPLETION,
        total_predicted_penalty_ns=plan.total_predicted_penalty_ns,
        peak_predicted_penalty_ns=plan.peak_predicted_penalty_ns,
        signature=plan.signature,
    )
    return runtime_profile, runtime_plan


def _load() -> tuple[EpochProfile, EpochPlan, dict[str, Any], str]:
    global _PROFILE, _PLAN
    raw = os.environ.get(PLAN_ENV)
    if not raw:
        raise SystemExit(f"{PLAN_ENV} must name a signed artifact")
    root = Path(__file__).resolve().parents[2]
    candidate = Path(raw)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if resolved == root or root not in resolved.parents:
        raise SystemExit(f"{PLAN_ENV} must resolve inside the repository")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("artifact must be an object")
        profile, plan = compiled.load_artifact(payload)
        runtime_profile, runtime_plan = _adapt(profile, plan)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid {PLAN_ENV}: {exc}") from exc
    _PROFILE, _PLAN = profile, plan
    return runtime_profile, runtime_plan, payload, str(resolved.relative_to(root))


def _run_block(*args: Any, plan: EpochPlan, **kwargs: Any) -> dict[str, Any]:
    if _PROFILE is None or _PLAN is None:
        raise RuntimeError("hybrid artifact was not loaded")
    if plan.width_by_token != compiled.RUNTIME_WIDTH_BY_TOKEN or kwargs.get("tokens") != 64:
        raise RuntimeError("hybrid runtime calendar changed")
    result = _ORIGINAL_RUN_BLOCK(
        *args, plan=replace(plan, completion_token_exclusive=64), **kwargs
    )
    if result["mode"] != "tempo_epoch":
        return result
    finish_ns = round(float(result["background_finish_from_block_start_ms"]) * 1_000_000)
    records = result["transfer_records"]
    if records:
        last = max(int(record["finished_ns"]) for record in records)
        block_start = last - finish_ns
        deadline = block_start + compiled.pilot.DEADLINE_NS
        for record in records:
            record["finished_by_plan_deadline"] = int(record["finished_ns"]) <= deadline
            record["deadline_semantics"] = "absolute_from_block_start"
    deadline_met = finish_ns <= compiled.pilot.DEADLINE_NS and all(
        bool(record["finished_by_plan_deadline"]) for record in records
    )
    lag_valid = round(float(result["max_descriptor_start_lag_ms"]) * 1_000_000) <= compiled.pilot.START_LAG_CAP_NS
    no_drain = float(result["post_foreground_drain_ms"]) == 0.0
    service_valid = bool(result["correctness_met"]) and bool(
        result["schedule_start_adherence_met"]
    ) and deadline_met and no_drain
    performance_valid = service_valid and lag_valid
    result.update({
        "execution": "same_allocation_adaptive_hybrid_absolute_deadline",
        "active_service_plan_signature": _PLAN.signature,
        "active_service_derivation": compiled.DERIVATION,
        "runtime_width_by_token": list(compiled.RUNTIME_WIDTH_BY_TOKEN),
        "grouped_pulse_tokens": list(compiled.GROUPED_PULSE_TOKENS),
        "plan_last_issue_token_exclusive": ISSUE_COMPLETION,
        "absolute_service_deadline_ns": compiled.pilot.DEADLINE_NS,
        "absolute_background_finish_from_block_start_ns": finish_ns,
        "absolute_service_deadline_met": deadline_met,
        "candidate_relative_token_deadline_used": False,
        "actual_start_lag_cap_ns": compiled.pilot.START_LAG_CAP_NS,
        "lag_model_validated": lag_valid,
        "no_post_foreground_drain_met": no_drain,
        "service_execution_valid": service_valid,
        "performance_screen_valid": performance_valid,
        "independent_validation": False,
        "promotion_valid": False,
        "plan_deadline_met": deadline_met,
    })
    return result


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = _ORIGINAL_AGGREGATE(records)
    ordered = sorted(records, key=lambda item: item["rank"])
    source_blocks = [
        rank_record["blocks"][index]
        for rank_record in ordered[:base.RANKS_PER_NODE]
        for index, mode in enumerate(base.BLOCK_MODES)
        if mode == "tempo_epoch"
    ]
    global_blocks = [block for block in result["blocks"] if block["mode"] == "tempo_epoch"]
    correct = bool(result["overall_correctness_met"]) and all(
        bool(block["correctness_met"]) for block in global_blocks
    )
    adherence = all(bool(block["schedule_start_adherence_met"]) for block in source_blocks)
    deadline = all(block.get("absolute_service_deadline_met") is True for block in source_blocks)
    no_drain = all(block.get("no_post_foreground_drain_met") is True for block in source_blocks)
    lag_valid = all(block.get("lag_model_validated") is True for block in source_blocks)
    service_valid = correct and adherence and deadline and no_drain
    performance_valid = service_valid and lag_valid
    result["modes"]["tempo_epoch"].update({
        "service_execution_valid": service_valid,
        "lag_model_validated": lag_valid,
        "performance_screen_valid": performance_valid,
        "independent_validation": False,
        "promotion_valid": False,
        "absolute_service_deadline_ns": compiled.pilot.DEADLINE_NS,
        "actual_start_lag_cap_ns": compiled.pilot.START_LAG_CAP_NS,
    })
    result["scheduler_semantics"].update({
        "name": "TEMPO same-allocation adaptive hybrid pulse calendar",
        "derivation": compiled.DERIVATION,
        "same_allocation_adaptive_pilot": True,
        "independent_validation": False,
        "promotion_evidence": False,
    })
    result["service_execution_valid"] = service_valid
    result["lag_model_validated"] = lag_valid
    result["performance_screen_valid"] = performance_valid
    result["independent_validation"] = False
    result["promotion_valid"] = False
    result["tempo_epoch_execution_valid"] = service_valid
    if not correct:
        result["screen_outcome"] = "invalid_correctness"
    elif not adherence:
        result["screen_outcome"] = "kill_descriptor_calendar_service_mismatch"
    elif not deadline:
        result["screen_outcome"] = "kill_absolute_service_deadline_miss"
    elif not no_drain:
        result["screen_outcome"] = "kill_post_foreground_drain"
    elif not lag_valid:
        result["screen_outcome"] = "valid_service_execution_but_lag_model_not_validated"
    else:
        result["screen_outcome"] = "valid_same_allocation_adaptive_performance_screen_not_promotion"
    return result


def main() -> None:
    install_geometry()
    base._load_plan = _load
    base._run_block = _run_block
    base.aggregate_rank_records = _aggregate
    base.main()


if __name__ == "__main__":
    main()
