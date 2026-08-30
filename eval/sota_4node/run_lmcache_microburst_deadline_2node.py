#!/usr/bin/env python3
"""Run microburst admission with an explicit service-completion deadline.

The generic epoch runner historically used the last issue token as the byte
completion deadline.  This wrapper keeps the signed issue calendar unchanged
but evaluates NIXL completion against ``profile.deadline_tokens`` instead.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from eval.sota_4node import run_lmcache_epoch_2node as base
from eval.sota_4node import run_lmcache_microburst_2node as microburst
from tempo.inference_epoch import EpochPlan, EpochProfile


_ORIGINAL_LOAD_PLAN = base._load_plan
_ORIGINAL_RUN_BLOCK = base._run_block
_SERVICE_DEADLINE_TOKEN_EXCLUSIVE: int | None = None


def _load_service_deadline_plan() -> tuple[EpochProfile, EpochPlan, dict[str, Any], str]:
    global _SERVICE_DEADLINE_TOKEN_EXCLUSIVE
    profile, plan, payload, path = _ORIGINAL_LOAD_PLAN()
    if profile.deadline_tokens <= int(plan.completion_token_exclusive or 0):
        raise SystemExit("service deadline must follow the final issue token")
    _SERVICE_DEADLINE_TOKEN_EXCLUSIVE = profile.deadline_tokens
    return profile, plan, payload, path


def _run_service_deadline_block(
    *args: Any,
    plan: EpochPlan,
    **kwargs: Any,
) -> dict[str, Any]:
    if _SERVICE_DEADLINE_TOKEN_EXCLUSIVE is None:
        raise RuntimeError("service deadline plan was not loaded")
    runtime_plan = replace(
        plan,
        completion_token_exclusive=_SERVICE_DEADLINE_TOKEN_EXCLUSIVE,
    )
    result = _ORIGINAL_RUN_BLOCK(*args, plan=runtime_plan, **kwargs)
    result["plan_last_issue_token_exclusive"] = plan.completion_token_exclusive
    result["service_deadline_token_exclusive"] = _SERVICE_DEADLINE_TOKEN_EXCLUSIVE
    if result["mode"] == "tempo_epoch":
        result["execution"] = "capacity_matched_microburst_service_deadline"
    return result


def main() -> None:
    microburst.install_microburst_geometry()
    base._load_plan = _load_service_deadline_plan
    base._run_block = _run_service_deadline_block
    base.main()


if __name__ == "__main__":
    main()
