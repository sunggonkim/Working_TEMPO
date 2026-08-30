#!/usr/bin/env python3
"""Stagger four coalesced official LMCache/NIXL rank batches across tokens.

Greedy launches one fully coalesced batch on every source rank at token zero.
This candidate preserves the same four total ``batched_write`` calls and exact
bytes, but admits source pairs 0,1,2,3 at tokens 0,1,2,3.  The compiled plan's
quantum is therefore a real rank-local service call, not a descriptor chunk.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

from eval.sota_4node import run_lmcache_epoch_2node as base
from tempo.inference_epoch import EpochPlan, EpochProfile, load_epoch_artifact


PAIR_BATCH_COUNT = base.PAIR_COUNT
_ORIGINAL_OBJECT_INDICES = base.object_indices_for_rank
_ORIGINAL_RUN_BLOCK = base._run_block
_RUNTIME_DEADLINE_TOKEN_EXCLUSIVE: int | None = None


def _load_pair_batch_plan() -> tuple[EpochProfile, EpochPlan, dict[str, Any], str]:
    global _RUNTIME_DEADLINE_TOKEN_EXCLUSIVE
    raw_path = os.environ.get("TEMPO_EPOCH_PLAN")
    if not raw_path:
        raise SystemExit("TEMPO_EPOCH_PLAN must name a compiled artifact")
    repo_root = Path(__file__).resolve().parents[2]
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve()
    if resolved == repo_root or repo_root not in resolved.parents:
        raise SystemExit("TEMPO_EPOCH_PLAN must resolve below the repository root")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("artifact must contain an object")
        profile, plan = load_epoch_artifact(payload)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid TEMPO_EPOCH_PLAN: {exc}") from exc
    if not plan.feasible or profile.total_quanta != PAIR_BATCH_COUNT:
        raise SystemExit("rank-stagger plan must contain four feasible service calls")
    flattened = tuple(
        index for assignments in plan.quantum_indices_by_token for index in assignments
    )
    if flattened != tuple(range(PAIR_BATCH_COUNT)) or profile.max_width != 1:
        raise SystemExit("rank-stagger plan must admit exactly one ordered pair per token")
    _RUNTIME_DEADLINE_TOKEN_EXCLUSIVE = profile.deadline_tokens
    return profile, plan, payload, str(resolved.relative_to(repo_root))


def _pair_batch_object_indices(
    plan: EpochPlan,
    mode: str,
    token_index: int,
    *,
    pair_index: int,
    requests: int,
) -> tuple[int, ...]:
    if mode != "tempo_epoch":
        return _ORIGINAL_OBJECT_INDICES(
            plan,
            mode,
            token_index,
            pair_index=pair_index,
            requests=requests,
        )
    if token_index >= len(plan.quantum_indices_by_token):
        return ()
    admitted_pairs = plan.quantum_indices_by_token[token_index]
    if pair_index not in admitted_pairs:
        return ()
    return tuple(
        request * base.CHUNKS_PER_REQUEST + chunk
        for chunk in range(base.CHUNKS_PER_REQUEST)
        for request in range(requests)
    )


def _run_pair_batch_block(*args: Any, plan: EpochPlan, **kwargs: Any) -> dict[str, Any]:
    if _RUNTIME_DEADLINE_TOKEN_EXCLUSIVE is None:
        raise RuntimeError("rank-stagger plan was not loaded")
    runtime_plan = replace(
        plan,
        completion_token_exclusive=_RUNTIME_DEADLINE_TOKEN_EXCLUSIVE,
    )
    result = _ORIGINAL_RUN_BLOCK(*args, plan=runtime_plan, **kwargs)
    result["plan_last_issue_token_exclusive"] = plan.completion_token_exclusive
    result["service_deadline_token_exclusive"] = _RUNTIME_DEADLINE_TOKEN_EXCLUSIVE
    if result["mode"] == "tempo_epoch":
        result["execution"] = "coalesced_rank_batch_stagger_no_hot_path_global_control"
    return result


def install_rank_stagger_policy() -> None:
    base._load_plan = _load_pair_batch_plan
    base.object_indices_for_rank = _pair_batch_object_indices
    base._run_block = _run_pair_batch_block


def main() -> None:
    install_rank_stagger_policy()
    base.main()


if __name__ == "__main__":
    main()
