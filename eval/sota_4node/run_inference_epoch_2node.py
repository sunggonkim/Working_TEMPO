#!/usr/bin/env python3
"""Run a compact four-mode screen with a compiled local TEMPO epoch plan.

This module reuses the bounded CUDA/NCCL data plane from
``run_inference_interconnect_2node`` but replaces its eleven-mode, token-level
control experiment with four explicit modes and sixteen balanced blocks.  The
candidate plan is loaded and validated before distributed initialization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from eval.sota_4node import run_inference_interconnect_2node as base
from tempo.inference_epoch import EpochPlan, EpochProfile, load_epoch_artifact


EPOCH_MODES = (
    "fg_only",
    "greedy_coalesced",
    "static_serial",
    "tempo_epoch",
)
EPOCH_LATIN_ROWS = tuple(
    tuple(
        EPOCH_MODES[(column + row) % len(EPOCH_MODES)]
        for column in range(len(EPOCH_MODES))
    )
    for row in range(len(EPOCH_MODES))
)
EPOCH_BLOCK_MODES = tuple(mode for row in EPOCH_LATIN_ROWS for mode in row)
CANONICAL_QUANTA = base.AOT_PAIR_CHUNKS
_ORIGINAL_AGGREGATE = base.aggregate_rank_records


def schedule_entries_for_plan(
    plan: EpochPlan,
    mode: str,
    token_index: int,
    *,
    requests_per_block: int = 1,
) -> tuple[tuple[int, int, int], ...]:
    """Return rank-identical ``(request, pair, chunk)`` token entries."""

    if mode not in EPOCH_MODES:
        raise ValueError(f"unknown epoch mode: {mode}")
    if isinstance(token_index, bool) or not isinstance(token_index, int) or token_index < 0:
        raise ValueError("token_index must be a non-negative int")
    if (
        isinstance(requests_per_block, bool)
        or not isinstance(requests_per_block, int)
        or requests_per_block <= 0
    ):
        raise ValueError("requests_per_block must be a positive int")
    if not plan.feasible:
        raise ValueError("cannot execute an infeasible epoch plan")

    if mode == "fg_only":
        selected: tuple[int, ...] = ()
    elif mode == "greedy_coalesced":
        selected = tuple(range(len(CANONICAL_QUANTA))) if token_index == 0 else ()
    elif mode == "static_serial":
        selected = (token_index,) if token_index < len(CANONICAL_QUANTA) else ()
    else:
        selected = (
            plan.quantum_indices_by_token[token_index]
            if token_index < len(plan.quantum_indices_by_token)
            else ()
        )
    return tuple(
        (request, CANONICAL_QUANTA[index][0], CANONICAL_QUANTA[index][1])
        for index in selected
        for request in range(requests_per_block)
    )


def _summary_for(
    schedule: Callable[..., tuple[tuple[int, int, int], ...]],
    mode: str,
    *,
    requests_per_block: int,
) -> dict[str, int]:
    horizon = max(16, len(CANONICAL_QUANTA))
    entries = [
        schedule(mode, token, requests_per_block=requests_per_block)
        for token in range(horizon)
    ]
    return {
        "chunks": sum(len(item) for item in entries),
        "max_active_pairs": max(
            (len({entry[1] for entry in item}) for item in entries), default=0
        ),
        "max_chunks_at_token": max((len(item) for item in entries), default=0),
    }


def install_epoch_scheme(
    profile: EpochProfile,
    plan: EpochPlan,
    artifact: dict[str, Any],
    *,
    artifact_path: str,
) -> None:
    """Install the compact experiment contract into the reused data plane."""

    if not plan.feasible:
        raise ValueError("refusing to install an infeasible epoch plan")
    if profile.total_quanta != len(CANONICAL_QUANTA):
        raise ValueError("epoch total_quanta differs from pair/chunk geometry")
    if len(plan.width_by_token) < len(CANONICAL_QUANTA):
        raise ValueError("epoch plan must cover at least sixteen token slots")

    def schedule(
        mode: str,
        token_index: int,
        *,
        requests_per_block: int = 1,
    ) -> tuple[tuple[int, int, int], ...]:
        return schedule_entries_for_plan(
            plan,
            mode,
            token_index,
            requests_per_block=requests_per_block,
        )

    def summary(mode: str, *, requests_per_block: int = 1) -> dict[str, int]:
        return _summary_for(
            schedule,
            mode,
            requests_per_block=requests_per_block,
        )

    def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
        result = _ORIGINAL_AGGREGATE(records)
        result.update(
            {
                "schema_version": "tempo-inference-epoch-2node-1",
                "claim_scope": "scheduler_screen_not_sota_promotion",
                "data_plane": "torch.distributed pair NCCL broadcast",
                "epoch_plan_source": artifact_path,
                "epoch_artifact": artifact,
                "mode_semantics": {
                    "fg_only": "foreground decoder only",
                    "greedy_coalesced": (
                        "all prepacked background quanta issued at token zero"
                    ),
                    "static_serial": (
                        "one canonical pair/chunk quantum per token"
                    ),
                    "tempo_epoch": (
                        "offline-compiled immutable local calendar; no token-hot-path "
                        "WORLD control"
                    ),
                },
            }
        )
        return result

    base.MODE_ORDER = EPOCH_MODES
    base.LATIN_ROWS = EPOCH_LATIN_ROWS
    base.BLOCK_MODES = EPOCH_BLOCK_MODES
    base.COALESCED_AOT_MODES = frozenset(EPOCH_MODES[1:])
    base.AOT_PAIR_CONCURRENCY_BY_MODE = {
        "tempo_epoch": tuple(plan.width_by_token),
    }
    base.schedule_entries = schedule
    base.schedule_summary = summary
    base.aggregate_rank_records = aggregate


def _load_artifact_from_environment() -> tuple[EpochProfile, EpochPlan, dict[str, Any], str]:
    raw_path = os.environ.get("TEMPO_EPOCH_PLAN")
    if not raw_path:
        raise SystemExit("TEMPO_EPOCH_PLAN must name a compiled plan artifact")
    repo_root = Path(__file__).resolve().parents[2]
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve()
    if resolved != repo_root and repo_root not in resolved.parents:
        raise SystemExit("TEMPO_EPOCH_PLAN must resolve inside the repository")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot load TEMPO_EPOCH_PLAN: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("TEMPO_EPOCH_PLAN must contain a JSON object")
    try:
        profile, plan = load_epoch_artifact(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid TEMPO_EPOCH_PLAN: {exc}") from exc
    return profile, plan, payload, str(resolved.relative_to(repo_root))


def main() -> None:
    profile, plan, artifact, artifact_path = _load_artifact_from_environment()
    install_epoch_scheme(
        profile,
        plan,
        artifact,
        artifact_path=artifact_path,
    )
    base.main()


if __name__ == "__main__":
    main()
