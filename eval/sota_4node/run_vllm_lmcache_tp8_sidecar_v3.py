#!/usr/bin/env python3
"""Add-only real-TP8 pair-staggered LMCache/NIXL pilot.

This thin adapter reuses the audited v1 data plane and measurements.  Its
fixed admission calendar selects one of four source/receiver pairs every two
decode tokens.  It is not a global single-flight scheduler: a slow physical
transfer may overlap later admissions on independent pair workers.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from eval.sota_4node import run_vllm_lmcache_tp8_sidecar as _v1


WORLD_SIZE = _v1.WORLD_SIZE
NODES = _v1.NODES
RANKS_PER_NODE = _v1.RANKS_PER_NODE
PAIR_COUNT = _v1.PAIR_COUNT
REQUESTS = _v1.REQUESTS
TOKENS = _v1.TOKENS
CHUNKS_PER_REQUEST = _v1.CHUNKS_PER_REQUEST
CHUNK_BYTES = _v1.CHUNK_BYTES
KV_BYTES_PER_RANK = _v1.KV_BYTES_PER_RANK

PAIR_MODE = "tempo_pair_stagger"
MODES = ("fg_only", "lmcache_greedy", PAIR_MODE)
LATIN_ROWS = tuple(
    tuple(MODES[(column + row) % len(MODES)] for column in range(len(MODES)))
    for row in range(len(MODES))
)
BLOCK_SPECS = tuple(
    (prompt_index, position, mode)
    for prompt_index, row in enumerate(LATIN_ROWS)
    for position, mode in enumerate(row)
)

SCHEDULE_NAME = "real_tp8_pair_stagger_v1"
POLICY_LABEL = "pair_staggered_admission_v1"
SCHEDULED_TOKENS = tuple(range(1, TOKENS, 2))
ABSOLUTE_DEADLINE_NS = 1_250_000_000
PAIR_STAGGER_SIGNATURE = (
    "b8d376650086e441e9dd0fbc034d52fc0fb5b9e7518da9cd0d0908e22babf520"
)
CONTRACT_PATH = Path("eval/sota_4node/real_tp8_pair_stagger_v1.json")


def _mapping_entry(scheduled_token: int) -> dict[str, Any]:
    slot = (scheduled_token - 1) // 2
    group = slot // PAIR_COUNT
    return {
        "scheduled_token": scheduled_token,
        "slot": slot,
        "active_pair": slot % PAIR_COUNT,
        "group": group,
        "chunks": [2 * group, 2 * group + 1],
    }


PAIR_STAGGER_SCHEDULE: dict[str, Any] = {
    "name": SCHEDULE_NAME,
    "scheduled_tokens": list(SCHEDULED_TOKENS),
    "deadline_ns": ABSOLUTE_DEADLINE_NS,
    "requests": REQUESTS,
    "chunks_per_request": CHUNKS_PER_REQUEST,
    "chunk_bytes": CHUNK_BYTES,
    "pair_count": PAIR_COUNT,
    "mapping": [_mapping_entry(token) for token in SCHEDULED_TOKENS],
}
EXPECTED_PROVENANCE = {
    "source": "same_allocation_real_tp8_observation",
    "adaptive_pilot": True,
    "promotion_valid": False,
    "deadline_predeclared": True,
    "policy_label": POLICY_LABEL,
    "global_single_flight": False,
    "physical_transfer_overlap_possible": True,
}


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def schedule_object_indices(
    mode: str,
    scheduled_token: int,
    *,
    pair_index: int,
) -> tuple[int, ...]:
    """Return objects admitted for pair ``pair_index`` at decode token ``t``."""

    if mode != PAIR_MODE:
        return _original_schedule_object_indices(
            mode, scheduled_token, pair_index=pair_index
        )
    if isinstance(scheduled_token, bool) or not isinstance(scheduled_token, int):
        raise ValueError("scheduled_token must be an int")
    if not 0 <= scheduled_token < TOKENS:
        raise ValueError(f"scheduled_token must be in 0..{TOKENS - 1}")
    if isinstance(pair_index, bool) or not isinstance(pair_index, int):
        raise ValueError("pair_index must be an int")
    if not 0 <= pair_index < PAIR_COUNT:
        raise ValueError("pair_index must be in 0..3")
    if scheduled_token % 2 != 1:
        return ()
    slot = (scheduled_token - 1) // 2
    if pair_index != slot % PAIR_COUNT:
        return ()
    group = slot // PAIR_COUNT
    chunks = (2 * group, 2 * group + 1)
    return tuple(
        request * CHUNKS_PER_REQUEST + chunk
        for chunk in chunks
        for request in range(REQUESTS)
    )


def validate_pair_stagger_schedule() -> None:
    if _canonical_sha256(PAIR_STAGGER_SCHEDULE) != PAIR_STAGGER_SIGNATURE:
        raise RuntimeError("pair-stagger code signature changed")
    total_calls = 0
    for token in SCHEDULED_TOKENS:
        active = [
            pair
            for pair in range(PAIR_COUNT)
            if schedule_object_indices(PAIR_MODE, token, pair_index=pair)
        ]
        if active != [((token - 1) // 2) % PAIR_COUNT]:
            raise RuntimeError(f"pair-stagger admission mapping changed at token {token}")
        total_calls += len(active)
    if total_calls != 32:
        raise RuntimeError("pair-stagger call count changed")
    for pair in range(PAIR_COUNT):
        pair_tokens = [
            token
            for token in SCHEDULED_TOKENS
            if schedule_object_indices(PAIR_MODE, token, pair_index=pair)
        ]
        if len(pair_tokens) != 8 or any(
            right - left != 8 for left, right in zip(pair_tokens, pair_tokens[1:])
        ):
            raise RuntimeError(f"pair {pair} token spacing changed")
        flattened = [
            index
            for token in SCHEDULED_TOKENS
            for index in schedule_object_indices(PAIR_MODE, token, pair_index=pair)
        ]
        if sorted(flattened) != list(range(REQUESTS * CHUNKS_PER_REQUEST)):
            raise RuntimeError(f"pair {pair} exact-once object coverage changed")


def load_pair_stagger_contract(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid pair-stagger contract: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("pair-stagger contract must contain an object")
    if payload.get("schema_version") != "tempo-real-tp8-pair-stagger-contract-1":
        raise ValueError("pair-stagger contract schema changed")
    schedule = payload.get("schedule")
    if schedule != PAIR_STAGGER_SCHEDULE:
        raise ValueError("pair-stagger schedule changed")
    signature = _canonical_sha256(schedule)
    if signature != PAIR_STAGGER_SIGNATURE:
        raise ValueError("pair-stagger schedule signature changed")
    if payload.get("code_signature_sha256") != signature:
        raise ValueError("pair-stagger signed envelope changed")
    if payload.get("provenance") != EXPECTED_PROVENANCE:
        raise ValueError("pair-stagger pilot provenance changed")
    return payload, signature


_original_parse_args = _v1._parse_args
_original_schedule_object_indices = _v1.schedule_object_indices
_original_run_block = _v1._run_block
_original_aggregate_rank_records = _v1.aggregate_rank_records
_pair_runtime = False
_suppress_initial_pair_lookup = False


def _parse_args() -> Any:
    args = _original_parse_args()
    old_default = Path(
        "results/lmcache_active_pulse_group2_job_56929977/"
        "active_pulse_group2_plan.json"
    )
    if args.plan == old_default:
        args.plan = CONTRACT_PATH
    if args.model == "models/TinyLlama-1.1B-Chat-v1.0":
        repo_root = Path(_v1.__file__).resolve().parents[2]
        args.model = str((repo_root / args.model).resolve())
    return args


def _runtime_schedule_object_indices(
    mode: str,
    token_index: int,
    *,
    pair_index: int,
) -> tuple[int, ...]:
    global _suppress_initial_pair_lookup
    if _pair_runtime and mode == "tempo_group2":
        # v1 probes token zero once at request start.  Suppress that probe;
        # event e after output-token e arrives admits scheduled token t=e+1.
        if _suppress_initial_pair_lookup:
            _suppress_initial_pair_lookup = False
            return ()
        scheduled_token = token_index + 1
        if scheduled_token >= TOKENS:
            return ()
        return schedule_object_indices(
            PAIR_MODE, scheduled_token, pair_index=pair_index
        )
    return _original_schedule_object_indices(mode, token_index, pair_index=pair_index)


def _run_block(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Adapt v1's per-token hook to the fixed t-1 -> t admission calendar."""

    global _pair_runtime, _suppress_initial_pair_lookup
    requested_mode = kwargs.get("mode")
    if requested_mode != PAIR_MODE:
        return _original_run_block(*args, **kwargs)
    if _pair_runtime:
        raise RuntimeError("nested pair-stagger block execution is not supported")
    adapted = dict(kwargs)
    adapted["mode"] = "tempo_group2"  # v1's internal per-token hook selector
    _pair_runtime = True
    _suppress_initial_pair_lookup = True
    try:
        result = _original_run_block(*args, **adapted)
        if _suppress_initial_pair_lookup:
            raise RuntimeError("v1 request-start schedule probe was not observed")
    finally:
        _pair_runtime = False
        _suppress_initial_pair_lookup = False
    result["mode"] = PAIR_MODE
    for record in result.get("transfer_records", []):
        event_token = int(record["scheduled_token"])
        record["triggered_after_token"] = event_token
        record["scheduled_token"] = event_token + 1
        record["trigger_semantics"] = "observed_t_minus_1_to_t_decode_boundary"
    result["sidecar_revision"] = 3
    result["admission_policy"] = POLICY_LABEL
    result["global_single_flight"] = False
    return result


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = _original_aggregate_rank_records(records)
    candidate = [block for block in result["blocks"] if block["mode"] == PAIR_MODE]
    expected_candidates = len(LATIN_ROWS)
    exact_bytes = len(candidate) == expected_candidates and all(
        bool(block["correctness_met"])
        and int(block["background_completed_bytes"])
        == int(block["expected_background_bytes"])
        and int(block["receiver_verified_bytes"])
        == int(block["expected_background_bytes"])
        for block in candidate
    )
    adherence = len(candidate) == expected_candidates and all(
        bool(block["schedule_start_adherence_met"]) for block in candidate
    )
    deadline = len(candidate) == expected_candidates and all(
        bool(block["absolute_service_deadline_met"]) for block in candidate
    )
    no_drain = len(candidate) == expected_candidates and all(
        float(block["post_foreground_drain_ms"]) == 0.0 for block in candidate
    )
    overall_correct = bool(result["overall_correctness_met"]) and exact_bytes
    if not overall_correct:
        outcome = "invalid_output_or_transfer_correctness"
    elif not adherence:
        outcome = "kill_external_token_trigger_adherence_miss"
    elif not deadline:
        outcome = "kill_predeclared_1p25s_service_deadline_miss"
    elif not no_drain:
        outcome = "kill_post_foreground_drain"
    else:
        outcome = "valid_same_allocation_adaptive_pilot_not_promotable"

    result.pop("frozen_group2", None)
    result.pop("candidate_start_lag_cap_met", None)
    result.update(
        {
            "schema_version": "tempo-vllm-tp8-lmcache-pair-stagger-screen-1",
            "evidence_state": "same_allocation_adaptive_real_tp8_component_pilot",
            "block_sequence": [mode for _, _, mode in BLOCK_SPECS],
            "latin_rows": [list(row) for row in LATIN_ROWS],
            "real_tp8_pair_stagger_v1": {
                "code_signature_sha256": PAIR_STAGGER_SIGNATURE,
                "policy_label": POLICY_LABEL,
                "scheduled_tokens": list(SCHEDULED_TOKENS),
                "per_pair_token_interval": 8,
                "objects_per_admission": 4,
                "bytes_per_source_admission": 4 * CHUNK_BYTES,
                "absolute_deadline_ns": ABSOLUTE_DEADLINE_NS,
                "deadline_origin": "request_start",
                "deadline_predeclared": True,
                "same_allocation_adaptive_pilot": True,
                "promotion_valid": False,
                "global_single_flight": False,
                "physical_transfer_overlap_possible": True,
                "admission_semantics": (
                    "one pair admitted per two-token slot; independent blocking "
                    "workers may overlap when service exceeds a slot"
                ),
                "mapping": PAIR_STAGGER_SCHEDULE["mapping"],
            },
            "candidate_exact_bytes_met": exact_bytes,
            "candidate_schedule_adherence_met": adherence,
            "candidate_absolute_deadline_met": deadline,
            "candidate_no_post_foreground_drain_met": no_drain,
            "overall_correctness_met": overall_correct,
            "same_allocation_adaptive_pilot": True,
            "promotion_valid": False,
            "screen_outcome": outcome,
        }
    )
    result["honesty_boundary"] = (
        str(result["honesty_boundary"])
        + "; the admission calendar does not impose a physical global "
        "single-flight or concurrency cap"
    )
    return result


def _install_corrections() -> None:
    _v1.MODES = MODES
    _v1.LATIN_ROWS = LATIN_ROWS
    _v1.BLOCK_SPECS = BLOCK_SPECS
    _v1.ABSOLUTE_DEADLINE_NS = ABSOLUTE_DEADLINE_NS
    _v1.EXPECTED_PLAN_SIGNATURE = PAIR_STAGGER_SIGNATURE
    _v1.validate_frozen_schedule = validate_pair_stagger_schedule
    _v1.load_frozen_plan = load_pair_stagger_contract
    _v1._parse_args = _parse_args
    _v1.schedule_object_indices = _runtime_schedule_object_indices
    _v1._run_block = _run_block
    _v1.aggregate_rank_records = aggregate_rank_records


def main() -> None:
    _install_corrections()
    _v1.main()
    # NixlChannel's receiver listener is non-daemon at the pinned revision.
    # A successful rank must terminate explicitly after the final Gloo barrier.
    try:
        sys.stdout.flush()
    except BaseException:
        pass
    try:
        sys.stderr.flush()
    except BaseException:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
