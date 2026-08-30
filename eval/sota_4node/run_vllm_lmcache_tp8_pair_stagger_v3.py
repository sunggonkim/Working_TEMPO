#!/usr/bin/env python3
"""Real-TP8 pair-staggered LMCache/NIXL admission pilot.

This add-only entrypoint reuses the audited vLLM/LMCache sidecar data path but
replaces the synthetic group2 calendar with a real-TP8 pilot contract.  One
pair is admitted every two output-token boundaries.  Calls may still overlap
physically when a prior pair has not completed; this is staggered admission,
not a global single-flight scheduler.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

from eval.sota_4node import run_vllm_lmcache_tp8_sidecar as _v1


POLICY_LABEL = "pair_staggered_admission_v1"
CONTRACT_SCHEMA = "tempo-real-tp8-pair-stagger-contract-1"
CONTRACT_SIGNATURE = "b8d376650086e441e9dd0fbc034d52fc0fb5b9e7518da9cd0d0908e22babf520"
DEADLINE_NS = 1_250_000_000
SCHEDULED_TOKENS = tuple(range(1, _v1.TOKENS, 2))

_original_schedule = _v1.schedule_object_indices
_original_run_block = _v1._run_block
_original_aggregate = _v1.aggregate_rank_records
_runtime_shift = False
_runtime_token_zero_calls = 0


def pair_stagger_object_indices(
    mode: str,
    scheduled_token: int,
    *,
    pair_index: int,
) -> tuple[int, ...]:
    """Return exact rank-local objects for a scheduled output token."""

    if mode != "tempo_group2":
        return _original_schedule(mode, scheduled_token, pair_index=pair_index)
    if isinstance(scheduled_token, bool) or not isinstance(scheduled_token, int):
        raise ValueError("scheduled_token must be an int")
    if not 0 <= scheduled_token < _v1.TOKENS:
        raise ValueError("scheduled_token is outside the 64-token horizon")
    if isinstance(pair_index, bool) or not isinstance(pair_index, int):
        raise ValueError("pair_index must be an int")
    if not 0 <= pair_index < _v1.PAIR_COUNT:
        raise ValueError("pair_index must be in 0..3")
    if scheduled_token % 2 == 0:
        return ()
    slot = (scheduled_token - 1) // 2
    active_pair = slot % _v1.PAIR_COUNT
    if pair_index != active_pair:
        return ()
    group = slot // _v1.PAIR_COUNT
    chunks = (2 * group, 2 * group + 1)
    return tuple(
        request * _v1.CHUNKS_PER_REQUEST + chunk
        for chunk in chunks
        for request in range(_v1.REQUESTS)
    )


def _runtime_schedule(
    mode: str,
    token_index: int,
    *,
    pair_index: int,
) -> tuple[int, ...]:
    """Translate arrival of token i into admission for scheduled token i+1."""

    global _runtime_token_zero_calls
    if mode != "tempo_group2" or not _runtime_shift:
        return pair_stagger_object_indices(
            mode, token_index, pair_index=pair_index
        )
    # _v1 invokes enqueue(0) once at HTTP request start and once after output
    # token 0 arrives.  Only the latter is the t-1 -> t decode boundary.
    if token_index == 0 and _runtime_token_zero_calls == 0:
        _runtime_token_zero_calls += 1
        return ()
    scheduled_token = token_index + 1
    if scheduled_token >= _v1.TOKENS:
        return ()
    return pair_stagger_object_indices(
        mode, scheduled_token, pair_index=pair_index
    )


def validate_stagger_schedule() -> None:
    if len(SCHEDULED_TOKENS) != 32:
        raise RuntimeError("pair-stagger schedule must contain 32 slots")
    active_pairs: list[int] = []
    for token in SCHEDULED_TOKENS:
        active = [
            pair
            for pair in range(_v1.PAIR_COUNT)
            if pair_stagger_object_indices(
                "tempo_group2", token, pair_index=pair
            )
        ]
        if len(active) != 1:
            raise RuntimeError(f"token {token} must admit exactly one pair")
        active_pairs.extend(active)
    if active_pairs != [slot % _v1.PAIR_COUNT for slot in range(32)]:
        raise RuntimeError("pair-stagger round-robin order changed")
    for pair in range(_v1.PAIR_COUNT):
        flattened = tuple(
            index
            for token in range(_v1.TOKENS)
            for index in pair_stagger_object_indices(
                "tempo_group2", token, pair_index=pair
            )
        )
        if sorted(flattened) != list(
            range(_v1.REQUESTS * _v1.CHUNKS_PER_REQUEST)
        ):
            raise RuntimeError(f"object coverage changed for pair {pair}")


def load_stagger_contract(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid pair-stagger contract: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("pair-stagger contract must be an object")
    if payload.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("pair-stagger contract schema changed")
    if payload.get("code_signature_sha256") != CONTRACT_SIGNATURE:
        raise ValueError("pair-stagger contract signature changed")
    schedule = payload.get("schedule")
    if not isinstance(schedule, dict):
        raise ValueError("pair-stagger schedule is missing")
    expected_mapping = [
        {
            "scheduled_token": 1 + 2 * slot,
            "slot": slot,
            "active_pair": slot % _v1.PAIR_COUNT,
            "group": slot // _v1.PAIR_COUNT,
            "chunks": [2 * (slot // _v1.PAIR_COUNT), 2 * (slot // _v1.PAIR_COUNT) + 1],
        }
        for slot in range(32)
    ]
    expected_fields = {
        "name": "real_tp8_pair_stagger_v1",
        "scheduled_tokens": list(SCHEDULED_TOKENS),
        "deadline_ns": DEADLINE_NS,
        "requests": _v1.REQUESTS,
        "chunks_per_request": _v1.CHUNKS_PER_REQUEST,
        "chunk_bytes": _v1.CHUNK_BYTES,
        "pair_count": _v1.PAIR_COUNT,
        "mapping": expected_mapping,
    }
    if schedule != expected_fields:
        raise ValueError("pair-stagger schedule contents changed")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("pair-stagger provenance is missing")
    required_provenance = {
        "adaptive_pilot": True,
        "promotion_valid": False,
        "policy_label": POLICY_LABEL,
        "global_single_flight": False,
        "physical_transfer_overlap_possible": True,
    }
    if any(provenance.get(key) != value for key, value in required_provenance.items()):
        raise ValueError("pair-stagger provenance changed")
    validate_stagger_schedule()
    return payload, CONTRACT_SIGNATURE


def _run_block(*args: Any, **kwargs: Any) -> dict[str, Any]:
    global _runtime_shift, _runtime_token_zero_calls
    if _runtime_shift:
        raise RuntimeError("nested sidecar block execution is not supported")
    _runtime_shift = True
    _runtime_token_zero_calls = 0
    try:
        result = _original_run_block(*args, **kwargs)
    finally:
        _runtime_shift = False
    if result.get("mode") == "tempo_group2":
        for record in result.get("transfer_records", []):
            record["triggered_after_token"] = int(record["scheduled_token"])
            record["scheduled_token"] = int(record["scheduled_token"]) + 1
            record["trigger_semantics"] = "observed_t_minus_1_to_t_decode_boundary"
    result["sidecar_revision"] = 3
    result["candidate_policy"] = POLICY_LABEL
    result["global_single_flight"] = False
    result["physical_transfer_overlap_possible"] = True
    return result


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = _original_aggregate(records)
    result["schema_version"] = "tempo-vllm-tp8-lmcache-pair-stagger-screen-1"
    result["evidence_state"] = "same_allocation_real_tp8_adaptive_component_pilot"
    result["claim_scope"] = "research_component_screen_not_promotion_not_end_to_end_kv_connector"
    result["candidate_policy"] = POLICY_LABEL
    result["contract_signature_sha256"] = CONTRACT_SIGNATURE
    result["adaptive_pilot"] = True
    result["promotion_valid"] = False
    result["global_single_flight"] = False
    result["physical_transfer_overlap_possible"] = True
    result["stagger_contract"] = {
        "scheduled_tokens": list(SCHEDULED_TOKENS),
        "admitted_pair_by_slot": [slot % _v1.PAIR_COUNT for slot in range(32)],
        "calls_per_source": 8,
        "bytes_per_source": _v1.REQUESTS * _v1.KV_BYTES_PER_RANK,
        "global_bytes": _v1.PAIR_COUNT * _v1.REQUESTS * _v1.KV_BYTES_PER_RANK,
        "absolute_deadline_ns": DEADLINE_NS,
    }
    result["frozen_group2"] = {
        "policy_label": POLICY_LABEL,
        "contract_signature_sha256": CONTRACT_SIGNATURE,
        "scheduled_tokens": list(SCHEDULED_TOKENS),
        "absolute_deadline_ns": DEADLINE_NS,
        "retuned_from_real_tp8_pilot": True,
        "promotion_valid": False,
    }
    return result


def _install() -> None:
    _v1.ABSOLUTE_DEADLINE_NS = DEADLINE_NS
    _v1.validate_frozen_schedule = validate_stagger_schedule
    _v1.load_frozen_plan = load_stagger_contract
    _v1.schedule_object_indices = _runtime_schedule
    _v1._run_block = _run_block
    _v1.aggregate_rank_records = aggregate_rank_records


def main() -> None:
    _install()
    _v1.main()
    # The pinned receiver NixlChannel owns a non-daemon listener whose close()
    # can block.  All ranks have completed the final distributed barrier here;
    # force a clean process exit instead of hanging the second node's torchrun.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
