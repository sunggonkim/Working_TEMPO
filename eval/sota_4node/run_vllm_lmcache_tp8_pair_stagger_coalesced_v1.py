#!/usr/bin/env python3
"""One coalesced LMCache/NIXL call per pair with staggered admission."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

from eval.sota_4node import run_vllm_lmcache_tp8_pair_stagger_v3 as base


POLICY = "pair_staggered_coalesced_admission_v1"
CONTRACT_ID = "real-tp8-pair-stagger-coalesced-v1"
SCHEMA = "tempo-real-tp8-pair-stagger-coalesced-contract-1"
TOKENS = (1, 3, 5, 7)
DEADLINE_NS = 1_250_000_000


def coalesced_indices(mode: str, token: int, *, pair_index: int) -> tuple[int, ...]:
    if mode != "tempo_group2":
        return base._original_schedule(mode, token, pair_index=pair_index)
    if token not in TOKENS:
        return ()
    if pair_index != TOKENS.index(token):
        return ()
    return tuple(range(base._v1.REQUESTS * base._v1.CHUNKS_PER_REQUEST))


def validate_schedule() -> None:
    for pair in range(base._v1.PAIR_COUNT):
        active = [
            token
            for token in range(base._v1.TOKENS)
            if coalesced_indices("tempo_group2", token, pair_index=pair)
        ]
        if active != [TOKENS[pair]]:
            raise RuntimeError(f"coalesced pair {pair} admission changed")
        if coalesced_indices("tempo_group2", active[0], pair_index=pair) != tuple(range(32)):
            raise RuntimeError(f"coalesced pair {pair} object coverage changed")


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA,
        "contract_id": CONTRACT_ID,
        "provenance": {
            "source": "same_allocation_real_tp8_pair_stagger_observation",
            "adaptive_pilot": True,
            "promotion_valid": False,
            "policy_label": POLICY,
            "global_single_flight": False,
            "physical_transfer_overlap_possible": True,
        },
        "schedule": {
            "scheduled_tokens": list(TOKENS),
            "active_pairs": list(range(4)),
            "objects_per_source_call": 32,
            "bytes_per_source_call": 16_777_216,
            "global_bytes": 67_108_864,
            "deadline_ns": DEADLINE_NS,
        },
    }
    if payload != expected:
        raise ValueError("coalesced pair-stagger contract changed")
    validate_schedule()
    return payload, CONTRACT_ID


def _run_block(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = base._run_block(*args, **kwargs)
    result["candidate_policy"] = POLICY
    result["coalesced_calls_per_source"] = 1
    return result


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = base._original_aggregate(records)
    result.update(
        {
            "schema_version": "tempo-vllm-tp8-lmcache-pair-stagger-coalesced-screen-1",
            "evidence_state": "same_allocation_real_tp8_adaptive_component_pilot",
            "claim_scope": "research_component_screen_not_promotion_not_end_to_end_kv_connector",
            "candidate_policy": POLICY,
            "contract_id": CONTRACT_ID,
            "adaptive_pilot": True,
            "promotion_valid": False,
            "global_single_flight": False,
            "physical_transfer_overlap_possible": True,
            "coalesced_contract": {
                "scheduled_tokens": list(TOKENS),
                "active_pairs": list(range(4)),
                "calls_per_source": 1,
                "bytes_per_source": 16_777_216,
                "global_bytes": 67_108_864,
                "absolute_deadline_ns": DEADLINE_NS,
            },
        }
    )
    result["frozen_group2"] = {
        "policy_label": POLICY,
        "contract_id": CONTRACT_ID,
        "scheduled_tokens": list(TOKENS),
        "promotion_valid": False,
    }
    return result


def _install() -> None:
    base.pair_stagger_object_indices = coalesced_indices
    base._install()
    base._v1.ABSOLUTE_DEADLINE_NS = DEADLINE_NS
    base._v1.EXPECTED_PLAN_SIGNATURE = CONTRACT_ID
    base._v1.validate_frozen_schedule = validate_schedule
    base._v1.load_frozen_plan = load_contract
    base._v1._run_block = _run_block
    base._v1.aggregate_rank_records = aggregate


def main() -> None:
    _install()
    base._v1.main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
