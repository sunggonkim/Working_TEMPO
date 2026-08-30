#!/usr/bin/env python3
"""Two-wave front-loaded TP16 screen with one real 16 MiB descriptor/source.

This revision changes only the candidate admission schedule from the corrected
v3/v4 campaign: source/receiver pairs 0..3 submit their complete logical
0..31 batch after output token 1, and pairs 4..7 submit it after token 2.
Greedy still submits the same complete batch at request start.  The v4 memory
and channel adapter preserve exactly one official NIXL call and one contiguous
16 MiB transfer descriptor per source, or eight descriptors / 128 MiB globally.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from eval.sota_4node import (
    run_vllm_lmcache_tp16_pair_stagger_coalesced_v4 as v4,
)


v3 = v4.v3
v2 = v4.v2
v1 = v4.v1
base = v1.base

CONTRACT_ID = "real-tp16-frontload16mib-v7"
CONTRACT_SCHEMA = "tempo-real-tp16-frontload16mib-contract-7"
RESULT_SCHEMA = "tempo-vllm-tp16-lmcache-frontload16mib-screen-7"
POLICY = "tp16_two_wave_frontloaded_contiguous_descriptor_admission_v7"
PLAN_NAME = "tp16_frontload16mib_v7"

SCHEDULED_TOKENS = (1, 2)
SOURCE_SCHEDULED_TOKENS = (1, 1, 1, 1, 2, 2, 2, 2)
LOGICAL_OBJECT_INDICES = tuple(range(v4.LOGICAL_CHUNKS_PER_SOURCE))
PHYSICAL_DESCRIPTORS_GLOBAL = v1.SOURCE_COUNT

_v4_install = v4._install
_v4_aggregate = v4.aggregate_rank_records
_v4_run_block = v4._run_block


def source_scheduled_token(pair_index: int) -> int:
    if isinstance(pair_index, bool) or not isinstance(pair_index, int):
        raise ValueError("pair_index must be an int")
    if not 0 <= pair_index < v1.PAIR_COUNT:
        raise ValueError("pair_index must be in 0..7")
    return SOURCE_SCHEDULED_TOKENS[pair_index]


def frontload_indices(
    mode: str,
    scheduled_token: int,
    *,
    pair_index: int,
) -> tuple[int, ...]:
    """Return one complete logical batch at request start or token wave 1/2."""

    if mode not in (*v1.MODES, "tempo_group2"):
        raise ValueError(f"unknown mode: {mode}")
    if isinstance(scheduled_token, bool) or not isinstance(scheduled_token, int):
        raise ValueError("scheduled_token must be an int")
    if not 0 <= scheduled_token < v1.TOKENS:
        raise ValueError(f"scheduled_token must be in 0..{v1.TOKENS - 1}")
    expected_token = source_scheduled_token(pair_index)
    if mode == "fg_only":
        return ()
    if mode == "lmcache_greedy":
        return LOGICAL_OBJECT_INDICES if scheduled_token == 0 else ()
    return LOGICAL_OBJECT_INDICES if scheduled_token == expected_token else ()


def validate_schedule() -> None:
    if SOURCE_SCHEDULED_TOKENS != (1, 1, 1, 1, 2, 2, 2, 2):
        raise RuntimeError("two-wave source trigger vector changed")
    if SCHEDULED_TOKENS != (1, 2):
        raise RuntimeError("front-loaded token boundaries changed")
    if v1.BYTES_PER_SOURCE != 16 << 20 or v1.GLOBAL_BYTES != 128 << 20:
        raise RuntimeError("TP16 byte geometry changed")
    if v4.PHYSICAL_DESCRIPTORS_PER_SOURCE_CALL != 1:
        raise RuntimeError("single-descriptor source geometry changed")
    if PHYSICAL_DESCRIPTORS_GLOBAL != 8:
        raise RuntimeError("global physical descriptor count changed")
    for pair in range(v1.PAIR_COUNT):
        active = [
            token
            for token in range(v1.TOKENS)
            if frontload_indices("tempo_coalesced", token, pair_index=pair)
        ]
        expected = source_scheduled_token(pair)
        if active != [expected]:
            raise RuntimeError(f"source {pair} trigger token changed")
        if frontload_indices(
            "tempo_coalesced", expected, pair_index=pair
        ) != LOGICAL_OBJECT_INDICES:
            raise RuntimeError(f"source {pair} full logical batch changed")
        if frontload_indices(
            "lmcache_greedy", 0, pair_index=pair
        ) != LOGICAL_OBJECT_INDICES:
            raise RuntimeError(f"source {pair} greedy request-start batch changed")
    for campaign_index in range(3):
        sequence = [
            mode for _, _, mode in v1.campaign_block_specs(campaign_index)
        ]
        if len(sequence) != 9 or any(
            sequence.count(mode) != 3 for mode in v1.MODES
        ):
            raise RuntimeError("three-by-three Latin campaign changed")


def _expected_contract() -> dict[str, Any]:
    payload = v4._expected_contract()
    payload["schema_version"] = CONTRACT_SCHEMA
    payload["contract_id"] = CONTRACT_ID
    payload["provenance"].update(
        {
            "policy_label": POLICY,
            "frontloaded_two_wave": True,
            "physical_transfer_overlap_possible": True,
        }
    )
    payload["schedule"].update(
        {
            "scheduled_tokens": list(SCHEDULED_TOKENS),
            "source_scheduled_tokens": list(SOURCE_SCHEDULED_TOKENS),
            "logical_object_indices_per_source_call": list(
                LOGICAL_OBJECT_INDICES
            ),
            "logical_chunks_per_source_call": len(LOGICAL_OBJECT_INDICES),
            "physical_descriptors_per_source_call": 1,
            "physical_descriptors_global": PHYSICAL_DESCRIPTORS_GLOBAL,
        }
    )
    return payload


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid TP16 frontload16mib v7 contract: {exc}") from exc
    if payload != _expected_contract():
        raise ValueError("TP16 frontload16mib v7 contract changed")
    validate_schedule()
    return payload, CONTRACT_ID


def install_nixl_descriptor_count_compatibility(
    descriptor_type: type | None = None,
) -> type:
    """Install v5's narrow ``descCount`` compatibility without overriding len."""

    if descriptor_type is None:
        from nixl import _api

        descriptor_type = _api.nixlBind.nixlXferDList
    if hasattr(descriptor_type, "__len__"):
        return descriptor_type
    desc_count = getattr(descriptor_type, "descCount", None)
    if not callable(desc_count):
        raise RuntimeError(
            "NIXL descriptor list exposes neither __len__ nor descCount()"
        )
    descriptor_type.__len__ = lambda self: int(self.descCount())
    if not hasattr(descriptor_type, "__len__"):
        raise RuntimeError("NIXL descriptor count compatibility was not installed")
    return descriptor_type


def _decorate_frontload_block(
    block: dict[str, Any], *, rank: int, requested_mode: str
) -> dict[str, Any]:
    is_source = rank < v1.SOURCE_COUNT
    background = requested_mode != "fg_only"
    pair_index = rank if is_source else rank - v1.RECEIVER_OFFSET
    expected_records = 1 if is_source and background else 0
    records = block.get("transfer_records", [])
    if len(records) != expected_records:
        raise RuntimeError("frontload logical source call count changed")

    for record in records:
        if list(record["object_indices"]) != list(LOGICAL_OBJECT_INDICES):
            raise RuntimeError("frontload logical 0..31 batch changed")
        if int(record["physical_transfer_descriptors"]) != 1:
            raise RuntimeError("frontload physical descriptor count changed")
        if int(record["physical_transfer_bytes"]) != v1.BYTES_PER_SOURCE:
            raise RuntimeError("frontload physical descriptor byte count changed")
        if int(record["official_batched_write_objects"]) != 1:
            raise RuntimeError("frontload official call object count changed")
        if requested_mode == "lmcache_greedy":
            if int(record["scheduled_token"]) != 0:
                raise RuntimeError("greedy is no longer request-start admitted")
        elif requested_mode == "tempo_coalesced":
            expected_token = source_scheduled_token(pair_index)
            if int(record["scheduled_token"]) != expected_token:
                raise RuntimeError("candidate source trigger token changed")
            if int(record["triggered_after_token_index"]) != expected_token - 1:
                raise RuntimeError("candidate decode-boundary trigger changed")

    expected_physical = expected_records
    if int(block["physical_transfer_calls"]) != expected_physical:
        raise RuntimeError("frontload physical call count changed")
    if int(block["physical_transfer_descriptors"]) != expected_physical:
        raise RuntimeError("frontload physical descriptor count changed")
    block["source_scheduled_token"] = source_scheduled_token(pair_index)
    block["frontload_full_logical_batch"] = bool(
        is_source and requested_mode == "tempo_coalesced"
    )
    block["frontload_schedule_exact"] = True
    return block


def _run_block(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = _v4_run_block(*args, **kwargs)
    return _decorate_frontload_block(
        result,
        rank=int(kwargs["rank"]),
        requested_mode=str(kwargs["mode"]),
    )


def _validate_rank_records(
    records: list[dict[str, Any]], block_count: int
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda item: int(item["rank"]))
    if len(ordered) != v1.WORLD_SIZE:
        raise ValueError("frontload aggregation requires all sixteen ranks")
    if [int(item["rank"]) for item in ordered] != list(range(v1.WORLD_SIZE)):
        raise ValueError("frontload aggregation rank set changed")
    if any(len(item["blocks"]) != block_count for item in ordered):
        raise ValueError("frontload rank block counts differ")
    return ordered


def _decorate_frontload_result(
    result: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    ordered = _validate_rank_records(records, len(result["blocks"]))
    for block_index, block in enumerate(result["blocks"]):
        rank_blocks = [item["blocks"][block_index] for item in ordered]
        if any(item["mode"] != block["mode"] for item in rank_blocks):
            raise ValueError("frontload rank block modes differ")
        background = block["mode"] != "fg_only"
        expected_global = PHYSICAL_DESCRIPTORS_GLOBAL if background else 0
        source_blocks = rank_blocks[: v1.SOURCE_COUNT]
        receiver_blocks = rank_blocks[v1.SOURCE_COUNT :]
        physical_calls = sum(
            int(item["physical_transfer_calls"]) for item in source_blocks
        )
        physical_descriptors = sum(
            int(item["physical_transfer_descriptors"]) for item in source_blocks
        )
        receiver_descriptors = sum(
            int(item["physical_transfer_descriptors"])
            for item in receiver_blocks
        )
        if (
            physical_calls != expected_global
            or physical_descriptors != expected_global
            or receiver_descriptors != 0
        ):
            raise ValueError("global physical call/descriptor total changed")
        if background:
            if int(block["background_completed_bytes"]) != v1.GLOBAL_BYTES:
                raise ValueError("frontload global completed bytes changed")
            if int(block["receiver_verified_bytes"]) != v1.GLOBAL_BYTES:
                raise ValueError("frontload global verified bytes changed")
        block.update(
            {
                "source_scheduled_tokens": list(SOURCE_SCHEDULED_TOKENS),
                "frontload_schedule_exact": True,
                "physical_nixl_calls_global": physical_calls,
                "physical_transfer_descriptors_global": physical_descriptors,
                "physical_source_descriptors_global": physical_descriptors,
            }
        )

    candidate_blocks = [
        block for block in result["blocks"] if block["mode"] == "tempo_coalesced"
    ]
    if len(candidate_blocks) != 3:
        raise ValueError("campaign must contain exactly three TEMPO replicates")
    gates = {
        "schedule_tokens_exact": all(
            bool(block["frontload_schedule_exact"]) for block in candidate_blocks
        ),
        "physical_descriptors_exact": all(
            int(block["physical_transfer_descriptors_global"])
            == PHYSICAL_DESCRIPTORS_GLOBAL
            for block in candidate_blocks
        ),
        "overall_correctness_met": bool(result["overall_correctness_met"]),
        "no_post_foreground_drain_met": bool(
            result["candidate_no_post_foreground_drain_met"]
        ),
        "absolute_deadline_met": bool(result["candidate_absolute_deadline_met"]),
        "schedule_start_adherence_met": bool(
            result["candidate_schedule_adherence_met"]
        ),
    }
    if not gates["schedule_tokens_exact"] or not gates["physical_descriptors_exact"]:
        raise ValueError("frontload structural gate failed")

    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    result["candidate_policy"] = POLICY
    result["promotion_valid"] = False
    result["frontload_schedule"] = {
        "scheduled_tokens": list(SCHEDULED_TOKENS),
        "source_scheduled_tokens": list(SOURCE_SCHEDULED_TOKENS),
        "logical_object_indices_per_source_call": list(LOGICAL_OBJECT_INDICES),
        "physical_descriptors_global": PHYSICAL_DESCRIPTORS_GLOBAL,
    }
    result["candidate_gates"] = gates
    result["config"]["plan_name"] = PLAN_NAME
    result["config"]["source_scheduled_tokens"] = list(
        SOURCE_SCHEDULED_TOKENS
    )
    result["background"].update(
        {
            "physical_operation": "one_contiguous_16mib_descriptor_per_source_in_two_token_waves",
            "physical_descriptors_per_source_call": 1,
            "physical_descriptors_global": PHYSICAL_DESCRIPTORS_GLOBAL,
        }
    )
    result["coalesced_contract"].update(
        {
            "scheduled_tokens": list(SCHEDULED_TOKENS),
            "source_scheduled_tokens": list(SOURCE_SCHEDULED_TOKENS),
            "logical_object_indices_per_source_call": list(
                LOGICAL_OBJECT_INDICES
            ),
            "physical_descriptors_per_source_call": 1,
            "physical_descriptors_global": PHYSICAL_DESCRIPTORS_GLOBAL,
        }
    )
    result["frozen_group2"].update(
        {
            "policy_label": POLICY,
            "contract_id": CONTRACT_ID,
            "scheduled_tokens": list(SCHEDULED_TOKENS),
            "source_scheduled_tokens": list(SOURCE_SCHEDULED_TOKENS),
        }
    )
    return result


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return _decorate_frontload_result(_v4_aggregate(records), records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("eval/sota_4node/real_tp16_frontload16mib_v7.json"),
    )
    parser.add_argument("--api-host", required=True)
    parser.add_argument("--api-port", type=int, required=True)
    parser.add_argument("--model", default="models/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--nixl-port-base", type=int, default=35100)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--campaign-index", type=int, choices=range(3), required=True)
    parser.add_argument("--allocation-id", default=os.environ.get("SLURM_JOB_ID"))
    args = parser.parse_args()
    if not args.allocation_id:
        parser.error("allocation-id is required outside Slurm")
    if not 1024 <= args.api_port <= 65535:
        parser.error("api-port must be a valid TCP port")
    if not 1024 <= args.nixl_port_base <= 65535 - v1.PAIR_COUNT:
        parser.error("nixl-port-base must leave eight valid TCP ports")
    if args.request_timeout_s <= 0:
        parser.error("request-timeout-s must be positive")
    v2._allocation_id = str(args.allocation_id)
    return args


def _install(campaign_index: int) -> None:
    _v4_install(campaign_index)
    v1.CONTRACT_ID = CONTRACT_ID
    v1.CONTRACT_SCHEMA = CONTRACT_SCHEMA
    v1.POLICY = POLICY
    v1.SCHEDULED_TOKENS = SCHEDULED_TOKENS
    v1.coalesced_indices = frontload_indices
    v1.validate_schedule = validate_schedule
    v1.load_contract = load_contract
    v1.aggregate_rank_records = aggregate_rank_records
    v1._run_block = _run_block
    base.EXPECTED_PLAN_SIGNATURE = CONTRACT_ID
    base.validate_frozen_schedule = validate_schedule
    base.load_frozen_plan = load_contract
    base.schedule_object_indices = v1._runtime_schedule
    base.aggregate_rank_records = aggregate_rank_records
    base._run_block = _run_block


def main() -> None:
    install_nixl_descriptor_count_compatibility()
    v1._parse_args = _parse_args
    v1._install = _install
    v1.CONTRACT_ID = CONTRACT_ID
    v1.CONTRACT_SCHEMA = CONTRACT_SCHEMA
    v1.POLICY = POLICY
    v1.load_contract = load_contract
    v1.aggregate_rank_records = aggregate_rank_records
    v1.main()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
