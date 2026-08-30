#!/usr/bin/env python3
"""512 KiB token-microburst profile over the audited v6 data path.

The v6 memory/channel implementation is geometry-parametric.  This add-only
entrypoint freezes it to 32 real 512 KiB descriptors and calls per source.
Sources 0..3 issue chunks 0..31 at odd output-token boundaries 1..63.
Sources 4..7 issue chunk zero at token 1 and chunks 1..31 at even boundaries
2..62.  Greedy uses the same 512 KiB data plane but starts all 32 sequential
writes at request start.  Both modes still move exactly 128 MiB globally.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

from eval.sota_4node import run_vllm_lmcache_tp16_pair_quantum2mib_v6 as v6


v1 = v6.v1

CONTRACT_ID = "real-tp16-token-microburst512kib-v8"
CONTRACT_SCHEMA = "tempo-real-tp16-token-microburst512kib-contract-8"
RESULT_SCHEMA = "tempo-vllm-tp16-lmcache-token-microburst512kib-screen-8"
POLICY = "tp16_token_microburst512kib_admission_v8"

LOGICAL_CHUNKS_PER_QUANTUM = 1
QUANTUM_BYTES = v1.CHUNK_BYTES
QUANTA_PER_SOURCE = v1.REQUESTS * v1.CHUNKS_PER_REQUEST
PHYSICAL_CALLS_PER_SOURCE = QUANTA_PER_SOURCE
PHYSICAL_CALLS_GLOBAL = v1.SOURCE_COUNT * PHYSICAL_CALLS_PER_SOURCE
REGISTERED_DESCRIPTORS_PER_RANK = QUANTA_PER_SOURCE
WAVE0_TOKENS = tuple(range(1, v1.TOKENS, 2))
WAVE1_TOKENS = (1,) + tuple(range(2, v1.TOKENS, 2))
SCHEDULED_TOKENS = tuple(range(1, v1.TOKENS))

DESCRIPTOR_GEOMETRY = {
    "registered_buffer_bytes_per_rank": v1.BYTES_PER_SOURCE,
    "registered_buffer_alignment_bytes": QUANTUM_BYTES,
    "registered_descriptors_per_rank": REGISTERED_DESCRIPTORS_PER_RANK,
    "nixl_transfer_descriptor_bytes": QUANTUM_BYTES,
    "logical_verification_chunks_per_rank": QUANTA_PER_SOURCE,
    "logical_chunks_per_descriptor": LOGICAL_CHUNKS_PER_QUANTUM,
    "logical_chunk_bytes": v1.CHUNK_BYTES,
    "physical_calls_per_source": PHYSICAL_CALLS_PER_SOURCE,
    "physical_calls_global": PHYSICAL_CALLS_GLOBAL,
    "physical_descriptors_global": PHYSICAL_CALLS_GLOBAL,
    "physical_bytes_per_source": v1.BYTES_PER_SOURCE,
    "physical_bytes_global": v1.GLOBAL_BYTES,
}


def source_scheduled_tokens(pair_index: int) -> tuple[int, ...]:
    if isinstance(pair_index, bool) or not isinstance(pair_index, int):
        raise ValueError("pair_index must be an int")
    if not 0 <= pair_index < v1.PAIR_COUNT:
        raise ValueError("pair_index must be in 0..7")
    return WAVE0_TOKENS if pair_index < 4 else WAVE1_TOKENS


def microburst_indices(
    mode: str, scheduled_token: int, *, pair_index: int
) -> tuple[int, ...]:
    if mode not in (*v1.MODES, "tempo_group2"):
        raise ValueError(f"unknown mode: {mode}")
    if isinstance(scheduled_token, bool) or not isinstance(scheduled_token, int):
        raise ValueError("scheduled_token must be an int")
    if not 0 <= scheduled_token < v1.TOKENS:
        raise ValueError(f"scheduled_token must be in 0..{v1.TOKENS - 1}")
    if mode == "fg_only":
        return ()
    if mode == "lmcache_greedy":
        return tuple(range(QUANTA_PER_SOURCE)) if scheduled_token == 0 else ()
    tokens = source_scheduled_tokens(pair_index)
    if scheduled_token not in tokens:
        return ()
    return (tokens.index(scheduled_token),)


def validate_schedule() -> None:
    if QUANTUM_BYTES != 512 << 10:
        raise RuntimeError("microburst must be exactly 512 KiB")
    if QUANTA_PER_SOURCE != 32 or PHYSICAL_CALLS_GLOBAL != 256:
        raise RuntimeError("microburst call geometry changed")
    if v1.BYTES_PER_SOURCE != 16 << 20 or v1.GLOBAL_BYTES != 128 << 20:
        raise RuntimeError("microburst byte geometry changed")
    if WAVE0_TOKENS[-1] != 63 or WAVE1_TOKENS[-1] != 62:
        raise RuntimeError("microburst terminal token changed")
    for pair in range(v1.PAIR_COUNT):
        tokens = source_scheduled_tokens(pair)
        batches = [
            microburst_indices("tempo_coalesced", token, pair_index=pair)
            for token in tokens
        ]
        if len(tokens) != 32 or tuple(item[0] for item in batches) != tuple(range(32)):
            raise RuntimeError(f"source {pair} microburst coverage changed")
        if any(len(item) != 1 for item in batches):
            raise RuntimeError(f"source {pair} microburst width changed")
    for campaign_index in range(3):
        sequence = [mode for _, _, mode in v1.campaign_block_specs(campaign_index)]
        if len(sequence) != 9 or any(sequence.count(mode) != 3 for mode in v1.MODES):
            raise RuntimeError("three-by-three Latin campaign changed")


def _install_profile() -> None:
    v6.CONTRACT_ID = CONTRACT_ID
    v6.CONTRACT_SCHEMA = CONTRACT_SCHEMA
    v6.RESULT_SCHEMA = RESULT_SCHEMA
    v6.POLICY = POLICY
    v6.LOGICAL_CHUNKS_PER_QUANTUM = LOGICAL_CHUNKS_PER_QUANTUM
    v6.QUANTUM_BYTES = QUANTUM_BYTES
    v6.QUANTA_PER_SOURCE = QUANTA_PER_SOURCE
    v6.PHYSICAL_CALLS_PER_SOURCE = PHYSICAL_CALLS_PER_SOURCE
    v6.PHYSICAL_CALLS_GLOBAL = PHYSICAL_CALLS_GLOBAL
    v6.REGISTERED_DESCRIPTORS_PER_RANK = REGISTERED_DESCRIPTORS_PER_RANK
    v6.WAVE0_TOKENS = WAVE0_TOKENS
    v6.WAVE1_TOKENS = WAVE1_TOKENS
    v6.SCHEDULED_TOKENS = SCHEDULED_TOKENS
    v6.DESCRIPTOR_GEOMETRY = dict(DESCRIPTOR_GEOMETRY)
    v6.source_scheduled_tokens = source_scheduled_tokens
    v6.quantum_indices = microburst_indices
    v6.validate_schedule = validate_schedule
    v6.load_contract = load_contract
    v6.aggregate_rank_records = aggregate_rank_records


def _expected_contract() -> dict[str, Any]:
    _install_profile()
    payload = v6._expected_contract()
    payload["provenance"]["token_microburst512kib"] = True
    payload["schedule"]["token_one_active_sources"] = list(range(v1.SOURCE_COUNT))
    payload["schedule"]["steady_active_sources_per_token"] = 4
    return payload


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid TP16 512 KiB microburst v8 contract: {exc}") from exc
    if payload != _expected_contract():
        raise ValueError("TP16 512 KiB microburst v8 contract changed")
    validate_schedule()
    return payload, CONTRACT_ID


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = v6._decorate_quantum_result(v6._v3_aggregate(records), records)
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    result["candidate_policy"] = POLICY
    result["background"]["physical_operation"] = (
        "thirty_two_sequential_or_token_admitted_512kib_nixl_writes_per_source"
    )
    result["microburst_schedule"] = {
        "source_scheduled_tokens": [
            list(source_scheduled_tokens(pair)) for pair in range(v1.PAIR_COUNT)
        ],
        "token_one_active_sources": list(range(v1.SOURCE_COUNT)),
        "steady_active_sources_per_token": 4,
        "physical_calls_per_source": PHYSICAL_CALLS_PER_SOURCE,
        "physical_calls_global": PHYSICAL_CALLS_GLOBAL,
    }
    return result


def main() -> None:
    _install_profile()
    v6.main()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
