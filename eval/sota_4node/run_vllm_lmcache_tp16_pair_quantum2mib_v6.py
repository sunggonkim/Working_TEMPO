#!/usr/bin/env python3
"""Four-node TP16 screen with eight real 2 MiB NIXL descriptors per source.

The foreground and clock-domain rules are inherited from the corrected v3
runner.  The sidecar still exposes 32 logical 512 KiB views so initialization
and receiver verification remain byte-exact, but every four adjacent views
share one contiguous 2 MiB ``TensorMemoryObj``.  LMCache/NIXL therefore
registers exactly eight transfer descriptors per rank.

Greedy submits its 32 logical views at request start; the adapter executes
eight sequential official 2 MiB writes.  TEMPO submits one four-view quantum
at each of eight token boundaries.  Sources 0..3 use tokens 1+7g and sources
4..7 use tokens 2+7g, for g=0..7.  Both modes move 16 MiB/source and
128 MiB globally through 64 physical NIXL calls/descriptors.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np

from eval.sota_4node import run_vllm_lmcache_tp16_pair_stagger_coalesced_v3 as v3


v2 = v3.v2
v1 = v3.v1
base = v1.base

CONTRACT_ID = "real-tp16-pair-quantum2mib-v6"
CONTRACT_SCHEMA = "tempo-real-tp16-pair-quantum2mib-contract-6"
RESULT_SCHEMA = "tempo-vllm-tp16-lmcache-pair-quantum2mib-screen-6"
POLICY = "tp16_two_wave_eight_quantum2mib_admission_v6"

LOGICAL_CHUNKS_PER_SOURCE = v1.REQUESTS * v1.CHUNKS_PER_REQUEST
LOGICAL_CHUNKS_PER_QUANTUM = 4
QUANTUM_BYTES = LOGICAL_CHUNKS_PER_QUANTUM * v1.CHUNK_BYTES
QUANTA_PER_SOURCE = LOGICAL_CHUNKS_PER_SOURCE // LOGICAL_CHUNKS_PER_QUANTUM
PHYSICAL_CALLS_PER_SOURCE = QUANTA_PER_SOURCE
PHYSICAL_CALLS_GLOBAL = v1.SOURCE_COUNT * PHYSICAL_CALLS_PER_SOURCE
REGISTERED_DESCRIPTORS_PER_RANK = QUANTA_PER_SOURCE

WAVE0_TOKENS = tuple(1 + 7 * group for group in range(QUANTA_PER_SOURCE))
WAVE1_TOKENS = tuple(2 + 7 * group for group in range(QUANTA_PER_SOURCE))
SCHEDULED_TOKENS = tuple(sorted(set(WAVE0_TOKENS + WAVE1_TOKENS)))

DESCRIPTOR_GEOMETRY = {
    "registered_buffer_bytes_per_rank": v1.BYTES_PER_SOURCE,
    "registered_buffer_alignment_bytes": QUANTUM_BYTES,
    "registered_descriptors_per_rank": REGISTERED_DESCRIPTORS_PER_RANK,
    "nixl_transfer_descriptor_bytes": QUANTUM_BYTES,
    "logical_verification_chunks_per_rank": LOGICAL_CHUNKS_PER_SOURCE,
    "logical_chunks_per_descriptor": LOGICAL_CHUNKS_PER_QUANTUM,
    "logical_chunk_bytes": v1.CHUNK_BYTES,
    "physical_calls_per_source": PHYSICAL_CALLS_PER_SOURCE,
    "physical_calls_global": PHYSICAL_CALLS_GLOBAL,
    "physical_descriptors_global": PHYSICAL_CALLS_GLOBAL,
    "physical_bytes_per_source": v1.BYTES_PER_SOURCE,
    "physical_bytes_global": v1.GLOBAL_BYTES,
}

_v3_install = v3._install
_v3_aggregate = v3.aggregate_rank_records
_v3_run_block = v3._run_block
_official_loader = base.official._load_official_lmcache


def source_scheduled_tokens(pair_index: int) -> tuple[int, ...]:
    if isinstance(pair_index, bool) or not isinstance(pair_index, int):
        raise ValueError("pair_index must be an int")
    if not 0 <= pair_index < v1.PAIR_COUNT:
        raise ValueError("pair_index must be in 0..7")
    return WAVE0_TOKENS if pair_index < 4 else WAVE1_TOKENS


def quantum_indices(
    mode: str,
    scheduled_token: int,
    *,
    pair_index: int,
) -> tuple[int, ...]:
    """Return logical views admitted at one declared schedule boundary."""

    if mode not in (*v1.MODES, "tempo_group2"):
        raise ValueError(f"unknown mode: {mode}")
    if isinstance(scheduled_token, bool) or not isinstance(scheduled_token, int):
        raise ValueError("scheduled_token must be an int")
    if not 0 <= scheduled_token < v1.TOKENS:
        raise ValueError(f"scheduled_token must be in 0..{v1.TOKENS - 1}")
    tokens = source_scheduled_tokens(pair_index)
    if mode == "fg_only":
        return ()
    if mode == "lmcache_greedy":
        return tuple(range(LOGICAL_CHUNKS_PER_SOURCE)) if scheduled_token == 0 else ()
    if scheduled_token not in tokens:
        return ()
    group = tokens.index(scheduled_token)
    start = group * LOGICAL_CHUNKS_PER_QUANTUM
    return tuple(range(start, start + LOGICAL_CHUNKS_PER_QUANTUM))


def validate_schedule() -> None:
    if QUANTUM_BYTES != 2 << 20:
        raise RuntimeError("physical transfer quantum is not exactly 2 MiB")
    if QUANTA_PER_SOURCE != 8 or REGISTERED_DESCRIPTORS_PER_RANK != 8:
        raise RuntimeError("descriptor geometry changed")
    if PHYSICAL_CALLS_GLOBAL != 64 or v1.GLOBAL_BYTES != 128 << 20:
        raise RuntimeError("global physical call/byte geometry changed")
    if WAVE0_TOKENS[-1] != 50 or WAVE1_TOKENS[-1] != 51:
        raise RuntimeError("two-wave terminal token changed")
    for pair in range(v1.PAIR_COUNT):
        active = [
            token
            for token in range(v1.TOKENS)
            if quantum_indices("tempo_coalesced", token, pair_index=pair)
        ]
        if tuple(active) != source_scheduled_tokens(pair):
            raise RuntimeError(f"source {pair} token schedule changed")
        batches = [
            quantum_indices("tempo_coalesced", token, pair_index=pair)
            for token in active
        ]
        flattened = tuple(index for batch in batches for index in batch)
        if flattened != tuple(range(LOGICAL_CHUNKS_PER_SOURCE)):
            raise RuntimeError(f"source {pair} logical coverage changed")
        if len(batches) != PHYSICAL_CALLS_PER_SOURCE:
            raise RuntimeError(f"source {pair} physical call count changed")
    for campaign_index in range(3):
        sequence = [mode for _, _, mode in v1.campaign_block_specs(campaign_index)]
        if len(sequence) != 9 or any(sequence.count(mode) != 3 for mode in v1.MODES):
            raise RuntimeError("three-by-three Latin campaign changed")


def _expected_contract() -> dict[str, Any]:
    payload = v3._expected_contract()
    payload["schema_version"] = CONTRACT_SCHEMA
    payload["contract_id"] = CONTRACT_ID
    payload["provenance"].update(
        {
            "policy_label": POLICY,
            "physical_transfer_overlap_possible": True,
        }
    )
    payload["schedule"].update(
        {
            "scheduled_tokens": list(SCHEDULED_TOKENS),
            "source_scheduled_tokens": [
                list(source_scheduled_tokens(pair)) for pair in range(v1.PAIR_COUNT)
            ],
            "calls_per_source": PHYSICAL_CALLS_PER_SOURCE,
            "source_calls_global": PHYSICAL_CALLS_GLOBAL,
            "logical_chunks_per_call": LOGICAL_CHUNKS_PER_QUANTUM,
            "bytes_per_source_call": QUANTUM_BYTES,
            "bytes_per_source": v1.BYTES_PER_SOURCE,
            "registered_descriptors_per_rank": REGISTERED_DESCRIPTORS_PER_RANK,
            "physical_descriptors_global": PHYSICAL_CALLS_GLOBAL,
        }
    )
    payload["descriptor_geometry"] = dict(DESCRIPTOR_GEOMETRY)
    return payload


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid TP16 2 MiB quantum v6 contract: {exc}") from exc
    if payload != _expected_contract():
        raise ValueError("TP16 2 MiB quantum v6 contract changed")
    validate_schedule()
    return payload, CONTRACT_ID


def _make_quantum2mib_memory(
    torch: Any,
    TensorMemoryObj: Any,
    MemoryObjMetadata: Any,
    MemoryFormat: Any,
    *,
    requests: int,
    chunk_bytes: int,
) -> tuple[Any, Any, list[Any], dict[int, int]]:
    """Create 32 logical views attached four-at-a-time to eight 2 MiB objects."""

    logical_count = requests * v1.CHUNKS_PER_REQUEST
    total_bytes = logical_count * chunk_bytes
    if (
        requests != v1.REQUESTS
        or chunk_bytes != v1.CHUNK_BYTES
        or logical_count != LOGICAL_CHUNKS_PER_SOURCE
        or total_bytes != v1.BYTES_PER_SOURCE
    ):
        raise RuntimeError("2 MiB quantum memory geometry changed")

    backing = torch.zeros(
        total_bytes + QUANTUM_BYTES - 1,
        dtype=torch.uint8,
        device="cuda",
    )
    offset = (-backing.data_ptr()) % QUANTUM_BYTES
    buffer = backing[offset : offset + total_bytes]
    if buffer.numel() != v1.BYTES_PER_SOURCE:
        raise RuntimeError("registered buffer byte length changed")
    if buffer.data_ptr() % QUANTUM_BYTES != 0:
        raise RuntimeError("registered buffer is not 2 MiB aligned")

    quantum_objects: list[Any] = []
    for quantum_index in range(QUANTA_PER_SOURCE):
        raw = buffer[
            quantum_index * QUANTUM_BYTES : (quantum_index + 1) * QUANTUM_BYTES
        ]
        shape = torch.Size([QUANTUM_BYTES])
        quantum_objects.append(
            TensorMemoryObj(
                raw_data=raw,
                metadata=MemoryObjMetadata(
                    shape=shape,
                    dtype=torch.uint8,
                    address=raw.data_ptr(),
                    phy_size=QUANTUM_BYTES,
                    ref_count=1,
                    pin_count=0,
                    fmt=MemoryFormat.BINARY,
                    shapes=[shape],
                    dtypes=[torch.uint8],
                ),
                parent_allocator=None,
            )
        )

    logical_objects: list[Any] = []
    descriptor_index_by_address: dict[int, int] = {}
    for logical_index in range(logical_count):
        raw = buffer[
            logical_index * chunk_bytes : (logical_index + 1) * chunk_bytes
        ]
        shape = torch.Size([chunk_bytes])
        logical_object = TensorMemoryObj(
            raw_data=raw,
            metadata=MemoryObjMetadata(
                shape=shape,
                dtype=torch.uint8,
                address=raw.data_ptr(),
                phy_size=chunk_bytes,
                ref_count=1,
                pin_count=0,
                fmt=MemoryFormat.BINARY,
                shapes=[shape],
                dtypes=[torch.uint8],
            ),
            parent_allocator=None,
        )
        quantum_index = logical_index // LOGICAL_CHUNKS_PER_QUANTUM
        logical_object._tempo_quantum_transfer_object = quantum_objects[quantum_index]
        logical_object._tempo_quantum_index = quantum_index
        logical_object._tempo_logical_index = logical_index
        logical_objects.append(logical_object)
        descriptor_index_by_address[raw.data_ptr()] = quantum_index

    for quantum_index, quantum_object in enumerate(quantum_objects):
        if descriptor_index_by_address.get(quantum_object.meta.address) != quantum_index:
            raise RuntimeError("quantum object does not map to its NIXL descriptor")
    return backing, buffer, logical_objects, descriptor_index_by_address


def _descriptor_count(channel: Any) -> int:
    """Read the real NIXL descriptor count through its API, never container len()."""

    descriptors = channel.nixl_wrapper.xfer_descs
    count_method = getattr(descriptors, "descCount", None)
    if not callable(count_method):
        raise RuntimeError("NIXL transfer descriptor list lacks descCount()")
    return int(count_method())


def _quantum2mib_channel_class(base_channel: Any) -> Any:
    """Force 2 MiB registration and translate logical batches to official writes."""

    class Quantum2MiBNixlChannel(base_channel):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            buffer_ptr = int(kwargs.get("buffer_ptr", -1))
            buffer_size = int(kwargs.get("buffer_size", -1))
            if buffer_size != v1.BYTES_PER_SOURCE:
                raise RuntimeError("NIXL registered buffer must be exactly 16 MiB")
            if buffer_ptr < 0 or buffer_ptr % QUANTUM_BYTES != 0:
                raise RuntimeError("NIXL registered buffer must be 2 MiB aligned")
            kwargs["align_bytes"] = QUANTUM_BYTES
            super().__init__(*args, **kwargs)
            descriptor_count = _descriptor_count(self)
            if descriptor_count != REGISTERED_DESCRIPTORS_PER_RANK:
                raise RuntimeError(
                    "official LMCache did not create exactly eight NIXL descriptors"
                )
            self.tempo_registered_descriptor_count = descriptor_count
            self.tempo_physical_write_calls = 0
            self.tempo_physical_write_events: list[dict[str, Any]] = []
            self._tempo_event_lock = threading.Lock()

        def batched_write(
            self,
            objects: list[Any],
            transfer_spec: dict[str, Any] | None = None,
        ) -> int:
            if transfer_spec is None:
                raise ValueError("2 MiB quantum write requires transfer_spec")
            if not objects:
                raise ValueError("2 MiB quantum write requires logical objects")

            try:
                logical_indices = [int(item._tempo_logical_index) for item in objects]
                quantum_indices = [int(item._tempo_quantum_index) for item in objects]
                quantum_objects = [item._tempo_quantum_transfer_object for item in objects]
            except AttributeError as exc:
                raise RuntimeError("write object is not an attached logical chunk") from exc
            supplied_remote = np.asarray(
                transfer_spec.get("remote_indexes", ()), dtype=np.uint64
            ).tolist()
            if supplied_remote != logical_indices:
                raise RuntimeError("logical local and remote chunk indexes differ")

            unique_quanta: list[int] = []
            physical_objects: list[Any] = []
            for quantum_index, quantum_object in zip(
                quantum_indices, quantum_objects, strict=True
            ):
                if not unique_quanta or unique_quanta[-1] != quantum_index:
                    if quantum_index in unique_quanta:
                        raise RuntimeError("logical batch revisits a noncontiguous quantum")
                    unique_quanta.append(quantum_index)
                    physical_objects.append(quantum_object)
                elif physical_objects[-1] is not quantum_object:
                    raise RuntimeError("logical chunks disagree on their quantum object")

            event = {
                "logical_indices": list(logical_indices),
                "quantum_indices": list(unique_quanta),
                "physical_calls": 0,
                "completed_physical_descriptors": 0,
                "error": None,
            }
            try:
                for quantum_index, quantum_object in zip(
                    unique_quanta, physical_objects, strict=True
                ):
                    collapsed_spec = dict(transfer_spec)
                    collapsed_spec["remote_indexes"] = np.asarray(
                        [quantum_index], dtype=np.uint64
                    )
                    event["physical_calls"] += 1
                    completed = int(
                        super().batched_write(
                            objects=[quantum_object],
                            transfer_spec=collapsed_spec,
                        )
                    )
                    if completed != 1:
                        raise RuntimeError("official 2 MiB write count changed")
                    event["completed_physical_descriptors"] += completed
            except BaseException as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                with self._tempo_event_lock:
                    self.tempo_physical_write_calls += int(event["physical_calls"])
                    self.tempo_physical_write_events.append(event)
            return len(logical_indices)

    Quantum2MiBNixlChannel.__name__ = "Quantum2MiBNixlChannel"
    Quantum2MiBNixlChannel.__qualname__ = "Quantum2MiBNixlChannel"
    return Quantum2MiBNixlChannel


def _load_quantum2mib_lmcache(repo_root: Path) -> tuple[Any, Any, Any, Any]:
    channel, tensor_obj, metadata, memory_format = _official_loader(repo_root)
    return (
        _quantum2mib_channel_class(channel),
        tensor_obj,
        metadata,
        memory_format,
    )


def _decorate_quantum_block(
    block: dict[str, Any],
    *,
    channel: Any,
    rank: int,
    requested_mode: str,
    write_events: list[dict[str, Any]],
) -> dict[str, Any]:
    registered = int(channel.tempo_registered_descriptor_count)
    if registered != REGISTERED_DESCRIPTORS_PER_RANK:
        raise RuntimeError("runtime registered descriptor count changed")
    is_source = rank < v1.SOURCE_COUNT
    background = requested_mode != "fg_only"
    expected_logical_batches = (
        (1 if requested_mode == "lmcache_greedy" else PHYSICAL_CALLS_PER_SOURCE)
        if is_source and background
        else 0
    )
    expected_physical_calls = PHYSICAL_CALLS_PER_SOURCE if is_source and background else 0
    records = block.get("transfer_records", [])
    if len(records) != expected_logical_batches or len(write_events) != expected_logical_batches:
        raise RuntimeError("logical background batch count changed")

    for record, event in zip(records, write_events, strict=True):
        logical_indices = [int(value) for value in record["object_indices"]]
        if logical_indices != event["logical_indices"]:
            raise RuntimeError("worker record and official write coverage differ")
        expected_quanta = sorted(
            set(index // LOGICAL_CHUNKS_PER_QUANTUM for index in logical_indices)
        )
        if event["quantum_indices"] != expected_quanta:
            raise RuntimeError("official write quantum mapping changed")
        physical_calls = int(event["physical_calls"])
        completed_descriptors = int(event["completed_physical_descriptors"])
        if event["error"] or completed_descriptors != physical_calls:
            raise RuntimeError("official 2 MiB descriptor completion changed")
        record.update(
            {
                "registered_nixl_descriptors": registered,
                "physical_nixl_calls": physical_calls,
                "physical_transfer_descriptors": completed_descriptors,
                "physical_transfer_bytes": completed_descriptors * QUANTUM_BYTES,
                "quantum_descriptor_indexes": list(event["quantum_indices"]),
                "logical_chunks_per_quantum": LOGICAL_CHUNKS_PER_QUANTUM,
            }
        )

    physical_calls = sum(int(event["physical_calls"]) for event in write_events)
    physical_descriptors = sum(
        int(event["completed_physical_descriptors"]) for event in write_events
    )
    if physical_calls != expected_physical_calls or physical_descriptors != expected_physical_calls:
        raise RuntimeError("physical NIXL call/descriptor count changed")
    if is_source and background:
        if int(block["background_completed_bytes"]) != v1.BYTES_PER_SOURCE:
            raise RuntimeError("source completed-byte accounting changed")
        if int(block["expected_source_bytes"]) != v1.BYTES_PER_SOURCE:
            raise RuntimeError("source expected-byte accounting changed")

    records_by_trigger = sorted(records, key=lambda item: int(item["trigger_ns"]))
    head_of_line = any(
        int(previous["finished_ns"]) > int(current["trigger_ns"])
        for previous, current in zip(records_by_trigger, records_by_trigger[1:])
    )
    peak_pending = int(block["peak_pending_batches"])
    block.update(
        {
            "registered_nixl_descriptors": registered,
            "logical_background_batch_calls": expected_logical_batches,
            "physical_nixl_calls": physical_calls,
            "physical_transfer_descriptors": physical_descriptors,
            "physical_transfer_bytes": physical_descriptors * QUANTUM_BYTES,
            "expected_physical_nixl_calls": expected_physical_calls,
            "expected_physical_transfer_descriptors": expected_physical_calls,
            "head_of_line_queue_detected": bool(head_of_line or peak_pending > 1),
            "all_calls_completed_before_next_trigger": not head_of_line,
        }
    )
    block["coalesced_calls_per_source"] = expected_physical_calls
    return block


def _run_block(*args: Any, **kwargs: Any) -> dict[str, Any]:
    channel = kwargs["channel"]
    event_start = len(channel.tempo_physical_write_events)
    result = _v3_run_block(*args, **kwargs)
    write_events = list(channel.tempo_physical_write_events[event_start:])
    return _decorate_quantum_block(
        result,
        channel=channel,
        rank=int(kwargs["rank"]),
        requested_mode=str(kwargs["mode"]),
        write_events=write_events,
    )


def _decorate_quantum_result(
    result: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: int(item["rank"]))
    if len(ordered) != v1.WORLD_SIZE:
        raise ValueError("descriptor aggregation requires all sixteen ranks")
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    result["candidate_policy"] = POLICY
    result["descriptor_geometry"] = dict(DESCRIPTOR_GEOMETRY)
    result["config"]["descriptor_geometry"] = dict(DESCRIPTOR_GEOMETRY)
    result["background"].update(
        {
            "physical_operation": "eight_sequential_or_token_admitted_2mib_nixl_writes_per_source",
            "registered_descriptors_per_rank": REGISTERED_DESCRIPTORS_PER_RANK,
            "physical_calls_per_source": PHYSICAL_CALLS_PER_SOURCE,
            "physical_calls_global": PHYSICAL_CALLS_GLOBAL,
        }
    )
    result["coalesced_contract"].update(
        {
            "scheduled_tokens": list(SCHEDULED_TOKENS),
            "source_scheduled_tokens": [
                list(source_scheduled_tokens(pair)) for pair in range(v1.PAIR_COUNT)
            ],
            "calls_per_source": PHYSICAL_CALLS_PER_SOURCE,
            "source_calls_global": PHYSICAL_CALLS_GLOBAL,
            "registered_descriptors_per_rank": REGISTERED_DESCRIPTORS_PER_RANK,
            "physical_descriptors_global": PHYSICAL_CALLS_GLOBAL,
        }
    )
    result["frozen_group2"].update(
        {
            "policy_label": POLICY,
            "contract_id": CONTRACT_ID,
            "scheduled_tokens": list(SCHEDULED_TOKENS),
        }
    )

    if len(result["blocks"]) != len(ordered[0]["blocks"]):
        raise ValueError("aggregate and rank block counts differ")
    for block_index, block in enumerate(result["blocks"]):
        rank_blocks = [item["blocks"][block_index] for item in ordered]
        source_blocks = rank_blocks[: v1.SOURCE_COUNT]
        registered_counts = [
            int(item["registered_nixl_descriptors"]) for item in rank_blocks
        ]
        if registered_counts != [REGISTERED_DESCRIPTORS_PER_RANK] * v1.WORLD_SIZE:
            raise ValueError("rank registered-descriptor counts changed")
        background = block["mode"] != "fg_only"
        expected_global = PHYSICAL_CALLS_GLOBAL if background else 0
        physical_calls = sum(int(item["physical_nixl_calls"]) for item in source_blocks)
        physical_descriptors = sum(
            int(item["physical_transfer_descriptors"]) for item in source_blocks
        )
        if physical_calls != expected_global or physical_descriptors != expected_global:
            raise ValueError("global physical call/descriptor total changed")
        expected_bytes = v1.GLOBAL_BYTES if background else 0
        if int(block["background_completed_bytes"]) != expected_bytes:
            raise ValueError("global completed-byte total changed")
        if int(block["receiver_verified_bytes"]) != expected_bytes:
            raise ValueError("global receiver-verified byte total changed")
        peak_pending = max(int(item["peak_pending_batches"]) for item in source_blocks)
        block.update(
            {
                "registered_nixl_descriptors_per_rank": REGISTERED_DESCRIPTORS_PER_RANK,
                "background_source_calls": expected_global,
                "expected_source_calls_global": expected_global,
                "source_calls_global": physical_calls,
                "physical_nixl_calls_global": physical_calls,
                "physical_transfer_descriptors_global": physical_descriptors,
                "physical_background_bytes": expected_bytes,
                "peak_pending_batches": peak_pending,
                "source_peak_pending_batches": [
                    int(item["peak_pending_batches"]) for item in source_blocks
                ],
                "head_of_line_queue_detected": any(
                    bool(item["head_of_line_queue_detected"]) for item in source_blocks
                ),
                "all_calls_completed_before_next_trigger": all(
                    bool(item["all_calls_completed_before_next_trigger"])
                    for item in source_blocks
                ),
            }
        )
    return result


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return _decorate_quantum_result(_v3_aggregate(records), records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("eval/sota_4node/real_tp16_pair_quantum2mib_v6.json"),
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
    _v3_install(campaign_index)
    v1.CONTRACT_ID = CONTRACT_ID
    v1.CONTRACT_SCHEMA = CONTRACT_SCHEMA
    v1.POLICY = POLICY
    v1.SCHEDULED_TOKENS = SCHEDULED_TOKENS
    v1.coalesced_indices = quantum_indices
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
    base.epoch._make_chunk_memory = _make_quantum2mib_memory
    base.official._load_official_lmcache = _load_quantum2mib_lmcache


def main() -> None:
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
