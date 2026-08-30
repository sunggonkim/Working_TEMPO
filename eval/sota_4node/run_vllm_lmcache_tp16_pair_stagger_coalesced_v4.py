#!/usr/bin/env python3
"""True single-descriptor revision of the four-node TP16 campaign.

The v1-v3 data path kept 32 logical 512 KiB objects per source.  Although
those objects were submitted by one ``batched_write`` call, LMCache's pinned
``NixlAgentWrapper`` created one NIXL transfer descriptor per ``align_bytes``
page, so the physical operation still contained 32 descriptors.

This revision keeps the logical chunk views for deterministic initialization
and receiver verification, but registers the 16 MiB buffer as exactly one
NIXL transfer descriptor.  Every physical write passed to the official
LMCache implementation contains one whole-buffer ``TensorMemoryObj`` and
remote descriptor index zero.  Existing logical accounting remains unchanged:
eight sources each move and verify exactly 16 MiB, or 128 MiB globally.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from eval.sota_4node import run_vllm_lmcache_tp16_pair_stagger_coalesced_v3 as v3


v2 = v3.v2
v1 = v3.v1

CONTRACT_ID = "real-tp16-pair-stagger-coalesced-v4"
CONTRACT_SCHEMA = "tempo-real-tp16-pair-stagger-coalesced-contract-4"
RESULT_SCHEMA = "tempo-vllm-tp16-lmcache-pair-stagger-coalesced-screen-4"
POLICY = "tp16_pair_staggered_contiguous_descriptor_admission_v4"

LOGICAL_CHUNKS_PER_SOURCE = v1.REQUESTS * v1.CHUNKS_PER_REQUEST
DESCRIPTOR_BYTES = v1.BYTES_PER_SOURCE
PHYSICAL_DESCRIPTORS_PER_SOURCE_CALL = 1
PHYSICAL_DESCRIPTORS_GLOBAL = v1.SOURCE_COUNT
REMOTE_DESCRIPTOR_INDEX = 0

DESCRIPTOR_GEOMETRY = {
    "registered_buffer_bytes_per_rank": DESCRIPTOR_BYTES,
    "registered_buffer_alignment_bytes": DESCRIPTOR_BYTES,
    "nixl_transfer_descriptors_per_rank": PHYSICAL_DESCRIPTORS_PER_SOURCE_CALL,
    "nixl_transfer_descriptor_bytes": DESCRIPTOR_BYTES,
    "official_batched_write_objects": 1,
    "local_descriptor_indexes": [REMOTE_DESCRIPTOR_INDEX],
    "remote_descriptor_indexes": [REMOTE_DESCRIPTOR_INDEX],
    "logical_verification_chunks_per_rank": LOGICAL_CHUNKS_PER_SOURCE,
    "logical_chunk_bytes": v1.CHUNK_BYTES,
    "physical_source_descriptors_global": PHYSICAL_DESCRIPTORS_GLOBAL,
    "physical_bytes_global": v1.GLOBAL_BYTES,
}

_v3_install = v3._install
_v3_aggregate = v3.aggregate_rank_records
_v3_run_block = v3._run_block
_official_loader = v1.base.official._load_official_lmcache


def _expected_contract() -> dict[str, Any]:
    payload = v3._expected_contract()
    payload["schema_version"] = CONTRACT_SCHEMA
    payload["contract_id"] = CONTRACT_ID
    payload["provenance"]["policy_label"] = POLICY
    payload["schedule"].update(
        {
            "logical_chunks_per_source_call": LOGICAL_CHUNKS_PER_SOURCE,
            "physical_descriptors_per_source_call": (
                PHYSICAL_DESCRIPTORS_PER_SOURCE_CALL
            ),
            "physical_descriptors_global": PHYSICAL_DESCRIPTORS_GLOBAL,
        }
    )
    payload["descriptor_geometry"] = dict(DESCRIPTOR_GEOMETRY)
    return payload


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid TP16 single-descriptor v4 contract: {exc}") from exc
    if payload != _expected_contract():
        raise ValueError("TP16 single-descriptor v4 contract changed")
    v1.validate_schedule()
    return payload, CONTRACT_ID


def _make_single_descriptor_memory(
    torch: Any,
    TensorMemoryObj: Any,
    MemoryObjMetadata: Any,
    MemoryFormat: Any,
    *,
    requests: int,
    chunk_bytes: int,
) -> tuple[Any, Any, list[Any], dict[int, int]]:
    """Build 32 logical views plus one attached 16 MiB transfer object.

    Only the logical views are returned in ``objects`` so the inherited
    initializer and receiver verifier keep their exact per-chunk byte checks.
    Each view retains the whole-buffer object through a private attribute used
    by :func:`_single_descriptor_channel_class` at write time.
    """

    logical_count = requests * v1.CHUNKS_PER_REQUEST
    total_bytes = logical_count * chunk_bytes
    if (
        logical_count != LOGICAL_CHUNKS_PER_SOURCE
        or chunk_bytes != v1.CHUNK_BYTES
        or total_bytes != DESCRIPTOR_BYTES
    ):
        raise RuntimeError("single-descriptor TP16 memory geometry changed")

    # Over-allocate once so the registered buffer can be aligned to the exact
    # transfer-descriptor size.  zeros() also makes the unmeasured full-buffer
    # warmup deterministic outside the two logical chunks it touches.
    backing = torch.zeros(
        total_bytes + DESCRIPTOR_BYTES - 1,
        dtype=torch.uint8,
        device="cuda",
    )
    offset = (-backing.data_ptr()) % DESCRIPTOR_BYTES
    buffer = backing[offset : offset + total_bytes]
    if buffer.numel() != DESCRIPTOR_BYTES:
        raise RuntimeError("single-descriptor buffer has the wrong byte length")
    if buffer.data_ptr() % DESCRIPTOR_BYTES != 0:
        raise RuntimeError("single-descriptor buffer is not 16 MiB aligned")

    whole_shape = torch.Size([DESCRIPTOR_BYTES])
    whole_object = TensorMemoryObj(
        raw_data=buffer,
        metadata=MemoryObjMetadata(
            shape=whole_shape,
            dtype=torch.uint8,
            address=buffer.data_ptr(),
            phy_size=DESCRIPTOR_BYTES,
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.BINARY,
            shapes=[whole_shape],
            dtypes=[torch.uint8],
        ),
        parent_allocator=None,
    )

    objects: list[Any] = []
    index_by_address: dict[int, int] = {}
    for index in range(logical_count):
        raw = buffer[index * chunk_bytes : (index + 1) * chunk_bytes]
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
        # TensorMemoryObj is a regular Python class at the pinned checkout.
        # Holding the whole object here also keeps it alive for every write.
        logical_object._tempo_whole_transfer_object = whole_object
        objects.append(logical_object)
        index_by_address[raw.data_ptr()] = index

    # The whole object begins at the same address as logical chunk zero.  The
    # inherited descriptor-index shim therefore resolves it to descriptor 0.
    if index_by_address.get(whole_object.meta.address) != REMOTE_DESCRIPTOR_INDEX:
        raise RuntimeError("whole-buffer object does not map to descriptor zero")
    return backing, buffer, objects, index_by_address


def _single_descriptor_channel_class(base_channel: Any) -> Any:
    """Wrap official NixlChannel while preserving its public return contract."""

    class SingleDescriptorNixlChannel(base_channel):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            buffer_ptr = int(kwargs.get("buffer_ptr", -1))
            buffer_size = int(kwargs.get("buffer_size", -1))
            if buffer_size != DESCRIPTOR_BYTES:
                raise RuntimeError("NIXL registered buffer must be exactly 16 MiB")
            if buffer_ptr < 0 or buffer_ptr % DESCRIPTOR_BYTES != 0:
                raise RuntimeError("NIXL registered buffer must be 16 MiB aligned")
            kwargs["align_bytes"] = DESCRIPTOR_BYTES
            super().__init__(*args, **kwargs)
            descriptor_count = len(self.nixl_wrapper.xfer_descs)
            if descriptor_count != PHYSICAL_DESCRIPTORS_PER_SOURCE_CALL:
                raise RuntimeError(
                    "official LMCache did not create exactly one NIXL descriptor"
                )
            self.tempo_nixl_transfer_descriptor_count = descriptor_count
            self.tempo_physical_write_calls = 0
            self.tempo_last_logical_object_count = 0
            self.tempo_last_remote_descriptor_indexes: list[int] = []

        def batched_write(
            self,
            objects: list[Any],
            transfer_spec: dict[str, Any] | None = None,
        ) -> int:
            if transfer_spec is None:
                raise ValueError("single-descriptor write requires transfer_spec")
            if not objects:
                raise ValueError("single-descriptor write requires logical objects")
            logical_count = len(objects)
            remote_indexes = np.asarray(
                transfer_spec.get("remote_indexes", ()), dtype=np.uint64
            )
            if len(remote_indexes) != logical_count:
                raise RuntimeError("logical objects and remote indexes differ")
            whole_objects = {
                id(getattr(item, "_tempo_whole_transfer_object", None))
                for item in objects
            }
            if len(whole_objects) != 1 or id(None) in whole_objects:
                raise RuntimeError("logical chunks do not share one transfer object")
            whole_object = objects[0]._tempo_whole_transfer_object
            collapsed_spec = dict(transfer_spec)
            collapsed_spec["remote_indexes"] = np.asarray(
                [REMOTE_DESCRIPTOR_INDEX], dtype=np.uint64
            )
            physical_count = int(
                super().batched_write(
                    objects=[whole_object],
                    transfer_spec=collapsed_spec,
                )
            )
            if physical_count != PHYSICAL_DESCRIPTORS_PER_SOURCE_CALL:
                raise RuntimeError("official single-descriptor write count changed")
            self.tempo_physical_write_calls += 1
            self.tempo_last_logical_object_count = logical_count
            self.tempo_last_remote_descriptor_indexes = [REMOTE_DESCRIPTOR_INDEX]
            # The inherited experiment accounts verified logical chunks.  The
            # official call above has already completed one 16 MiB descriptor.
            return logical_count

    SingleDescriptorNixlChannel.__name__ = "SingleDescriptorNixlChannel"
    SingleDescriptorNixlChannel.__qualname__ = "SingleDescriptorNixlChannel"
    return SingleDescriptorNixlChannel


def _load_single_descriptor_lmcache(repo_root: Path) -> tuple[Any, Any, Any, Any]:
    channel, tensor_obj, metadata, memory_format = _official_loader(repo_root)
    return (
        _single_descriptor_channel_class(channel),
        tensor_obj,
        metadata,
        memory_format,
    )


def _decorate_single_descriptor_block(
    block: dict[str, Any],
    *,
    channel: Any,
    rank: int,
    requested_mode: str,
) -> dict[str, Any]:
    descriptor_count = int(channel.tempo_nixl_transfer_descriptor_count)
    if descriptor_count != PHYSICAL_DESCRIPTORS_PER_SOURCE_CALL:
        raise RuntimeError("runtime NIXL descriptor count changed")
    is_source = rank < v1.SOURCE_COUNT
    background = requested_mode != "fg_only"
    expected_calls = 1 if is_source and background else 0
    records = block.get("transfer_records", [])
    if len(records) != expected_calls:
        raise RuntimeError("single-descriptor block call count changed")
    for record in records:
        if int(record["completed_objects"]) != LOGICAL_CHUNKS_PER_SOURCE:
            raise RuntimeError("logical chunk completion count changed")
        if list(record["object_indices"]) != list(range(LOGICAL_CHUNKS_PER_SOURCE)):
            raise RuntimeError("logical transfer coverage changed")
        if channel.tempo_last_logical_object_count != LOGICAL_CHUNKS_PER_SOURCE:
            raise RuntimeError("official write did not collapse the full logical batch")
        if channel.tempo_last_remote_descriptor_indexes != [REMOTE_DESCRIPTOR_INDEX]:
            raise RuntimeError("official write did not target remote descriptor zero")
        record.update(
            {
                "logical_chunk_objects": LOGICAL_CHUNKS_PER_SOURCE,
                "official_batched_write_objects": 1,
                "physical_transfer_descriptors": 1,
                "physical_transfer_bytes": DESCRIPTOR_BYTES,
                "local_descriptor_indexes": [REMOTE_DESCRIPTOR_INDEX],
                "remote_descriptor_indexes": [REMOTE_DESCRIPTOR_INDEX],
                "contiguous_single_descriptor": True,
            }
        )
    block["physical_transfer_calls"] = expected_calls
    block["physical_transfer_descriptors"] = expected_calls
    block["physical_transfer_bytes"] = expected_calls * DESCRIPTOR_BYTES
    block["contiguous_single_descriptor"] = bool(expected_calls)
    return block


def _run_block(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = _v3_run_block(*args, **kwargs)
    return _decorate_single_descriptor_block(
        result,
        channel=kwargs["channel"],
        rank=int(kwargs["rank"]),
        requested_mode=str(kwargs["mode"]),
    )


def _decorate_result(result: dict[str, Any]) -> dict[str, Any]:
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    result["candidate_policy"] = POLICY
    result["descriptor_geometry"] = dict(DESCRIPTOR_GEOMETRY)
    result["config"]["descriptor_geometry"] = dict(DESCRIPTOR_GEOMETRY)
    result["background"].update(
        {
            "physical_operation": "one_contiguous_16mib_nixl_descriptor_per_source",
            "official_batched_write_objects_per_source_call": 1,
            "physical_descriptors_per_source_call": 1,
        }
    )
    result["coalesced_contract"].update(
        {
            "logical_chunks_per_source_call": LOGICAL_CHUNKS_PER_SOURCE,
            "physical_descriptors_per_source_call": 1,
            "physical_descriptors_global": PHYSICAL_DESCRIPTORS_GLOBAL,
        }
    )
    result["frozen_group2"].update(
        {
            "policy_label": POLICY,
            "contract_id": CONTRACT_ID,
        }
    )
    for block in result["blocks"]:
        background = block["mode"] != "fg_only"
        expected_descriptors = PHYSICAL_DESCRIPTORS_GLOBAL if background else 0
        block["physical_source_descriptors_global"] = expected_descriptors
        block["physical_background_bytes"] = v1.GLOBAL_BYTES if background else 0
        if background:
            if int(block["background_completed_bytes"]) != v1.GLOBAL_BYTES:
                raise ValueError("single-descriptor completed byte total changed")
            if int(block["receiver_verified_bytes"]) != v1.GLOBAL_BYTES:
                raise ValueError("single-descriptor verified byte total changed")
    return result


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return _decorate_result(_v3_aggregate(records))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("eval/sota_4node/real_tp16_pair_stagger_coalesced_v4.json"),
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
    v1.load_contract = load_contract
    v1.aggregate_rank_records = aggregate_rank_records
    v1._run_block = _run_block
    v1.base.EXPECTED_PLAN_SIGNATURE = CONTRACT_ID
    v1.base.load_frozen_plan = load_contract
    v1.base.aggregate_rank_records = aggregate_rank_records
    v1.base._run_block = _run_block
    v1.base.epoch._make_chunk_memory = _make_single_descriptor_memory
    v1.base.official._load_official_lmcache = _load_single_descriptor_lmcache


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
