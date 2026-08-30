#!/usr/bin/env python3
"""Identical-path 4-node checkpoint comparison for TEMPO and open baselines."""

from __future__ import annotations

import argparse
import atexit
import csv
import datetime as dt
import functools
import hashlib
import importlib.util
import json
import math
import os
import queue
import random
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import msgpack
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.tensor import DTensor

import tempo as _tempo_runtime_package
from tempo import group_credit_checkpoint as _group_credit_checkpoint_module
from tempo import split_guard as _split_guard_module


GroupCreditCheckpointer = _group_credit_checkpoint_module.GroupCreditCheckpointer
D2HCausalGuard = _split_guard_module.D2HCausalGuard
NodePFSLane = _split_guard_module.NodePFSLane
transition_deltas = _split_guard_module.transition_deltas


POLICIES = (
    "none",
    "torch_async",
    "torchsnapshot",
    "datastates",
    "tempo",
    "tempo_v2",
    "tempo_v3",
    "v4_open",
    "tempo_v4",
)

TIER_MODES = (
    "fg_only",
    "open_combined",
    "d2h_only",
    "persist_only",
    "combined",
)

MIB = 1 << 20
UINT64_MAX = (1 << 64) - 1
V4_D2H_REQUEST_MIB = 1
V4_PAYLOAD_REGION_MIB = 4
V4_PFS_REQUEST_MIB = 4
# A full-compute candidate may issue several <=1 MiB D2H subrequests while
# one non-installable compute interval is active. This is a cumulative phase
# budget, not a larger physical request and never applies to a collective
# boundary. Four payload regions let the host producer get ahead of PFS on
# multi-node runs; the C++ worker still allocates/fills each <=4 MiB region
# serially and the hard physical request bound remains 1 MiB.
V4_FULL_COMPUTE_D2H_BURST_MIB = 4 * V4_PAYLOAD_REGION_MIB
V4_MAX_COLLECTIVE_D2H_REQUESTS = 16
V4_MAX_COLLECTIVE_PFS_REQUESTS = 4
V4_MAX_COLLECTIVE_D2H_CREDIT_BYTES = (
    V4_MAX_COLLECTIVE_D2H_REQUESTS * V4_D2H_REQUEST_MIB * MIB
)
V4_MAX_COLLECTIVE_PFS_CREDIT_BYTES = (
    V4_MAX_COLLECTIVE_PFS_REQUESTS * V4_PFS_REQUEST_MIB * MIB
)
V4_COMPUTE_REALIZATION_HISTORY = 8
V4_COMPUTE_REALIZATION_PERCENTILE_PPM = 250_000
V4_CONTROLLER_PACKET_SCHEMA = (
    "tempo-v4-controller-packet-compact-profile-deterministic-msgpack-sha256-3"
)
V4_CONTROLLER_PACKET_SCHEMA_ID = 3
V4_CONTROLLER_PROFILE_SCHEMA = "tempo-v4-profile-sufficient-statistics-1"
V4_PROFILE_OPERATIONS = ("all_gather_into_tensor", "reduce_scatter_tensor")
V4_PROFILE_ARRIVAL_SOURCES = (
    "planned_lead_in",
    "stream_ordered_lead_in",
    "stream_ordered_zero_compute",
    "previous_execution_hold",
    "skipped_noninstallable_lead_in",
    "finalize_no_data_work",
    "deadline_drain",
    "",
)
V4_NONINSTALLABLE_ARRIVAL_SOURCES = frozenset(
    {
        "stream_ordered_zero_compute",
        "previous_execution_hold",
        "skipped_noninstallable_lead_in",
    }
)
V4_CONTROLLER_MAX_PACKET_BYTES = 32 * 1024
V4_CONTROLLER_PACKET_MAGIC = b"TMPV4PKT"
V4_CONTROLLER_PACKET_CODEC = "msgpack"
V4_CONTROLLER_PACKET_CODEC_VERSION = "1.1.1"
V4_CONTROLLER_PACKET_CANONICALIZATION = (
    "recursive_utf8_key_sort_unique_decode"
)
if getattr(msgpack, "__version__", "") != V4_CONTROLLER_PACKET_CODEC_VERSION:
    raise RuntimeError(
        "TEMPO v4 requires exact msgpack==1.1.1, found "
        f"{getattr(msgpack, '__version__', '<unknown>')}"
    )
# Network byte order makes the fixed header independent of host endianness:
# magic, uint32 schema/header/payload/rank, uint64 step, SHA-256(payload).
V4_CONTROLLER_PACKET_HEADER = struct.Struct(">8sIIIIQ32s")
V4_STAGE_CALIBRATION_SCHEMA = "tempo-v4-stage-calibration-selection-1"
V4_STAGE_FLOOR_PROVENANCE_SCHEMA = "tempo-v4-stage-floor-provenance-1"
V4_ARCHIVE_D2H_SEED_BPS = 5_936_536_675
V4_ARCHIVE_PFS_SEED_BPS = 2_211_348_539
V4_STAGE_CALIBRATION_FILENAME = "stage_service_selection.json"
V4_STAGE_CALIBRATION_HASH_FILENAME = "stage_service_selection.sha256"
_V4_CONTROLLER_MODULE: Any | None = None
_V4_CONTROLLER_PATH: Path | None = None
_V4_CONTROLLER_SHA256 = ""


class V4ControllerPacketError(RuntimeError):
    """A controller packet failed deterministic wire validation."""


def _v4_exact_json_int(mapping: dict[str, Any], key: str, *, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{label} field {key!r} is not a non-negative integer")
    return value


def _v4_configured_stage_floor_bps(args: argparse.Namespace) -> tuple[int, int]:
    return (
        round(float(args.tempo_v4_d2h_floor_gbps) * 1e9),
        round(float(args.tempo_v4_pfs_floor_gbps) * 1e9),
    )


def _v4_open_c0_rate_bps(args: argparse.Namespace) -> int:
    if str(args.policy) != "v4_open" or not bool(getattr(args, "v4_open_c0", False)):
        return 0
    return round(float(args.tempo_v4_d2h_floor_gbps) * 1e9)


def load_v4_stage_floor_provenance(
    args: argparse.Namespace,
    output_dir: Path,
    world_size: int,
) -> dict[str, Any]:
    """Bind v4's configured rates to the one-shot matched calibration.

    Unit tests may instantiate the adapter without the harness environment; those
    records are explicitly marked ``cli_unverified`` and can never pass the
    production analyzer.  A partially configured environment is always fatal.
    """

    configured_d2h_bps, configured_pfs_bps = _v4_configured_stage_floor_bps(args)
    selection_name = os.environ.get("TEMPO_V4_CALIBRATION_SELECTION_JSON", "")
    selection_sha256 = os.environ.get("TEMPO_V4_CALIBRATION_SHA256", "")
    if not selection_name and not selection_sha256:
        return {
            "schema_version": V4_STAGE_FLOOR_PROVENANCE_SCHEMA,
            "source": "cli_unverified",
            "selection_schema_version": "",
            "selection_path": "",
            "selection_file_sha256": "",
            "selected_d2h_bps": configured_d2h_bps,
            "selected_pfs_bps": configured_pfs_bps,
            "archive_cap_d2h_bps": V4_ARCHIVE_D2H_SEED_BPS,
            "archive_cap_pfs_bps": V4_ARCHIVE_PFS_SEED_BPS,
            "group_min_raw_d2h_bps": 0,
            "group_min_full_pipeline_bps": 0,
            "haircut_numerator": 0,
            "haircut_denominator": 0,
            "never_raise": True,
            "hard_gate": "unverified_cli_only",
            "calibration_consumers": ["v4_open", "tempo_v4"],
            "baseline_consumers": [],
        }
    if not selection_name or not selection_sha256:
        raise RuntimeError(
            "TEMPO v4 calibration path and SHA-256 must either both be set or both be absent"
        )
    if len(selection_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in selection_sha256
    ):
        raise RuntimeError("TEMPO v4 calibration SHA-256 is not 64 lowercase hex digits")

    selection_path = Path(selection_name).resolve(strict=True)
    expected_path = (
        output_dir.resolve(strict=True).parent / V4_STAGE_CALIBRATION_FILENAME
    ).resolve(strict=True)
    if selection_path != expected_path:
        raise RuntimeError(
            "TEMPO v4 calibration is not the job result's canonical selection: "
            f"actual={selection_path} expected={expected_path}"
        )
    raw = selection_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != selection_sha256:
        raise RuntimeError(
            "TEMPO v4 calibration file hash differs from the harness selection: "
            f"actual={actual_sha256} expected={selection_sha256}"
        )
    sidecar_path = selection_path.with_name(V4_STAGE_CALIBRATION_HASH_FILENAME)
    try:
        sidecar = sidecar_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("TEMPO v4 calibration SHA-256 sidecar is unreadable") from exc
    if sidecar != selection_sha256 + "\n":
        raise RuntimeError("TEMPO v4 calibration SHA-256 sidecar differs")
    try:
        selection = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("TEMPO v4 calibration selection is malformed JSON") from exc
    if not isinstance(selection, dict):
        raise RuntimeError("TEMPO v4 calibration selection is not a JSON object")
    try:
        canonical = (
            json.dumps(
                selection,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("TEMPO v4 calibration selection is not canonicalizable") from exc
    if raw != canonical:
        raise RuntimeError("TEMPO v4 calibration selection bytes are not canonical")

    if selection.get("schema_version") != V4_STAGE_CALIBRATION_SCHEMA:
        raise RuntimeError("TEMPO v4 calibration selection schema differs")
    if selection.get("passed") is not True or selection.get("invariant_errors") != []:
        raise RuntimeError("TEMPO v4 calibration did not pass its invariants")
    if (
        _v4_exact_json_int(selection, "expected_world_size", label="calibration")
        != world_size
        or _v4_exact_json_int(selection, "rank_count", label="calibration")
        != world_size
    ):
        raise RuntimeError("TEMPO v4 calibration rank count differs from this run")

    method = selection.get("method")
    archive = selection.get("archive_caps")
    observed = selection.get("observed")
    rates = selection.get("down_only_selection")
    deadline = selection.get("deadline_feasibility")
    geometry = selection.get("geometry")
    if not all(isinstance(value, dict) for value in (method, archive, observed, rates, deadline, geometry)):
        raise RuntimeError("TEMPO v4 calibration selection sections are incomplete")
    assert isinstance(method, dict)
    assert isinstance(archive, dict)
    assert isinstance(observed, dict)
    assert isinstance(rates, dict)
    assert isinstance(deadline, dict)
    assert isinstance(geometry, dict)
    if (
        method.get("high_level_api") != "datastates.CheckpointEngine.save"
        or method.get("barrier_synchronized") is not True
        or method.get("wait_sequence") != [False, True]
        or method.get("attempts_per_rank") != 1
        or method.get("warmups_per_rank") != 0
        or method.get("retries_per_rank") != 0
    ):
        raise RuntimeError("TEMPO v4 calibration method is not the matched one-shot path")
    archive_d2h = _v4_exact_json_int(archive, "d2h_bps", label="archive cap")
    archive_pfs = _v4_exact_json_int(archive, "pfs_bps", label="archive cap")
    if (archive_d2h, archive_pfs) != (
        V4_ARCHIVE_D2H_SEED_BPS,
        V4_ARCHIVE_PFS_SEED_BPS,
    ):
        raise RuntimeError("TEMPO v4 calibration archive caps differ from the frozen caps")
    group_min_d2h = _v4_exact_json_int(
        observed, "group_min_raw_d2h_rate_bps", label="observed calibration"
    )
    group_min_pfs = _v4_exact_json_int(
        observed, "group_min_full_pipeline_rate_bps", label="observed calibration"
    )
    haircut_numerator = _v4_exact_json_int(
        rates, "haircut_numerator", label="calibration selection"
    )
    haircut_denominator = _v4_exact_json_int(
        rates, "haircut_denominator", label="calibration selection"
    )
    selected_d2h = _v4_exact_json_int(
        rates, "selected_d2h_bps", label="calibration selection"
    )
    selected_pfs = _v4_exact_json_int(
        rates, "selected_pfs_bps", label="calibration selection"
    )
    if (haircut_numerator, haircut_denominator) != (9, 10):
        raise RuntimeError("TEMPO v4 calibration haircut differs from 9/10")
    if (
        selected_d2h
        != min(archive_d2h, group_min_d2h * haircut_numerator // haircut_denominator)
        or selected_pfs
        != min(archive_pfs, group_min_pfs * haircut_numerator // haircut_denominator)
    ):
        raise RuntimeError("TEMPO v4 selected rates do not reproduce the down-only rule")
    if rates.get("never_raise") is not True:
        raise RuntimeError("TEMPO v4 calibration is not marked down-only")
    if rates.get("consumers") != ["v4_open", "tempo_v4"] or rates.get(
        "baseline_consumers"
    ) != []:
        raise RuntimeError("TEMPO v4 calibration consumers differ")
    if (selected_d2h, selected_pfs) != (configured_d2h_bps, configured_pfs_bps):
        raise RuntimeError(
            "TEMPO v4 CLI stage rates differ from the calibrated selection: "
            f"configured=({configured_d2h_bps},{configured_pfs_bps}) "
            f"selected=({selected_d2h},{selected_pfs})"
        )
    if deadline.get("hard_gate") != "no_drain_thresholds":
        raise RuntimeError("TEMPO v4 calibration hard gate differs")
    if selected_d2h < _v4_exact_json_int(
        deadline, "d2h_no_drain_threshold_bps", label="deadline feasibility"
    ) or selected_pfs < _v4_exact_json_int(
        deadline, "pfs_no_drain_threshold_bps", label="deadline feasibility"
    ):
        raise RuntimeError("TEMPO v4 calibrated rate is below the no-DRAIN threshold")
    expected_geometry = {
        "d2h_request_bytes": int(args.tempo_v4_d2h_chunk_mb) * MIB,
        "payload_region_bytes": V4_PAYLOAD_REGION_MIB * MIB,
        "pfs_request_bytes": int(args.tempo_v4_pfs_chunk_mb) * MIB,
        "max_pfs_inflight_bytes": int(args.tempo_v4_max_pfs_inflight_mb) * MIB,
        "max_pfs_inflight_requests": (
            int(args.tempo_v4_max_pfs_inflight_mb)
            // int(args.tempo_v4_pfs_chunk_mb)
        ),
        "odirect_required": True,
    }
    if any(geometry.get(key) != value for key, value in expected_geometry.items()):
        raise RuntimeError("TEMPO v4 calibration geometry differs from the measured run")

    return {
        "schema_version": V4_STAGE_FLOOR_PROVENANCE_SCHEMA,
        "source": "matched_stage_calibration",
        "selection_schema_version": V4_STAGE_CALIBRATION_SCHEMA,
        "selection_path": str(selection_path),
        "selection_file_sha256": selection_sha256,
        "selected_d2h_bps": selected_d2h,
        "selected_pfs_bps": selected_pfs,
        "archive_cap_d2h_bps": archive_d2h,
        "archive_cap_pfs_bps": archive_pfs,
        "group_min_raw_d2h_bps": group_min_d2h,
        "group_min_full_pipeline_bps": group_min_pfs,
        "haircut_numerator": haircut_numerator,
        "haircut_denominator": haircut_denominator,
        "never_raise": True,
        "hard_gate": "no_drain_thresholds",
        "calibration_consumers": ["v4_open", "tempo_v4"],
        "baseline_consumers": [],
    }


class V4ControllerPacketCodec:
    """Deterministically sorted MessagePack in one fixed-size CPU frame."""

    SCHEMA = V4_CONTROLLER_PACKET_SCHEMA
    SCHEMA_ID = V4_CONTROLLER_PACKET_SCHEMA_ID
    HEADER = V4_CONTROLLER_PACKET_HEADER
    MAX_PACKET_BYTES = V4_CONTROLLER_MAX_PACKET_BYTES
    MAX_PAYLOAD_BYTES = MAX_PACKET_BYTES - HEADER.size
    CODEC = V4_CONTROLLER_PACKET_CODEC
    CODEC_VERSION = V4_CONTROLLER_PACKET_CODEC_VERSION
    CANONICALIZATION = V4_CONTROLLER_PACKET_CANONICALIZATION
    MAX_ARRAY_ITEMS = 4_096
    MAX_MAP_ITEMS = 4_096
    MAX_NESTING_DEPTH = 64
    _ZERO_FRAME = bytes(MAX_PACKET_BYTES)
    _ENVELOPE_KEYS = frozenset(("packet", "rank", "schema", "step"))

    @classmethod
    def _canonical_value(cls, value: Any, *, depth: int = 0) -> Any:
        """Validate the JSON-compatible domain and sort maps in place.

        The common controller semantics intentionally exclude MessagePack-only
        binary, extension, tuple, and non-string-map-key values.  Reordering a
        dict changes insertion order only, not mapping semantics, and avoids a
        full object-graph copy on every scheduled rendezvous.
        """

        if depth > cls.MAX_NESTING_DEPTH:
            raise V4ControllerPacketError(
                f"controller packet nesting exceeds {cls.MAX_NESTING_DEPTH}"
            )
        if value is None:
            return None
        value_type = type(value)
        if value_type is bool or value_type is str:
            return value
        if value_type is int:
            if value < -(1 << 63) or value > UINT64_MAX:
                raise V4ControllerPacketError(
                    "controller packet integer is outside int64/uint64"
                )
            return value
        if value_type is float:
            if not math.isfinite(value):
                raise V4ControllerPacketError("controller packet float is non-finite")
            return value
        if value_type is list:
            if len(value) > cls.MAX_ARRAY_ITEMS:
                raise V4ControllerPacketError(
                    "controller packet array exceeds the item limit"
                )
            for item in value:
                cls._canonical_value(item, depth=depth + 1)
            return value
        if value_type is dict:
            if len(value) > cls.MAX_MAP_ITEMS:
                raise V4ControllerPacketError(
                    "controller packet map exceeds the item limit"
                )
            previous_key: str | None = None
            requires_sort = False
            for key, item in value.items():
                if type(key) is not str:
                    raise V4ControllerPacketError(
                        "controller packet map key is not a string"
                    )
                if previous_key is not None and key <= previous_key:
                    requires_sort = True
                previous_key = key
                cls._canonical_value(item, depth=depth + 1)
            if requires_sort:
                ordered_items = sorted(value.items(), key=lambda item: item[0])
                value.clear()
                value.update(ordered_items)
            return value
        raise V4ControllerPacketError(
            "controller packet has unsupported MessagePack value "
            f"{type(value).__name__}"
        )

    @classmethod
    def _canonical_msgpack(cls, value: Any) -> bytes:
        canonical = cls._canonical_value(value)
        try:
            return msgpack.packb(
                canonical,
                use_bin_type=True,
                strict_types=True,
                use_single_float=False,
            )
        except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
            raise V4ControllerPacketError(
                f"controller packet canonical MessagePack encoding failed: {exc}"
            ) from exc

    @staticmethod
    def _require_index(value: Any, name: str, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise V4ControllerPacketError(f"controller packet {name} is not an integer")
        if value < 0 or value > maximum:
            raise V4ControllerPacketError(
                f"controller packet {name} is outside [0, {maximum}]"
            )
        return value

    @classmethod
    def encode(cls, packet: dict[str, Any], *, rank: int, step: int) -> bytes:
        rank = cls._require_index(rank, "rank", (1 << 32) - 1)
        step = cls._require_index(step, "step", UINT64_MAX)
        if not isinstance(packet, dict):
            raise V4ControllerPacketError("controller packet is not an object")
        if type(packet.get("rank")) is not int or packet.get("rank") != rank:
            raise V4ControllerPacketError("controller packet payload rank mismatch")
        if type(packet.get("step")) is not int or packet.get("step") != step:
            raise V4ControllerPacketError("controller packet payload step mismatch")
        envelope = {
            "packet": packet,
            "rank": rank,
            "schema": cls.SCHEMA,
            "step": step,
        }
        payload = cls._canonical_msgpack(envelope)
        if len(payload) > cls.MAX_PAYLOAD_BYTES:
            raise V4ControllerPacketError(
                "controller packet payload overflow: "
                f"{len(payload)} > {cls.MAX_PAYLOAD_BYTES} bytes"
            )
        header = cls.HEADER.pack(
            V4_CONTROLLER_PACKET_MAGIC,
            cls.SCHEMA_ID,
            cls.HEADER.size,
            len(payload),
            rank,
            step,
            hashlib.sha256(payload).digest(),
        )
        padding_bytes = cls.MAX_PACKET_BYTES - len(header) - len(payload)
        return header + payload + cls._ZERO_FRAME[:padding_bytes]

    @classmethod
    def encode_into(
        cls,
        packet: dict[str, Any],
        destination: torch.Tensor,
        *,
        rank: int,
        step: int,
    ) -> int:
        if (
            destination.device.type != "cpu"
            or destination.dtype is not torch.uint8
            or not destination.is_contiguous()
            or destination.numel() != cls.MAX_PACKET_BYTES
        ):
            raise V4ControllerPacketError(
                "controller packet destination must be one contiguous fixed-size CPU uint8 tensor"
            )
        frame = cls.encode(packet, rank=rank, step=step)
        destination_view = destination.numpy()
        # encode() already supplies zero padding; assigning the complete frame
        # prevents stale bytes when this preallocated tensor is reused.
        destination_view[:] = np.frombuffer(frame, dtype=np.uint8)
        payload_bytes = cls.HEADER.unpack_from(frame)[3]
        return cls.HEADER.size + int(payload_bytes)

    @staticmethod
    def _object_without_duplicates(
        pairs: list[tuple[Any, Any]]
    ) -> tuple[dict[str, Any], int]:
        result: dict[str, Any] = {}
        previous_key: str | None = None
        nesting = 1
        for key, value in pairs:
            if type(key) is not str:
                raise V4ControllerPacketError(
                    "controller packet MessagePack map key is not a string"
                )
            # UTF-8 preserves Unicode scalar ordering. raw=False has already
            # rejected invalid UTF-8, so direct string comparison implements
            # the encoder's UTF-8 byte order without allocating key bytes.
            if previous_key is not None and key <= previous_key:
                if key == previous_key:
                    raise V4ControllerPacketError(
                        f"controller packet MessagePack has duplicate key {key!r}"
                    )
                raise V4ControllerPacketError(
                    "controller packet payload is not canonical MessagePack"
                )
            previous_key = key
            value_type = type(value)
            value_nesting = 0
            if value_type is tuple:
                value, value_nesting = value
                value_type = type(value)
            if value_type is float and not math.isfinite(value):
                raise V4ControllerPacketError(
                    "controller packet float is non-finite"
                )
            if value_type is bytes:
                raise V4ControllerPacketError(
                    "controller packet MessagePack binary value is unsupported"
                )
            result[key] = value
            candidate_nesting = value_nesting + 1
            if candidate_nesting > nesting:
                nesting = candidate_nesting
        if nesting > V4ControllerPacketCodec.MAX_NESTING_DEPTH:
            raise V4ControllerPacketError(
                "controller packet nesting exceeds "
                f"{V4ControllerPacketCodec.MAX_NESTING_DEPTH}"
            )
        return result, nesting

    @staticmethod
    def _list_without_unsupported_values(
        values: list[Any],
    ) -> tuple[list[Any], int]:
        nesting = 1
        for index, value in enumerate(values):
            value_type = type(value)
            value_nesting = 0
            if value_type is tuple:
                value, value_nesting = value
                values[index] = value
                value_type = type(value)
            if value_type is float and not math.isfinite(value):
                raise V4ControllerPacketError(
                    "controller packet float is non-finite"
                )
            if value_type is bytes:
                raise V4ControllerPacketError(
                    "controller packet MessagePack binary value is unsupported"
                )
            candidate_nesting = value_nesting + 1
            if candidate_nesting > nesting:
                nesting = candidate_nesting
        if nesting > V4ControllerPacketCodec.MAX_NESTING_DEPTH:
            raise V4ControllerPacketError(
                "controller packet nesting exceeds "
                f"{V4ControllerPacketCodec.MAX_NESTING_DEPTH}"
            )
        return values, nesting

    @staticmethod
    def _reject_extension(code: int, data: bytes) -> Any:
        del code, data
        raise V4ControllerPacketError(
            "controller packet MessagePack extension values are unsupported"
        )

    @classmethod
    def decode(
        cls,
        frame: bytes | bytearray | memoryview | np.ndarray,
        *,
        expected_rank: int,
        expected_step: int,
    ) -> dict[str, Any]:
        expected_rank = cls._require_index(expected_rank, "expected rank", (1 << 32) - 1)
        expected_step = cls._require_index(expected_step, "expected step", UINT64_MAX)
        try:
            view = memoryview(frame).cast("B")
        except (TypeError, ValueError) as exc:
            raise V4ControllerPacketError("controller packet frame is not bytes-like") from exc
        if len(view) != cls.MAX_PACKET_BYTES:
            raise V4ControllerPacketError(
                "controller packet frame capacity mismatch: "
                f"{len(view)} != {cls.MAX_PACKET_BYTES}"
            )
        try:
            magic, schema_id, header_bytes, payload_bytes, rank, step, digest = (
                cls.HEADER.unpack_from(view)
            )
        except struct.error as exc:
            raise V4ControllerPacketError("controller packet header is truncated") from exc
        if magic != V4_CONTROLLER_PACKET_MAGIC:
            raise V4ControllerPacketError("controller packet magic mismatch")
        if schema_id != cls.SCHEMA_ID:
            raise V4ControllerPacketError(
                f"controller packet schema mismatch: {schema_id} != {cls.SCHEMA_ID}"
            )
        if header_bytes != cls.HEADER.size:
            raise V4ControllerPacketError(
                f"controller packet header length mismatch: {header_bytes} != {cls.HEADER.size}"
            )
        if rank != expected_rank:
            raise V4ControllerPacketError(
                f"controller packet rank mismatch: {rank} != {expected_rank}"
            )
        if step != expected_step:
            raise V4ControllerPacketError(
                f"controller packet step mismatch: {step} != {expected_step}"
            )
        if payload_bytes <= 0 or payload_bytes > cls.MAX_PAYLOAD_BYTES:
            raise V4ControllerPacketError(
                f"controller packet payload length is invalid: {payload_bytes}"
            )
        payload_end = cls.HEADER.size + payload_bytes
        payload = bytes(view[cls.HEADER.size:payload_end])
        # Compare against a preallocated all-zero suffix in C rather than
        # iterating over up to 32 KiB in Python.
        padding_bytes = cls.MAX_PACKET_BYTES - payload_end
        if bytes(view[payload_end:]) != cls._ZERO_FRAME[:padding_bytes]:
            raise V4ControllerPacketError("controller packet has nonzero padding")
        if hashlib.sha256(payload).digest() != digest:
            raise V4ControllerPacketError("controller packet SHA-256 mismatch")
        try:
            decoded_with_nesting = msgpack.unpackb(
                payload,
                raw=False,
                use_list=True,
                strict_map_key=True,
                object_pairs_hook=cls._object_without_duplicates,
                list_hook=cls._list_without_unsupported_values,
                ext_hook=cls._reject_extension,
                timestamp=0,
                max_str_len=cls.MAX_PAYLOAD_BYTES,
                max_bin_len=0,
                max_array_len=cls.MAX_ARRAY_ITEMS,
                max_map_len=cls.MAX_MAP_ITEMS,
                max_ext_len=0,
            )
        except V4ControllerPacketError:
            raise
        except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
            raise V4ControllerPacketError(
                f"controller packet MessagePack decoding failed: {exc}"
            ) from exc
        if (
            type(decoded_with_nesting) is not tuple
            or len(decoded_with_nesting) != 2
            or type(decoded_with_nesting[1]) is not int
        ):
            raise V4ControllerPacketError(
                "controller packet top-level value is not a map"
            )
        decoded, _decoded_nesting = decoded_with_nesting
        if not isinstance(decoded, dict) or set(decoded) != cls._ENVELOPE_KEYS:
            raise V4ControllerPacketError("controller packet envelope keys mismatch")
        if decoded.get("schema") != cls.SCHEMA:
            raise V4ControllerPacketError("controller packet envelope schema mismatch")
        if type(decoded.get("rank")) is not int or decoded.get("rank") != expected_rank:
            raise V4ControllerPacketError("controller packet envelope rank mismatch")
        if type(decoded.get("step")) is not int or decoded.get("step") != expected_step:
            raise V4ControllerPacketError("controller packet envelope step mismatch")
        packet = decoded.get("packet")
        if not isinstance(packet, dict):
            raise V4ControllerPacketError("controller packet payload is not an object")
        if type(packet.get("rank")) is not int or packet.get("rank") != expected_rank:
            raise V4ControllerPacketError("controller packet payload rank mismatch")
        if type(packet.get("step")) is not int or packet.get("step") != expected_step:
            raise V4ControllerPacketError("controller packet payload step mismatch")
        return packet


def v4_controller_module() -> Any:
    """Load the snapshotted controller when the run script provides one."""

    global _V4_CONTROLLER_MODULE, _V4_CONTROLLER_PATH, _V4_CONTROLLER_SHA256
    if _V4_CONTROLLER_MODULE is not None:
        return _V4_CONTROLLER_MODULE
    configured = (
        os.environ.get("TEMPO_V4_CONTROLLER_SNAPSHOT", "")
        or os.environ.get("V4_CONTROLLER_SNAPSHOT", "")
    ).strip()
    path = Path(configured).resolve() if configured else (Path(__file__).resolve().parents[2] / "tempo" / "v4_controller.py")
    if not path.is_file():
        raise RuntimeError(f"TEMPO v4 controller source does not exist: {path}")
    spec = importlib.util.spec_from_file_location("tempo_v4_controller_executed", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load TEMPO v4 controller source: {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their defining module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _V4_CONTROLLER_MODULE = module
    _V4_CONTROLLER_PATH = path
    _V4_CONTROLLER_SHA256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument(
        "--tier-mode",
        choices=TIER_MODES,
        default="",
        help=(
            "TEMPO-RD attribution mode.  The default preserves the legacy "
            "backend path; d2h_only requires a node-local sink and "
            "persist_only host-preloads tensors before persistence."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--restore-only", action="store_true")
    parser.add_argument("--steps", type=int, default=72)
    parser.add_argument("--warmup-steps", type=int, default=12)
    parser.add_argument("--checkpoint-step", type=int, default=24)
    parser.add_argument(
        "--checkpoint-steps",
        default="",
        help="Comma-separated checkpoint steps; overrides --checkpoint-step",
    )
    parser.add_argument("--window-steps", type=int, default=16)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-size", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--probe-mb", type=int, default=64)
    parser.add_argument("--credit-mb", type=int, default=4)
    parser.add_argument("--target-slowdown", type=float, default=1.10)
    parser.add_argument("--deadline-seconds", type=float, default=120.0)
    parser.add_argument("--datastates-cache-gb", type=float, default=1.0)
    parser.add_argument("--tempo-v2-d2h-gbps", type=float, default=1.0)
    parser.add_argument("--tempo-v2-chunk-mb", type=int, default=4)
    parser.add_argument("--tempo-v3-d2h-gbps", type=float, default=2.0)
    parser.add_argument("--tempo-v3-chunk-mb", type=int, default=1)
    parser.add_argument("--tempo-v3-deadline-reserve-ms", type=float, default=50.0)
    parser.add_argument("--tempo-v3-collective-reserve", type=float, default=0.15)
    parser.add_argument("--tempo-v3-epoch-lead-ms", type=float, default=20.0)
    parser.add_argument("--tempo-v4-d2h-chunk-mb", type=int, default=1)
    parser.add_argument("--tempo-v4-pfs-chunk-mb", type=int, default=4)
    parser.add_argument("--tempo-v4-max-pfs-inflight-mb", type=int, default=16)
    parser.add_argument("--tempo-v4-watchdog-ms", type=float, default=250.0)
    parser.add_argument("--tempo-v4-low-slack-ms", type=float, default=50.0)
    parser.add_argument("--tempo-v4-recovery-slack-ms", type=float, default=250.0)
    parser.add_argument("--tempo-v4-high-slack-ms", type=float, default=300.0)
    parser.add_argument("--tempo-v4-deadline-margin-ms", type=float, default=25.0)
    parser.add_argument("--tempo-v4-finalization-reserve-ms", type=float, default=50.0)
    parser.add_argument("--tempo-v4-pipeline-reserve-ms", type=float, default=25.0)
    parser.add_argument(
        "--tempo-v4-d2h-floor-gbps",
        type=float,
        default=V4_ARCHIVE_D2H_SEED_BPS / 1e9,
    )
    parser.add_argument(
        "--tempo-v4-pfs-floor-gbps",
        type=float,
        default=V4_ARCHIVE_PFS_SEED_BPS / 1e9,
    )
    parser.add_argument(
        "--v4-open-c0",
        action="store_true",
        help="enable the fixed-rate C0 v4_open ablation",
    )
    parser.add_argument("--tempo-v4-controller-timeout-ms", type=float, default=2000.0)
    parser.add_argument(
        "--tempo-v4-control-mode",
        choices=("scheduled", "split_guard", "work_conserving"),
        default="scheduled",
        help=(
            "diagnostic control path; split_guard keeps PFS continuous and "
            "gates D2H to compute/residual windows; work_conserving permits "
            "at most one 1 MiB D2H residual request at a collective boundary"
        ),
    )
    parser.add_argument(
        "--tempo-v4-telemetry",
        choices=("required", "off"),
        default="required",
        help="disable the deleted journal publisher for a local prototype screen",
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--clock-calibration-samples", type=int, default=21)
    args = parser.parse_args()
    if not args.tier_mode:
        args.tier_mode = "fg_only" if args.policy == "none" else "combined"
    if args.policy == "none" and args.tier_mode != "fg_only":
        parser.error("policy=none only supports tier-mode=fg_only")
    if args.policy in ("v4_open", "tempo_v4") and args.tier_mode not in (
        "combined",
        "open_combined",
    ):
        parser.error(
            "v4 policies only support combined/open_combined; isolated tier "
            "modes must use the unpaced DataStates adapter"
        )
    try:
        checkpoint_steps = (
            [int(value.strip()) for value in args.checkpoint_steps.split(",") if value.strip()]
            if args.checkpoint_steps
            else [args.checkpoint_step]
        )
    except ValueError:
        parser.error("checkpoint-steps must be a comma-separated integer list")
    if not checkpoint_steps or checkpoint_steps != sorted(set(checkpoint_steps)):
        parser.error("checkpoint-steps must be non-empty, unique, and increasing")
    if any(step < args.warmup_steps or step >= args.steps for step in checkpoint_steps):
        parser.error("every checkpoint step must be in [warmup-steps, steps)")
    if any(step + args.window_steps + 1 >= args.steps for step in checkpoint_steps):
        parser.error("every checkpoint window and terminal status step must fit inside the run")
    if any(
        right - left <= args.window_steps + 1
        for left, right in zip(checkpoint_steps, checkpoint_steps[1:])
    ):
        parser.error("checkpoint windows and terminal status steps must not overlap")
    args.checkpoint_steps = checkpoint_steps
    args.checkpoint_step = checkpoint_steps[-1]
    if args.hidden_size % args.heads:
        parser.error("hidden-size must be divisible by heads")
    if args.tempo_v2_d2h_gbps <= 0 or args.tempo_v2_chunk_mb <= 0:
        parser.error("TEMPO v2 D2H rate and chunk size must be positive")
    if args.tempo_v3_d2h_gbps <= 0 or args.tempo_v3_chunk_mb <= 0:
        parser.error("TEMPO v3 D2H rate and chunk size must be positive")
    if args.tempo_v3_deadline_reserve_ms < 0:
        parser.error("TEMPO v3 deadline reserve must be non-negative")
    if args.tempo_v3_epoch_lead_ms < 0:
        parser.error("TEMPO v3 epoch lead must be non-negative")
    if not 0 <= args.tempo_v3_collective_reserve < 1:
        parser.error("TEMPO v3 collective reserve must be in [0, 1)")
    if (
        args.tempo_v4_d2h_chunk_mb != V4_D2H_REQUEST_MIB
        or args.tempo_v4_pfs_chunk_mb != V4_PFS_REQUEST_MIB
    ):
        parser.error(
            "TEMPO v4 requires 1 MiB D2H admissions inside retained "
            "4 MiB payload/PFS regions"
        )
    if args.tempo_v4_max_pfs_inflight_mb < args.tempo_v4_pfs_chunk_mb:
        parser.error("TEMPO v4 PFS in-flight cap must cover at least one request")
    if args.tempo_v4_max_pfs_inflight_mb % args.tempo_v4_pfs_chunk_mb:
        parser.error("TEMPO v4 PFS in-flight cap must be a multiple of its request quantum")
    if args.tempo_v4_watchdog_ms <= 0 or args.tempo_v4_controller_timeout_ms <= 0:
        parser.error("TEMPO v4 watchdog and controller timeout must be positive")
    if not 0 <= args.tempo_v4_low_slack_ms < args.tempo_v4_high_slack_ms:
        parser.error("TEMPO v4 slack watermarks must satisfy 0 <= low < high")
    if not (
        args.tempo_v4_low_slack_ms
        <= args.tempo_v4_recovery_slack_ms
        < args.tempo_v4_high_slack_ms
    ):
        parser.error("TEMPO v4 recovery watermark must satisfy low <= recovery < high")
    if min(
        args.tempo_v4_deadline_margin_ms,
        args.tempo_v4_finalization_reserve_ms,
        args.tempo_v4_pipeline_reserve_ms,
    ) < 0:
        parser.error("TEMPO v4 deadline reserves must be non-negative")
    if (
        not math.isfinite(args.tempo_v4_d2h_floor_gbps)
        or not math.isfinite(args.tempo_v4_pfs_floor_gbps)
        or args.tempo_v4_d2h_floor_gbps <= 0
        or args.tempo_v4_pfs_floor_gbps <= 0
    ):
        parser.error("TEMPO v4 stage service floors must be positive")
    if args.v4_open_c0 and args.policy != "v4_open":
        parser.error("--v4-open-c0 is only valid with policy=v4_open")
    if args.clock_calibration_samples < 1:
        parser.error("clock-calibration-samples must be positive")
    return args


class DecoderBlock(nn.Module):
    def __init__(self, hidden: int, ffn: int, heads: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.norm2 = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, 2 * ffn, bias=False)
        self.down = nn.Linear(ffn, hidden, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        batch, sequence, _ = value.shape
        qkv = self.qkv(self.norm1(value)).view(
            batch, sequence, 3, self.heads, self.head_dim
        )
        query, key, val = qkv.unbind(dim=2)
        attention = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            val.transpose(1, 2),
            is_causal=True,
        )
        attention = attention.transpose(1, 2).reshape(batch, sequence, self.hidden)
        value = residual + self.proj(attention)
        residual = value
        gate, up = self.up(self.norm2(value)).chunk(2, dim=-1)
        return residual + self.down(F.silu(gate) * up)


class MiniGPT(nn.Module):
    def __init__(self, layers: int, hidden: int, ffn: int, heads: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [DecoderBlock(hidden, ffn, heads) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(hidden)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return self.final_norm(value)


def init_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(local_rank), WORLD_SIZE=str(world_size))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=dt.timedelta(minutes=10),
        device_id=torch.device("cuda", local_rank),
    )
    return rank, local_rank, world_size


def calibrate_wall_clock(
    rank: int,
    world_size: int,
    group: dist.ProcessGroup,
    samples: int,
) -> tuple[int, int]:
    """Estimate each rank's realtime offset to rank 0 with minimum-RTT pings."""
    offsets = torch.zeros(world_size, dtype=torch.int64)
    round_trip_ns = torch.zeros(world_size, dtype=torch.int64)
    ping = torch.zeros(1, dtype=torch.int64)
    for peer in range(1, world_size):
        dist.barrier(group=group)
        if rank == peer:
            best_rtt: int | None = None
            best_offset = 0
            for sample in range(samples):
                response = torch.zeros(1, dtype=torch.int64)
                begin_ns = time.time_ns()
                dist.send(ping, dst=0, group=group, tag=peer * 100 + sample)
                dist.recv(response, src=0, group=group, tag=10_000 + peer * 100 + sample)
                end_ns = time.time_ns()
                rtt = end_ns - begin_ns
                offset = int(response.item()) - (begin_ns + end_ns) // 2
                if best_rtt is None or rtt < best_rtt:
                    best_rtt = rtt
                    best_offset = offset
            result = torch.tensor([best_offset, best_rtt or 0], dtype=torch.int64)
            dist.send(result, dst=0, group=group, tag=20_000 + peer)
        elif rank == 0:
            for sample in range(samples):
                dist.recv(ping, src=peer, group=group, tag=peer * 100 + sample)
                response = torch.tensor([time.time_ns()], dtype=torch.int64)
                dist.send(response, dst=peer, group=group, tag=10_000 + peer * 100 + sample)
            result = torch.zeros(2, dtype=torch.int64)
            dist.recv(result, src=peer, group=group, tag=20_000 + peer)
            offsets[peer] = result[0]
            round_trip_ns[peer] = result[1]
        dist.barrier(group=group)
    dist.broadcast(offsets, src=0, group=group)
    dist.broadcast(round_trip_ns, src=0, group=group)
    return int(offsets[rank].item()), int(round_trip_ns[rank].item())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _phase_hsn_snapshot() -> dict[str, Any]:
    """Read node-level HSN byte counters for one collective slice.

    These counters are deliberately opt-in and are only emitted when the raw
    fabric runner requests them.  They are host/node scoped (not per-rank
    traffic attribution); the phase interval and source binding make them
    useful for a later node-slice causal gate without pretending they are
    GPU/NVLink or per-rank counters.
    """

    values: dict[str, int] = {}
    for directory in sorted(Path("/sys/class/net").glob("hsn*/statistics")):
        interface = directory.parent.name
        for field in ("rx_bytes", "tx_bytes", "rx_packets", "tx_packets"):
            path = directory / field
            try:
                values[f"{interface}.{field}"] = int(
                    path.read_text(encoding="utf-8").strip()
                )
            except (OSError, ValueError):
                continue
    return {
        "source": "sysfs:/sys/class/net/hsn*/statistics;host_device_sum",
        "timestamp_monotonic_ns": time.monotonic_ns(),
        "hostname": socket.gethostname(),
        "interfaces": values,
        "rx_bytes": sum(
            value for key, value in values.items() if key.endswith(".rx_bytes")
        ),
        "tx_bytes": sum(
            value for key, value in values.items() if key.endswith(".tx_bytes")
        ),
    }


def atomic_durable_json(path: Path, value: Any) -> None:
    """Atomically publish a JSON commit record and fsync its directory entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _v4_compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Convert a rich observer profile to self-contained wire statistics."""

    if not isinstance(profile, dict):
        raise RuntimeError("controller profile is not an object")
    notifications = profile.get("notifications")
    windows = profile.get("windows")
    profile_step = profile.get("step")
    if (
        type(profile_step) is not int
        or profile_step < 0
        or not isinstance(notifications, list)
        or not isinstance(windows, list)
        or not notifications
        or len(notifications) > 26
        or len(windows) != 2 * len(notifications) + 1
    ):
        raise RuntimeError("controller profile has an invalid 2N+1 shape")
    signatures: list[str] = []
    for index, notification in enumerate(notifications):
        if not isinstance(notification, dict):
            raise RuntimeError("controller notification is not an object")
        signature = notification.get("signature")
        window = windows[2 * index + 1]
        if (
            type(signature) is not str
            or not signature
            or not isinstance(window, dict)
            or str(window.get("signature")) != signature
        ):
            raise RuntimeError("controller profile layout signature mismatch")
        signatures.append(signature)
    layout = {"phase_count": len(signatures), "collective_signatures": signatures}
    starts: list[int] = []
    ends: list[int] = []
    installable_mask = 0
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise RuntimeError("controller profile window is not an object")
        start = window.get("start_corrected_ns")
        end = window.get("end_corrected_ns")
        installable = bool(window.get("installable", False))
        # Enqueue-ahead legitimately makes a skipped lead-in noncausal:
        # the next collective can be observed ready before the preceding
        # CUDA completion estimate.  Preserve those bounds so the receiver
        # reconstructs the same zero-capacity window; only an installable
        # window is required to have a positive-time ordering.
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or (installable and end < start)
        ):
            raise RuntimeError("controller profile window bounds are invalid")
        starts.append(start)
        ends.append(end)
        if installable:
            installable_mask |= 1 << index
    gpu_ms: list[float] = []
    gate_wait_ms: list[float] = []
    for notification in notifications:
        gpu = notification.get("gpu_ms")
        gate = notification.get("gate_wait_ms", 0.0)
        if (
            type(gpu) not in (int, float)
            or type(gate) not in (int, float)
            or not math.isfinite(float(gpu))
            or not math.isfinite(float(gate))
            or float(gpu) < 0.0
            or float(gate) < 0.0
        ):
            raise RuntimeError("controller profile timing is invalid")
        gpu_ms.append(float(gpu))
        gate_wait_ms.append(float(gate))
    return {
        "schema": V4_CONTROLLER_PROFILE_SCHEMA,
        "profile_step": profile_step,
        "layout": layout,
        "layout_sha256": canonical_sha256(layout),
        "window_start_corrected_ns": starts,
        "window_end_corrected_ns": ends,
        "window_installable_mask": installable_mask,
        "notification_gpu_ms": gpu_ms,
        "notification_gate_wait_ms": gate_wait_ms,
    }


def _v4_compact_progress(relative: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(relative, dict):
        raise RuntimeError("event-relative progress is absent")
    fields = (
        "total_bytes", "queued_bytes", "ready_bytes", "admitted_bytes",
        "completed_bytes", "inflight_bytes", "last_progress_monotonic_ns",
    )
    result: dict[str, Any] = {
        # Short keys are intentional: these fields repeat for every rank and
        # stage.  The fixed schema keeps the representation self-contained;
        # this is not a stateful delta/cache protocol.
        "s": int(relative.get("snapshot_monotonic_ns", 0)),
        "w": bool(relative.get("watchdog_fail_open", False)),
    }
    for name in ("d2h", "pfs"):
        stage = relative.get(name)
        if not isinstance(stage, dict):
            raise RuntimeError(f"event-relative {name} progress is absent")
        values = []
        for field in fields:
            value = stage.get(field, 0)
            if type(value) is not int or value < 0 or value > UINT64_MAX:
                raise RuntimeError(f"event-relative {name}.{field} is invalid")
            values.append(value)
        result["d" if name == "d2h" else "p"] = values
    return result


def _v4_wire_packet(local: dict[str, Any]) -> dict[str, Any]:
    """Build the compact packet without mutating the local semantic snapshot."""

    wire = {
        key: local.get(key)
        for key in (
            "rank",
            "step",
            "active",
            "checkpoint_id",
            "now_corrected_ns",
            "deadline_corrected_ns",
            "observer_healthy",
            "error",
            "envelope_breach",
            "clock_uncertainty_ns",
        )
    }
    full_profile = local.get("profile")
    wire["profile"] = (
        None if full_profile is None else _v4_compact_profile(full_profile)
    )
    relative = local.get("event_relative_stats")
    wire["progress"] = (
        None if relative is None else _v4_compact_progress(relative)
    )
    return wire


def _v4_expand_compact_profile(profile: dict[str, Any], expected_step: int) -> dict[str, Any]:
    if not isinstance(profile, dict) or profile.get("schema") != V4_CONTROLLER_PROFILE_SCHEMA:
        raise RuntimeError("controller profile schema mismatch")
    profile_step = profile.get("profile_step")
    if type(profile_step) is not int or profile_step != expected_step - 1:
        raise RuntimeError("controller profile step is stale")
    layout = profile.get("layout")
    if not isinstance(layout, dict):
        raise RuntimeError("controller profile layout is absent")
    phase_count = layout.get("phase_count")
    signatures = layout.get("collective_signatures")
    if (
        type(phase_count) is not int or phase_count <= 0 or phase_count > 26
        or not isinstance(signatures, list) or len(signatures) != phase_count
        or any(type(value) is not str or not value for value in signatures)
        or profile.get("layout_sha256") != canonical_sha256(layout)
    ):
        raise RuntimeError("controller profile layout is invalid")
    count = 2 * phase_count + 1
    starts = profile.get("window_start_corrected_ns")
    ends = profile.get("window_end_corrected_ns")
    mask = profile.get("window_installable_mask")
    gpu_ms = profile.get("notification_gpu_ms")
    gate_wait_ms = profile.get("notification_gate_wait_ms")
    if (
        not isinstance(starts, list) or not isinstance(ends, list)
        or len(starts) != count or len(ends) != count
        or type(mask) is not int or mask < 0 or mask >> count
        or not isinstance(gpu_ms, list) or not isinstance(gate_wait_ms, list)
        or len(gpu_ms) != phase_count or len(gate_wait_ms) != phase_count
    ):
        raise RuntimeError("controller profile vectors are invalid")
    for index, (start, end) in enumerate(zip(starts, ends)):
        installable = bool(mask & (1 << index))
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or (installable and end < start)
        ):
            raise RuntimeError("controller profile window bounds are invalid")
    if any(
        type(value) not in (int, float)
        or not math.isfinite(float(value)) or float(value) < 0.0
        for value in (*gpu_ms, *gate_wait_ms)
    ):
        raise RuntimeError("controller profile timing vectors are invalid")
    windows: list[dict[str, Any]] = []
    notifications: list[dict[str, Any]] = []
    for index, signature in enumerate(signatures):
        lead_index = 2 * index
        execution_index = lead_index + 1
        ready = int(starts[execution_index])
        completion = int(ends[execution_index])
        lead_installable = bool(mask & (1 << lead_index))
        windows.append({
            "phase_id": lead_index,
            "signature": f"lead-in:{signature}",
            "kind": "compute",
            "duration_ns": max(1_000, int(ends[lead_index]) - int(starts[lead_index])),
            "installable": lead_installable,
            "start_corrected_ns": int(starts[lead_index]),
            "end_corrected_ns": int(ends[lead_index]),
        })
        windows.append({
            "phase_id": execution_index,
            "signature": signature,
            "kind": "collective",
            "duration_ns": max(1_000, completion - ready),
            "installable": True,
            "start_corrected_ns": ready,
            "end_corrected_ns": completion,
        })
        notifications.append({
            "sequence": index,
            "phase_index": index,
            "signature": signature,
            "ready_unix_ns": ready,
            "ready_corrected_ns": ready,
            "gpu_ms": float(gpu_ms[index]),
            "phase_install_ms": 0.0,
            "gate_wait_ms": float(gate_wait_ms[index]),
            "completion_unix_ns": completion,
            "callback_unix_ns": completion,
            "arrival_plan_source": (
                "stream_ordered_lead_in" if lead_installable else "stream_ordered_zero_compute"
            ),
        })
    exit_index = count - 1
    windows.append({
        "phase_id": exit_index,
        "signature": "compute:step-exit",
        "kind": "compute",
        "duration_ns": max(1_000, int(ends[-1]) - int(starts[-1])),
        "installable": bool(mask & (1 << exit_index)),
        "start_corrected_ns": int(starts[-1]),
        "end_corrected_ns": int(ends[-1]),
    })
    return {"step": profile_step, "windows": windows, "notifications": notifications, "phase_count": phase_count}


def _v4_expand_compact_progress(progress: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(progress, dict):
        raise RuntimeError("compact progress is absent")
    fields = (
        "total_bytes", "queued_bytes", "ready_bytes", "admitted_bytes",
        "completed_bytes", "inflight_bytes", "last_progress_monotonic_ns",
    )
    if set(progress) != {"s", "w", "d", "p"}:
        raise RuntimeError("compact progress keys are invalid")
    if type(progress["s"]) is not int or not 0 <= progress["s"] <= UINT64_MAX:
        raise RuntimeError("compact progress snapshot is invalid")
    if type(progress["w"]) is not bool:
        raise RuntimeError("compact progress watchdog flag is invalid")
    result: dict[str, Any] = {
        "snapshot_monotonic_ns": progress["s"],
        "watchdog_fail_open": progress["w"],
    }
    for name, key in (("d2h", "d"), ("pfs", "p")):
        values = progress[key]
        if (
            not isinstance(values, list)
            or len(values) != len(fields)
            or any(type(value) is not int or not 0 <= value <= UINT64_MAX for value in values)
        ):
            raise RuntimeError(f"compact {name} progress vector is invalid")
        result[name] = dict(zip(fields, values))
    return result


def rng_state_sha256(value: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(value):
        tensor = value[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


RUNTIME_PYTHON_MODULES_SCHEMA = "tempo-runtime-python-modules-1"


def _capture_runtime_python_modules() -> dict[str, dict[str, str]]:
    """Capture the source bytes that actually defined imported TEMPO helpers.

    This is evaluated once during module import.  Hashing again only at summary
    time could accidentally bless source bytes changed after Python had already
    executed the original module.
    """

    modules = {
        "tempo": _tempo_runtime_package,
        "tempo.group_credit_checkpoint": _group_credit_checkpoint_module,
        "tempo.split_guard": _split_guard_module,
    }
    captured: dict[str, dict[str, str]] = {}
    for name, module in modules.items():
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not raw_path:
            raise RuntimeError(f"runtime TEMPO module lacks a source path: {name}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"runtime TEMPO module source is missing: {name}={path}")
        captured[name] = {
            "source_path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    if os.environ.get("TEMPO_PYTHON_SNAPSHOT_ROOT", "").strip():
        expected_variables = {
            "tempo": ("TEMPO_INIT_SNAPSHOT", "TEMPO_INIT_SHA256"),
            "tempo.group_credit_checkpoint": (
                "CHECKPOINTER_SNAPSHOT",
                "CHECKPOINTER_SHA256",
            ),
            "tempo.split_guard": ("SPLIT_GUARD_SNAPSHOT", "SPLIT_GUARD_SHA256"),
        }
        for name, (path_variable, hash_variable) in expected_variables.items():
            expected_path_raw = os.environ.get(path_variable, "").strip()
            expected_sha256 = os.environ.get(hash_variable, "").strip()
            if not expected_path_raw or not re.fullmatch(
                r"[0-9a-f]{64}", expected_sha256
            ):
                raise RuntimeError(
                    f"frozen TEMPO provenance environment is incomplete for {name}"
                )
            if (
                captured[name]["source_path"] != str(Path(expected_path_raw).resolve())
                or captured[name]["sha256"] != expected_sha256
            ):
                raise RuntimeError(
                    f"runtime TEMPO module differs from required snapshot: {name}"
                )
    return captured


_RUNTIME_PYTHON_MODULES = _capture_runtime_python_modules()


def runtime_python_module_provenance() -> dict[str, dict[str, str]]:
    """Return an unaliased JSON-safe copy of import-time helper provenance."""

    return {name: dict(value) for name, value in _RUNTIME_PYTHON_MODULES.items()}


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


@torch.no_grad()
def model_checksum(model: nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64, device="cuda")
    for parameter in model.parameters():
        if parameter.numel() and parameter.is_floating_point():
            total += parameter.detach().double().sum()
    dist.all_reduce(total)
    return float(total.item())


@torch.no_grad()
def optimizer_checksum(optimizer: torch.optim.Optimizer) -> float:
    total = torch.zeros((), dtype=torch.float64, device="cuda")
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and value.numel():
                total += value.detach().to(device="cuda", dtype=torch.float64).sum()
    dist.all_reduce(total)
    return float(total.item())


def local_tensor_leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, DTensor):
        yield prefix, value.to_local()
    elif isinstance(value, ShardedTensor):
        for index, shard in enumerate(value.local_shards()):
            yield f"{prefix}/shard{index}", shard.tensor
    elif isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from local_tensor_leaves(item, f"{prefix}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from local_tensor_leaves(item, f"{prefix}/{index}")


def rank_rng_state(rng: torch.Generator, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "rng_cpu": torch.get_rng_state().clone(),
        "rng_cuda": torch.cuda.get_rng_state(device).clone(),
        "input_rng": rng.get_state().clone(),
    }


def gathered_rng_state(
    local: dict[str, torch.Tensor], world_size: int, group: dist.ProcessGroup
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in local.items():
        gathered = [torch.empty_like(value) for _ in range(world_size)]
        dist.all_gather(gathered, value, group=group)
        result[key + "_by_rank"] = torch.stack(gathered)
    return result


@dataclass
class BackendMetrics:
    policy: str
    tier_mode: str = ""
    tier_endpoint: str = ""
    tier_host_preloaded: bool = False
    tier_gpu_transfer: bool = False
    # Raw engine snapshots are retained for the TEMPO-RD tier matrix.  They
    # are evidence inputs, not causal conclusions; the post-run validator
    # still requires observed paths, supported counters, and intervention
    # metrics.
    tier_stage_stats_start: dict[str, Any] = field(default_factory=dict)
    tier_stage_stats_end: dict[str, Any] = field(default_factory=dict)
    # Native DataStates runs may not expose the TEMPO engine's admission
    # counters.  The tier screen therefore also records a separate logical
    # stage ledger at the wait(False)->wait(True) boundaries.  These are
    # logical bytes and active intervals, not hardware PCIe/NIC counters.
    tier_logical_stage_schema: str = "tempo-rd-logical-stage-timing-1"
    state_bytes_local: int = 0
    trigger_ms: float = 0.0
    consistency_block_ms: float = 0.0
    durable_ms: float = 0.0
    restore_ms: float = 0.0
    deadline_met: bool = False
    checkpoint_path: str = ""
    d2h_rate_gbps: float = 0.0
    d2h_chunks: int = 0
    d2h_phase_us: float = 0.0
    d2h_epoch_unix_ns: int = 0
    d2h_first_issue_unix_ns: int = 0
    d2h_first_issue_corrected_ns: int = 0
    clock_offset_ns: int = 0
    clock_calibration_rtt_ns: int = 0
    prepare_ms: float = 0.0
    validation_ms: float = 0.0
    d2h_wait_ms: float = 0.0
    collectives_gated: int = 0
    gate_block_ms: float = 0.0
    gate_hold_ms: float = 0.0
    state_dict_ms: float = 0.0
    shadow_copy_ms: float = 0.0
    payload_build_ms: float = 0.0
    engine_save_ms: float = 0.0
    epoch_sync_ms: float = 0.0
    durability_barrier_ms: float = 0.0
    checkpoint_file_bytes: int = 0
    checkpoint_allocated_bytes: int = 0
    logical_file_extent_bytes: int = 0
    logical_layout_publication_sequence: int = 0
    logical_layout_version: int = -1
    commit_marker_path: str = ""
    commit_manifest_sha256: str = ""
    commit_validated: bool = False
    fsync_evidence_valid: bool = False
    v4_scheduled: bool = False
    v4_mode: str = ""
    v4_global_slack_ns: int = 0
    v4_projected_completion_ns: int = 0
    v4_deadline_feasible: bool = False
    v4_force_drain: bool = False
    v4_force_drain_reason: str = ""
    v4_force_drain_reason_class: str = ""
    v4_controller_ms: float = 0.0
    v4_plan_count: int = 0
    v4_control_gather_count: int = 0
    v4_control_terminal_gather_count: int = 0
    v4_control_common_terminal_mode: str = ""
    v4_control_window_exhausted: bool = False
    v4_controller_packet_bytes_last: int = 0
    v4_controller_packet_bytes_max: int = 0
    v4_controller_packet_frame_bytes: int = 0
    v4_controller_packet_failures: int = 0
    v4_phase_install_count: int = 0
    v4_signature_mismatch_count: int = 0
    v4_watchdog_trip_count: int = 0
    v4_rejected_plan_count: int = 0
    v4_envelope_breach: bool = False
    v4_envelope_breach_reason: str = ""
    v4_controller_sha256: str = ""


class TrainingState:
    """TorchSnapshot Stateful for step and rank-local RNG streams."""

    def __init__(self, step: int, rng: torch.Generator, device: torch.device) -> None:
        self.step = step
        self.rng = rng
        self.device = device

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"step": torch.tensor(self.step, dtype=torch.int64), **rank_rng_state(self.rng, self.device)}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.step = int(state_dict["step"].item())
        torch.set_rng_state(state_dict["rng_cpu"].cpu())
        torch.cuda.set_rng_state(state_dict["rng_cuda"].cpu(), device=self.device)
        self.rng.set_state(state_dict["input_rng"].cpu())


class CheckpointBackend:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        rng: torch.Generator,
        device: torch.device,
        rank: int,
        world_size: int,
        control_group: dist.ProcessGroup,
        controller_group: dist.ProcessGroup,
        dcp_group: dist.ProcessGroup,
    ) -> None:
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.rng = rng
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.control_group = control_group
        self.controller_group = controller_group
        self.dcp_group = dcp_group
        self.root = Path(args.checkpoint_dir)
        self.output_dir = Path(args.output_dir)
        self.options = StateDictOptions(full_state_dict=False, cpu_offload=False)
        self.metrics = BackendMetrics(
            policy=args.policy,
            tier_mode=str(getattr(args, "tier_mode", "")),
        )
        self.expected_checksum: float | None = None
        self.expected_optimizer_checksum: float | None = None
        self.expected_rng: dict[str, torch.Tensor] | None = None
        self.checkpoint_step: int | None = None
        self.trigger_ns: int | None = None
        self.checkpoint_events: list[dict[str, Any]] = []
        self.event_recorded = True

    def prepare_common(self, step: int) -> None:
        if not self.event_recorded:
            raise RuntimeError("previous checkpoint event was not finalized")
        prepare_ms = self.metrics.prepare_ms
        self.metrics = BackendMetrics(
            policy=self.args.policy,
            tier_mode=str(getattr(self.args, "tier_mode", "")),
            prepare_ms=prepare_ms,
        )
        self.metrics.clock_offset_ns = int(self.args.clock_offset_ns)
        self.metrics.clock_calibration_rtt_ns = int(self.args.clock_calibration_rtt_ns)
        self.event_recorded = False
        begin = time.perf_counter()
        self.checkpoint_step = step
        self.expected_checksum = model_checksum(self.model)
        self.expected_optimizer_checksum = optimizer_checksum(self.optimizer)
        self.expected_rng = rank_rng_state(self.rng, self.device)
        self.metrics.validation_ms = (time.perf_counter() - begin) * 1000
        self.trigger_ns = time.time_ns()

    def start(self, step: int) -> None:
        raise NotImplementedError

    def prepare(self) -> None:
        return None

    def before_optimizer_step(self) -> float:
        return 0.0

    def after_optimizer_step(self) -> float:
        return 0.0

    def set_compute(self) -> None:
        return None

    def set_collective(self) -> None:
        return None

    def observe_collective(self, latency_ms: float, baseline_sample: bool) -> None:
        return None

    def attach_observer(self, observer: "CudaCollectiveObserver") -> None:
        return None

    def on_step_begin(self, step: int) -> float:
        return 0.0

    def on_step_end(self, step: int) -> None:
        return None

    def close_step_credit_before_probe(self, step: int) -> float:
        """Close any stream-ordered checkpoint credit before the probe."""

        del step
        return 0.0

    def record_step(self, row: dict[str, Any]) -> None:
        return None

    def active(self) -> bool:
        return False

    def d2h_active(self) -> bool:
        return False

    def close(self) -> None:
        return None

    def finish_event(self) -> None:
        if self.checkpoint_step is None or self.event_recorded:
            return
        self.wait_durable()
        self.checkpoint_events.append(asdict(self.metrics))
        self.event_recorded = True

    def wait_durable(self) -> None:
        raise NotImplementedError

    def restore(self) -> dict[str, Any]:
        raise NotImplementedError

    def select_restore_step(self, step: int) -> None:
        self.checkpoint_step = step

    def _finish_durability(self) -> None:
        assert self.trigger_ns is not None
        self.metrics.durable_ms = (time.time_ns() - self.trigger_ns) / 1e6
        self.metrics.deadline_met = self.metrics.durable_ms <= self.args.deadline_seconds * 1000

    def _verify(self, loaded_step: int) -> dict[str, Any]:
        assert self.expected_checksum is not None and self.expected_optimizer_checksum is not None and self.expected_rng is not None
        actual_checksum = model_checksum(self.model)
        actual_optimizer_checksum = optimizer_checksum(self.optimizer)
        actual_rng = rank_rng_state(self.rng, self.device)
        rng_match = all(torch.equal(actual_rng[key].cpu(), self.expected_rng[key].cpu()) for key in self.expected_rng)
        error = abs(actual_checksum - self.expected_checksum)
        tolerance = 1e-7 * max(1.0, abs(self.expected_checksum))
        optimizer_error = abs(actual_optimizer_checksum - self.expected_optimizer_checksum)
        optimizer_tolerance = 1e-7 * max(1.0, abs(self.expected_optimizer_checksum))
        return {
            "attempted": True,
            "passed": bool(
                loaded_step == self.checkpoint_step
                and error <= tolerance
                and optimizer_error <= optimizer_tolerance
                and rng_match
            ),
            "loaded_step": loaded_step,
            "expected_checksum": self.expected_checksum,
            "actual_checksum": actual_checksum,
            "absolute_error": error,
            "expected_optimizer_checksum": self.expected_optimizer_checksum,
            "actual_optimizer_checksum": actual_optimizer_checksum,
            "optimizer_absolute_error": optimizer_error,
            "rng_match": rng_match,
        }


class DCPBackend(CheckpointBackend):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.future: Any = None
        self.path: Path | None = None

    def start(self, step: int) -> None:
        if self.active():
            raise RuntimeError("DCP does not support overlapping checkpoints")
        self.prepare_common(step)
        begin = time.perf_counter()
        model_state, optim_state = get_state_dict(self.model, self.optimizer, options=self.options)
        rng_by_rank = gathered_rng_state(self.expected_rng or {}, self.world_size, self.control_group)
        state = {"model": model_state, "optimizer": optim_state, "step": torch.tensor(step, dtype=torch.int64), **rng_by_rank}
        self.metrics.state_bytes_local = sum(t.numel() * t.element_size() for _, t in local_tensor_leaves(state))
        self.path = self.root / f"step_{step:07d}"
        self.metrics.checkpoint_path = str(self.path)
        writer = dcp.FileSystemWriter(self.path, single_file_per_rank=True, sync_files=True, thread_count=1, overwrite=True)
        self.future = dcp.async_save(state, storage_writer=writer, process_group=self.dcp_group)
        self.metrics.trigger_ms = (time.perf_counter() - begin) * 1000

    def active(self) -> bool:
        return bool(self.future is not None and not self.future.done())

    def wait_durable(self) -> None:
        self.future.result()
        dist.barrier(group=self.dcp_group)
        self._finish_durability()

    def select_restore_step(self, step: int) -> None:
        super().select_restore_step(step)
        self.path = self.root / f"step_{step:07d}"

    def restore(self) -> dict[str, Any]:
        assert self.path is not None
        begin = time.perf_counter()
        model_state, optim_state = get_state_dict(self.model, self.optimizer, options=self.options)
        current_rng = rank_rng_state(self.rng, self.device)
        state = {
            "model": model_state,
            "optimizer": optim_state,
            "step": torch.tensor(-1, dtype=torch.int64),
            **{key + "_by_rank": torch.empty((self.world_size, *value.shape), dtype=value.dtype) for key, value in current_rng.items()},
        }
        dcp.load(state, checkpoint_id=self.path)
        set_state_dict(self.model, self.optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"], options=self.options)
        for key in current_rng:
            restored = state[key + "_by_rank"][self.rank]
            if key == "rng_cpu":
                torch.set_rng_state(restored.cpu())
            elif key == "rng_cuda":
                torch.cuda.set_rng_state(restored.cpu(), device=self.device)
            else:
                self.rng.set_state(restored.cpu())
        self.metrics.restore_ms = (time.perf_counter() - begin) * 1000
        return self._verify(int(state["step"].item()))


class TempoBackend(DCPBackend):
    def __init__(self, **kwargs: Any) -> None:
        CheckpointBackend.__init__(self, **kwargs)
        self.impl = GroupCreditCheckpointer(
            policy="group",
            checkpoint_root=self.root,
            rank=self.rank,
            world_size=self.world_size,
            control_group=self.control_group,
            dcp_group=self.dcp_group,
            credit_bytes=self.args.credit_mb * 1024 * 1024,
            target_slowdown=self.args.target_slowdown,
            deadline_seconds=self.args.deadline_seconds,
        )
        self.path: Path | None = None

    def start(self, step: int) -> None:
        self.prepare_common(step)
        begin = time.perf_counter()
        model_state, optim_state = get_state_dict(self.model, self.optimizer, options=self.options)
        rng_by_rank = gathered_rng_state(self.expected_rng or {}, self.world_size, self.control_group)
        state = {"model": model_state, "optimizer": optim_state, "step": torch.tensor(step, dtype=torch.int64), **rng_by_rank}
        self.impl.start(state, step)
        self.metrics.state_bytes_local = self.impl.metrics.state_bytes if self.impl.metrics else 0
        self.metrics.trigger_ms = (time.perf_counter() - begin) * 1000
        self.path = self.root / f"step_{step:07d}"
        self.metrics.checkpoint_path = str(self.path)

    def before_optimizer_step(self) -> float:
        blocked = self.impl.before_optimizer_step()
        self.metrics.consistency_block_ms += blocked
        return blocked

    def after_optimizer_step(self) -> float:
        blocked = self.impl.after_optimizer_step()
        self.metrics.consistency_block_ms += blocked
        return blocked

    def set_compute(self) -> None:
        self.impl.phase.set_compute()

    def set_collective(self) -> None:
        self.impl.phase.set_collective()

    def observe_collective(self, latency_ms: float, baseline_sample: bool) -> None:
        self.impl.observe_collective(latency_ms, baseline_sample=baseline_sample)

    def active(self) -> bool:
        return self.impl.active

    def wait_durable(self) -> None:
        result = self.impl.wait_durable()
        self.metrics.durable_ms = result.durable_ms
        self.metrics.deadline_met = result.deadline_met


class TorchSnapshotBackend(CheckpointBackend):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from torchsnapshot import Snapshot
        from torchsnapshot.tricks.fsdp import FSDPOptimizerAdapter

        self.Snapshot = Snapshot
        self.optimizer_adapter = FSDPOptimizerAdapter(self.model, self.optimizer)
        self.training_state: TrainingState | None = None
        self.pending: Any = None
        self.path: Path | None = None

    def start(self, step: int) -> None:
        self.prepare_common(step)
        begin = time.perf_counter()
        self.training_state = TrainingState(step, self.rng, self.device)
        self.path = self.root / f"step_{step:07d}"
        self.metrics.checkpoint_path = str(self.path)
        app_state = {"model": self.model, "optimizer": self.optimizer_adapter, "training": self.training_state}
        # SHARDED_STATE_DICT preserves original parameter keys with
        # use_orig_params=True; LOCAL_STATE_DICT exposes _flat_param keys that
        # PyTorch 2.8 cannot load back into this FSDP configuration.
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
            self.pending = self.Snapshot.async_take(path=str(self.path), app_state=app_state, pg=self.control_group)
        self.metrics.trigger_ms = (time.perf_counter() - begin) * 1000
        local_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        local_bytes += sum(v.numel() * v.element_size() for state in self.optimizer.state.values() for v in state.values() if torch.is_tensor(v))
        self.metrics.state_bytes_local = local_bytes

    def active(self) -> bool:
        return bool(self.pending is not None and not self.pending.done())

    def wait_durable(self) -> None:
        self.pending.wait()
        dist.barrier(group=self.control_group)
        self._finish_durability()

    def select_restore_step(self, step: int) -> None:
        super().select_restore_step(step)
        self.path = self.root / f"step_{step:07d}"

    def restore(self) -> dict[str, Any]:
        assert self.path is not None
        begin = time.perf_counter()
        restored_training = TrainingState(-1, self.rng, self.device)
        app_state = {"model": self.model, "optimizer": self.optimizer_adapter, "training": restored_training}
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
            self.Snapshot(path=str(self.path), pg=self.control_group).restore(app_state)
        self.metrics.restore_ms = (time.perf_counter() - begin) * 1000
        return self._verify(restored_training.step)


class DataStatesBackend(CheckpointBackend):
    COMMIT_POLICIES = frozenset(("datastates", "v4_open", "tempo_v4"))

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from datastates import CheckpointEngine

        self.engine = CheckpointEngine({"host_cache_size": self.args.datastates_cache_gb, "engine_type": "state_engine"}, rank=self.rank)
        self.path: Path | None = None
        self.saved_names: list[str] = []
        self.done = threading.Event()
        self.done.set()
        self.d2h_done = threading.Event()
        self.d2h_done.set()
        self.worker: threading.Thread | None = None
        self.error: BaseException | None = None
        self.closed = False
        self.prewarmed = False
        self.stage_stats_at_start: dict[str, Any] = {}
        self._logical_d2h_busy_ns = 0
        self._logical_pfs_busy_ns = 0

    def _capture_tier_stage_stats(self, *, phase: str) -> None:
        """Persist an exact engine counter snapshot for tier attribution.

        Isolated G1 modes use the same DataStates engine as the legacy/open
        paths.  Keeping the raw snapshots in each checkpoint event lets the
        post-run validator join service bytes with path evidence instead of
        inferring a domain from topology or a duration alone.
        """

        get_stats = getattr(self.engine._engine.ckpt_engine, "get_stage_stats", None)
        if get_stats is None:
            return
        raw = json.loads(get_stats())
        if not isinstance(raw, dict):
            raise RuntimeError("DataStates stage stats must be a JSON object")
        if phase == "start":
            self.metrics.tier_stage_stats_start = raw
        elif phase == "end":
            self.metrics.tier_stage_stats_end = raw
        else:
            raise ValueError(f"unknown tier stage-stat phase: {phase}")

    @staticmethod
    def _logical_stage_snapshot(
        *,
        d2h_bytes: int,
        pfs_bytes: int,
        d2h_busy_ns: int,
        pfs_busy_ns: int,
        d2h_requests: int = 1,
        pfs_requests: int = 1,
    ) -> dict[str, Any]:
        """Build a logical stage ledger without inventing hardware counters.

        Native DataStates can legitimately return zero TEMPO admission stats
        because it is not using the TEMPO controller.  The G1 tier matrix
        still needs to distinguish the logical D2H and persistence stages.
        This record uses exact state/file bytes and measured wait intervals;
        it must never be interpreted as PCIe, CXI, or OST hardware traffic.
        """

        def stage(total: int, busy_ns: int, requests: int) -> dict[str, int]:
            total = int(total)
            busy_ns = int(busy_ns)
            requests = int(requests) if total else 0
            now = time.monotonic_ns() if total else 0
            return {
                "total_bytes": total,
                "queued_bytes": 0,
                "ready_bytes": 0,
                "admitted_bytes": total,
                "completed_bytes": total,
                "inflight_bytes": 0,
                "inflight_requests": 0,
                "admitted_requests": requests,
                "max_request_bytes": total,
                "peak_inflight_bytes": total,
                "peak_inflight_requests": 1 if total else 0,
                "last_progress_monotonic_ns": now,
                "last_completion_monotonic_ns": now,
                "busy_ns": busy_ns,
            }

        return {
            "schema": "tempo-rd-logical-stage-timing-1",
            "counter_semantics": "logical_bytes_and_wait_interval",
            "hardware_counter": False,
            "d2h": stage(d2h_bytes, d2h_busy_ns, d2h_requests),
            "pfs": stage(pfs_bytes, pfs_busy_ns, pfs_requests),
        }

    def prepare(self) -> None:
        """Exercise the same FSDP state-dict path before measured checkpoints."""

        if self.prewarmed:
            return
        begin = time.perf_counter()
        get_state_dict(self.model, self.optimizer, options=self.options)
        self.metrics.prepare_ms = (time.perf_counter() - begin) * 1000
        self.prewarmed = True

    def _tier_mode(self) -> str:
        mode = str(getattr(self.args, "tier_mode", "combined") or "combined")
        if mode not in {"open_combined", "combined", "d2h_only", "persist_only"}:
            raise RuntimeError(
                f"DataStates backend does not implement tier mode {mode!r}; "
                "unsupported modes must fail closed"
            )
        return mode

    def _checkpoint_path_for_tier(self, step: int, mode: str) -> Path:
        if mode != "d2h_only":
            self.metrics.tier_endpoint = "persistent_endpoint"
            return self.root / f"step_{step:07d}" / f"rank_{self.rank:05d}.ds"
        local_root = str(os.environ.get("TEMPO_RD_LOCAL_SINK_ROOT", "")).strip()
        if not local_root:
            raise RuntimeError(
                "d2h_only requires TEMPO_RD_LOCAL_SINK_ROOT; refusing to "
                "silently write the attribution mode to persistent storage"
            )
        local_path = Path(local_root)
        if not local_path.is_absolute() or "lustre" in local_path.as_posix().lower():
            raise RuntimeError("d2h_only sink must be an absolute non-Lustre path")
        self.metrics.tier_endpoint = "node_local_sink"
        return local_path / f"step_{step:07d}" / f"rank_{self.rank:05d}.ds"

    @staticmethod
    def _host_preload_payload(value: Any) -> Any:
        """Synchronously materialize tensors on host memory for PFS-only mode."""

        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: DataStatesBackend._host_preload_payload(item) for key, item in value.items()}
        if isinstance(value, list):
            return [DataStatesBackend._host_preload_payload(item) for item in value]
        if isinstance(value, tuple):
            return tuple(DataStatesBackend._host_preload_payload(item) for item in value)
        return value

    def start(self, step: int) -> None:
        if self.active():
            raise RuntimeError("DataStates does not support overlapping checkpoints")
        if self.worker is not None:
            self.worker.join()
        self.error = None
        self.prepare_common(step)
        tier_mode = self._tier_mode()
        # G1 attribution keeps the persistent modes on the DataStates engine,
        # but d2h_only is a host-copy endpoint rather than a persistence
        # endpoint.  Perlmutter's node-local tmpfs does not support O_DIRECT;
        # sending that endpoint through the controlled host tier would make a
        # nominally D2H-only measurement fail before the copy is observed.
        self.metrics.tier_mode = tier_mode
        self.metrics.tier_gpu_transfer = tier_mode != "persist_only"
        self.metrics.tier_host_preloaded = tier_mode == "persist_only"
        begin = time.perf_counter()
        state_dict_begin = time.perf_counter()
        model_state, optim_state = get_state_dict(self.model, self.optimizer, options=self.options)
        self.metrics.state_dict_ms = (time.perf_counter() - state_dict_begin) * 1000
        payload_begin = time.perf_counter()
        leaves = list(local_tensor_leaves({"model": model_state, "optimizer": optim_state}))
        self.saved_names = [name for name, _ in leaves]
        payload: dict[str, Any] = {f"tensor_{index:06d}": tensor for index, (_, tensor) in enumerate(leaves)}
        payload.update(
            {
                "tensor_names": self.saved_names,
                "optimizer_param_groups": optim_state.get("param_groups", []),
                "step": step,
                **(self.expected_rng or {}),
            }
        )
        if tier_mode == "persist_only":
            payload = self._host_preload_payload(payload)
        self.metrics.state_bytes_local = sum(t.numel() * t.element_size() for _, t in leaves)
        self.metrics.payload_build_ms = (time.perf_counter() - payload_begin) * 1000
        self.path = self._checkpoint_path_for_tier(step, tier_mode)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics.checkpoint_path = str(self.path)
        self.done.clear()
        self.d2h_done.clear()
        if tier_mode == "d2h_only":
            self.stage_stats_at_start = self._d2h_only_stage_snapshot(0, 0)
            self.stage_stats_at_start["logical_stage"] = self._logical_stage_snapshot(
                d2h_bytes=0, pfs_bytes=0, d2h_busy_ns=0, pfs_busy_ns=0
            )
            self.metrics.tier_stage_stats_start = dict(self.stage_stats_at_start)
            self.worker = threading.Thread(
                target=self._complete_d2h_only,
                args=(payload,),
                name=f"datastates-d2h-only-rank{self.rank}",
                daemon=False,
            )
            self.worker.start()
            return
        self.engine._engine.ckpt_engine.configure_d2h_pacing(0.0, 0)
        get_stats = getattr(self.engine._engine.ckpt_engine, "get_stage_stats", None)
        if get_stats is not None:
            self.stage_stats_at_start = json.loads(get_stats())
            self.metrics.tier_stage_stats_start = dict(self.stage_stats_at_start)
        self.metrics.tier_stage_stats_start["logical_stage"] = self._logical_stage_snapshot(
            d2h_bytes=0,
            pfs_bytes=0,
            d2h_busy_ns=0,
            pfs_busy_ns=0,
        )
        save_begin = time.perf_counter()
        self.engine.save(payload, str(self.path))
        self.metrics.engine_save_ms = (time.perf_counter() - save_begin) * 1000
        self.metrics.trigger_ms = (time.perf_counter() - begin) * 1000
        self.worker = threading.Thread(target=self._complete, name=f"datastates-rank{self.rank}", daemon=False)
        self.worker.start()

    @staticmethod
    def _d2h_only_stage_snapshot(total_bytes: int, completed_bytes: int) -> dict[str, Any]:
        """Return an engine-shaped snapshot for the non-persistent host endpoint."""

        now = time.monotonic_ns()
        d2h = {
            "total_bytes": int(total_bytes),
            "queued_bytes": 0,
            "ready_bytes": 0,
            "admitted_bytes": int(completed_bytes),
            "completed_bytes": int(completed_bytes),
            "inflight_bytes": 0,
            "inflight_requests": 0,
            "admitted_requests": 1 if completed_bytes else 0,
            "max_request_bytes": int(completed_bytes),
            "peak_inflight_bytes": int(completed_bytes),
            "peak_inflight_requests": 1 if completed_bytes else 0,
            "last_progress_monotonic_ns": now if completed_bytes else 0,
            "last_completion_monotonic_ns": now if completed_bytes else 0,
        }
        pfs = {key: 0 for key in d2h}
        return {
            "d2h": d2h,
            "pfs": pfs,
            "pfs_odirect_required": False,
            "pfs_odirect_verified": False,
            "pfs_fsync_complete": False,
            "pfs_fsync_monotonic_ns": 0,
            "tier_endpoint": "node_local_sink",
        }

    def _complete_d2h_only(self, payload: dict[str, Any]) -> None:
        """Copy CUDA state to a node-local regular-file sink without PFS/O_DIRECT."""

        try:
            copy_begin_ns = time.perf_counter_ns()
            host_payload = self._host_preload_payload(payload)
            copy_busy_ns = time.perf_counter_ns() - copy_begin_ns
            self.metrics.tier_d2h_copy_ms = copy_busy_ns / 1_000_000
            self.d2h_done.set()
            assert self.path is not None
            torch.save(host_payload, self.path)
            logical_bytes = self.path.stat().st_size
            self.metrics.logical_file_extent_bytes = int(logical_bytes)
            self.metrics.tier_stage_stats_end = self._d2h_only_stage_snapshot(
                self.metrics.state_bytes_local, self.metrics.state_bytes_local
            )
            self.metrics.tier_stage_stats_end["logical_stage"] = self._logical_stage_snapshot(
                d2h_bytes=self.metrics.state_bytes_local,
                pfs_bytes=0,
                d2h_busy_ns=copy_busy_ns,
                pfs_busy_ns=0,
            )
            self._finish_durability()
        except BaseException as exc:
            self.error = exc
        finally:
            self.d2h_done.set()
            self.done.set()

    def _complete(self) -> None:
        try:
            # DataStates may lazily read live model/optimizer tensors only while
            # they are immutable.  Signal the training thread as soon as D2H is
            # complete, then continue the host-to-PFS drain in the background.
            d2h_begin_ns = time.perf_counter_ns()
            self.engine.wait(False)
            self._logical_d2h_busy_ns = time.perf_counter_ns() - d2h_begin_ns
            self.d2h_done.set()
            if self._tier_mode() == "d2h_only":
                # This mode is intentionally a transfer attribution endpoint,
                # not a durability experiment.  Do not silently wait for or
                # report a local-sink fsync as if it were the persistent stage.
                self._capture_tier_stage_stats(phase="end")
                self._finish_durability()
                return
            pfs_begin_ns = time.perf_counter_ns()
            self.engine.wait(True)
            self._logical_pfs_busy_ns = time.perf_counter_ns() - pfs_begin_ns
            barrier_begin = time.perf_counter()
            if self.args.policy in self.COMMIT_POLICIES:
                self._publish_global_commit()
            else:
                dist.barrier(group=self.control_group)
            self.metrics.durability_barrier_ms = (time.perf_counter() - barrier_begin) * 1000
            logical_pfs_bytes = int(self.metrics.checkpoint_file_bytes)
            self.metrics.tier_stage_stats_end["logical_stage"] = self._logical_stage_snapshot(
                d2h_bytes=self.metrics.state_bytes_local if self._tier_mode() != "persist_only" else 0,
                pfs_bytes=logical_pfs_bytes,
                d2h_busy_ns=self._logical_d2h_busy_ns,
                pfs_busy_ns=self._logical_pfs_busy_ns,
            )
            self._finish_durability()
        except BaseException as exc:
            self.error = exc
        finally:
            self.d2h_done.set()
            self.done.set()

    def _local_durability_evidence(self) -> dict[str, Any]:
        get_stats = getattr(self.engine._engine.ckpt_engine, "get_stage_stats", None)
        evidence: dict[str, Any] = {"kind": "engine_wait_true"}
        if get_stats is not None:
            raw = json.loads(get_stats())
            self.metrics.tier_stage_stats_end = dict(raw)
            previous_fsync_ns = int(self.stage_stats_at_start.get("pfs_fsync_monotonic_ns", 0))
            fsync_ns = int(raw.get("pfs_fsync_monotonic_ns", 0))
            evidence.update(
                {
                    "pfs_fsync_complete": bool(raw.get("pfs_fsync_complete", False)),
                    "pfs_fsync_monotonic_ns": fsync_ns,
                    "event_start_pfs_fsync_monotonic_ns": previous_fsync_ns,
                }
            )
            if not bool(raw.get("pfs_fsync_complete", False)) or fsync_ns <= previous_fsync_ns:
                raise RuntimeError("DataStates wait(True) lacks fresh fsync evidence")
        self.metrics.fsync_evidence_valid = True
        return evidence

    def _publish_global_commit(self) -> None:
        assert self.path is not None and self.checkpoint_step is not None
        assert self.expected_checksum is not None
        assert self.expected_optimizer_checksum is not None
        assert self.expected_rng is not None
        checkpoint_stat = self.path.stat()
        checkpoint_bytes = checkpoint_stat.st_size
        checkpoint_allocated_bytes = checkpoint_stat.st_blocks * 512
        self._validate_checkpoint_file_extent(checkpoint_bytes)
        if checkpoint_allocated_bytes < checkpoint_bytes:
            raise RuntimeError(
                f"checkpoint file is sparse after fsync: logical={checkpoint_bytes} "
                f"allocated={checkpoint_allocated_bytes} path={self.path}"
            )
        self.metrics.checkpoint_file_bytes = checkpoint_bytes
        self.metrics.checkpoint_allocated_bytes = checkpoint_allocated_bytes
        # The tier-attribution DataStates path predates the synchronous logical
        # layout publication API used by v4.  Its physical extent is already
        # validated after wait(True)+fsync, so bind that exact extent as the
        # logical extent instead of leaving a misleading zero in the G1 record.
        if self.metrics.logical_file_extent_bytes == 0:
            self.metrics.logical_file_extent_bytes = int(checkpoint_bytes)
        manifest = {
            "schema_version": "tempo-global-commit-1",
            "policy": self.args.policy,
            "step": self.checkpoint_step,
            "rank": self.rank,
            "world_size": self.world_size,
            "checkpoint_path": str(self.path),
            "checkpoint_file_bytes": checkpoint_bytes,
            "checkpoint_allocated_bytes": checkpoint_allocated_bytes,
            "logical_file_extent_bytes": self.metrics.logical_file_extent_bytes,
            "logical_layout_publication_sequence": (
                self.metrics.logical_layout_publication_sequence
            ),
            "logical_layout_version": self.metrics.logical_layout_version,
            "state_bytes_local": self.metrics.state_bytes_local,
            "tensor_names_sha256": canonical_sha256(self.saved_names),
            "model_checksum": self.expected_checksum,
            "optimizer_checksum": self.expected_optimizer_checksum,
            "rng_sha256": rng_state_sha256(self.expected_rng),
            "controller_sha256": self.metrics.v4_controller_sha256,
            "durability_evidence": self._local_durability_evidence(),
        }
        # Gather the already-fsynced rank evidence in memory.  Writing sixteen
        # separate manifest files before the marker either performs sixteen
        # metadata fsyncs or makes the marker's directory fsync flush sixteen
        # dirty files.  Both were measured at 170--440 ms on Lustre.  A fixed
        # CPU frame keeps the commit proof self-contained and makes the one
        # durable marker below the only metadata commit point.
        manifest_bytes = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        manifest_frame_bytes = 16 << 10
        manifest_header_bytes = 4 + hashlib.sha256().digest_size
        if len(manifest_bytes) > manifest_frame_bytes - manifest_header_bytes:
            raise RuntimeError("durability manifest exceeds its fixed gather frame")
        manifest_frame = bytearray(manifest_frame_bytes)
        manifest_frame[:4] = len(manifest_bytes).to_bytes(4, "big")
        manifest_frame[4:manifest_header_bytes] = hashlib.sha256(manifest_bytes).digest()
        manifest_frame[manifest_header_bytes:manifest_header_bytes + len(manifest_bytes)] = (
            manifest_bytes
        )
        manifest_send = torch.from_numpy(
            np.frombuffer(bytes(manifest_frame), dtype=np.uint8).copy()
        )
        manifest_receive = [torch.empty_like(manifest_send) for _ in range(self.world_size)]
        dist.all_gather(manifest_receive, manifest_send, group=self.control_group)
        manifests: list[dict[str, Any]] = []
        for peer, tensor in enumerate(manifest_receive):
            frame = tensor.numpy().tobytes()
            length = int.from_bytes(frame[:4], "big")
            digest_bytes = frame[4:manifest_header_bytes]
            if not 0 < length <= manifest_frame_bytes - manifest_header_bytes:
                raise RuntimeError(f"invalid durability manifest length from rank {peer}")
            encoded = frame[manifest_header_bytes:manifest_header_bytes + length]
            if (
                hashlib.sha256(encoded).digest() != digest_bytes
                or any(frame[manifest_header_bytes + length:])
            ):
                raise RuntimeError(f"corrupt durability manifest frame from rank {peer}")
            peer_manifest = json.loads(encoded)
            if (
                not isinstance(peer_manifest, dict)
                or json.dumps(
                    peer_manifest, sort_keys=True, separators=(",", ":")
                ).encode("utf-8") != encoded
                or int(peer_manifest.get("rank", -1)) != peer
                or int(peer_manifest.get("step", -1)) != self.checkpoint_step
                or int(peer_manifest.get("world_size", -1)) != self.world_size
                or str(peer_manifest.get("policy", "")) != self.args.policy
                or int(peer_manifest.get("checkpoint_file_bytes", 0)) <= 0
                or int(peer_manifest.get("checkpoint_allocated_bytes", 0))
                < int(peer_manifest.get("checkpoint_file_bytes", 0))
                or (
                    self.args.policy in ("v4_open", "tempo_v4")
                    and int(peer_manifest.get("logical_file_extent_bytes", 0))
                    != int(peer_manifest.get("checkpoint_file_bytes", 0))
                )
            ):
                raise RuntimeError(f"invalid durability manifest from rank {peer}")
            manifests.append(peer_manifest)

        marker_path = self.path.parent / "GLOBAL_COMMIT.json"
        identity = canonical_sha256(manifests)
        if self.rank == 0:
            marker = {
                "schema_version": "tempo-global-commit-1",
                "policy": self.args.policy,
                "step": self.checkpoint_step,
                "world_size": self.world_size,
                "manifest_sha256": identity,
                "manifest_count": len(manifests),
                "manifests": manifests,
                "controller_sha256": self.metrics.v4_controller_sha256,
                "committed_unix_ns": time.time_ns(),
            }
            atomic_durable_json(marker_path, marker)
        dist.barrier(group=self.control_group)
        marker = json.loads(marker_path.read_text())
        embedded_manifests = marker.get("manifests")
        if (
            marker.get("schema_version") != "tempo-global-commit-1"
            or marker.get("policy") != self.args.policy
            or int(marker.get("step", -1)) != self.checkpoint_step
            or int(marker.get("world_size", -1)) != self.world_size
            or not isinstance(embedded_manifests, list)
            or len(embedded_manifests) != self.world_size
            or embedded_manifests != manifests
            or canonical_sha256(embedded_manifests) != identity
            or marker.get("controller_sha256", "") != self.metrics.v4_controller_sha256
        ):
            raise RuntimeError(f"global checkpoint commit marker is invalid: {marker_path}")
        self.metrics.commit_marker_path = str(marker_path)
        self.metrics.commit_manifest_sha256 = identity
        self.metrics.commit_validated = True

    def _validate_checkpoint_file_extent(self, checkpoint_bytes: int) -> None:
        """Validate a durable physical file against any pre-I/O layout claim."""

        if checkpoint_bytes <= 0:
            raise RuntimeError("checkpoint file is empty after DataStates wait(True)")
        announced = int(self.metrics.logical_file_extent_bytes)
        if announced > 0 and checkpoint_bytes != announced:
            raise RuntimeError(
                "durable checkpoint extent differs from the synchronously published "
                f"logical layout: physical={checkpoint_bytes} logical={announced} "
                f"path={self.path}"
            )

    def active(self) -> bool:
        return not self.done.is_set()

    def d2h_active(self) -> bool:
        return not self.d2h_done.is_set()

    def before_optimizer_step(self) -> float:
        if self.d2h_done.is_set():
            if self.error is not None:
                raise RuntimeError("DataStates D2H staging failed") from self.error
            return 0.0
        begin = time.perf_counter()
        self.d2h_done.wait()
        blocked_ms = (time.perf_counter() - begin) * 1000
        self.metrics.consistency_block_ms += blocked_ms
        self.metrics.d2h_wait_ms += blocked_ms
        if self.error is not None:
            raise RuntimeError("DataStates D2H staging failed") from self.error
        return blocked_ms

    def wait_durable(self) -> None:
        self.done.wait()
        if self.worker is not None:
            self.worker.join()
        if self.error is not None:
            raise RuntimeError("DataStates background persistence failed") from self.error
        if self._tier_mode() == "d2h_only":
            self.metrics.d2h_first_issue_unix_ns = 0
            self.metrics.d2h_first_issue_corrected_ns = 0
            return
        first_issue_ns = int(
            self.engine._engine.ckpt_engine.get_d2h_first_issue_unix_ns()
        )
        self.metrics.d2h_first_issue_unix_ns = first_issue_ns
        self.metrics.d2h_first_issue_corrected_ns = (
            first_issue_ns + int(self.args.clock_offset_ns) if first_issue_ns else 0
        )

    def restore(self) -> dict[str, Any]:
        assert self.path is not None
        if self._tier_mode() == "d2h_only":
            raise RuntimeError("d2h_only attribution has no persistent restore endpoint")
        if self.args.policy in self.COMMIT_POLICIES:
            self._verify_global_commit()
        begin = time.perf_counter()
        restored = self.engine.load(str(self.path))
        model_state, optim_state = get_state_dict(self.model, self.optimizer, options=self.options)
        destinations = list(local_tensor_leaves({"model": model_state, "optimizer": optim_state}))
        names = restored["tensor_names"]
        if names != [name for name, _ in destinations]:
            raise RuntimeError("DataStates tensor layout changed between save and restore")
        with torch.no_grad():
            for index, (_, destination) in enumerate(destinations):
                destination.copy_(restored[f"tensor_{index:06d}"].to(destination.device))
        optim_state["param_groups"] = restored["optimizer_param_groups"]
        set_state_dict(self.model, self.optimizer, model_state_dict=model_state, optim_state_dict=optim_state, options=self.options)
        torch.set_rng_state(restored["rng_cpu"].cpu())
        torch.cuda.set_rng_state(restored["rng_cuda"].cpu(), device=self.device)
        self.rng.set_state(restored["input_rng"].cpu())
        self.metrics.restore_ms = (time.perf_counter() - begin) * 1000
        return self._verify(int(restored["step"]))

    def _verify_global_commit(self) -> None:
        assert self.path is not None and self.checkpoint_step is not None
        marker_path = self.path.parent / "GLOBAL_COMMIT.json"
        if not marker_path.is_file():
            raise RuntimeError(f"missing global checkpoint commit marker: {marker_path}")
        marker = json.loads(marker_path.read_text())
        manifests = marker.get("manifests")
        if not isinstance(manifests, list) or len(manifests) != self.world_size:
            raise RuntimeError(
                f"global checkpoint marker lacks embedded rank manifests: {marker_path}"
            )
        if any(
            not isinstance(manifest, dict)
            or int(manifest.get("rank", -1)) != peer
            for peer, manifest in enumerate(manifests)
        ):
            raise RuntimeError(
                f"global checkpoint embedded manifest order is invalid: {marker_path}"
            )
        identity = canonical_sha256(manifests)
        expected_controller = _V4_CONTROLLER_SHA256 if self.args.policy in ("v4_open", "tempo_v4") else ""
        if (
            marker.get("schema_version") != "tempo-global-commit-1"
            or marker.get("policy") != self.args.policy
            or int(marker.get("step", -1)) != self.checkpoint_step
            or int(marker.get("world_size", -1)) != self.world_size
            or int(marker.get("manifest_count", -1)) != self.world_size
            or marker.get("manifest_sha256") != identity
            or marker.get("controller_sha256", "") != expected_controller
        ):
            raise RuntimeError(f"global checkpoint commit marker mismatch: {marker_path}")
        local = manifests[self.rank]
        expected_rng_sha = (
            rng_state_sha256(self.expected_rng) if self.expected_rng is not None else ""
        )
        if (
            int(local.get("rank", -1)) != self.rank
            or Path(str(local.get("checkpoint_path", ""))) != self.path
            or int(local.get("checkpoint_file_bytes", 0)) != self.path.stat().st_size
            or int(local.get("checkpoint_allocated_bytes", 0)) < self.path.stat().st_size
            or (
                self.args.policy in ("v4_open", "tempo_v4")
                and int(local.get("logical_file_extent_bytes", 0))
                != self.path.stat().st_size
            )
            or float(local.get("model_checksum", math.nan)) != self.expected_checksum
            or float(local.get("optimizer_checksum", math.nan))
            != self.expected_optimizer_checksum
            or str(local.get("rng_sha256", "")) != expected_rng_sha
        ):
            raise RuntimeError(f"rank checkpoint manifest mismatch: {self.path}")
        self.metrics.commit_marker_path = str(marker_path)
        self.metrics.commit_manifest_sha256 = identity
        self.metrics.commit_validated = True

    def select_restore_step(self, step: int) -> None:
        super().select_restore_step(step)
        self.path = self.root / f"step_{step:07d}" / f"rank_{self.rank:05d}.ds"

    def close(self) -> None:
        if self.closed:
            return
        self.engine.shutdown()
        self.closed = True


class TempoV2Backend(DataStatesBackend):
    """DataStates data plane with an immutable GPU shadow and paced D2H.

    The shadow removes the copy-on-update stall from TEMPO v1.  D2H chunks are
    deterministically phase-shifted within each node and paced at the minimum
    configured rate that still leaves 20% of the durability deadline for the
    final storage drain.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.shadow_names: list[str] = []
        self.shadow_tensors: list[torch.Tensor] = []
        self.chunk_counts: list[int] = []
        self.prepared = False

    def prepare(self) -> None:
        if self.prepared:
            return
        begin = time.perf_counter()
        model_state, optim_state = get_state_dict(self.model, self.optimizer, options=self.options)
        leaves = list(local_tensor_leaves({"model": model_state, "optimizer": optim_state}))
        if any(not tensor.is_contiguous() for _, tensor in leaves):
            raise RuntimeError("TEMPO v2 requires contiguous local state tensors")
        self.shadow_names = [name for name, _ in leaves]
        self.shadow_tensors = [torch.empty_like(tensor, device=self.device) for _, tensor in leaves]
        self.metrics.prepare_ms = (time.perf_counter() - begin) * 1000
        self.prepared = True

    def set_compute(self) -> None:
        self.engine._engine.ckpt_engine.set_d2h_paused(False)

    def set_collective(self) -> None:
        self.engine._engine.ckpt_engine.set_d2h_paused(True)

    def set_transfer_paused(self, paused: bool) -> None:
        self.engine._engine.ckpt_engine.set_d2h_paused(paused)

    def before_optimizer_step(self) -> float:
        # The immutable GPU shadow, rather than live optimizer state, is the
        # source of D2H, so training may update the live tensors immediately.
        return 0.0

    def chunk_mb(self) -> int:
        return self.args.tempo_v2_chunk_mb

    def pacing_rate(self, state_bytes: int, elapsed_seconds: float) -> float:
        del elapsed_seconds
        deadline_copy_budget = max(0.001, self.args.deadline_seconds * 0.8)
        deadline_rate = state_bytes / deadline_copy_budget
        return max(self.args.tempo_v2_d2h_gbps * 1e9, deadline_rate)

    def pacing_start_delay_ns(self, chunk_bytes: int, rate_bytes_per_second: float) -> int:
        local_slots = max(1, torch.cuda.device_count())
        return int(
            (chunk_bytes / rate_bytes_per_second)
            * 1e9
            * (self.device.index or 0)
            / local_slots
        )

    def defer_pacing_until_after_save(self) -> bool:
        return False

    def configure_transfer_before_save(
        self, state_bytes: int, chunk_bytes: int
    ) -> tuple[bool, float, int]:
        deferred = self.defer_pacing_until_after_save()
        if deferred:
            self.set_transfer_paused(True)
            return True, 0.0, 0
        assert self.trigger_ns is not None
        elapsed_seconds = max(0.0, (time.time_ns() - self.trigger_ns) / 1e9)
        rate_bytes_per_second = self.pacing_rate(state_bytes, elapsed_seconds)
        phase_ns = self.pacing_start_delay_ns(chunk_bytes, rate_bytes_per_second)
        self.engine._engine.ckpt_engine.configure_d2h_pacing(rate_bytes_per_second, phase_ns)
        return False, rate_bytes_per_second, phase_ns

    def configure_transfer_after_save(
        self, state_bytes: int, chunk_bytes: int
    ) -> tuple[float, int]:
        assert self.trigger_ns is not None
        elapsed_seconds = max(0.0, (time.time_ns() - self.trigger_ns) / 1e9)
        rate_bytes_per_second = self.pacing_rate(state_bytes, elapsed_seconds)
        phase_ns = self.pacing_start_delay_ns(chunk_bytes, rate_bytes_per_second)
        self.engine._engine.ckpt_engine.configure_d2h_pacing(
            rate_bytes_per_second, phase_ns
        )
        self.set_transfer_paused(False)
        return rate_bytes_per_second, phase_ns

    def after_engine_save(self) -> None:
        return None

    def start(self, step: int) -> None:
        if self.active():
            raise RuntimeError("TEMPO v2/v3 does not support overlapping checkpoints")
        if self.worker is not None:
            self.worker.join()
        self.error = None
        self.prepare_common(step)
        tier_mode = self._tier_mode()
        self.metrics.tier_mode = tier_mode
        self.metrics.tier_gpu_transfer = tier_mode != "persist_only"
        self.metrics.tier_host_preloaded = tier_mode == "persist_only"
        begin = time.perf_counter()
        if not self.prepared:
            self.prepare()
        state_dict_begin = time.perf_counter()
        model_state, optim_state = get_state_dict(self.model, self.optimizer, options=self.options)
        self.metrics.state_dict_ms = (time.perf_counter() - state_dict_begin) * 1000
        leaves = list(local_tensor_leaves({"model": model_state, "optimizer": optim_state}))
        names = [name for name, _ in leaves]
        if names != self.shadow_names or len(leaves) != len(self.shadow_tensors):
            raise RuntimeError("TEMPO v2 tensor layout changed after shadow preparation")
        if any(not tensor.is_contiguous() for _, tensor in leaves):
            raise RuntimeError("TEMPO v2 requires contiguous local state tensors")

        shadow_begin = time.perf_counter()
        with torch.no_grad():
            for (_, source), shadow in zip(leaves, self.shadow_tensors):
                if source.shape != shadow.shape or source.dtype != shadow.dtype:
                    raise RuntimeError("TEMPO v2 tensor metadata changed after shadow preparation")
                shadow.copy_(source, non_blocking=True)
        torch.cuda.synchronize(self.device)
        self.metrics.shadow_copy_ms = (time.perf_counter() - shadow_begin) * 1000

        payload_begin = time.perf_counter()
        chunk_bytes = self.chunk_mb() * 1024 * 1024
        payload: dict[str, Any] = {}
        self.chunk_counts = []
        chunk_total = 0
        for tensor_index, shadow in enumerate(self.shadow_tensors):
            flat = shadow.reshape(-1)
            elements_per_chunk = max(1, chunk_bytes // shadow.element_size())
            count = 0
            for offset in range(0, flat.numel(), elements_per_chunk):
                payload[f"tensor_{tensor_index:06d}_chunk_{count:04d}"] = flat[
                    offset : offset + elements_per_chunk
                ]
                count += 1
            self.chunk_counts.append(count)
            chunk_total += count

        payload.update(
            {
                "tensor_names": self.shadow_names,
                "tensor_chunk_counts": self.chunk_counts,
                "optimizer_param_groups": optim_state.get("param_groups", []),
                "step": step,
                **(self.expected_rng or {}),
            }
        )
        # PFS-only attribution must begin with a host-resident payload.  The
        # recursive conversion is deliberately performed before engine.save;
        # otherwise the DataStates engine would silently include the GPU->host
        # leg and the mode would be mislabeled as a persistence-only sample.
        if tier_mode == "persist_only":
            payload = self._host_preload_payload(payload)
        self.metrics.payload_build_ms = (time.perf_counter() - payload_begin) * 1000
        state_bytes = sum(t.numel() * t.element_size() for t in self.shadow_tensors)
        if tier_mode == "persist_only":
            self.set_transfer_paused(True)
            deferred_pacing, rate_bytes_per_second, phase_ns = False, 0.0, 0
        else:
            deferred_pacing, rate_bytes_per_second, phase_ns = self.configure_transfer_before_save(
                state_bytes, chunk_bytes
            )

        self.saved_names = self.shadow_names
        self.metrics.state_bytes_local = state_bytes
        self.metrics.d2h_rate_gbps = rate_bytes_per_second / 1e9
        self.metrics.d2h_chunks = chunk_total
        self.metrics.d2h_phase_us = phase_ns / 1000.0
        self.path = self._checkpoint_path_for_tier(step, tier_mode)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics.checkpoint_path = str(self.path)
        self.done.clear()
        self.d2h_done.clear()
        save_begin = time.perf_counter()
        self.engine.save(payload, str(self.path))
        self.metrics.engine_save_ms = (time.perf_counter() - save_begin) * 1000
        self.after_engine_save()
        if deferred_pacing:
            epoch_begin = time.perf_counter()
            rate_bytes_per_second, phase_ns = self.configure_transfer_after_save(
                state_bytes, chunk_bytes
            )
            self.metrics.d2h_rate_gbps = rate_bytes_per_second / 1e9
            self.metrics.d2h_phase_us = phase_ns / 1000.0
            self.metrics.epoch_sync_ms = (time.perf_counter() - epoch_begin) * 1000
        self.metrics.trigger_ms = (time.perf_counter() - begin) * 1000
        self.worker = threading.Thread(target=self._complete, name=f"tempo-v2-rank{self.rank}", daemon=False)
        self.worker.start()

    def restore(self) -> dict[str, Any]:
        assert self.path is not None
        begin = time.perf_counter()
        restored = self.engine.load(str(self.path))
        model_state, optim_state = get_state_dict(self.model, self.optimizer, options=self.options)
        destinations = list(local_tensor_leaves({"model": model_state, "optimizer": optim_state}))
        if restored["tensor_names"] != [name for name, _ in destinations]:
            raise RuntimeError("TEMPO v2 tensor layout changed between save and restore")
        restored_chunk_counts = [int(value) for value in restored["tensor_chunk_counts"]]
        if self.chunk_counts and restored_chunk_counts != self.chunk_counts:
            raise RuntimeError("TEMPO v2 chunk layout changed between save and restore")
        self.chunk_counts = restored_chunk_counts
        if len(self.chunk_counts) != len(destinations):
            raise RuntimeError("TEMPO v2 checkpoint has an invalid chunk-count vector")
        with torch.no_grad():
            for tensor_index, (_, destination) in enumerate(destinations):
                offset = 0
                flat_destination = destination.reshape(-1)
                for chunk_index in range(self.chunk_counts[tensor_index]):
                    chunk = restored[f"tensor_{tensor_index:06d}_chunk_{chunk_index:04d}"]
                    elements = chunk.numel()
                    flat_destination[offset : offset + elements].copy_(chunk.to(destination.device))
                    offset += elements
                if offset != destination.numel():
                    raise RuntimeError("TEMPO v2 restored chunk size mismatch")
        optim_state["param_groups"] = restored["optimizer_param_groups"]
        set_state_dict(self.model, self.optimizer, model_state_dict=model_state, optim_state_dict=optim_state, options=self.options)
        torch.set_rng_state(restored["rng_cpu"].cpu())
        torch.cuda.set_rng_state(restored["rng_cuda"].cpu(), device=self.device)
        self.rng.set_state(restored["input_rng"].cpu())
        self.metrics.restore_ms = (time.perf_counter() - begin) * 1000
        return self._verify(int(restored["step"]))


class CudaCollectiveObserver:
    """Measure CUDA collectives and publish nonblocking local phase updates."""

    COLLECTIVES = ("all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor")

    def __init__(
        self,
        *,
        device: torch.device,
        rank: int,
        clock_offset_ns: int = 0,
        backend: "TempoV3Backend | None" = None,
        activity_backend: Any | None = None,
        phase_listener: Any | None = None,
    ) -> None:
        self.device = device
        self.rank = rank
        self.clock_offset_ns = clock_offset_ns
        self.backend = backend
        # Activity sampling is independent of v3's binary gate.  DataStates,
        # v4_open, and tempo_v4 all need the exact checkpoint-active state at
        # collective ready time even when no legacy pause gate is installed.
        self.activity_backend = activity_backend
        self.phase_listener = phase_listener
        self.originals: dict[str, Any] = {}
        self.lock = threading.Lock()
        self.metadata_lock = threading.Lock()
        self.lifecycle = threading.Condition()
        self.active_wrappers = 0
        self.closed = False
        self.worker_error: BaseException | None = None
        self.inflight = 0
        self.hold_begin: float | None = None
        self.gated_collectives = 0
        self.block_ms = 0.0
        self.hold_ms = 0.0
        self.current_step = -1
        self.current_phase_index = 0
        self.sequence = 0
        self.rows: list[dict[str, Any]] = []
        self.phase_fabric_counters_enabled = (
            os.environ.get("TEMPO_RD_PHASE_FABRIC_COUNTERS", "") == "1"
        )
        self.phase_fabric_counters: list[dict[str, Any]] = []
        # Keep the actual stream objects used for prepared credit callbacks so
        # event-final trace retirement can synchronize those streams even when
        # durability evidence is assembled by DataStates' background thread.
        self.control_streams: dict[int, Any] = {}
        # FSDP may enqueue collectives on multiple CUDA streams.  Credit tokens
        # are rank-global, so placing callbacks directly on those streams can
        # let a numerically later token execute first and make the earlier one
        # stale.  A single transition stream plus event dependencies provides
        # the total order required by the controller without a host sync.
        self.transition_stream: Any | None = None
        self.step_begin_unix_ns: dict[int, int] = {}
        self.step_finish_unix_ns: dict[int, int] = {}
        self.step_notifications: dict[int, list[dict[str, Any]]] = {}
        self.events: queue.Queue[
            tuple[
                int,
                str,
                int,
                bool,
                int,
                float,
                int,
                int,
                int,
                str,
                dict[str, Any],
                torch.cuda.Event,
                torch.cuda.Event,
                torch.cuda.Event,
            ]
            | None
        ] = queue.Queue()
        # A failed rank must not spend the remainder of a debug allocation
        # blocked on this diagnostic queue.  Successful runs still call close()
        # and join deterministically.
        self.worker = threading.Thread(target=self._wait_events, name=f"collective-observer-rank{rank}", daemon=True)
        self.worker.start()
        self._install()

    def set_step(self, step: int) -> None:
        with self.metadata_lock:
            self.current_step = step
            self.current_phase_index = 0
            self.step_begin_unix_ns[step] = time.time_ns()
            self.step_notifications[step] = []

    @staticmethod
    def _signature(phase_index: int, name: str, input_bytes: int, output_bytes: int) -> str:
        return f"{phase_index}:{name}:{input_bytes}:{output_bytes}"

    @staticmethod
    def _profile_completion_unix_ns(
        ready_unix_ns: int, gpu_ms: float, gate_wait_ms: float
    ) -> int:
        """Estimate exposed completion without observer-thread scheduling lag."""

        # start.record() precedes the close token, so CUDA elapsed time already
        # contains the stream callback plus NCCL.  Adding host enqueue time
        # would double-count close and incorrectly charge the post-end open.
        exposed_ns = max(1_000, round((gate_wait_ms + gpu_ms) * 1e6))
        return ready_unix_ns + exposed_ns

    def finish_step(self, step: int, finish_unix_ns: int | None = None) -> None:
        with self.metadata_lock:
            self.step_finish_unix_ns[step] = int(finish_unix_ns or time.time_ns())

    def profile_snapshot(self, step: int | None = None) -> dict[str, Any] | None:
        """Return the newest complete step profile without waiting on CUDA."""

        with self.metadata_lock:
            candidates = [
                item
                for item in self.step_finish_unix_ns
                if item in self.step_notifications and item >= 0
            ]
            if step is None:
                if not candidates:
                    return None
                step = max(candidates)
            if step not in self.step_finish_unix_ns or step not in self.step_notifications:
                return None
            begin_ns = self.step_begin_unix_ns[step]
            finish_ns = self.step_finish_unix_ns[step]
            notifications = [dict(item) for item in self.step_notifications[step]]
        if not notifications:
            return None
        if any(notification.get("completion_unix_ns") is None for notification in notifications):
            return None
        windows: list[dict[str, Any]] = []
        clock_offset_ns = int(getattr(self, "clock_offset_ns", 0))
        previous_completion_ns = begin_ns
        for index, notification in enumerate(notifications):
            ready_ns = int(notification["ready_unix_ns"])
            completion_ns = int(notification["completion_unix_ns"])
            raw_lead_in_ns = ready_ns - previous_completion_ns
            lead_in_installable = bool(
                index == 0
                and
                raw_lead_in_ns > 0
                and str(notification.get("arrival_plan_source", ""))
                not in V4_NONINSTALLABLE_ARRIVAL_SOURCES
            )
            windows.append(
                {
                    "phase_id": 2 * index,
                    "signature": f"lead-in:{notification['signature']}",
                    "kind": "compute",
                    # Arrival skew is charged to the preceding lead-in; the
                    # ready callback cannot retroactively repair it.
                    "duration_ns": max(1_000, raw_lead_in_ns),
                    # Only the entry lead-in is host-visible before the first
                    # FSDP wrapper. Internal wrappers are CUDA-enqueued ahead;
                    # their host timestamp gaps are not admission windows.
                    "installable": lead_in_installable,
                    "start_corrected_ns": previous_completion_ns + clock_offset_ns,
                    "end_corrected_ns": ready_ns + clock_offset_ns,
                }
            )
            windows.append(
                {
                    "phase_id": 2 * index + 1,
                    "signature": str(notification["signature"]),
                    "kind": "collective",
                    "duration_ns": max(1_000, completion_ns - ready_ns),
                    "installable": True,
                    "start_corrected_ns": ready_ns + clock_offset_ns,
                    "end_corrected_ns": completion_ns + clock_offset_ns,
                }
            )
            previous_completion_ns = completion_ns
        windows.append(
            {
                "phase_id": 2 * len(notifications),
                "signature": "compute:step-exit",
                "kind": "compute",
                # Two live 16-rank runs showed that the stream-ordered OPEN to
                # terminal CLOSE interval is <=1 ms even though host profile
                # timestamps report 9--14 ms (544/544 observed transitions).
                # Cap the schedulable interval to the actual token lifetime.
                "duration_ns": min(
                    1_000_000, max(1_000, finish_ns - previous_completion_ns)
                ),
                "installable": finish_ns > previous_completion_ns,
                "start_corrected_ns": previous_completion_ns + clock_offset_ns,
                "end_corrected_ns": finish_ns + clock_offset_ns,
            }
        )
        return {
            "step": step,
            "windows": windows,
            "notifications": notifications,
            "phase_count": len(notifications),
        }

    def healthy(self) -> bool:
        with self.metadata_lock:
            return self.worker_error is None

    @staticmethod
    def _tensor_bytes(value: Any) -> int:
        return int(value.numel() * value.element_size()) if torch.is_tensor(value) and value.is_cuda else 0

    @classmethod
    def _collective_tensor_bytes(
        cls, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[int, int]:
        def argument(index: int, key: str) -> Any:
            return args[index] if len(args) > index else kwargs.get(key)

        if name == "all_reduce":
            value = argument(0, "tensor")
            size = cls._tensor_bytes(value)
            return size, size
        output = argument(0, "output_tensor")
        input_value = argument(1, "input_tensor")
        return cls._tensor_bytes(input_value), cls._tensor_bytes(output)

    @staticmethod
    def _async_requested(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        if "async_op" in kwargs:
            return bool(kwargs["async_op"])
        async_index = {"all_reduce": 3, "all_gather_into_tensor": 3, "reduce_scatter_tensor": 4}[name]
        return len(args) > async_index and bool(args[async_index])

    def _install(self) -> None:
        for name in self.COLLECTIVES:
            original = getattr(dist, name)
            self.originals[name] = original
            setattr(dist, name, self._make_wrapper(name, original))

    def _make_wrapper(self, name: str, original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            input_bytes, output_bytes = self._collective_tensor_bytes(name, args, kwargs)
            if input_bytes == 0 and output_bytes == 0:
                return original(*args, **kwargs)
            if self._async_requested(name, args, kwargs):
                raise RuntimeError(
                    "CudaCollectiveObserver requires synchronous collectives so the quiet-window hold covers completion"
                )
            with self.lifecycle:
                if self.closed:
                    return original(*args, **kwargs)
                self.active_wrappers += 1
            owns_gate = False
            try:
                with self.metadata_lock:
                    sequence = self.sequence
                    self.sequence += 1
                    step = self.current_step
                    is_training_fsdp = (
                        name in ("all_gather_into_tensor", "reduce_scatter_tensor")
                        and step not in self.step_finish_unix_ns
                    )
                    phase_index = self.current_phase_index if is_training_fsdp else -1
                    if is_training_fsdp:
                        self.current_phase_index += 1
                ready_ns = time.time_ns()
                fabric_start = (
                    _phase_hsn_snapshot()
                    if getattr(self, "phase_fabric_counters_enabled", False)
                    and is_training_fsdp
                    else None
                )
                signature = (
                    self._signature(phase_index, name, input_bytes, output_bytes)
                    if is_training_fsdp
                    else f"diagnostic:{name}:{input_bytes}:{output_bytes}"
                )
                phase_metadata: dict[str, Any] = {}
                with self.metadata_lock:
                    # Validation/checkpoint collectives can run after the
                    # training interval has been closed.  Keep measuring them,
                    # but do not let them contaminate the next-step FSDP
                    # signature template.
                    if is_training_fsdp:
                        self.step_notifications.setdefault(step, []).append(
                            {
                                "sequence": sequence,
                                "phase_index": phase_index,
                                "signature": signature,
                                "ready_unix_ns": ready_ns,
                                "ready_corrected_ns": ready_ns + self.clock_offset_ns,
                                "gpu_ms": None,
                                "completion_unix_ns": None,
                            }
                        )
                # Gate while either checkpoint stage is active.  Once D2H has
                # drained, host-to-storage may still be submitting io_uring
                # writes that share Perlmutter's Slingshot fabric with NCCL.
                # The host gate stops new submissions; it does not cancel an
                # I/O request that was already handed to the kernel.
                sampled_activity = bool(
                    self.activity_backend is not None
                    and self.activity_backend.active()
                )
                # v4's listener distinguishes byte/control liveness from the
                # later commit-record lifecycle. Other policies use the exact
                # ready-time backend sample above.
                checkpoint_active_at_ready = bool(
                    phase_metadata.get(
                        "checkpoint_active_at_ready", sampled_activity
                    )
                )
                phase_metadata.setdefault(
                    "checkpoint_active_at_ready", checkpoint_active_at_ready
                )
                should_gate = bool(
                    self.backend is not None and checkpoint_active_at_ready
                )
                gate_wait_ms = self._enter() if should_gate else 0.0
                owns_gate = should_gate
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                control_done = torch.cuda.Event(enable_timing=False)
                stream = torch.cuda.current_stream(self.device)
                callback_stream = stream
                # The measured/exposed interval deliberately contains the
                # stream-ordered close transition and NCCL, but not the next
                # compute/open transition.  control_done is a non-timing fence
                # used only to make plan replacement and trace retirement safe.
                start.record(stream)
                if self.phase_listener is not None and is_training_fsdp:
                    callback_stream = self.get_transition_stream()
                    stream_ready = torch.cuda.Event(enable_timing=False)
                    close_done = torch.cuda.Event(enable_timing=False)
                    stream_ready.record(stream)
                    callback_stream.wait_event(stream_ready)
                    callback_stream_ptr = self.register_control_stream(
                        callback_stream
                    )
                    try:
                        result_metadata = self.phase_listener.on_collective_phase(
                            step=step,
                            phase_index=phase_index,
                            sequence=sequence,
                            signature=signature,
                            ready_unix_ns=ready_ns,
                            cuda_stream_ptr=callback_stream_ptr,
                        )
                        if result_metadata:
                            phase_metadata = result_metadata
                    except BaseException as exc:
                        phase_metadata = {"phase_callback_error": repr(exc)}
                        try:
                            self.phase_listener.on_observer_error(
                                f"phase close callback failed at {signature}: {exc!r}"
                            )
                        except BaseException:
                            pass
                    close_done.record(callback_stream)
                    # The collective cannot begin until its CLOSE token has
                    # executed on the rank-global transition stream.
                    stream.wait_event(close_done)
                result = original(*args, **kwargs)
                fabric_end = (
                    _phase_hsn_snapshot()
                    if fabric_start is not None
                    else None
                )
                if fabric_start is not None and fabric_end is not None:
                    with self.metadata_lock:
                        self.phase_fabric_counters.append(
                            {
                                "rank": self.rank,
                                "step": step,
                                "phase_index": phase_index,
                                "sequence": sequence,
                                "phase_signature": signature,
                                "collective": name,
                                "hostname": fabric_start["hostname"],
                                "source": fabric_start["source"],
                                "start_monotonic_ns": fabric_start[
                                    "timestamp_monotonic_ns"
                                ],
                                "end_monotonic_ns": fabric_end[
                                    "timestamp_monotonic_ns"
                                ],
                                "start_rx_bytes": fabric_start["rx_bytes"],
                                "start_tx_bytes": fabric_start["tx_bytes"],
                                "end_rx_bytes": fabric_end["rx_bytes"],
                                "end_tx_bytes": fabric_end["tx_bytes"],
                                "delta_rx_bytes": max(
                                    0,
                                    fabric_end["rx_bytes"]
                                    - fabric_start["rx_bytes"],
                                ),
                                "delta_tx_bytes": max(
                                    0,
                                    fabric_end["tx_bytes"]
                                    - fabric_start["tx_bytes"],
                                ),
                            }
                        )
                end.record(stream)
                if self.phase_listener is not None and is_training_fsdp:
                    # OPEN is ordered after this collective and after every
                    # prior rank-local transition, even when FSDP switches
                    # CUDA streams between calls.
                    callback_stream.wait_event(end)
                    try:
                        self.phase_listener.on_collective_enqueued(
                            step=step,
                            phase_index=phase_index,
                            sequence=sequence,
                            signature=signature,
                            cuda_stream_ptr=callback_stream_ptr,
                            phase_metadata=phase_metadata,
                        )
                    except BaseException as exc:
                        phase_metadata["phase_open_callback_error"] = repr(exc)
                        try:
                            self.phase_listener.on_observer_error(
                                f"phase open callback failed at {signature}: {exc!r}"
                            )
                        except BaseException:
                            pass
                    control_done.record(callback_stream)
                else:
                    control_done.record(stream)
                self.events.put(
                    (
                        sequence,
                        name,
                        step,
                        should_gate,
                        ready_ns,
                        gate_wait_ms,
                        input_bytes,
                        output_bytes,
                        phase_index,
                        signature,
                        phase_metadata,
                        start,
                        end,
                        control_done,
                    )
                )
                owns_gate = False  # the event worker now owns the hold
                return result
            finally:
                if owns_gate:
                    self._leave_without_event()
                with self.lifecycle:
                    self.active_wrappers -= 1
                    self.lifecycle.notify_all()

        return wrapped

    def _enter(self) -> float:
        assert self.backend is not None
        with self.lock:
            begin = time.perf_counter()
            if self.inflight == 0:
                self.backend.set_transfer_paused(True)
                self.hold_begin = time.perf_counter()
            self.inflight += 1
            self.gated_collectives += 1
            wait_ms = (time.perf_counter() - begin) * 1000
            self.block_ms += wait_ms
            return wait_ms

    def _resume_if_idle(self) -> None:
        self.inflight -= 1
        if self.inflight < 0:
            raise RuntimeError("TEMPO collective gate reference count became negative")
        if self.inflight == 0:
            assert self.backend is not None
            self.backend.set_transfer_paused(False)
            if self.hold_begin is not None:
                self.hold_ms += (time.perf_counter() - self.hold_begin) * 1000
                self.hold_begin = None

    def _leave_without_event(self) -> None:
        with self.lock:
            self._resume_if_idle()

    def _wait_events(self) -> None:
        while True:
            item = self.events.get()
            try:
                if item is None:
                    return
                (
                    sequence,
                    name,
                    step,
                    gated,
                    ready_ns,
                    gate_wait_ms,
                    input_bytes,
                    output_bytes,
                    phase_index,
                    signature,
                    phase_metadata,
                    start,
                    end,
                    control_done,
                ) = item
                try:
                    # The open transition is enqueued after the timing event.
                    # Waiting on this extra fence makes observer-idle imply all
                    # stream callbacks for the phase have completed.
                    control_done.synchronize()
                    gpu_ms = start.elapsed_time(end)
                    completion_unix_ns = time.time_ns()
                    if self.phase_listener is not None and phase_index >= 0:
                        try:
                            self.phase_listener.on_collective_complete(
                                step=step,
                                phase_index=phase_index,
                                sequence=sequence,
                                signature=signature,
                                completion_unix_ns=completion_unix_ns,
                            )
                        except BaseException as exc:
                            try:
                                self.phase_listener.on_observer_error(
                                    f"completion callback failed at {signature}: {exc!r}"
                                )
                            except BaseException:
                                pass
                    with self.metadata_lock:
                        phase_install_ms = float(
                            phase_metadata.get("phase_install_ms", 0.0)
                        )
                        profile_completion_unix_ns = self._profile_completion_unix_ns(
                            ready_ns, gpu_ms, gate_wait_ms
                        )
                        for notification in self.step_notifications.get(step, []):
                            if int(notification["sequence"]) == sequence:
                                notification["gpu_ms"] = gpu_ms
                                notification["phase_install_ms"] = phase_install_ms
                                notification["gate_wait_ms"] = gate_wait_ms
                                notification["completion_unix_ns"] = (
                                    profile_completion_unix_ns
                                )
                                notification["callback_unix_ns"] = completion_unix_ns
                                notification["arrival_plan_source"] = str(
                                    phase_metadata.get(
                                        "arrival_plan_source", "planned_lead_in"
                                    )
                                )
                                break
                        self.rows.append({
                            "rank": self.rank,
                            "sequence": sequence,
                            "step": step,
                            "phase_index": phase_index,
                            "phase_signature": signature,
                            "collective": name,
                            "gated": int(gated),
                            "checkpoint_active_at_ready": int(
                                gated or bool(phase_metadata.get("checkpoint_active_at_ready", False))
                            ),
                            "controlled_at_ready": int(
                                bool(phase_metadata.get("controlled_at_ready", False))
                            ),
                            "controller_plan_version": int(
                                phase_metadata.get("controller_plan_version", 0)
                            ),
                            "logical_phase_id": int(
                                phase_metadata.get("logical_phase_id", -1)
                            ),
                            "runtime_phase_id": int(
                                phase_metadata.get("runtime_phase_id", 0)
                            ),
                            "credit_accepted": int(
                                bool(phase_metadata.get("credit_accepted", False))
                            ),
                            "arrival_runtime_phase_id": int(
                                phase_metadata.get("arrival_runtime_phase_id", 0)
                            ),
                            "arrival_credit_accepted": int(
                                bool(phase_metadata.get("arrival_credit_accepted", False))
                            ),
                            "execution_credit_accepted": int(
                                bool(phase_metadata.get("execution_credit_accepted", False))
                            ),
                            "completion_callback_lag": int(
                                bool(phase_metadata.get("completion_callback_lag", False))
                            ),
                            "arrival_plan_source": str(
                                phase_metadata.get("arrival_plan_source", "")
                            ),
                            "drain_active_at_ready": int(
                                bool(phase_metadata.get("drain_active_at_ready", False))
                            ),
                            "finalize_at_ready": int(
                                bool(phase_metadata.get("finalize_at_ready", False))
                            ),
                            "phase_install_ms": float(
                                phase_metadata.get("phase_install_ms", 0.0)
                            ),
                            "phase_close_enqueue_ms": float(
                                phase_metadata.get("phase_close_enqueue_ms", 0.0)
                            ),
                            "phase_open_enqueue_ms": float(
                                phase_metadata.get("phase_open_enqueue_ms", 0.0)
                            ),
                            "ready_unix_ns": ready_ns,
                            "profile_completion_unix_ns": profile_completion_unix_ns,
                            "completion_callback_unix_ns": completion_unix_ns,
                            "ready_corrected_ns": ready_ns + self.clock_offset_ns,
                            "clock_offset_ns": self.clock_offset_ns,
                            "gate_wait_ms": gate_wait_ms,
                            "tensor_bytes": input_bytes,
                            "input_tensor_bytes": input_bytes,
                            "output_tensor_bytes": output_bytes,
                            "gpu_ms": gpu_ms,
                        })
                except BaseException as exc:
                    with self.metadata_lock:
                        if self.worker_error is None:
                            self.worker_error = exc
                finally:
                    if gated:
                        try:
                            with self.lock:
                                self._resume_if_idle()
                        except BaseException as exc:
                            with self.metadata_lock:
                                if self.worker_error is None:
                                    self.worker_error = exc
            finally:
                self.events.task_done()

    def wait_idle(self) -> None:
        self.events.join()
        with self.metadata_lock:
            error = self.worker_error
        if error is not None:
            raise RuntimeError("CUDA collective observer failed") from error

    def wait_idle_bounded(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while self.events.unfinished_tasks:
            with self.metadata_lock:
                if self.worker_error is not None:
                    return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)
        with self.metadata_lock:
            return self.worker_error is None

    def synchronize_control_streams(self) -> None:
        """Wait for every CUDA stream that carried a credit transition."""

        with self.metadata_lock:
            streams = list(self.control_streams.values())
        for stream in streams:
            stream.synchronize()

    def register_control_stream(self, stream: Any) -> int:
        stream_ptr = int(stream.cuda_stream)
        with self.metadata_lock:
            self.control_streams[stream_ptr] = stream
        return stream_ptr

    def get_transition_stream(self) -> Any:
        """Return the one stream that serializes all rank-local credit tokens."""

        stream = self.transition_stream
        if stream is None:
            stream = torch.cuda.Stream(device=self.device)
            self.transition_stream = stream
        return stream

    def close(self) -> None:
        with self.lifecycle:
            if self.closed:
                return
            self.closed = True
            self.lifecycle.wait_for(lambda: self.active_wrappers == 0)
        for name, original in self.originals.items():
            setattr(dist, name, original)
        self.originals.clear()
        error: BaseException | None = None
        try:
            self.wait_idle()
        except BaseException as exc:
            error = exc
        finally:
            if self.backend is not None:
                try:
                    self.backend.set_transfer_paused(False)
                except BaseException as exc:
                    if error is None:
                        error = exc
            self.events.put(None)
            self.events.join()
            self.worker.join()
        if error is not None:
            raise RuntimeError("failed to close CUDA collective observer") from error


class TempoV3Backend(TempoV2Backend):
    """Deadline-aware staging with exact FSDP collective quiet windows."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gate: CudaCollectiveObserver | None = None
        self.gate_baseline = (0, 0.0, 0.0)

    def attach_gate(self, gate: CudaCollectiveObserver) -> None:
        self.gate = gate

    def chunk_mb(self) -> int:
        return self.args.tempo_v3_chunk_mb

    def pacing_rate(self, state_bytes: int, elapsed_seconds: float) -> float:
        reserve_seconds = self.args.tempo_v3_deadline_reserve_ms / 1000.0
        reserve_seconds += self.args.tempo_v3_epoch_lead_ms / 1000.0
        reserve_seconds += self.args.deadline_seconds * self.args.tempo_v3_collective_reserve
        available = max(0.001, self.args.deadline_seconds - elapsed_seconds - reserve_seconds)
        return max(self.args.tempo_v3_d2h_gbps * 1e9, state_bytes / available)

    def defer_pacing_until_after_save(self) -> bool:
        return True

    def pacing_start_delay_ns(self, chunk_bytes: int, rate_bytes_per_second: float) -> int:
        del chunk_bytes, rate_bytes_per_second
        # Wait until every rank has completed engine.save() before rank 0
        # chooses the future epoch.  Choosing it before this rendezvous can
        # make a slow rank observe an already-expired epoch and launch
        # immediately, defeating group alignment.
        dist.barrier(group=self.control_group)
        # All ranks receive one future wall-clock epoch.  This is a
        # group-synchronized credit schedule, not a claim that the underlying
        # PCIe/NUMA/NIC paths are physically independent.
        epoch = torch.zeros(1, dtype=torch.int64)
        if self.rank == 0:
            epoch[0] = time.time_ns() + int(self.args.tempo_v3_epoch_lead_ms * 1e6)
        dist.broadcast(epoch, src=0, group=self.control_group)
        target_ns = int(epoch.item())
        self.metrics.d2h_epoch_unix_ns = target_ns
        corrected_now_ns = time.time_ns() + int(self.args.clock_offset_ns)
        return max(0, target_ns - corrected_now_ns)

    def start(self, step: int) -> None:
        if self.gate is None:
            raise RuntimeError("TEMPO v3 collective observer was not attached")
        self.gate_baseline = (
            self.gate.gated_collectives,
            self.gate.block_ms,
            self.gate.hold_ms,
        )
        super().start(step)

    def set_compute(self) -> None:
        # Real distributed collectives are intercepted by CudaCollectiveObserver.
        return None

    def set_collective(self) -> None:
        return None

    def set_transfer_paused(self, paused: bool) -> None:
        if paused:
            self.engine._engine.ckpt_engine.set_persistence_paused(True)
            self.engine._engine.ckpt_engine.set_d2h_paused(True)
        else:
            self.engine._engine.ckpt_engine.set_d2h_paused(False)
            self.engine._engine.ckpt_engine.set_persistence_paused(False)

    def wait_durable(self) -> None:
        super().wait_durable()
        if self.gate is None:
            raise RuntimeError("TEMPO v3 collective gate was not attached")
        self.gate.wait_idle()
        gated, block_ms, hold_ms = self.gate_baseline
        self.metrics.collectives_gated = self.gate.gated_collectives - gated
        self.metrics.gate_block_ms = self.gate.block_ms - block_ms
        self.metrics.gate_hold_ms = self.gate.hold_ms - hold_ms


class TempoV4Backend(TempoV2Backend):
    """Group-tail, deadline-projected admission over an identical open path."""

    TELEMETRY_SCHEMA = "tempo-v4-runtime-2"
    TELEMETRY_JOURNAL_SCHEMA = "tempo-v4-local-telemetry-journal-1"
    TELEMETRY_MAX_BYTES = 64 * 1024 * 1024
    TELEMETRY_SUPPRESSED_RECORD_TYPES = frozenset(
        ("phase_complete", "step", "training_step")
    )
    LOGICAL_LAYOUT_SCHEMA_VERSION = 1
    LOGICAL_LAYOUT_KEYS = frozenset(
        (
            "schema_version",
            "publication_sequence",
            "version",
            "path",
            "payload_extent_bytes",
            "metadata_bytes",
            "logical_file_extent_bytes",
            "fs_block_alignment_bytes",
        )
    )

    @staticmethod
    def _commit_durability_evidence(
        full_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the bounded marker commitment for full local evidence."""

        return {
            "kind": "v4_stage_counters_and_fsync_commitment",
            "checkpoint_file_bytes": int(full_evidence["checkpoint_file_bytes"]),
            "event_start_monotonic_ns": int(
                full_evidence["event_start_monotonic_ns"]
            ),
            "pfs_fsync_monotonic_ns": int(
                full_evidence["pfs_fsync_monotonic_ns"]
            ),
            "full_evidence_sha256": canonical_sha256(full_evidence),
        }

    @staticmethod
    def _configured_control_reuse_stride() -> int:
        """Read the gather stride without permitting stale-plan reuse.

        A skipped gather does not advance the controller generation/step
        ledger.  Re-enabling the old stride knob would therefore make the
        next real plan stale and could reopen an already-issued event prefix
        against a fresh runtime base.  Keep the parser explicit so tests and
        manifests can prove that this unsafe mode is rejected rather than
        silently normalized.
        """

        try:
            stride = max(1, int(os.environ.get("TEMPO_V4_GATHER_STRIDE", "1")))
        except (TypeError, ValueError):
            stride = 1
        if stride > 1:
            raise ValueError(
                "TEMPO_V4_GATHER_STRIDE>1 is unsupported until controller "
                "heartbeat/credit-ledger advancement is implemented"
            )
        return stride

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.c0_d2h_rate_bps = _v4_open_c0_rate_bps(self.args)
        self.v4 = v4_controller_module()
        self.config = self.v4.ControllerConfig(
            d2h_quantum_bytes=self.args.tempo_v4_d2h_chunk_mb * MIB,
            pfs_quantum_bytes=self.args.tempo_v4_pfs_chunk_mb * MIB,
            max_pfs_inflight_bytes=self.args.tempo_v4_max_pfs_inflight_mb * MIB,
            max_collective_d2h_requests=V4_MAX_COLLECTIVE_D2H_REQUESTS,
            max_collective_pfs_requests=V4_MAX_COLLECTIVE_PFS_REQUESTS,
            low_slack_ns=int(self.args.tempo_v4_low_slack_ms * 1e6),
            bounded_recovery_slack_ns=int(
                getattr(self.args, "tempo_v4_recovery_slack_ms", 250.0) * 1e6
            ),
            high_slack_ns=int(self.args.tempo_v4_high_slack_ms * 1e6),
            deadline_margin_ns=int(self.args.tempo_v4_deadline_margin_ms * 1e6),
            max_plan_staleness_steps=1,
            watchdog_timeout_ns=int(self.args.tempo_v4_watchdog_ms * 1e6),
        )
        self.controller = self.v4.TempoV4Controller(self.config)
        # Scheduled TEMPO uses a producer-only compute lane: if the horizon
        # target is larger than the currently installable compute prefix, the
        # remainder stays as next-plan debt instead of starting in a
        # collective.  v4_open remains the matched unrestricted reference.
        self.work_conserving_mode = (
            self.args.policy == "tempo_v4"
            and getattr(self.args, "tempo_v4_control_mode", "scheduled")
            == "work_conserving"
        )
        self.controller.compute_only_d2h = (
            self.args.policy == "tempo_v4" and not self.work_conserving_mode
        )
        self.controller.policy_name = str(self.args.policy)
        self.tail_feedback = self.v4.TailFeedback()
        self.scheduled = self.args.policy == "tempo_v4"
        self.split_guard_mode = (
            self.args.policy == "tempo_v4"
            and getattr(self.args, "tempo_v4_control_mode", "scheduled")
            in {"split_guard", "work_conserving"}
        )
        self.split_guard_d2h = D2HCausalGuard(
            quantum_bytes=V4_D2H_REQUEST_MIB * MIB
        )
        # NodePFSLane is retained as an allocation-free reference oracle for
        # unit tests. The live path below is enforced by the C++ rank-local
        # cumulative PFS prefix; no process-shared node daemon is claimed by
        # this experiment.
        self.split_guard_pfs = NodePFSLane(
            quantum_bytes=V4_PFS_REQUEST_MIB * MIB,
            max_inflight_bytes=4 * V4_MAX_COLLECTIVE_PFS_CREDIT_BYTES,
            max_inflight_requests=4 * V4_MAX_COLLECTIVE_PFS_REQUESTS,
        )
        self.stage_floor_provenance = load_v4_stage_floor_provenance(
            self.args,
            self.output_dir,
            self.world_size,
        )
        self.observer: CudaCollectiveObserver | None = None
        self.checkpoint_id = ""
        self.event_base_stats: dict[str, Any] | None = None
        self.event_start_monotonic_ns = 0
        self.event_expected_state_bytes = 0
        self.event_expected_pfs_bytes = 0
        # Rolling PFS leases follow host-ready inventory, not the amount of
        # cumulative credit already issued by an earlier runtime plan.
        self.current_event_host_ready_bytes = 0
        self.current_group_host_ready_bytes = 0
        self.current_group_host_ready_valid = False
        # SplitGuard carries both stage ceilings across controller plan
        # versions.  Without the D2H counterpart, a later gather could reopen
        # the full event payload relative to a new C++ base and duplicate the
        # same event's GPU-facing allowance.
        self.split_guard_d2h_cumulative_ceiling = 0
        self.split_guard_pfs_cumulative_ceiling = 0
        # Once split-guard has entered its finite recovery lane, repeatedly
        # running the full O(W) planner at every scheduled gather only burns
        # the deadline budget without changing the causal lane.  This latch is
        # event-local and is reset for every checkpoint; gathers still happen,
        # terminal/fail-open decisions still happen, and each stream transition
        # receives a fresh runtime slot.
        self.split_guard_recovery_plan_latched = False
        self.split_guard_recovery_replans = 0
        self.split_guard_recovery_plan_step = -1
        self.split_guard_last_full_plan_step = -1
        self.event_layout_pre_save_publication_sequence = 0
        self.event_logical_layout: dict[str, Any] = {}
        self.final_durability_evidence: dict[str, Any] | None = None
        self.event_generation = 0
        self.current_controller_plan: Any | None = None
        self.current_rank_plan: Any | None = None
        self.current_plan_step = -1
        self.current_expected_signatures: list[str] = []
        self.current_seen_phases = 0
        self.current_runtime_plan_version = 0
        self.next_runtime_plan_version = 1
        self.next_runtime_phase_id = 1
        self.current_phase_slots: dict[int, dict[str, Any]] = {}
        self.current_installable_phases: set[int] = set()
        self.current_terminal_logical_phase_id = -1
        self.current_terminal_credit_closed = False
        self.installed_logical_phase_id = -1
        self.installed_runtime_phase_id = 0
        self.installed_credit_accepted = False
        self.installed_d2h_budget_bytes = 0
        self.installed_pfs_budget_bytes = 0
        self.event_installed_runtime_phases: set[tuple[int, int]] = set()
        self.event_runtime_slots: dict[tuple[int, int], dict[str, Any]] = {}
        self.event_terminal_runtime_phases: set[tuple[int, int]] = set()
        self.control_retiring = False
        self.control_finalize_event = threading.Event()
        self.control_gather_schedule = self._build_control_gather_schedule(
            self.args.warmup_steps,
            self.args.checkpoint_steps,
            self.args.window_steps,
        )
        # Explicit work-conserving experiment knob.  The default remains one
        # common packet gather per scheduled step.  A stride >1 lets the
        # controller reuse a validated group plan between gather points while
        # still installing fresh CUDA-ordered transitions and retaining the
        # terminal common gather.  This is intentionally disabled for the
        # strict/split paths.
        self.control_reuse_stride = self._configured_control_reuse_stride()
        if not self.work_conserving_mode:
            self.control_reuse_stride = 1
        # The full-compute and PFS-compute-only lanes are stream-ordered
        # placement experiments, not synonyms for the legacy
        # ``work_conserving`` mode.  They must also be usable with the strict
        # split_guard state machine: otherwise the only way to exercise the
        # exposure-reduction candidate is to enable the collective residual
        # path that it is intended to remove.  The knobs remain opt-in and
        # the default split_guard/scheduled behavior is unchanged.
        self.work_full_compute_lane = bool(
            self.split_guard_mode
            and os.environ.get("TEMPO_V4_FULL_COMPUTE_D2H", "0") == "1"
        )
        # Optional post-save event lease used to diagnose the long-lived
        # reset-first expiry seen in the 4-node work-conserving run.  The
        # bounded 4 MiB bootstrap remains unchanged; after logical-layout
        # publication, only the first ordinary plan may open the remaining
        # event D2H extent.  Later controller plans observe the admitted
        # event-relative counter and open no duplicate lease.  This is kept
        # opt-in until the strict tail/skew gates are rechecked offline.
        self.work_event_d2h_lease = bool(
            self.work_conserving_mode
            and os.environ.get("TEMPO_V4_EVENT_D2H_LEASE", "0") == "1"
        )
        self.work_adaptive_d2h_recovery = bool(
            self.work_conserving_mode
            and os.environ.get("TEMPO_V4_ADAPTIVE_D2H_RECOVERY", "0") == "1"
        )
        self.d2h_event_lease_used = False
        self.d2h_event_lease_plan_version = 0
        # Optional producer/consumer isolation experiment.  The existing
        # work-conserving path opens one 4 MiB PFS lease at every stream
        # boundary, including immediately before NCCL.  That was observable
        # in live traces as a ~0.85 s PFS admission ramp versus ~0.26 s for
        # v4_open.  Keep the default unchanged; this opt-in lane only opens
        # finite PFS prefixes on compute windows and carries the remainder to
        # the next common plan.  It is a measured candidate, not a claim.
        self.pfs_compute_only_lane = bool(
            self.split_guard_mode
            and os.environ.get("TEMPO_V4_PFS_COMPUTE_ONLY", "0") == "1"
        )
        # A future PFS lease is a cumulative *readiness-gated* ceiling.  It
        # lets the host worker start persistence as soon as D2H produces the
        # next extent, without pretending that the whole checkpoint was
        # already host-ready in the producer-lead snapshot.  The physical
        # request path still checks ready_bytes, qP, and the 16 MiB/4-request
        # cap.  Keep this opt-in until it passes the strict analyzer.
        self.pfs_future_lease = bool(
            self.work_conserving_mode
            and os.environ.get("TEMPO_V4_PFS_FUTURE_LEASE", "0") == "1"
        )
        self.current_pfs_future_lease_bytes = 0
        self.current_pfs_ready_budget_bytes = 0
        self.control_gather_calls = 0
        self.control_terminal_gather_calls = 0
        self.control_gather_skips = 0
        self.control_common_terminal_origin: int | None = None
        self.control_common_terminal_mode = ""
        self.control_common_terminal_reason = ""
        self.fail_open_reason = ""
        self.controller_group_failed = False
        self.controller_packet_error = ""
        self.observer_error = ""
        self._controller_packet_send_buffer = torch.zeros(
            V4ControllerPacketCodec.MAX_PACKET_BYTES, dtype=torch.uint8, device="cpu"
        )
        self._controller_packet_recv_buffers = [
            torch.empty_like(self._controller_packet_send_buffer)
            for _ in range(self.world_size)
        ]
        self._controller_packet_last_encoded_bytes = 0
        self.control_lock = threading.RLock()
        self.telemetry_lock = threading.Lock()
        self.telemetry_pending: list[dict[str, Any]] = []
        self.telemetry_last_monotonic_ns = -1
        self.telemetry_records_emitted = 0
        self.telemetry_write_calls = 0
        self.telemetry_shared_write_calls_during_measurement = 0
        self.telemetry_handle: Any | None = None
        self.telemetry_published = False
        self.telemetry_enabled = (
            not bool(self.args.restore_only)
            and str(getattr(self.args, "tempo_v4_telemetry", "required")) == "required"
        )
        if self.telemetry_enabled:
            self._configure_telemetry_journal()
        else:
            self.telemetry_local_path = Path()
            self.telemetry_path = Path()
            self.telemetry_publisher_path = Path()
            self.telemetry_publisher_sha256 = ""
            self.telemetry_preflight_evidence: dict[str, Any] = {}
        self.run_id = f"{time.time_ns()}-{os.getpid()}"
        self.telemetry_structural_marker = self._structural_marker()
        self.telemetry_structural_sha256 = canonical_sha256(
            self.telemetry_structural_marker
        )
        self.baseline_latency_ms: dict[str, list[float]] = {}
        self.baseline_skew_ms: dict[str, list[float]] = {}
        # Previous-step corrected intersections are predictions, not current
        # guarantees.  Live traces showed that 68--70% shrink on the following
        # step and roughly one third disappear entirely.  Learn a deterministic
        # lower-quartile realization ratio during the existing warmup gathers
        # and keep only a short rolling history.  Every rank receives the same
        # packets, so this state evolves identically without another collective.
        self.compute_intersection_ratios_ppm: dict[str, list[int]] = {}
        self.previous_compute_intersections_ns: dict[str, int] = {}
        self.previous_compute_intersection_profile_step: int | None = None
        self.current_projected_intersections: dict[str, int] = {}
        self.last_intersection_diagnostics: list[dict[str, Any]] = []
        self.metrics.v4_controller_sha256 = _V4_CONTROLLER_SHA256
        self.metrics.v4_controller_packet_frame_bytes = (
            V4ControllerPacketCodec.MAX_PACKET_BYTES
        )

    def _configure_telemetry_journal(self) -> None:
        """Bind this rank to a preflighted node-local journal and publisher."""

        environment_names = (
            "TEMPO_V4_TELEMETRY_LOCAL_PATH",
            "TEMPO_V4_TELEMETRY_FINAL_PATH",
            "TEMPO_V4_TELEMETRY_PREFLIGHT_JSON",
            "TEMPO_V4_TELEMETRY_PUBLISHER_PATH",
            "TEMPO_V4_TELEMETRY_PUBLISHER_SHA256",
        )
        values = {name: os.environ.get(name, "") for name in environment_names}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise RuntimeError(
                "v4 requires a preflighted rank-local telemetry journal; missing "
                + ", ".join(missing)
            )
        self.telemetry_local_path = Path(
            values["TEMPO_V4_TELEMETRY_LOCAL_PATH"]
        ).resolve(strict=False)
        self.telemetry_path = Path(
            values["TEMPO_V4_TELEMETRY_FINAL_PATH"]
        ).resolve(strict=False)
        expected_final = (
            self.output_dir.resolve(strict=True)
            / f"tempo_v4_telemetry_rank{self.rank}.jsonl"
        )
        if self.telemetry_path != expected_final:
            raise RuntimeError(
                "v4 telemetry final path differs from the canonical policy artifact: "
                f"actual={self.telemetry_path} expected={expected_final}"
            )
        self.telemetry_preflight_path = Path(
            values["TEMPO_V4_TELEMETRY_PREFLIGHT_JSON"]
        ).resolve(strict=True)
        self.telemetry_publisher_path = Path(
            values["TEMPO_V4_TELEMETRY_PUBLISHER_PATH"]
        ).resolve(strict=True)
        self.telemetry_publisher_sha256 = values[
            "TEMPO_V4_TELEMETRY_PUBLISHER_SHA256"
        ]
        if (
            len(self.telemetry_publisher_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.telemetry_publisher_sha256
            )
            or hashlib.sha256(self.telemetry_publisher_path.read_bytes()).hexdigest()
            != self.telemetry_publisher_sha256
        ):
            raise RuntimeError("v4 telemetry publisher snapshot/hash is invalid")
        try:
            evidence = json.loads(
                self.telemetry_preflight_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("v4 telemetry preflight evidence is unreadable") from exc
        if not isinstance(evidence, dict):
            raise RuntimeError("v4 telemetry preflight evidence is not an object")
        expected_evidence = {
            "schema_version": self.TELEMETRY_JOURNAL_SCHEMA,
            "policy": self.args.policy,
            "rank": self.rank,
            "local_path": str(self.telemetry_local_path),
            "final_path": str(self.telemetry_path),
            "local_filesystem_approved": True,
            "local_non_lustre": True,
            "distinct_device": True,
            "max_journal_bytes": self.TELEMETRY_MAX_BYTES,
            "shared_writes_during_measurement": 0,
            "publication_protocol": "hidden_temp_fsync_replace_dir_fsync",
            "publisher_sha256": self.telemetry_publisher_sha256,
        }
        differences = {
            key: (evidence.get(key), expected)
            for key, expected in expected_evidence.items()
            if evidence.get(key) != expected
        }
        local_mount = evidence.get("local_mount")
        final_mount = evidence.get("final_mount")
        if (
            differences
            or not isinstance(local_mount, dict)
            or not isinstance(final_mount, dict)
            or str(local_mount.get("filesystem_type", ""))
            not in {"btrfs", "ext2", "ext3", "ext4", "overlay", "ramfs", "tmpfs", "xfs"}
            or "lustre" in str(local_mount.get("filesystem_type", "")).lower()
        ):
            raise RuntimeError(
                "v4 telemetry preflight evidence violates the local-journal contract: "
                + json.dumps(differences, sort_keys=True)
            )
        if self.telemetry_local_path.exists() or self.telemetry_path.exists():
            raise RuntimeError("v4 telemetry journal/final target was not fresh")
        self.telemetry_preflight_evidence = evidence
        self.telemetry_handle = self.telemetry_local_path.open(
            "x", encoding="utf-8", buffering=64 * 1024
        )

    @property
    def ckpt_engine(self) -> Any:
        return self.engine._engine.ckpt_engine

    @staticmethod
    def _require_layout_uint64(layout: dict[str, Any], field: str) -> int:
        value = layout.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"DataStates logical layout field {field!r} is not an integer")
        if value < 0 or value > UINT64_MAX:
            raise RuntimeError(f"DataStates logical layout field {field!r} is outside uint64")
        return value

    def _read_logical_layout_envelope(self) -> dict[str, Any]:
        getter = getattr(self.ckpt_engine, "get_last_checkpoint_layout", None)
        if getter is None:
            raise RuntimeError(
                "DataStates lacks the synchronous get_last_checkpoint_layout API"
            )
        try:
            layout = json.loads(getter())
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("DataStates returned malformed logical-layout JSON") from exc
        if not isinstance(layout, dict):
            raise RuntimeError("DataStates logical layout is not a JSON object")
        if set(layout) != self.LOGICAL_LAYOUT_KEYS:
            raise RuntimeError(
                "DataStates logical layout schema keys differ: "
                f"expected={sorted(self.LOGICAL_LAYOUT_KEYS)} actual={sorted(layout)}"
            )
        schema_version = self._require_layout_uint64(layout, "schema_version")
        if schema_version != self.LOGICAL_LAYOUT_SCHEMA_VERSION:
            raise RuntimeError(
                "unsupported DataStates logical layout schema: "
                f"{schema_version}"
            )
        self._require_layout_uint64(layout, "publication_sequence")
        return layout

    def _validate_published_logical_layout(
        self, layout: dict[str, Any]
    ) -> dict[str, Any]:
        if self.path is None:
            raise RuntimeError("v4 checkpoint path is unavailable after engine.save")
        sequence = self._require_layout_uint64(layout, "publication_sequence")
        expected_sequence = self.event_layout_pre_save_publication_sequence + 1
        if sequence != expected_sequence:
            raise RuntimeError(
                "DataStates logical layout publication is stale or skipped: "
                f"before={self.event_layout_pre_save_publication_sequence} "
                f"after={sequence} expected={expected_sequence}"
            )
        version = self._require_layout_uint64(layout, "version")
        engine_version = getattr(self.engine._engine, "last_ckpt_version", None)
        if isinstance(engine_version, bool) or not isinstance(engine_version, int):
            raise RuntimeError("DataStates last checkpoint version is unavailable")
        if version != engine_version:
            raise RuntimeError(
                "DataStates logical layout version mismatch: "
                f"layout={version} engine={engine_version}"
            )
        published_path = layout.get("path")
        expected_path = str(self.path)
        if not isinstance(published_path, str) or published_path != expected_path:
            raise RuntimeError(
                "DataStates logical layout path mismatch: "
                f"layout={published_path!r} expected={expected_path!r}"
            )
        if os.path.realpath(published_path) != os.path.realpath(expected_path):
            raise RuntimeError("DataStates logical layout canonical path mismatch")
        payload_extent = self._require_layout_uint64(layout, "payload_extent_bytes")
        metadata_bytes = self._require_layout_uint64(layout, "metadata_bytes")
        logical_extent = self._require_layout_uint64(
            layout, "logical_file_extent_bytes"
        )
        alignment = self._require_layout_uint64(
            layout, "fs_block_alignment_bytes"
        )
        if (
            payload_extent <= 0
            or metadata_bytes <= 0
            or logical_extent <= 0
            or alignment <= 0
        ):
            raise RuntimeError("DataStates published a non-positive logical file layout")
        if any(
            component % alignment
            for component in (payload_extent, metadata_bytes, logical_extent)
        ):
            raise RuntimeError(
                "DataStates logical layout is not aligned to its published filesystem "
                f"block size: alignment={alignment} payload={payload_extent} "
                f"metadata={metadata_bytes} logical={logical_extent}"
            )
        if payload_extent < self.event_expected_state_bytes:
            raise RuntimeError(
                "DataStates payload extent is smaller than the checkpoint tensor bytes: "
                f"payload={payload_extent} state={self.event_expected_state_bytes}"
            )
        if payload_extent > UINT64_MAX - metadata_bytes:
            raise RuntimeError("DataStates logical layout component sum overflows uint64")
        if logical_extent != payload_extent + metadata_bytes:
            raise RuntimeError(
                "DataStates logical file extent does not equal payload plus metadata: "
                f"logical={logical_extent} payload={payload_extent} "
                f"metadata={metadata_bytes}"
            )
        return dict(layout)

    def attach_observer(self, observer: CudaCollectiveObserver) -> None:
        self.observer = observer

    @staticmethod
    def _build_control_gather_schedule(
        warmup_steps: int,
        checkpoint_steps: list[int],
        window_steps: int,
    ) -> dict[int, tuple[str, int | None]]:
        """Build the rank-invariant bounded controller rendezvous schedule.

        on_step_begin(step) observes profile ``step - 1``.  Warmup therefore
        gathers at 1..warmup_steps inclusive.  Each checkpoint has exactly
        window_steps controlled gathers followed by one status-only terminal
        gather.  No rank-local event/done/error state participates here.
        """

        if warmup_steps < 0 or window_steps < 1:
            raise ValueError("invalid deterministic control-gather bounds")
        schedule: dict[int, tuple[str, int | None]] = {
            step: ("warmup", None) for step in range(1, warmup_steps + 1)
        }
        for checkpoint_step in checkpoint_steps:
            for step in range(
                checkpoint_step + 1,
                checkpoint_step + window_steps + 1,
            ):
                if step in schedule:
                    raise ValueError(f"overlapping control-gather step {step}")
                schedule[step] = ("controlled", checkpoint_step)
            terminal_step = checkpoint_step + window_steps + 1
            if terminal_step in schedule:
                raise ValueError(f"overlapping terminal control-gather step {terminal_step}")
            schedule[terminal_step] = ("terminal", checkpoint_step)
        return schedule

    def chunk_mb(self) -> int:
        # Payload/PFS regions stay at 4 MiB.  The C++ GPU worker fills each
        # region with dynamically granted <=1 MiB D2H subcopies.
        return V4_PAYLOAD_REGION_MIB

    def set_compute(self) -> None:
        # v4 never drives the legacy binary pause switch.
        return None

    def set_collective(self) -> None:
        return None

    def set_transfer_paused(self, paused: bool) -> None:
        del paused
        return None

    def _structural_marker(self) -> dict[str, Any]:
        cached = getattr(self, "telemetry_structural_marker", None)
        if cached is not None:
            return cached
        provenance = self.stage_floor_provenance
        return {
            "immutable_gpu_shadow": True,
            "shadow_copy": True,
            "payload_region_bytes": V4_PAYLOAD_REGION_MIB * MIB,
            "c0_enabled": bool(self.c0_d2h_rate_bps),
            "c0_d2h_rate_bps": int(self.c0_d2h_rate_bps),
            "c0_max_inflight_bytes": V4_D2H_REQUEST_MIB * MIB,
            "telemetry_mode": str(getattr(self.args, "tempo_v4_telemetry", "required")),
            "pinned_region_bytes": V4_PAYLOAD_REGION_MIB * MIB,
            "d2h_chunk_bytes": self.args.tempo_v4_d2h_chunk_mb * MIB,
            "d2h_quantum_bytes": self.args.tempo_v4_d2h_chunk_mb * MIB,
            "pfs_request_bytes": self.args.tempo_v4_pfs_chunk_mb * MIB,
            "pfs_quantum_bytes": self.args.tempo_v4_pfs_chunk_mb * MIB,
            "max_pfs_inflight_bytes": self.args.tempo_v4_max_pfs_inflight_mb * MIB,
            "pfs_inflight_cap_bytes": self.args.tempo_v4_max_pfs_inflight_mb * MIB,
            "max_collective_d2h_credit_bytes": (
                V4_MAX_COLLECTIVE_D2H_CREDIT_BYTES
            ),
            "max_collective_pfs_credit_bytes": (
                V4_MAX_COLLECTIVE_PFS_CREDIT_BYTES
            ),
            "max_collective_d2h_requests": V4_MAX_COLLECTIVE_D2H_REQUESTS,
            "max_collective_pfs_requests": V4_MAX_COLLECTIVE_PFS_REQUESTS,
            "compute_intersection_realization_history": (
                V4_COMPUTE_REALIZATION_HISTORY
            ),
            "compute_intersection_realization_percentile_ppm": (
                V4_COMPUTE_REALIZATION_PERCENTILE_PPM
            ),
            "credit_control_enabled": True,
            "credit_enabled": True,
            "scheduled": self.scheduled,
            "controller_group_separate_from_durability": True,
            "controller_packet_schema": V4ControllerPacketCodec.SCHEMA,
            "controller_packet_codec": V4ControllerPacketCodec.CODEC,
            "controller_packet_codec_version": V4ControllerPacketCodec.CODEC_VERSION,
            "controller_packet_canonicalization": (
                V4ControllerPacketCodec.CANONICALIZATION
            ),
            "controller_packet_frame_bytes": V4ControllerPacketCodec.MAX_PACKET_BYTES,
            "controller_packet_header_bytes": V4ControllerPacketCodec.HEADER.size,
            "controller_packet_collective": "gloo_all_gather_cpu_uint8",
            "control_gather_schedule": "warmup_1_to_n_then_checkpoint_1_to_window_plus_terminal",
            "control_gather_max_calls": len(self.control_gather_schedule),
            "control_gather_schedule_sha256": canonical_sha256(
                sorted(
                    (step, kind, origin)
                    for step, (kind, origin) in self.control_gather_schedule.items()
                )
            ),
            "causal_phase_schedule": (
                "stream_close_then_collective_then_stream_open_then_step_close"
            ),
            "phase_credit_semantics": "prepared_prefix_transport_phase_local_delta",
            # SplitGuard never mints a new GPU->host allowance at a collective
            # boundary.  v4_open deliberately keeps the matched, finite
            # non-throttling allowance instead; make this distinction an
            # explicit structural contract so historical scheduled traces
            # cannot be mistaken for the repaired path.
            "d2h_collective_semantics": (
                (
                    "bounded_one_request_collective_residual"
                    if getattr(self, "work_conserving_mode", False)
                    else "zero_fresh_credit_residual_only"
                )
                if getattr(self, "split_guard_mode", False)
                else "non_throttling_full_event"
            ),
            "d2h_event_lease": bool(
                getattr(self, "work_event_d2h_lease", False)
            ),
            "d2h_event_lease_semantics": (
                "single_post_save_event_prefix_no_reopen"
                if getattr(self, "work_event_d2h_lease", False)
                else "disabled"
            ),
            "d2h_adaptive_recovery_lease": bool(
                getattr(self, "work_adaptive_d2h_recovery", False)
            ),
            "pfs_compute_only_lane": bool(
                getattr(self, "pfs_compute_only_lane", False)
            ),
            "pfs_future_lease": bool(
                getattr(self, "pfs_future_lease", False)
            ),
            "pfs_future_lease_semantics": (
                "cumulative_readiness_gated_ceiling"
                if getattr(self, "pfs_future_lease", False)
                else "disabled"
            ),
            "pfs_admission_semantics": (
                "compute_only_bounded_16mib_four_requests"
                if getattr(self, "pfs_compute_only_lane", False)
                else (
                    "bounded_all_stream_boundaries"
                    if getattr(self, "split_guard_mode", False)
                    else "non_throttling_full_event"
                )
            ),
            "bootstrap_d2h_semantics": (
                (
                    "bounded_prefix_before_save"
                    if os.environ.get("TEMPO_V4_BOOTSTRAP_D2H_MIB", "").strip()
                    else "full_event_before_save"
                )
                if getattr(self, "work_full_compute_lane", False)
                else "zero_until_first_plan"
            ),
            "terminal_step_credit_close": True,
            "admission_trace_schema": "tempo-v4-admission-trace-2",
            "pfs_odirect_required": True,
            "stage_floor_provenance_schema": provenance["schema_version"],
            "stage_floor_source": provenance["source"],
            "stage_calibration_selection_sha256": provenance[
                "selection_file_sha256"
            ],
            "d2h_floor_bps": provenance["selected_d2h_bps"],
            "pfs_floor_bps": provenance["selected_pfs_bps"],
            "telemetry_journal_schema": self.TELEMETRY_JOURNAL_SCHEMA,
            "telemetry_measurement_sink": "rank_local_non_lustre_jsonl",
            "telemetry_shared_writes_during_measurement": 0,
            "telemetry_publication_protocol": (
                "hidden_temp_fsync_replace_dir_fsync"
            ),
            "telemetry_max_journal_bytes": self.TELEMETRY_MAX_BYTES,
            "telemetry_publisher_sha256": self.telemetry_publisher_sha256,
            "telemetry_suppressed_record_types": sorted(
                self.TELEMETRY_SUPPRESSED_RECORD_TYPES
            ),
        }

    def _emit(self, record_type: str, *, urgent: bool = False, **fields: Any) -> None:
        if (
            not self.telemetry_enabled
            or record_type in self.TELEMETRY_SUPPRESSED_RECORD_TYPES
        ):
            return
        record = {
            "schema_version": self.TELEMETRY_SCHEMA,
            "run_id": self.run_id,
            "record_type": record_type,
            "policy": self.args.policy,
            "rank": self.rank,
            "checkpoint_id": self.checkpoint_id,
            "event_step": -1 if self.checkpoint_step is None else self.checkpoint_step,
            "monotonic_ns": 0,
            "unix_ns": time.time_ns(),
            "runtime_plan_version": int(
                getattr(self, "current_runtime_plan_version", 0)
            ),
            "runtime_phase_id": int(
                getattr(self, "installed_runtime_phase_id", 0)
            ),
            "runtime_python_modules_schema": RUNTIME_PYTHON_MODULES_SCHEMA,
            "runtime_python_modules": runtime_python_module_provenance(),
            "controller_source": "" if _V4_CONTROLLER_PATH is None else str(_V4_CONTROLLER_PATH),
            "controller_sha256": _V4_CONTROLLER_SHA256,
            "structural_sha256": self.telemetry_structural_sha256,
            **fields,
        }
        if record_type == "start":
            record["structural"] = self.telemetry_structural_marker
        with self.telemetry_lock:
            record["monotonic_ns"] = max(
                time.perf_counter_ns(), self.telemetry_last_monotonic_ns + 1
            )
            self.telemetry_last_monotonic_ns = int(record["monotonic_ns"])
            self.telemetry_pending.append(record)
            self.telemetry_records_emitted += 1
        if urgent:
            self._flush_telemetry()

    def _flush_telemetry(self) -> None:
        if not self.telemetry_enabled:
            return
        with self.telemetry_lock:
            pending, self.telemetry_pending = self.telemetry_pending, []
            if not pending:
                return
            payload = "".join(
                json.dumps(record, sort_keys=True, default=str) + "\n"
                for record in pending
            )
            if self.telemetry_handle is None:
                current_bytes = (
                    self.telemetry_local_path.stat().st_size
                    if self.telemetry_local_path.exists()
                    else 0
                )
            else:
                current_bytes = int(self.telemetry_handle.tell())
            encoded_bytes = len(payload.encode("utf-8"))
            if current_bytes + encoded_bytes > self.TELEMETRY_MAX_BYTES:
                self.telemetry_pending = pending + self.telemetry_pending
                raise RuntimeError(
                    "rank-local v4 telemetry exceeded its preflighted byte cap: "
                    f"current={current_bytes} append={encoded_bytes} "
                    f"cap={self.TELEMETRY_MAX_BYTES}"
                )
            if self.telemetry_handle is None:
                # Unit-test adapters use the same local path without the Slurm
                # preflight/publisher. Production keeps one node-local handle.
                self.telemetry_local_path.parent.mkdir(parents=True, exist_ok=True)
                with self.telemetry_local_path.open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(payload)
                    handle.flush()
            else:
                self.telemetry_handle.write(payload)
                self.telemetry_handle.flush()
            self.telemetry_write_calls += 1

    def _close_local_telemetry(self) -> None:
        self._flush_telemetry()
        with self.telemetry_lock:
            handle, self.telemetry_handle = self.telemetry_handle, None
            if handle is not None:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()

    def _publish_local_telemetry(self) -> None:
        if self.telemetry_published:
            return
        if (
            hashlib.sha256(self.telemetry_publisher_path.read_bytes()).hexdigest()
            != self.telemetry_publisher_sha256
        ):
            raise RuntimeError("v4 telemetry publisher changed after preflight")
        completed = subprocess.run(
            (
                sys.executable,
                str(self.telemetry_publisher_path),
                "publish",
                "--policy",
                self.args.policy,
                "--rank",
                str(self.rank),
                "--job-id",
                str(self.telemetry_preflight_evidence["job_id"]),
                "--step-id",
                str(self.telemetry_preflight_evidence["step_id"]),
                "--max-bytes",
                str(self.TELEMETRY_MAX_BYTES),
                "--expected-helper-sha256",
                self.telemetry_publisher_sha256,
                "--local",
                str(self.telemetry_local_path),
                "--final",
                str(self.telemetry_path),
            ),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10.0,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "v4 telemetry atomic publication failed: "
                f"rc={completed.returncode} stderr={completed.stderr.strip()}"
            )
        self.telemetry_published = True

    @staticmethod
    def _require_counter(mapping: dict[str, Any], name: str) -> int:
        value = mapping.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"invalid DataStates counter {name}={value!r}")
        return value

    def _read_stage_stats(self) -> dict[str, Any]:
        raw_value = self.ckpt_engine.get_stage_stats()
        raw = json.loads(raw_value)
        if not isinstance(raw, dict):
            raise RuntimeError("DataStates get_stage_stats did not return a JSON object")
        for name in (
            "watchdog_trip_count",
            "rejected_plan_count",
            "invalid_plan_count",
            "invariant_violation_count",
            "plan_version",
            "phase_id",
            "d2h_budget_bytes",
            "pfs_budget_bytes",
            "max_pfs_inflight_bytes",
            "max_d2h_request_bytes",
            "max_pfs_request_bytes",
            "max_pfs_inflight_requests",
            "watchdog_timeout_ns",
            "plan_updated_monotonic_ns",
            "snapshot_monotonic_ns",
            "d2h_phase_base_admitted_bytes",
            "d2h_phase_admitted_bytes",
            "pfs_cumulative_ceiling_bytes",
            "pfs_phase_base_admitted_bytes",
            "pfs_phase_admitted_bytes",
            "pfs_fsync_monotonic_ns",
            "pfs_odirect_open_count",
        ):
            self._require_counter(raw, name)
        for name in (
            "enabled",
            "force_drain",
            "watchdog_fail_open",
            "pfs_fsync_complete",
            "pfs_odirect_required",
            "pfs_odirect_verified",
        ):
            if not isinstance(raw.get(name), bool):
                raise RuntimeError(f"invalid DataStates boolean {name}={raw.get(name)!r}")
        for stage_name in ("d2h", "pfs"):
            stage = raw.get(stage_name)
            if not isinstance(stage, dict):
                raise RuntimeError(f"DataStates stats lacks {stage_name} object")
            for name in (
                "total_bytes",
                "queued_bytes",
                "ready_bytes",
                "admitted_bytes",
                "completed_bytes",
                "inflight_bytes",
                "inflight_requests",
                "admitted_requests",
                "max_request_bytes",
                "peak_inflight_bytes",
                "peak_inflight_requests",
                "last_progress_monotonic_ns",
                "last_completion_monotonic_ns",
            ):
                self._require_counter(stage, name)
        return raw

    def _take_admission_trace(self) -> dict[str, Any]:
        raw_value = self.ckpt_engine.take_admission_trace()
        trace = json.loads(raw_value)
        if not isinstance(trace, dict):
            raise RuntimeError("DataStates admission trace did not return a JSON object")
        if trace.get("schema_version") != "tempo-v4-admission-trace-2":
            raise RuntimeError("DataStates admission trace schema mismatch")
        for name in (
            "active_token",
            "enqueued_callbacks",
            "completed_callbacks",
            "stale_callback_count",
            "blocked_callback_count",
            "enqueue_failure_count",
        ):
            self._require_counter(trace, name)
        entries, transitions = trace.get("entries"), trace.get("transitions")
        if not isinstance(entries, list) or not isinstance(transitions, list):
            raise RuntimeError("DataStates admission trace lacks v2 trace arrays")
        transition_counters = (
            "token",
            "plan_version",
            "phase_id",
            "d2h_cumulative_ceiling_bytes",
            "pfs_cumulative_ceiling_bytes",
            "d2h_active_budget_bytes",
            "pfs_active_budget_bytes",
            "not_before_monotonic_ns",
            "expires_monotonic_ns",
            "activation_monotonic_ns",
            "activation_unix_ns",
            "callback_exit_monotonic_ns",
            "callback_duration_ns",
        )
        for index, transition in enumerate(transitions):
            if not isinstance(transition, dict):
                raise RuntimeError(
                    f"DataStates admission transition {index} is not an object"
                )
            for name in transition_counters:
                self._require_counter(transition, name)
            if not isinstance(transition.get("status"), str):
                raise RuntimeError(
                    f"DataStates admission transition {index} lacks status"
                )
        entry_counters = (
            "token",
            "plan_version",
            "phase_id",
            "d2h_cumulative_ceiling_bytes",
            "pfs_cumulative_ceiling_bytes",
            "d2h_active_budget_bytes",
            "pfs_active_budget_bytes",
            "activation_monotonic_ns",
            "activation_unix_ns",
            "applied_monotonic_ns",
            "not_before_monotonic_ns",
            "expires_monotonic_ns",
            "d2h_base_admitted_bytes",
            "pfs_base_admitted_bytes",
            "d2h_expired_before_bytes",
            "pfs_expired_before_bytes",
            "d2h_unconsumed_bytes",
            "pfs_unconsumed_bytes",
            "d2h_bytes",
            "d2h_requests",
            "d2h_controlled_bytes",
            "d2h_controlled_requests",
            "d2h_drain_bytes",
            "d2h_drain_requests",
            "pfs_bytes",
            "pfs_requests",
            "pfs_controlled_bytes",
            "pfs_controlled_requests",
            "pfs_drain_bytes",
            "pfs_drain_requests",
        )
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise RuntimeError(f"DataStates admission trace entry {index} is not an object")
            for name in entry_counters:
                self._require_counter(entry, name)
            if not isinstance(entry.get("stream_ordered"), bool):
                raise RuntimeError(
                    f"DataStates admission trace entry {index} lacks stream ordering"
                )
        return trace

    def _validate_admission_trace(
        self, trace: dict[str, Any], relative: dict[str, Any]
    ) -> None:
        entries = trace["entries"]
        transitions = trace["transitions"]
        expected_pairs = self.event_installed_runtime_phases
        if int(trace["enqueued_callbacks"]) != int(trace["completed_callbacks"]):
            raise RuntimeError("admission trace has incomplete CUDA callbacks")
        if int(trace["stale_callback_count"]) or int(trace["enqueue_failure_count"]):
            raise RuntimeError("admission trace has stale or failed CUDA callbacks")
        previous_token = 0
        seen_transition_tokens: set[int] = set()
        applied_tokens: set[int] = set()
        transition_pairs_by_plan: dict[int, list[tuple[int, int]]] = {}
        transition_status_by_pair: dict[tuple[int, int], str] = {}
        terminal_statuses = {
            "PREPARED_NOT_ENQUEUED",
            "APPLIED",
            "BLOCKED_DRAIN",
        }
        for transition in transitions:
            token = int(transition["token"])
            pair = (int(transition["plan_version"]), int(transition["phase_id"]))
            slot = self.event_runtime_slots.get(pair)
            if (
                token <= previous_token
                or token in seen_transition_tokens
                or slot is None
                or int(slot["token"]) != token
            ):
                raise RuntimeError("admission trace transition is not a prepared runtime slot")
            status = str(transition["status"])
            if status not in terminal_statuses:
                raise RuntimeError(f"admission trace transition is nonterminal: {status}")
            for stage in ("d2h", "pfs"):
                cumulative_name = f"{stage}_cumulative_ceiling_bytes"
                active_name = f"{stage}_active_budget_bytes"
                if int(transition[cumulative_name]) != int(slot[cumulative_name]):
                    raise RuntimeError("admission trace cumulative ceiling differs from prepared slot")
                if int(transition[active_name]) != int(slot[active_name]):
                    raise RuntimeError("admission trace active budget differs from phase-local delta")
            if status == "PREPARED_NOT_ENQUEUED":
                if bool(slot["enqueued"]):
                    raise RuntimeError("enqueued transition was reported as prepare-only")
                if any(
                    int(transition[name])
                    for name in (
                        "activation_monotonic_ns",
                        "activation_unix_ns",
                        "callback_exit_monotonic_ns",
                        "callback_duration_ns",
                    )
                ):
                    raise RuntimeError("prepare-only transition was unexpectedly activated")
            elif status == "APPLIED":
                if not bool(slot["enqueued"]):
                    raise RuntimeError("un-enqueued transition was reported as applied")
                applied_tokens.add(token)
            elif status == "BLOCKED_DRAIN" and not bool(slot["enqueued"]):
                raise RuntimeError("un-enqueued transition was reported as drain-blocked")
            seen_transition_tokens.add(token)
            transition_pairs_by_plan.setdefault(pair[0], []).append(pair)
            transition_status_by_pair[pair] = status
            previous_token = token
        expected_transition_tokens = {
            int(slot["token"])
            for slot in self.event_runtime_slots.values()
            if int(slot["token"]) != 0
        }
        if seen_transition_tokens != expected_transition_tokens:
            raise RuntimeError("admission trace is missing a prepared transition slot")
        terminal_pairs = set(self.event_terminal_runtime_phases)
        if transitions and not terminal_pairs:
            raise RuntimeError("admission trace lacks terminal step credit CLOSE slots")
        if {pair[0] for pair in terminal_pairs} != set(transition_pairs_by_plan):
            raise RuntimeError(
                "each prepared runtime plan must contain one terminal step credit CLOSE"
            )
        for plan_version, plan_pairs in transition_pairs_by_plan.items():
            closes = [pair for pair in plan_pairs if pair in terminal_pairs]
            if len(closes) != 1 or closes[0] != plan_pairs[-1]:
                raise RuntimeError(
                    f"runtime plan {plan_version} does not end in exactly one terminal CLOSE"
                )
            close_pair = closes[0]
            close_slot = self.event_runtime_slots[close_pair]
            terminal_pfs_release = int(
                close_slot.get("terminal_pfs_release_bytes", 0)
            )
            if (
                int(close_slot["d2h_active_budget_bytes"]) != 0
                or int(close_slot["pfs_active_budget_bytes"])
                != terminal_pfs_release
                or (
                    terminal_pfs_release
                    and not getattr(self, "split_guard_mode", False)
                )
                or not bool(close_slot["enqueued"])
                or transition_status_by_pair.get(close_pair) != "APPLIED"
            ):
                raise RuntimeError("terminal step credit CLOSE was not applied")
        if terminal_pairs:
            final_terminal_token = max(
                int(self.event_runtime_slots[pair]["token"]) for pair in terminal_pairs
            )
            if int(trace["active_token"]) != final_terminal_token:
                raise RuntimeError("terminal admission trace did not remain at terminal CLOSE")
        d2h_bytes = 0
        d2h_requests = 0
        pfs_bytes = 0
        pfs_requests = 0
        seen_pairs: set[tuple[int, int]] = set()
        previous_pair = (0, 0)
        for entry in entries:
            pair = (int(entry["plan_version"]), int(entry["phase_id"]))
            token = int(entry["token"])
            slot = self.event_runtime_slots.get(pair)
            if pair[0] <= 0 or pair[1] <= 0 or pair not in expected_pairs or slot is None:
                raise RuntimeError(
                    f"admission trace references an uninstalled runtime phase {pair}"
                )
            if pair < previous_pair or pair in seen_pairs:
                raise RuntimeError("admission trace runtime phases are not strictly ordered")
            previous_pair = pair
            seen_pairs.add(pair)
            if token != int(slot["token"]):
                raise RuntimeError("admission trace token differs from its runtime slot")
            if bool(entry["stream_ordered"]) != bool(token):
                raise RuntimeError("admission trace stream-order flag disagrees with token")
            if token and token not in applied_tokens:
                raise RuntimeError("admission entry references a non-applied transition")
            for stage in ("d2h", "pfs"):
                if int(entry[f"{stage}_cumulative_ceiling_bytes"]) != int(
                    slot[f"{stage}_cumulative_ceiling_bytes"]
                ):
                    raise RuntimeError("admission entry cumulative ceiling differs from runtime slot")
                if int(entry[f"{stage}_active_budget_bytes"]) != int(
                    slot[f"{stage}_active_budget_bytes"]
                ):
                    raise RuntimeError("admission entry active budget differs from phase-local delta")
            entry_d2h_bytes = int(entry["d2h_bytes"])
            entry_d2h_requests = int(entry["d2h_requests"])
            entry_pfs_bytes = int(entry["pfs_bytes"])
            entry_pfs_requests = int(entry["pfs_requests"])
            if (entry_d2h_bytes == 0) != (entry_d2h_requests == 0):
                raise RuntimeError("admission trace has inconsistent D2H byte/request totals")
            if (entry_pfs_bytes == 0) != (entry_pfs_requests == 0):
                raise RuntimeError("admission trace has inconsistent PFS byte/request totals")
            if entry_d2h_bytes != int(entry["d2h_controlled_bytes"]) + int(
                entry["d2h_drain_bytes"]
            ):
                raise RuntimeError("admission trace has inconsistent D2H control/drain bytes")
            if entry_pfs_bytes != int(entry["pfs_controlled_bytes"]) + int(
                entry["pfs_drain_bytes"]
            ):
                raise RuntimeError("admission trace has inconsistent PFS control/drain bytes")
            if entry_d2h_requests != int(entry["d2h_controlled_requests"]) + int(
                entry["d2h_drain_requests"]
            ):
                raise RuntimeError("admission trace has inconsistent D2H control/drain requests")
            if entry_pfs_requests != int(entry["pfs_controlled_requests"]) + int(
                entry["pfs_drain_requests"]
            ):
                raise RuntimeError("admission trace has inconsistent PFS control/drain requests")
            if entry_d2h_bytes > entry_d2h_requests * self.config.d2h_quantum_bytes:
                raise RuntimeError("admission trace exceeds the 1 MiB D2H request bound")
            if entry_pfs_bytes > entry_pfs_requests * self.config.pfs_quantum_bytes:
                raise RuntimeError("admission trace exceeds the 4 MiB PFS request bound")
            d2h_bytes += entry_d2h_bytes
            d2h_requests += entry_d2h_requests
            pfs_bytes += entry_pfs_bytes
            pfs_requests += entry_pfs_requests
        if not terminal_pairs.issubset(seen_pairs):
            raise RuntimeError("admission trace is missing an applied terminal CLOSE entry")
        for pair in terminal_pairs:
            slot = self.event_runtime_slots[pair]
            terminal_pfs_release = int(slot.get("terminal_pfs_release_bytes", 0))
            if (
                int(slot["d2h_active_budget_bytes"]) != 0
                or int(slot["pfs_active_budget_bytes"]) != terminal_pfs_release
                or (
                    terminal_pfs_release
                    and not getattr(self, "split_guard_mode", False)
                )
            ):
                raise RuntimeError("terminal CLOSE entry carried invalid credit")
        observed = {
            "d2h_bytes": d2h_bytes,
            "d2h_requests": d2h_requests,
            "pfs_bytes": pfs_bytes,
            "pfs_requests": pfs_requests,
        }
        expected = {
            "d2h_bytes": int(relative["d2h"]["admitted_bytes"]),
            "d2h_requests": int(relative["d2h"]["admitted_requests"]),
            "pfs_bytes": int(relative["pfs"]["admitted_bytes"]),
            "pfs_requests": int(relative["pfs"]["admitted_requests"]),
        }
        if observed != expected:
            raise RuntimeError(
                "admission trace totals differ from event-relative stage counters: "
                f"observed={observed} expected={expected}"
            )
        if (d2h_bytes or pfs_bytes) and not entries:
            raise RuntimeError("nonempty event has an empty admission trace")

    def _assert_structural_stats(self, raw: dict[str, Any]) -> None:
        expected_d2h_request = self.args.tempo_v4_d2h_chunk_mb * MIB
        expected_pfs_request = self.args.tempo_v4_pfs_chunk_mb * MIB
        expected_pfs_requests = (
            self.args.tempo_v4_max_pfs_inflight_mb
            // self.args.tempo_v4_pfs_chunk_mb
        )
        if not bool(raw.get("enabled", False)):
            raise RuntimeError("v4 credit control was not enabled before engine.save")
        if int(raw["max_d2h_request_bytes"]) != expected_d2h_request:
            raise RuntimeError("v4 D2H request bound is not 1 MiB")
        if int(raw["max_pfs_request_bytes"]) != expected_pfs_request:
            raise RuntimeError("v4 PFS request regionization is not 4 MiB")
        if int(raw["max_pfs_inflight_requests"]) != expected_pfs_requests:
            raise RuntimeError("v4 PFS in-flight request cap does not match 16 MiB / 4 MiB")
        if int(raw["pfs"]["inflight_requests"]) > expected_pfs_requests:
            raise RuntimeError("v4 observed PFS in-flight request count above its cap")
        if int(raw.get("max_pfs_inflight_bytes", 0)) != self.config.max_pfs_inflight_bytes:
            raise RuntimeError("v4 PFS byte cap differs from controller configuration")
        if not bool(raw.get("pfs_odirect_required", False)):
            raise RuntimeError("v4 controlled persistence does not require O_DIRECT")
        d2h = raw["d2h"]
        pfs = raw["pfs"]
        if (
            int(d2h["max_request_bytes"]) > expected_d2h_request
            or int(d2h["peak_inflight_bytes"]) > expected_d2h_request
            or int(d2h["peak_inflight_requests"]) > 1
        ):
            raise RuntimeError("v4 observed a D2H residual above one 1 MiB request")
        if (
            int(pfs["max_request_bytes"]) > expected_pfs_request
            or int(pfs["peak_inflight_bytes"]) > self.config.max_pfs_inflight_bytes
            or int(pfs["peak_inflight_requests"]) > expected_pfs_requests
        ):
            raise RuntimeError("v4 observed PFS work above its 4-request/16 MiB cap")

    def _event_relative_stats(self, raw: dict[str, Any]) -> dict[str, Any]:
        if self.event_base_stats is None:
            raise RuntimeError("TEMPO v4 event counter baseline is missing")
        result: dict[str, Any] = {
            name: raw.get(name)
            for name in (
                "enabled",
                "force_drain",
                "watchdog_fail_open",
                "plan_version",
                "phase_id",
                "pfs_fsync_complete",
                "pfs_fsync_monotonic_ns",
                "pfs_odirect_required",
                "pfs_odirect_verified",
                "snapshot_monotonic_ns",
                "max_d2h_request_bytes",
                "max_pfs_request_bytes",
                "max_pfs_inflight_bytes",
                "max_pfs_inflight_requests",
            )
        }
        for counter in (
            "watchdog_trip_count",
            "rejected_plan_count",
            "invalid_plan_count",
            "invariant_violation_count",
            "pfs_odirect_open_count",
        ):
            delta = int(raw[counter]) - int(self.event_base_stats[counter])
            if delta < 0:
                raise RuntimeError(f"lifetime counter {counter} regressed")
            result[counter] = delta
        for stage_name in ("d2h", "pfs"):
            stage = raw[stage_name]
            base = self.event_base_stats[stage_name]
            normalized: dict[str, int] = {}
            for counter in (
                "total_bytes",
                "admitted_bytes",
                "completed_bytes",
                "admitted_requests",
            ):
                delta = int(stage[counter]) - int(base[counter])
                if delta < 0:
                    raise RuntimeError(f"{stage_name}.{counter} lifetime counter regressed")
                normalized[counter] = delta
            for gauge in (
                "queued_bytes",
                "ready_bytes",
                "inflight_bytes",
                "inflight_requests",
                "max_request_bytes",
                "peak_inflight_bytes",
                "peak_inflight_requests",
            ):
                normalized[gauge] = int(stage[gauge])
            normalized["last_progress_monotonic_ns"] = int(stage["last_progress_monotonic_ns"])
            normalized["last_completion_monotonic_ns"] = int(stage["last_completion_monotonic_ns"])
            expected = (
                self.event_expected_state_bytes
                if stage_name == "d2h"
                else self.event_expected_pfs_bytes
            )
            normalized["total_bytes"] = max(
                normalized["total_bytes"],
                normalized["admitted_bytes"],
                normalized["completed_bytes"],
                expected,
            )
            if normalized["completed_bytes"] > normalized["admitted_bytes"]:
                raise RuntimeError(f"{stage_name} completion exceeds event admission")
            if normalized["inflight_bytes"] > normalized["admitted_bytes"] - normalized["completed_bytes"]:
                raise RuntimeError(f"{stage_name} in-flight bytes violate admission accounting")
            result[stage_name] = normalized
        return result

    def _install_synchronous_bootstrap(
        self,
        *,
        logical_phase_id: int,
        d2h_budget_bytes: int,
        pfs_budget_bytes: int,
        watchdog_timeout_ns: int | None = None,
    ) -> bool:
        """Arm the closed event-start data path before DataStates save()."""

        with self.control_lock:
            self.current_runtime_plan_version = self.next_runtime_plan_version
            self.next_runtime_plan_version += 1
            runtime_phase_id = self.next_runtime_phase_id
            self.next_runtime_phase_id += 1
            self.current_phase_slots = {}
            slot = {
                "token": 0,
                "plan_version": int(self.current_runtime_plan_version),
                "phase_id": int(runtime_phase_id),
                "logical_phase_id": int(logical_phase_id),
                "installable": True,
                "enqueued": False,
                "d2h_cumulative_ceiling_bytes": int(d2h_budget_bytes),
                "pfs_cumulative_ceiling_bytes": int(pfs_budget_bytes),
                "d2h_active_budget_bytes": int(d2h_budget_bytes),
                "pfs_active_budget_bytes": int(pfs_budget_bytes),
            }
            pair = (int(self.current_runtime_plan_version), int(runtime_phase_id))
            self.current_phase_slots[int(logical_phase_id)] = slot
            self.event_runtime_slots[pair] = slot
            self.event_installed_runtime_phases.add(pair)
            accepted = bool(
                self.ckpt_engine.install_credit_plan(
                    self.current_runtime_plan_version,
                    runtime_phase_id,
                    int(d2h_budget_bytes),
                    int(pfs_budget_bytes),
                    self.config.max_pfs_inflight_bytes,
                    self.config.watchdog_timeout_ns
                    if watchdog_timeout_ns is None
                    else int(watchdog_timeout_ns),
                )
            )
            self.installed_logical_phase_id = logical_phase_id
            self.installed_runtime_phase_id = runtime_phase_id
            self.installed_credit_accepted = accepted
            self.installed_d2h_budget_bytes = int(d2h_budget_bytes)
            self.installed_pfs_budget_bytes = int(pfs_budget_bytes)
            self.metrics.v4_phase_install_count += 1
        if not accepted:
            self._force_drain("DataStates rejected the synchronous event bootstrap")
        return accepted

    def _prepare_plan_transitions(self) -> bool:
        """Prepare every step slot as a stream-ordered phase-local delta.

        DataStates transports a cumulative PFS ceiling so it can reject missing
        or reordered slots, while the C++ contract resets D2H active budget at
        each new plan version. Group-noninstallable TEMPO compute slots are explicit zero
        deltas, which expire the preceding execution allowance without admitting
        new bytes. v4_open renews an event-sized allowance in every ordinary
        slot, which is non-throttling because no stage can have more than one
        event's bytes left. Both policies append the same zero-delta terminal
        CLOSE so step-exit credit cannot leak into the synthetic probe.
        """

        if self.current_rank_plan is None:
            raise RuntimeError("phase transition preparation requires a rank plan")
        credits = self.current_rank_plan.windows
        if not credits:
            raise RuntimeError("phase transition preparation received an empty plan")
        with self.control_lock:
            self.current_runtime_plan_version = self.next_runtime_plan_version
            self.next_runtime_plan_version += 1
            self.current_phase_slots = {}
            self.current_terminal_logical_phase_id = -1
            self.current_terminal_credit_closed = False
        # Unlike PFS, the C++ controller resets D2H active budget at a new
        # plan version.  Keep the event-global issued ceiling separately for
        # Python admission accounting, but pass only this plan's new delta to
        # the reset-first C++ contract.
        prior_split_d2h_issued = int(
            getattr(self, "split_guard_d2h_cumulative_ceiling", 0)
        ) if getattr(self, "split_guard_mode", False) else 0
        d2h_cumulative = 0
        pfs_cumulative = 0
        effective_window_deltas: list[tuple[int, int]] = []
        # A new runtime version must recapture the actual admitted base.  A
        # previous version may have opened a prefix that was not consumed
        # before the CUDA boundary; carrying the planned prefix instead makes
        # that credit expire and defers the tail to terminal drain.
        event_admitted_d2h: int | None = None
        event_admitted_pfs: int | None = None
        current_stats_for_base: dict[str, Any] | None = None
        if (
            getattr(self, "split_guard_mode", False)
            and getattr(self, "event_base_stats", None) is not None
        ):
            try:
                current_stats = self._read_stage_stats()
                current_stats_for_base = current_stats
                base_stats = self.event_base_stats or {}
                for stage_name, expected in (
                    ("d2h", int(getattr(self, "event_expected_state_bytes", 0))),
                    ("pfs", int(getattr(self, "event_expected_pfs_bytes", 0))),
                ):
                    current_stage = current_stats[stage_name]
                    base_stage = base_stats.get(stage_name, {})
                    current_admitted_raw = int(current_stage["admitted_bytes"])
                    base_admitted_raw = int(base_stage.get("admitted_bytes", 0))
                    if current_admitted_raw < base_admitted_raw:
                        raise ValueError(
                            f"{stage_name}.admitted_bytes regressed across event baseline"
                        )
                    admitted = max(
                        0,
                        current_admitted_raw - base_admitted_raw,
                    )
                    value = min(max(0, expected), admitted)
                    if stage_name == "d2h":
                        event_admitted_d2h = value
                    else:
                        event_admitted_pfs = value
            except (KeyError, TypeError, ValueError, RuntimeError):
                # Telemetry validation remains the fail-closed authority; do
                # not mint credit when the admitted gauge is malformed.
                event_admitted_d2h = None
                event_admitted_pfs = None
        # D2H is reset-first at a new runtime plan, so its cumulative base is
        # deliberately plan-local (zero).  The event-relative admitted gauge
        # only sizes split_d2h_remaining below.
        # PFS uses a monotone cumulative prefix across runtime plan versions.
        # The event-relative admitted gauge is used below to size the
        # remaining work, but must not replace the previously prepared
        # cumulative ceiling: doing so makes C++ finite_delta_ underflow when
        # an earlier prefix has not yet been consumed.
        # CPU fixtures may replace the engine with an empty lifetime snapshot
        # while deliberately setting a prior extent ceiling.  That snapshot
        # is not evidence that a real event regressed to zero; preserve the
        # already-closed extent in that synthetic case.  Real runs have a
        # positive event total and therefore take the admitted-base path.
        if (
            event_admitted_pfs == 0
            and prior_split_d2h_issued >= 0
            and int(getattr(self, "split_guard_pfs_cumulative_ceiling", 0))
            >= int(getattr(self, "event_expected_pfs_bytes", 0))
            and current_stats_for_base is not None
            and int(current_stats_for_base.get("pfs", {}).get("total_bytes", 0)) == 0
        ):
            event_admitted_pfs = None
            pfs_cumulative = int(
                getattr(self, "split_guard_pfs_cumulative_ceiling", 0)
            )
        split_compute_slots_remaining = sum(
            1
            for credit in credits
            if (
                int(credit.phase_id) != int(credits[0].phase_id)
                and str(getattr(credit.kind, "value", credit.kind)) == "compute"
            )
        )
        if getattr(self, "work_full_compute_lane", False):
            # The full-compute experiment intentionally keeps an unconsumed
            # event D2H tail alive across reused runtime plan versions.  The
            # first plan exposes the whole event extent; later plans subtract
            # the event-relative *admitted* counter before opening their new
            # C++ reset-first base.  Without that subtraction, each new plan
            # could mint the same allowance twice.
            event_expected = int(getattr(self, "event_expected_state_bytes", 0))
            if event_admitted_d2h is not None:
                split_d2h_remaining = max(
                    0, event_expected - int(event_admitted_d2h)
                )
            elif prior_split_d2h_issued > 0:
                try:
                    current_stats = self._read_stage_stats()
                    base_stats = self.event_base_stats or {}
                    current_admitted = int(current_stats["d2h"]["admitted_bytes"])
                    base_admitted = int(
                        base_stats.get("d2h", {}).get("admitted_bytes", 0)
                    )
                    event_admitted = max(0, current_admitted - base_admitted)
                except (KeyError, TypeError, ValueError, RuntimeError):
                    # A malformed gauge must not mint an extra allowance. The
                    # prior planned prefix is a conservative fallback until
                    # the normal telemetry path trips DRAIN.
                    event_admitted = min(event_expected, prior_split_d2h_issued)
                split_d2h_remaining = max(0, event_expected - event_admitted)
            else:
                split_d2h_remaining = event_expected
        elif getattr(self, "split_guard_mode", False):
            split_d2h_remaining = max(
                0,
                int(getattr(self, "event_expected_state_bytes", 0))
                - (
                    int(event_admitted_d2h)
                    if event_admitted_d2h is not None
                    else prior_split_d2h_issued
                ),
            )
        else:
            split_d2h_remaining = int(
                getattr(self, "event_expected_state_bytes", 0)
            )
        # Keep each PROTECT plan within the controller's snapshot/group-fair
        # host-ready frontier.  The stream path may spread that finite plan
        # across several phase boundaries, but it must not mint a fresh 16 MiB
        # PFS allowance at every boundary: doing so makes the serialized plan
        # exceed the same fair cap that validated the controller plan.
        split_initial_subquantum_plan = (
            int(getattr(self, "current_plan_step", -1))
            == int(getattr(self, "checkpoint_step", -2)) + 1
        )
        # The producer snapshot is intentionally still recorded separately
        # from the runtime lease.  In the opt-in future-lease lane, the first
        # plan may prepare the full finite cumulative PFS ceiling while the
        # physical producer-ready prefix remains request-aligned.  C++ never
        # issues a request until the corresponding ready_bytes is present.
        self.current_pfs_future_lease_bytes = 0
        self.current_pfs_ready_budget_bytes = 0
        if getattr(self, "split_guard_mode", False) or not self.scheduled:
            # The C++ PFS lane carries this absolute prefix across controller
            # plan versions.  This is required for both the finite split lane
            # and v4_open: open still grants a full-event allowance at every
            # ordinary phase, but its transported prefix must remain monotone
            # when the runtime plan version advances.
            pfs_cumulative = int(
                getattr(self, "split_guard_pfs_cumulative_ceiling", 0)
            )
            # D2H is reset-first at a new runtime plan.  Keep its cumulative
            # base plan-local (zero); event_admitted_d2h only determines the
            # remaining event work opened by this plan.
        else:
            pfs_cumulative = 0
        # PFS is host/storage work and must remain work-conserving in the
        # split lane.  A planner snapshot can legitimately report a zero
        # *new* target while host-ready bytes from the previous prefix are
        # still pending; tying the runtime prefix to that target starves the
        # consumer and was the cause of the repeated ~1s deadline misses.
        # Continue opening the event's remaining PFS extent in bounded 4 MiB
        # deltas.  The cumulative ceiling and C++ 16 MiB/4-request admission
        # cap still bound outstanding physical work; D2H remains causal.
        if getattr(self, "split_guard_mode", False):
            # PFS is a consumer lane: open the finite remaining event prefix
            # from ordinary plans so host-ready bytes can drain while D2H
            # proceeds. Re-open at most one bounded 4 MiB request per phase;
            # once the event extent is reached, later plans add no prefix.
            # The cumulative ceiling remains monotone across runtime plan
            # versions, while physical admission is still limited by total
            # bytes, 16 MiB, and four requests.
            # A cumulative prefix can be prepared before the corresponding
            # host-ready bytes exist.  Keep that already-issued ceiling in
            # place; do not add a second lease after the logical event extent
            # has been reached, or a late gauge could authorize bytes beyond
            # the published checkpoint.
            # PFS is deliberately not phase-gated.  The C++ admission path
            # still requires the corresponding host-ready bytes before it can
            # issue a physical request, so opening a finite cumulative prefix
            # here does not authorize a write of data that D2H has not
            # produced.  Using only the currently observed host-ready gauge
            # caused the consumer to starve whenever the first snapshot raced
            # the host worker; the next gather then had to recover an entire
            # event tail.  Keep the prefix finite and request-aligned, while
            # allowing the work-conserving lane to drain it as inventory
            # arrives.
            event_pfs_extent = max(
                0, int(getattr(self, "event_expected_pfs_bytes", 0))
            )
            # A later controller plan carries the previous cumulative prefix.
            # Only the still-unopened event extent may be added; repeated
            # gathers must never mint more PFS ceiling than the checkpoint
            # contains.  The logical lease may cover the full remaining
            # event; physical admission remains bounded independently by the
            # 4 MiB request and 16 MiB/four-request in-flight limits.
            if pfs_cumulative >= event_pfs_extent:
                split_pfs_remaining = 0
            else:
                split_pfs_remaining = max(
                    0, event_pfs_extent - pfs_cumulative
                )
        else:
            # Before recovery, preserve the controller's producer-lead
            # snapshot cap.  Opening the whole event extent here would make
            # the telemetry plan claim future PFS work that is not host-ready
            # yet, even though the C++ admission predicate would eventually
            # reject it.  Recovery is the only phase allowed to use the
            # bounded rolling lease above.
            planned_pfs = int(
                getattr(self.current_rank_plan, "target_pfs_bytes", 0)
            )
            split_pfs_remaining = max(0, planned_pfs - int(pfs_cumulative))
        # The initial pre-save/bootstrap path has no published logical
        # layout, so it must keep PFS closed.  After publication, the first
        # ordinary plan may open only a request-sized prefix when enough
        # host-ready inventory is actually visible.  A 4 KiB header/early
        # snapshot must remain zero-target; otherwise the plan claims a large
        # PFS prefix against no producer inventory and violates the
        # snapshot/group-fair producer-lead contract.  Later plans reopen the
        # bounded rolling prefix once the host-ready gauge reaches qP.
        published = bool(getattr(self, "event_logical_layout", {}))
        event_d2h_lease_first_plan = bool(
            (
                getattr(self, "work_event_d2h_lease", False)
                and split_initial_subquantum_plan
                and int(event_admitted_d2h or 0) == 0
            )
            or (
                getattr(self, "work_adaptive_d2h_recovery", False)
                and bool(
                    getattr(self, "split_guard_recovery_plan_latched", False)
                )
                and not bool(getattr(self, "d2h_event_lease_used", False))
            )
        )
        event_d2h_lease_first_plan = bool(
            event_d2h_lease_first_plan
            and published
            and int(split_d2h_remaining) > 0
        )
        if event_d2h_lease_first_plan:
            self.d2h_event_lease_used = True
            self.d2h_event_lease_plan_version = int(
                self.current_runtime_plan_version
            )
        keep_event_d2h_active = bool(
            getattr(self, "d2h_event_lease_used", False)
        )
        if bool(getattr(self, "current_group_host_ready_valid", False)):
            host_ready = max(
                0, int(getattr(self, "current_group_host_ready_bytes", 0))
            )
        else:
            host_ready = max(
                0, int(getattr(self, "current_event_host_ready_bytes", 0))
            )
        # The analyzer/controller group-fair frontier is request aligned for
        # ordinary regions.  The logical file may end in a smaller,
        # filesystem-block-aligned request, however.  Preserve that exact
        # final suffix once the carried prefix plus the group-visible ready
        # inventory can close the published extent; flooring it to qP strands
        # the suffix forever and leaves watchdog fail-open as the only drain.
        # This exception cannot authorize an intermediate partial region: it
        # applies only when the remaining cumulative prefix is itself < qP.
        pfs_quantum = V4_PFS_REQUEST_MIB * MIB
        if pfs_quantum > 0:
            aligned_host_ready = host_ready - host_ready % pfs_quantum
            final_prefix_tail = max(
                0,
                int(getattr(self, "event_expected_pfs_bytes", 0))
                - int(pfs_cumulative),
            )
            layout_alignment = int(
                getattr(self, "event_logical_layout", {}).get(
                    "fs_block_alignment_bytes", 0
                )
            )
            final_partial_ready = bool(
                getattr(self, "split_guard_mode", False)
                and published
                and 0 < final_prefix_tail < pfs_quantum
                and host_ready >= final_prefix_tail
                and layout_alignment > 0
                and final_prefix_tail % layout_alignment == 0
            )
            host_ready = max(
                aligned_host_ready,
                final_prefix_tail if final_partial_ready else 0,
            )
            if (
                getattr(self, "pfs_future_lease", False)
                and split_initial_subquantum_plan
                and published
            ):
                # This is evidence only for the producer-ready part of the
                # plan; the cumulative lease itself is opened below.
                self.current_pfs_ready_budget_bytes = min(
                    max(0, int(split_pfs_remaining)), int(host_ready)
                )
            # In the compute-only PFS lane, a sub-quantum remainder is a
            # special case rather than an ordinary request.  Do not let the
            # regular 16 MiB logical lease path turn an unaligned, missing,
            # or not-yet-ready suffix into a phantom grant.  The exact-tail
            # path above is the only admissible way to open a remainder below
            # qP; otherwise the cumulative prefix must stay unchanged and
            # the caller can retry once a valid aligned layout/ready gauge is
            # available.
            if (
                getattr(self, "pfs_compute_only_lane", False)
                and 0 < final_prefix_tail < pfs_quantum
                and not final_partial_ready
            ):
                split_pfs_remaining = 0
        initial_quantum = min(
            V4_PFS_REQUEST_MIB * MIB,
            max(0, int(getattr(self, "event_expected_pfs_bytes", 0))),
        )
        if split_initial_subquantum_plan and (
            not getattr(self, "pfs_future_lease", False)
            or not published
        ) and (
            not published or host_ready < initial_quantum
        ):
            split_pfs_remaining = 0
        elif split_initial_subquantum_plan and not getattr(
            self, "pfs_future_lease", False
        ):
            # Even after publication, the first producer-lead plan must not
            # claim more than the group-visible host-ready inventory.  The
            # controller target can be larger when D2H completed concurrently
            # with save(); the admission prefix is therefore clamped here and
            # expanded by later gathers as the inventory grows.
            # The full-compute D2H lane is not an exception for the PFS
            # producer.  It may keep D2H debt alive, but it must not advertise
            # consumer bytes that the common host-ready snapshot has not
            # published yet.  The C++ ready predicate is a physical safety
            # check; this clamp is the separate group-fair planning contract.
            split_pfs_remaining = min(split_pfs_remaining, host_ready)
        elif getattr(self, "split_guard_mode", False) and published:
            # A published event may have a transiently empty host-ready gauge
            # while the host worker is still materializing the next extent.
            # Do not freeze the *cumulative* consumer prefix at that sample:
            # the C++ admission predicate independently requires ready_bytes
            # for every physical request, so advancing this bounded prefix is
            # safe and lets later phases drain the inventory as it appears.
            # Clamping here stranded the whole event until terminal CLOSE,
            # producing a ~400 MiB last-phase release and 1 s deadline misses.
            # The initial pre-save plan remains host-ready clamped above.
            split_pfs_remaining = min(
                split_pfs_remaining,
                max(
                    0,
                    int(getattr(self, "event_expected_pfs_bytes", 0))
                    - int(pfs_cumulative),
                ),
            )
        if (
            getattr(self, "pfs_future_lease", False)
            and split_initial_subquantum_plan
            and published
            and self.current_pfs_ready_budget_bytes == 0
        ):
            self.current_pfs_ready_budget_bytes = min(
                max(0, int(split_pfs_remaining)), int(host_ready)
            )
        if (
            getattr(self, "pfs_future_lease", False)
            and split_initial_subquantum_plan
            and published
        ):
            self.current_pfs_future_lease_bytes = max(
                0, int(split_pfs_remaining)
            )
        # Once a logical layout is published, the storage worker may consume
        # the whole event as host-ready inventory appears.  Open one
        # event-wide *cumulative* PFS lease at the first eligible ordinary
        # phase; the C++ path still enforces the physical 4 MiB request and
        # 16 MiB/four-request in-flight caps.  Keeping a separate flag avoids
        # reopening that lease at every later phase.
        pfs_opened_this_plan = False
        for credit in credits:
            logical_phase_id = int(credit.phase_id)
            installable = logical_phase_id in self.current_installable_phases
            is_compute_window = (
                str(getattr(credit.kind, "value", credit.kind)) == "compute"
            )
            preserve_d2h_on_close = bool(
                getattr(self, "work_conserving_mode", False)
                and not keep_event_d2h_active
                and not is_compute_window
            )
            if getattr(self, "split_guard_mode", False):
                # Open one continuous PFS ceiling at phase zero.  D2H is
                # admitted only at group-installable compute windows, but use
                # the planner's finite per-window budget (not one quantum per
                # window).  A
                # projection DRAIN carries UINT64_MAX credits; distribute the
                # remaining event bytes evenly over the eligible compute slots
                # instead of converting the whole event to fail-open.  C++
                # still issues <=1 MiB requests, so the larger grant is only a
                # cumulative ceiling and preserves the causal boundary.
                group_installable = logical_phase_id in self.current_installable_phases
                if (
                    logical_phase_id != int(credits[0].phase_id)
                    and is_compute_window
                    and group_installable
                ):
                    if split_compute_slots_remaining <= 0:
                        d2h_delta = 0
                    else:
                        per_slot = (
                            split_d2h_remaining + split_compute_slots_remaining - 1
                        ) // split_compute_slots_remaining
                        d2h_delta = min(
                            split_d2h_remaining,
                            per_slot
                            if getattr(self, "work_full_compute_lane", False)
                            else min(per_slot, V4_MAX_COLLECTIVE_D2H_CREDIT_BYTES),
                        )
                    split_d2h_remaining -= d2h_delta
                    split_compute_slots_remaining -= 1
                elif logical_phase_id != int(credits[0].phase_id):
                    # A collective boundary is not a causal D2H interval.
                    # Do not open a new GPU->host allowance immediately before
                    # NCCL; any request already issued in the preceding
                    # compute interval remains the only non-preemptible
                    # residual and is accounted by the C++ admission trace.
                    d2h_delta = 0
                    if is_compute_window:
                        # Work-conserving residual lane: a group-
                        # noninstallable compute interval may launch at most
                        # one qD request.  The request is issued at the
                        # compute boundary and may finish during the following
                        # collective, but no fresh credit is opened at that
                        # collective.  This keeps the physical residual at
                        # <=1 MiB while preventing the entire event from
                        # being deferred to the terminal drain.
                        if getattr(self, "split_guard_mode", False) and not getattr(
                            self, "work_conserving_mode", False
                        ):
                            compute_burst = (
                                V4_FULL_COMPUTE_D2H_BURST_MIB * MIB
                                if getattr(self, "work_full_compute_lane", False)
                                else V4_D2H_REQUEST_MIB * MIB
                            )
                            d2h_delta = min(
                                split_d2h_remaining,
                                compute_burst,
                            )
                            split_d2h_remaining -= d2h_delta
                        elif getattr(self, "work_conserving_mode", False):
                            # In the sparse work-conserving lane, a
                            # non-installable compute window is not a safe
                            # place for a new GPU-facing request.  Leave the
                            # remainder for the guarded collective residual
                            # or a later common compute window; this avoids
                            # turning every low-confidence lead-in into an
                            # active-overlap group.
                            d2h_delta = 0
                        split_compute_slots_remaining -= 1
                    elif getattr(self, "work_conserving_mode", False):
                        # The strict split lane above refuses all fresh
                        # collective D2H.  On the archived trace that lane
                        # admitted only about 185 MiB before terminal drain.
                        # The work-conserving experiment permits one bounded
                        # qD request at a collective boundary, never a larger
                        # grant and never beyond the event extent.  Prefer
                        # that residual only when the *next* compute lead-in
                        # is not group-installable: if the next lead-in is a
                        # safe common window, keep the D2H work there instead
                        # of adding avoidable collective exposure.  This is a
                        # causal, deterministic middle lane between strict
                        # compute-only placement and residuals at every
                        # collective, while preserving the one-request cap.
                        next_compute_installable = (
                            int(logical_phase_id) + 1
                            in self.current_installable_phases
                        )
                        if next_compute_installable:
                            d2h_delta = 0
                        else:
                            d2h_delta = min(
                                max(0, split_d2h_remaining),
                                V4_D2H_REQUEST_MIB * MIB,
                            )
                        split_d2h_remaining -= d2h_delta
                else:
                    if event_d2h_lease_first_plan and logical_phase_id == int(
                        credits[0].phase_id
                    ):
                        # A single post-save event lease avoids reopening the
                        # same large D2H prefix at every reset-first plan.
                        # Physical requests remain <=qD; collective windows
                        # may carry only the existing residual through the
                        # CUDA-ordered transition.
                        d2h_delta = max(0, int(split_d2h_remaining))
                        split_d2h_remaining = 0
                    elif getattr(self, "work_full_compute_lane", False):
                        # In the explicit full-compute experiment, distribute the
                        # event D2H extent across all compute lead-ins instead of
                        # placing the whole payload before phase zero.  The old
                        # burst was deadline-feasible but amplified the first
                        # collective's rank-arrival skew.  Each lead-in still
                        # carries a bounded cumulative credit and the C++ lane
                        # issues only <=1 MiB subcopies, so this preserves the
                        # causal stream order while smoothing the GPU-facing
                        # pressure.  ``split_compute_slots_remaining`` counts
                        # the remaining compute windows after phase zero; include
                        # the current phase in the divisor for an exact partition.
                        if split_initial_subquantum_plan:
                            # Do not front-load the whole event at the first
                            # post-save lead-in.  The C++ worker is requestized
                            # (<=1 MiB), but a full-event ceiling here leaves a
                            # large dormant lease that can survive several
                            # collective boundaries and widen the measured
                            # window skew.  Partition the finite event prefix
                            # over all compute lead-ins, including this one;
                            # later runtime plans recapture actual admission
                            # and only reopen the remaining event tail.
                            slots_left = max(
                                1, int(split_compute_slots_remaining) + 1
                            )
                            d2h_delta = min(
                                max(0, int(split_d2h_remaining)),
                                (
                                    max(0, int(split_d2h_remaining))
                                    + slots_left
                                    - 1
                                )
                                // slots_left,
                            )
                            split_d2h_remaining -= d2h_delta
                            split_compute_slots_remaining = max(
                                0, int(split_compute_slots_remaining) - 1
                            )
                        else:
                            slots_left = max(
                                1, int(split_compute_slots_remaining) + 1
                            )
                            d2h_delta = min(
                                max(0, int(split_d2h_remaining)),
                                (
                                    max(0, int(split_d2h_remaining))
                                    + slots_left
                                    - 1
                                )
                                // slots_left,
                            )
                            split_d2h_remaining -= d2h_delta
                    else:
                        d2h_delta = 0
                # Spread the *finite plan target* over phase boundaries in
                # bounded four-request windows.  The previous implementation
                # reopened 16 MiB at every phase, which violated the
                # controller's group-fair snapshot cap even though each
                # individual phase stayed below its local 16 MiB ceiling.
                # PFS is the non-preemptible lane.  Keep its bounded grants
                # on compute windows so a storage request cannot be opened at
                # the collective boundary it is meant to protect.  The
                # cumulative prefix still carries unused allowance across
                # controller plans; only the placement is changed here.
                # Persistence is host/storage work rather than a GPU-facing
                # D2H transfer.  Open its bounded prefix at every boundary so
                # host-ready inventory can drain immediately, including while
                # a collective is executing; D2H itself remains restricted to
                # compute intervals with no fresh collective spill.
                if getattr(self, "pfs_compute_only_lane", False):
                    if not is_compute_window:
                        # Keep storage admission closed across NCCL.  A
                        # compute boundary may, however, open the whole
                        # rank-local residual cap (four 4 MiB requests).
                        # The C++ admission path still enforces both the
                        # 16 MiB byte cap and four-request cap, so this is a
                        # bounded work-conserving grant rather than a larger
                        # physical request.
                        pfs_delta = 0
                    else:
                        pfs_delta = min(
                            split_pfs_remaining,
                            int(self.config.max_pfs_inflight_bytes),
                        )
                else:
                    if published and not pfs_opened_this_plan:
                        pfs_delta = split_pfs_remaining
                    else:
                        pfs_delta = min(
                            split_pfs_remaining,
                            V4_PFS_REQUEST_MIB * MIB,
                        )
                split_pfs_remaining -= pfs_delta
                if pfs_delta:
                    pfs_opened_this_plan = True
                next_d2h_cumulative = d2h_cumulative + d2h_delta
                next_pfs_cumulative = pfs_cumulative + pfs_delta
                d2h_active = d2h_delta
                pfs_active = pfs_delta
                installable = bool(d2h_delta or pfs_delta)
            elif self.scheduled:
                d2h_delta = int(credit.d2h_budget_bytes) if installable else 0
                pfs_delta = int(credit.pfs_budget_bytes) if installable else 0
                next_d2h_cumulative = d2h_cumulative + d2h_delta
                next_pfs_cumulative = pfs_cumulative + pfs_delta
                d2h_active = next_d2h_cumulative - d2h_cumulative
                pfs_active = next_pfs_cumulative - pfs_cumulative
            else:
                # UINT64_MAX has a deliberately renewing meaning in the C++
                # prefix protocol, so repeating it cannot encode a final zero
                # delta. A full-event allowance is equally non-throttling and
                # remains an ordinary finite prefix that can be closed by
                # repeating it once.
                d2h_active = int(self.event_expected_state_bytes)
                pfs_active = int(self.event_expected_pfs_bytes)
                if d2h_active <= 0 or pfs_active <= 0:
                    raise RuntimeError("v4_open requires positive event byte extents")
                if (
                    d2h_cumulative > UINT64_MAX - d2h_active
                    or pfs_cumulative > UINT64_MAX - pfs_active
                ):
                    raise RuntimeError("v4_open phase-credit prefix overflows uint64")
                next_d2h_cumulative = d2h_cumulative + d2h_active
                next_pfs_cumulative = pfs_cumulative + pfs_active
            effective_window_deltas.append((int(d2h_active), int(pfs_active)))
            runtime_phase_id = self.next_runtime_phase_id
            self.next_runtime_phase_id += 1
            token = runtime_phase_id
            slot = {
                "token": int(token),
                "plan_version": int(self.current_runtime_plan_version),
                "phase_id": int(runtime_phase_id),
                "logical_phase_id": logical_phase_id,
                "installable": installable,
                "terminal_close": False,
                "enqueued": False,
                "d2h_cumulative_ceiling_bytes": int(next_d2h_cumulative),
                "pfs_cumulative_ceiling_bytes": int(next_pfs_cumulative),
                "d2h_active_budget_bytes": int(d2h_active),
                "pfs_active_budget_bytes": int(pfs_active),
                "preserve_d2h_on_close": preserve_d2h_on_close,
                "keep_d2h_active_on_close": bool(keep_event_d2h_active),
            }
            pair = (int(self.current_runtime_plan_version), int(runtime_phase_id))
            self.current_phase_slots[logical_phase_id] = slot
            self.event_runtime_slots[pair] = slot
            self.event_installed_runtime_phases.add(pair)
            preserve_pfs_on_close = bool(
                getattr(self, "split_guard_mode", False)
                and not keep_event_d2h_active
            )
            accepted = bool(
                self.ckpt_engine.prepare_credit_transition(
                    int(token),
                    int(self.current_runtime_plan_version),
                    int(runtime_phase_id),
                    int(next_d2h_cumulative),
                    int(next_pfs_cumulative),
                    self.config.max_pfs_inflight_bytes,
                    self.config.watchdog_timeout_ns,
                    0,
                    UINT64_MAX,
                    preserve_pfs_on_close,
                    preserve_d2h_on_close,
                    bool(keep_event_d2h_active),
                )
            )
            if not accepted:
                self._force_drain(
                    f"DataStates rejected prepared transition logical_phase={logical_phase_id}"
                )
                return False
            d2h_cumulative = next_d2h_cumulative
            pfs_cumulative = next_pfs_cumulative

        if getattr(self, "split_guard_mode", False):
            # The plan record must describe the exact stream ceilings that the
            # admission trace will prove.  In particular, a controller DRAIN
            # carries UINT64_MAX placeholders; exposing those placeholders
            # after we deliberately retain a finite split lane would make the
            # analyzer see a false plan/trace mismatch.
            effective_windows = tuple(
                replace(
                    credit,
                    d2h_budget_bytes=d2h_delta,
                    pfs_budget_bytes=pfs_delta,
                    d2h_spill_bytes=0,
                    pfs_spill_bytes=0,
                )
                for credit, (d2h_delta, pfs_delta) in zip(
                    credits, effective_window_deltas
                )
            )
            planned_d2h = sum(item[0] for item in effective_window_deltas)
            planned_pfs = sum(item[1] for item in effective_window_deltas)
            if hasattr(self.current_rank_plan, "__dataclass_fields__"):
                self.current_rank_plan = replace(
                    self.current_rank_plan,
                    target_d2h_bytes=planned_d2h,
                    target_pfs_bytes=(
                        int(getattr(self, "current_pfs_ready_budget_bytes", 0))
                        if getattr(self, "pfs_future_lease", False)
                        else planned_pfs
                    ),
                    planned_d2h_bytes=planned_d2h,
                    planned_pfs_bytes=planned_pfs,
                    windows=effective_windows,
                )
            else:
                # Lightweight CPU fixtures use a mutable namespace instead
                # of the production frozen RankCreditPlan.
                self.current_rank_plan.windows = effective_windows

        if getattr(self, "split_guard_mode", False) or not self.scheduled:
            self.split_guard_pfs_cumulative_ceiling = int(pfs_cumulative)
        if getattr(self, "split_guard_mode", False):
            # ``d2h_cumulative`` is deliberately plan-local because each
            # runtime plan version recaptures the C++ D2H base.  The Python
            # fallback ceiling is event-global, however: if a later stage
            # stats snapshot is malformed, it must conservatively remember
            # every prefix that an earlier plan could already have admitted
            # instead of resetting to zero and minting the event again.
            # Keep this monotone and bounded by the published event extent.
            event_d2h_extent = max(
                0, int(getattr(self, "event_expected_state_bytes", 0))
            )
            self.split_guard_d2h_cumulative_ceiling = min(
                event_d2h_extent,
                max(0, int(prior_split_d2h_issued))
                + max(0, int(d2h_cumulative)),
            )

        terminal_logical_phase_id = max(int(credit.phase_id) for credit in credits) + 1
        if terminal_logical_phase_id in self.current_phase_slots:
            raise RuntimeError("terminal credit CLOSE collides with a planned phase")
        runtime_phase_id = self.next_runtime_phase_id
        self.next_runtime_phase_id += 1
        token = runtime_phase_id
        # The terminal CLOSE is a bookkeeping boundary, not a bulk PFS
        # admission.  Earlier code promoted this slot to the full event
        # extent and updated the event-global ceiling here; that made every
        # subsequent ordinary plan believe the PFS prefix was already open,
        # then physically admitted the entire remaining event at terminal.
        # Ordinary stream phases must advance the cumulative prefix; terminal
        # may carry only the prefix already prepared by those phases.
        terminal_pfs_cumulative = int(pfs_cumulative)
        terminal_pfs_release = max(0, terminal_pfs_cumulative - int(pfs_cumulative))
        terminal_slot = {
            "token": int(token),
            "plan_version": int(self.current_runtime_plan_version),
            "phase_id": int(runtime_phase_id),
            "logical_phase_id": int(terminal_logical_phase_id),
            "installable": False,
            "terminal_close": True,
            "enqueued": False,
            "d2h_cumulative_ceiling_bytes": int(d2h_cumulative),
            "pfs_cumulative_ceiling_bytes": int(terminal_pfs_cumulative),
            "d2h_active_budget_bytes": 0,
            # This is a terminal PFS release, not fresh collective credit.
            # The C++ prefix protocol records the cumulative delta in the
            # admission trace, so retain it in the slot for exact binding.
            "pfs_active_budget_bytes": int(terminal_pfs_release),
            "preserve_d2h_on_close": False,
            "terminal_pfs_release_bytes": int(terminal_pfs_release),
        }
        terminal_pair = (
            int(self.current_runtime_plan_version),
            int(runtime_phase_id),
        )
        self.current_phase_slots[terminal_logical_phase_id] = terminal_slot
        self.event_runtime_slots[terminal_pair] = terminal_slot
        self.event_installed_runtime_phases.add(terminal_pair)
        self.event_terminal_runtime_phases.add(terminal_pair)
        accepted = bool(
            self.ckpt_engine.prepare_credit_transition(
                int(token),
                int(self.current_runtime_plan_version),
                int(runtime_phase_id),
                int(d2h_cumulative),
                int(terminal_pfs_cumulative),
                self.config.max_pfs_inflight_bytes,
                self.config.watchdog_timeout_ns,
                0,
                UINT64_MAX,
                bool(getattr(self, "split_guard_mode", False)),
                False,
            )
        )
        if not accepted:
            self._force_drain("DataStates rejected the terminal step credit CLOSE")
            return False
        self.current_terminal_logical_phase_id = terminal_logical_phase_id
        return True

    def _enqueue_phase_transition(
        self,
        logical_phase_id: int,
        cuda_stream_ptr: int,
    ) -> bool:
        with self.control_lock:
            if self.control_retiring or self.fail_open_reason:
                return False
            slot = self.current_phase_slots.get(int(logical_phase_id))
            if slot is None:
                self._force_drain(
                    f"missing prepared transition for logical phase {logical_phase_id}"
                )
                return False
            if bool(slot["enqueued"]):
                self._force_drain(
                    f"duplicate transition enqueue for logical phase {logical_phase_id}"
                )
                return False
            accepted = bool(
                self.ckpt_engine.enqueue_credit_transition(
                    int(slot["token"]), int(cuda_stream_ptr)
                )
            )
            slot["enqueued"] = accepted
            self.installed_logical_phase_id = int(logical_phase_id)
            self.installed_runtime_phase_id = int(slot["phase_id"])
            self.installed_credit_accepted = accepted
            self.installed_d2h_budget_bytes = int(slot["d2h_active_budget_bytes"])
            self.installed_pfs_budget_bytes = int(slot["pfs_active_budget_bytes"])
            self.metrics.v4_phase_install_count += 1
        if not accepted:
            self._force_drain(
                f"DataStates failed stream transition enqueue logical_phase={logical_phase_id}"
            )
        return accepted

    def _retire_credit_transitions(self) -> None:
        """Quiesce all callback streams and retire this event's slots once."""

        with self.control_lock:
            if self.control_retiring:
                return
            # Once latched, wrappers may still measure collectives but can no
            # longer enqueue a token after the terminal stream snapshot.
            self.control_retiring = True
        if self.observer is not None:
            self.observer.synchronize_control_streams()
        elif torch.cuda.is_available():
            torch.cuda.current_stream(self.device).synchronize()
        pending = int(self.ckpt_engine.pending_credit_transition_callbacks())
        if pending != 0:
            self._force_drain(
                f"credit transition retirement observed {pending} pending callbacks"
            )
            raise RuntimeError("DataStates credit callbacks remained pending after stream sync")
        if not bool(self.ckpt_engine.retire_credit_transition_callbacks()):
            self._force_drain("DataStates rejected terminal credit transition retirement")
            raise RuntimeError("DataStates credit transition retirement failed")

    def configure_transfer_before_save(
        self, state_bytes: int, chunk_bytes: int
    ) -> tuple[bool, float, int]:
        if chunk_bytes != V4_PAYLOAD_REGION_MIB * MIB:
            raise RuntimeError("v4 shadow payload must use 4 MiB chunks")
        self.ckpt_engine.configure_d2h_pacing(
            float(self.c0_d2h_rate_bps), 0
        )
        self.checkpoint_id = f"step-{self.checkpoint_step}"
        self.event_expected_state_bytes = state_bytes
        self.event_expected_pfs_bytes = state_bytes
        self.current_group_host_ready_bytes = 0
        self.current_group_host_ready_valid = False
        self.split_guard_d2h_cumulative_ceiling = 0
        self.split_guard_pfs_cumulative_ceiling = 0
        self.split_guard_recovery_plan_latched = False
        self.split_guard_recovery_replans = 0
        self.split_guard_recovery_plan_step = -1
        self.split_guard_last_full_plan_step = -1
        self.final_durability_evidence = None
        prior_layout = self._read_logical_layout_envelope()
        self.event_layout_pre_save_publication_sequence = self._require_layout_uint64(
            prior_layout, "publication_sequence"
        )
        self.event_logical_layout = {}
        stale_trace = self._take_admission_trace()
        if stale_trace["entries"] or stale_trace["transitions"]:
            raise RuntimeError("stale DataStates admission trace remained at event start")
        self.event_installed_runtime_phases = set()
        self.event_runtime_slots = {}
        self.event_terminal_runtime_phases = set()
        self.event_base_stats = self._read_stage_stats()
        self.event_start_monotonic_ns = int(self.event_base_stats["snapshot_monotonic_ns"])
        self.stage_stats_at_start = dict(self.event_base_stats)
        self.event_generation = 0
        self.d2h_event_lease_used = False
        self.d2h_event_lease_plan_version = 0
        self.current_controller_plan = None
        self.current_rank_plan = None
        self.current_plan_step = -1
        self.current_expected_signatures = []
        self.current_seen_phases = 0
        self.current_phase_slots = {}
        self.current_installable_phases = set()
        self.current_terminal_logical_phase_id = -1
        self.current_terminal_credit_closed = False
        self.current_projected_intersections = {}
        self.last_intersection_diagnostics = []
        self.control_retiring = False
        self.control_finalize_event.clear()
        self.control_common_terminal_origin = None
        self.control_common_terminal_mode = ""
        self.control_common_terminal_reason = ""
        self.fail_open_reason = ""
        self.metrics.v4_scheduled = self.scheduled
        self.metrics.v4_controller_sha256 = _V4_CONTROLLER_SHA256
        # The normal/strict path deliberately starts with a zero-credit
        # bootstrap and opens data work only after the first common plan.  The
        # explicit full-compute experiment is different: its contract is to
        # copy the whole event in the pre-collective lead-in, so the D2H
        # ceiling must already be armed before engine.save() enqueues regions.
        # PFS remains closed until the ordinary producer/consumer plan.
        bootstrap_d2h_budget = 0
        if getattr(self, "work_full_compute_lane", False):
            # The historical full-compute experiment opened the entire event
            # before the first CUDA phase.  That is useful as a liveness
            # stress test, but it creates a GPU-facing burst that is not
            # comparable to v4_open's stream-ordered phase schedule.  Allow a
            # bounded bootstrap prefix for matched performance runs while
            # preserving the legacy whole-event default for existing unit
            # fixtures and explicit stress tests.
            bootstrap_override = os.environ.get(
                "TEMPO_V4_BOOTSTRAP_D2H_MIB", ""
            ).strip()
            if bootstrap_override:
                try:
                    bootstrap_mib = int(bootstrap_override)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "TEMPO_V4_BOOTSTRAP_D2H_MIB must be an integer MiB value"
                    ) from exc
                if bootstrap_mib < 0:
                    raise RuntimeError(
                        "TEMPO_V4_BOOTSTRAP_D2H_MIB must be non-negative"
                    )
                bootstrap_d2h_budget = min(
                    int(state_bytes), bootstrap_mib * MIB
                )
            else:
                bootstrap_d2h_budget = int(state_bytes)
        bootstrap_pfs_budget = 0
        bootstrap_watchdog_ns = 2_000_000_000
        # Save() may consume a bounded bootstrap request before returning.
        # Keep the armed budget separate from the post-save remaining gauge.
        self.bootstrap_d2h_budget_bytes = int(bootstrap_d2h_budget)
        self.bootstrap_pfs_budget_bytes = int(bootstrap_pfs_budget)
        accepted = self._install_synchronous_bootstrap(
            logical_phase_id=0,
            d2h_budget_bytes=bootstrap_d2h_budget,
            pfs_budget_bytes=bootstrap_pfs_budget,
            watchdog_timeout_ns=bootstrap_watchdog_ns,
        )
        if not accepted:
            raise RuntimeError("v4 bootstrap credit was rejected before engine.save")
        armed = self._read_stage_stats()
        self._assert_structural_stats(armed)
        self._emit(
            "start",
            urgent=True,
            generation=self.event_generation,
            stage_service_calibration_sha256=self.stage_floor_provenance[
                "selection_file_sha256"
            ],
            stage_service_selected_d2h_bps=self.stage_floor_provenance[
                "selected_d2h_bps"
            ],
            stage_service_selected_pfs_bps=self.stage_floor_provenance[
                "selected_pfs_bps"
            ],
            c0_d2h_rate_bps=int(self.c0_d2h_rate_bps),
            stage_service_calibration_consumer=self.args.policy,
            stage_floor_provenance=self.stage_floor_provenance,
            controller_config=asdict(self.config),
            bootstrap={
                "runtime_plan_version": self.current_runtime_plan_version,
                "runtime_phase_id": self.installed_runtime_phase_id,
                "d2h_budget_bytes": bootstrap_d2h_budget,
                "pfs_budget_bytes": bootstrap_pfs_budget,
                "event_d2h_extent_bytes": int(state_bytes),
                "event_pfs_extent_bytes": int(self.event_expected_pfs_bytes),
                "watchdog_timeout_ns": bootstrap_watchdog_ns,
                "accepted": accepted,
            },
            raw_stats=armed,
        )
        return False, float(self.c0_d2h_rate_bps), 0

    def after_engine_save(self) -> None:
        # Keep the identical closed 2 s bootstrap through the post-save
        # controller rendezvous.  The first phase-plan install in
        # on_step_begin() changes to the normal 250 ms watchdog and releases
        # finite (tempo_v4) or unlimited (v4_open) credit.  This prevents the
        # open baseline from gaining a transfer-start head start in save().
        bootstrap_watchdog_ns = 2_000_000_000
        layout = self._validate_published_logical_layout(
            self._read_logical_layout_envelope()
        )
        self.event_logical_layout = layout
        self.event_expected_pfs_bytes = int(layout["logical_file_extent_bytes"])
        self.metrics.logical_file_extent_bytes = self.event_expected_pfs_bytes
        self.metrics.logical_layout_publication_sequence = int(
            layout["publication_sequence"]
        )
        self.metrics.logical_layout_version = int(layout["version"])
        raw = self._read_stage_stats()
        self._assert_structural_stats(raw)
        relative = self._event_relative_stats(raw)
        full_bootstrap = bool(getattr(self, "work_full_compute_lane", False))
        expected_d2h_bootstrap = int(
            getattr(
                self,
                "bootstrap_d2h_budget_bytes",
                int(self.event_expected_state_bytes) if full_bootstrap else 0,
            )
        )
        expected_pfs_bootstrap = int(
            getattr(self, "bootstrap_pfs_budget_bytes", 0)
        )
        bootstrap_valid = bool(
            0 <= int(raw.get("d2h_budget_bytes", -1)) <= expected_d2h_bootstrap
            and 0 <= int(raw.get("pfs_budget_bytes", -1)) <= expected_pfs_bootstrap
            and int(relative["d2h"].get("admitted_bytes", 0))
            <= expected_d2h_bootstrap
            and int(relative["pfs"].get("admitted_bytes", 0))
            <= expected_pfs_bootstrap
            and int(raw.get("watchdog_timeout_ns", -1)) == bootstrap_watchdog_ns
            and int(relative["watchdog_trip_count"]) == 0
            and not bool(raw.get("watchdog_fail_open", False))
        )
        if not bootstrap_valid:
            raise RuntimeError("v4 closed bootstrap changed or tripped during engine.save")
        self._emit(
            "post_save",
            urgent=True,
            bootstrap_watchdog_ns=bootstrap_watchdog_ns,
            bootstrap_valid=bootstrap_valid,
            pre_save_layout_publication_sequence=(
                self.event_layout_pre_save_publication_sequence
            ),
            logical_layout=layout,
            raw_stats=raw,
            event_relative_stats=relative,
        )

    def _event_live(self) -> bool:
        return bool(self.checkpoint_id and not self.event_recorded)

    def _control_live(self) -> bool:
        """Whether checkpoint byte/finalization work can still affect training."""

        return self._event_live() and not self.done.is_set()

    def _phase_schedule_active(self) -> bool:
        """Whether a common step plan must still run on every rank.

        Rank-local durability completion is intentionally ignored. Only a
        group-derived FINALIZE plan at a step boundary (or terminal training
        shutdown) may stop the already-prepared token sequence.
        """

        return bool(
            self._event_live()
            and self.current_controller_plan is not None
            and self.current_plan_step >= 0
            and self.current_controller_plan.mode is not self.v4.AdmissionMode.FINALIZE
            and not self.fail_open_reason
            and not self.control_retiring
        )

    def _envelope_breach(self, relative: dict[str, Any]) -> str:
        del relative
        # Lifetime counters expose completion but not stage-active service
        # time.  Dividing by event wall time would misclassify intentional
        # protected-window admission as a hardware service-rate breach.  Until
        # first-admit/active-time telemetry exists, the configured archive
        # floors are used only in deadline projection; zero progress,
        # watchdog, deadline, and invariant failures remain hard gates.
        return ""

    def _local_packet(self, step: int, observer_idle: bool) -> dict[str, Any]:
        errors: list[str] = []
        raw: dict[str, Any] = {}
        relative: dict[str, Any] | None = None
        envelope_breach = ""
        try:
            raw = self._read_stage_stats()
            if self._event_live():
                self._assert_structural_stats(raw)
                relative = self._event_relative_stats(raw)
                envelope_breach = self._envelope_breach(relative)
        except BaseException as exc:
            errors.append(f"stage telemetry error: {exc!r}")
        if self.observer_error:
            # This must travel in the gathered packet.  Appending a local-only
            # observer error after the gather would make controller inputs and
            # plan digests rank-divergent.
            errors.append(self.observer_error)
        if self.fail_open_reason:
            errors.append(self.fail_open_reason)
        controller_packet_error = getattr(self, "controller_packet_error", "")
        if controller_packet_error:
            errors.append(controller_packet_error)
        # Keep the local packet semantically rich.  The wire encoder below
        # projects it to the compact sufficient-statistics representation;
        # retaining the full snapshot here is required by direct/mock paths
        # and by local telemetry emitted for post-hoc verification.
        profile = (
            self.observer.profile_snapshot(step - 1)
            if self.observer is not None
            else None
        )
        progress = relative
        now_corrected_ns = time.time_ns() + int(self.args.clock_offset_ns)
        trigger_corrected_ns = (
            0
            if self.trigger_ns is None
            else self.trigger_ns + int(self.args.clock_offset_ns)
        )
        return {
            "rank": self.rank,
            "step": step,
            "active": self._event_live(),
            "checkpoint_id": self.checkpoint_id,
            "now_corrected_ns": now_corrected_ns,
            "deadline_corrected_ns": trigger_corrected_ns + int(self.args.deadline_seconds * 1e9),
            "raw_stats": raw,
            "event_relative_stats": relative,
            "profile": profile,
            "progress": progress,
            "observer_healthy": bool(
                observer_idle and self.observer is not None and self.observer.healthy()
            ),
            "error": "; ".join(errors),
            "envelope_breach": envelope_breach,
            "clock_uncertainty_ns": int(self.args.clock_calibration_rtt_ns) // 2,
        }

    @staticmethod
    def _controller_packet_failure(rank: int, step: int, reason: str) -> dict[str, Any]:
        # This is deliberately a valid semantic packet.  It lets every peer
        # finish the single scheduled collective and feed the same persistent
        # fail-open reason into the normal common DRAIN decision.
        bounded_reason = reason.encode("ascii", errors="backslashreplace").decode("ascii")[:512]
        return {
            "rank": rank,
            "step": step,
            "active": False,
            "checkpoint_id": "",
            "now_corrected_ns": 0,
            "deadline_corrected_ns": 0,
            "raw_stats": {},
            "event_relative_stats": None,
            "progress": None,
            "profile": None,
            "observer_healthy": False,
            "error": (
                f"controller packet rendezvous rejected rank {rank}: {bounded_reason}"
            ),
            "envelope_breach": "",
            "clock_uncertainty_ns": 0,
        }

    def _ensure_controller_packet_buffers(self) -> None:
        send = getattr(self, "_controller_packet_send_buffer", None)
        receive = getattr(self, "_controller_packet_recv_buffers", None)
        valid = bool(
            torch.is_tensor(send)
            and send.device.type == "cpu"
            and send.dtype is torch.uint8
            and send.is_contiguous()
            and send.numel() == V4ControllerPacketCodec.MAX_PACKET_BYTES
            and isinstance(receive, list)
            and len(receive) == self.world_size
            and all(
                torch.is_tensor(item)
                and item.device.type == "cpu"
                and item.dtype is torch.uint8
                and item.is_contiguous()
                and item.numel() == V4ControllerPacketCodec.MAX_PACKET_BYTES
                for item in receive
            )
        )
        if valid:
            return
        self._controller_packet_send_buffer = torch.zeros(
            V4ControllerPacketCodec.MAX_PACKET_BYTES, dtype=torch.uint8, device="cpu"
        )
        self._controller_packet_recv_buffers = [
            torch.empty_like(self._controller_packet_send_buffer)
            for _ in range(self.world_size)
        ]

    def _decode_controller_packet_buffers(self, step: int) -> list[dict[str, Any]]:
        packets: list[dict[str, Any]] = []
        for expected_rank, tensor in enumerate(self._controller_packet_recv_buffers):
            try:
                packet = V4ControllerPacketCodec.decode(
                    tensor.numpy(), expected_rank=expected_rank, expected_step=step
                )
                if packet.get("profile") is not None:
                    packet["profile"] = _v4_expand_compact_profile(
                        packet["profile"], expected_step=step
                    )
                if packet.get("progress") is not None:
                    packet["event_relative_stats"] = _v4_expand_compact_progress(
                        packet["progress"]
                    )
                else:
                    packet["event_relative_stats"] = None
                packet.setdefault("raw_stats", {})
            except BaseException as exc:
                packet = self._controller_packet_failure(
                    expected_rank,
                    step,
                    f"{type(exc).__name__}: {ascii(exc)}",
                )
            packets.append(packet)
        wire_errors = sorted(
            {
                str(packet.get("error", ""))
                for packet in packets
                if str(packet.get("error", "")).startswith(
                    "controller packet rendezvous rejected rank "
                )
            }
        )
        if wire_errors:
            # Keep the persistent diagnostic comfortably inside a subsequent
            # production-shaped packet; full frame details are deterministic
            # but must not trigger a secondary overflow cascade.
            common_error = "; ".join(wire_errors)[:2048]
            if not getattr(self, "controller_packet_error", ""):
                self.controller_packet_error = common_error
        if hasattr(self, "metrics"):
            # Count both byte-level decode rejection and a peer's valid
            # overflow/encode-rejection frame.
            self.metrics.v4_controller_packet_failures += len(wire_errors)
        return packets

    def _gather_packets(
        self, local: dict[str, Any], expected_step: int
    ) -> list[dict[str, Any]]:
        self._ensure_controller_packet_buffers()
        step = V4ControllerPacketCodec._require_index(
            expected_step, "scheduled step", UINT64_MAX
        )
        try:
            encoded_bytes = V4ControllerPacketCodec.encode_into(
                _v4_wire_packet(local),
                self._controller_packet_send_buffer,
                rank=self.rank,
                step=step,
            )
        except BaseException as exc:
            failure = self._controller_packet_failure(
                self.rank, step, f"{type(exc).__name__}: {ascii(exc)}"
            )
            # The fixed failure packet is intentionally small and entirely
            # primitive; local encode rejection must still reach the one common
            # collective instead of stranding peers before it.
            encoded_bytes = V4ControllerPacketCodec.encode_into(
                failure,
                self._controller_packet_send_buffer,
                rank=self.rank,
                step=step,
            )
        self._controller_packet_last_encoded_bytes = encoded_bytes
        if hasattr(self, "metrics"):
            self.metrics.v4_controller_packet_bytes_last = encoded_bytes
            self.metrics.v4_controller_packet_bytes_max = max(
                self.metrics.v4_controller_packet_bytes_max, encoded_bytes
            )
            self.metrics.v4_controller_packet_frame_bytes = (
                V4ControllerPacketCodec.MAX_PACKET_BYTES
            )
        # The controller process group has the short controller timeout.  All
        # tensors are equal, preallocated CPU uint8 frames, which is the Gloo
        # all_gather API supported across the target PyTorch versions.  This is
        # exactly one collective per deterministic scheduled rendezvous: no
        # length exchange, barrier, pickle, dynamic shape, or CUDA tensor.
        dist.all_gather(
            self._controller_packet_recv_buffers,
            self._controller_packet_send_buffer,
            group=self.controller_group,
        )
        return self._decode_controller_packet_buffers(step)

    @staticmethod
    def _profiles_match(packets: list[dict[str, Any]]) -> bool:
        profiles = [packet.get("profile") for packet in packets]
        if any(not isinstance(profile, dict) for profile in profiles):
            return False
        signatures = [
            [str(window["signature"]) for window in profile["windows"]]
            for profile in profiles
        ]
        return all(value == signatures[0] for value in signatures[1:])

    @staticmethod
    def _external_fail_open_reason(packets: list[dict[str, Any]]) -> str:
        reasons = [
            str(packet.get(key, ""))
            for packet in packets
            for key in ("error", "envelope_breach")
            if packet.get(key)
        ]
        return "; ".join(sorted(set(reasons)))

    def _update_tail_feedback(self, packets: list[dict[str, Any]]) -> None:
        if not self._profiles_match(packets):
            return
        profiles = [packet["profile"] for packet in packets]
        profile_step = int(profiles[0]["step"])
        notification_count = len(profiles[0]["notifications"])
        clock_uncertainty_ms = max(
            int(packet.get("clock_uncertainty_ns", 0)) for packet in packets
        ) / 1e6
        for index in range(notification_count):
            notifications = [profile["notifications"][index] for profile in profiles]
            signature = str(notifications[0]["signature"])
            if any(str(item["signature"]) != signature for item in notifications):
                continue
            gpu_values = [item.get("gpu_ms") for item in notifications]
            if any(value is None for value in gpu_values):
                continue
            # CUDA start precedes the close transition.  Its elapsed interval
            # already covers close+NCCL; only an actual host gate is additive.
            latency_ms = max(
                float(gpu_value)
                + float(notification.get("gate_wait_ms", 0.0))
                for gpu_value, notification in zip(gpu_values, notifications)
            )
            arrivals = [int(item["ready_corrected_ns"]) for item in notifications]
            skew_ms = (max(arrivals) - min(arrivals)) / 1e6
            if profile_step < self.args.warmup_steps:
                self.baseline_latency_ms.setdefault(signature, []).append(latency_ms)
                self.baseline_skew_ms.setdefault(signature, []).append(skew_ms)
                self.baseline_latency_ms[signature] = self.baseline_latency_ms[signature][-64:]
                self.baseline_skew_ms[signature] = self.baseline_skew_ms[signature][-64:]
                continue
            latency_reference = self.baseline_latency_ms.get(signature, [])
            skew_reference = self.baseline_skew_ms.get(signature, [])
            if not latency_reference or not skew_reference:
                continue
            baseline_latency = max(1e-6, percentile(latency_reference, 99))
            baseline_skew = max(0.0, percentile(skew_reference, 99))
            # Completion-tail debt belongs to the execution window. Arrival
            # skew already exists when the wrapper is reached. Charge it to
            # the lead-in only when every rank can actually install that
            # window.  Under enqueue-ahead the runtime holds the preceding
            # execution credit instead, so charging an uninstalled lead-in
            # would make feedback a no-op on the common callback-lag path.
            self.tail_feedback.observe(
                signature,
                latency_ms=latency_ms,
                baseline_latency_ms=baseline_latency,
                skew_ms=0.0,
                baseline_skew_ms=baseline_skew,
                clock_uncertainty_ms=max(1e-6, clock_uncertainty_ms),
            )
            lead_in_installable = all(
                bool(
                    profile["windows"][2 * index].get(
                        "installable",
                        str(notification.get("arrival_plan_source", ""))
                        not in V4_NONINSTALLABLE_ARRIVAL_SOURCES,
                    )
                )
                for profile, notification in zip(profiles, notifications)
            )
            arrival_feedback_signature = (
                f"lead-in:{signature}" if lead_in_installable else ""
            )
            if not arrival_feedback_signature and index > 0:
                prior_signatures = [
                    str(profile["notifications"][index - 1]["signature"])
                    for profile in profiles
                ]
                if all(value == prior_signatures[0] for value in prior_signatures[1:]):
                    arrival_feedback_signature = prior_signatures[0]
            if arrival_feedback_signature:
                self.tail_feedback.observe(
                    arrival_feedback_signature,
                    latency_ms=baseline_latency,
                    baseline_latency_ms=baseline_latency,
                    skew_ms=skew_ms,
                    baseline_skew_ms=baseline_skew,
                    clock_uncertainty_ms=max(1e-6, clock_uncertainty_ms),
                )

    @staticmethod
    def _raw_compute_intersections(
        packets: list[dict[str, Any]],
    ) -> tuple[int, dict[str, int]]:
        """Return one profile step and its corrected group intersections."""

        if not TempoV4Backend._profiles_match(packets):
            return -1, {}
        profiles = [packet["profile"] for packet in packets]
        profile_step = int(profiles[0]["step"])
        intersections: dict[str, int] = {}
        for index, template in enumerate(profiles[0]["windows"]):
            if str(template["kind"]) != "compute":
                continue
            if not all(
                "start_corrected_ns" in profile["windows"][index]
                and "end_corrected_ns" in profile["windows"][index]
                for profile in profiles
            ):
                continue
            signature = str(template["signature"])
            intersections[signature] = max(
                0,
                min(
                    int(profile["windows"][index]["end_corrected_ns"])
                    for profile in profiles
                )
                - max(
                    int(profile["windows"][index]["start_corrected_ns"])
                    for profile in profiles
                ),
            )
        return profile_step, intersections

    def _record_compute_intersection_realization(
        self, packets: list[dict[str, Any]]
    ) -> None:
        """Calibrate one-step compute-capacity realization from common data."""

        profile_step, current = self._raw_compute_intersections(packets)
        if profile_step < 0:
            return
        previous_step = getattr(
            self, "previous_compute_intersection_profile_step", None
        )
        previous = getattr(self, "previous_compute_intersections_ns", {})
        histories = getattr(self, "compute_intersection_ratios_ppm", {})
        if previous_step is not None and profile_step == previous_step + 1:
            for signature, predicted_ns in previous.items():
                if predicted_ns <= 0 or signature not in current:
                    continue
                ratio_ppm = min(
                    1_000_000,
                    int(current[signature]) * 1_000_000 // int(predicted_ns),
                )
                values = histories.setdefault(signature, [])
                values.append(ratio_ppm)
                del values[:-V4_COMPUTE_REALIZATION_HISTORY]
        self.compute_intersection_ratios_ppm = histories
        self.previous_compute_intersections_ns = current
        self.previous_compute_intersection_profile_step = profile_step

    def _compute_realization_ppm(self, signature: str) -> int:
        values = sorted(
            getattr(self, "compute_intersection_ratios_ppm", {}).get(signature, ())
        )
        if not values:
            # Unit fixtures and a genuinely new signature retain the prior
            # semantics.  Production FSDP signatures have eleven consecutive
            # warmup comparisons before the first checkpoint plan.
            return 1_000_000
        index = (len(values) - 1) * V4_COMPUTE_REALIZATION_PERCENTILE_PPM // 1_000_000
        return int(values[index])

    def _windows_from_packets(self, packets: list[dict[str, Any]]) -> tuple[Any, ...]:
        if not self._profiles_match(packets):
            self.last_intersection_diagnostics = []
            return ()
        profiles = [packet["profile"] for packet in packets]
        windows: list[Any] = []
        intersection_diagnostics: list[dict[str, Any]] = []
        for index, template in enumerate(profiles[0]["windows"]):
            durations = [int(profile["windows"][index]["duration_ns"]) for profile in profiles]
            is_compute = str(template["kind"]) == "compute"
            corrected_intersection_ns: int | None = None
            if is_compute and all(
                "start_corrected_ns" in profile["windows"][index]
                and "end_corrected_ns" in profile["windows"][index]
                for profile in profiles
            ):
                corrected_intersection_ns = max(
                    0,
                    min(
                        int(profile["windows"][index]["end_corrected_ns"])
                        for profile in profiles
                    )
                    - max(
                        int(profile["windows"][index]["start_corrected_ns"])
                        for profile in profiles
                    ),
                )
            raw_corrected_intersection_ns = corrected_intersection_ns
            realization_ppm = (
                self._compute_realization_ppm(str(template["signature"]))
                if is_compute and corrected_intersection_ns is not None
                else 1_000_000
            )
            if corrected_intersection_ns is not None:
                corrected_intersection_ns = (
                    corrected_intersection_ns * realization_ppm // 1_000_000
                )
            installable = all(
                bool(
                    profile["windows"][index].get(
                        "installable",
                        int(profile["windows"][index]["duration_ns"]) > 1_000,
                    )
                )
                for profile in profiles
            ) and (corrected_intersection_ns is None or corrected_intersection_ns > 0)
            if is_compute:
                intersection_diagnostics.append(
                    {
                        "profile_step": int(profiles[0]["step"]),
                        "phase_id": int(template["phase_id"]),
                        "signature": str(template["signature"]),
                        "min_rank_local_ns": min(durations),
                        "projected_intersection_ns": int(
                            0
                            if corrected_intersection_ns is None
                            else corrected_intersection_ns
                        ),
                        "raw_projected_intersection_ns": int(
                            0
                            if raw_corrected_intersection_ns is None
                            else raw_corrected_intersection_ns
                        ),
                        "realization_ppm": int(realization_ppm),
                        "scheduled_capacity_positive": installable,
                    }
                )
            # ``duration_ns`` may deliberately be stricter than the raw
            # corrected timestamp interval.  In particular, step-exit is
            # capped to the measured stream-token lifetime (1 ms) even when
            # host-side probe timestamps are much farther apart.  A group
            # intersection can only reduce a rank-local admission window; it
            # must never enlarge that explicit cap again.
            effective_duration_ns = min(durations)
            if corrected_intersection_ns is not None:
                effective_duration_ns = min(
                    effective_duration_ns, corrected_intersection_ns
                )
            safe_capacity = 1_000_000 if is_compute and installable else 0
            hard_capacity = 1_000_000 if installable else 0
            windows.append(
                self.v4.WindowSpec(
                    phase_id=index,
                    signature=str(template["signature"]),
                    kind=self.v4.WindowKind.COMPUTE if is_compute else self.v4.WindowKind.COLLECTIVE,
                    duration_ns=max(1_000, effective_duration_ns),
                    d2h_risk_ppm=10_000 if is_compute else 900_000,
                    pfs_risk_ppm=10_000 if is_compute else 950_000,
                    safe_d2h_capacity_ppm=safe_capacity,
                    safe_pfs_capacity_ppm=safe_capacity,
                    hard_d2h_capacity_ppm=hard_capacity,
                    hard_pfs_capacity_ppm=hard_capacity,
                )
            )
        self.last_intersection_diagnostics = intersection_diagnostics
        return self.v4.apply_tail_feedback(tuple(windows), self.tail_feedback)

    def _rank_progress(self, packet: dict[str, Any]) -> Any:
        relative = packet.get("event_relative_stats")
        if not isinstance(relative, dict):
            zero_stage = {
                "total_bytes": 0,
                "queued_bytes": 0,
                "ready_bytes": 0,
                "admitted_bytes": 0,
                "completed_bytes": 0,
                "inflight_bytes": 0,
                "last_progress_monotonic_ns": 0,
            }
            relative = {
                "d2h": zero_stage,
                "pfs": zero_stage,
                "watchdog_fail_open": True,
                "snapshot_monotonic_ns": 0,
            }

        def stage(name: str) -> Any:
            item = relative[name]
            return self.v4.StageProgress(
                total_bytes=int(item["total_bytes"]),
                queued_bytes=int(item["queued_bytes"]),
                ready_bytes=int(item["ready_bytes"]),
                admitted_bytes=int(item["admitted_bytes"]),
                completed_bytes=int(item["completed_bytes"]),
                inflight_bytes=int(item["inflight_bytes"]),
                last_progress_monotonic_ns=int(item["last_progress_monotonic_ns"]),
            )

        d2h = stage("d2h")
        pfs = stage("pfs")
        watchdog_ns = self.config.watchdog_timeout_ns
        snapshot_ns = int(relative.get("snapshot_monotonic_ns", 0))
        progress_stalled = False
        for progress in (d2h, pfs):
            if (
                progress.inflight_bytes > 0
                and progress.last_progress_monotonic_ns > 0
                and snapshot_ns - progress.last_progress_monotonic_ns > 4 * watchdog_ns
            ):
                progress_stalled = True
        # ``pfs.ready_bytes`` is a worker-queue gauge, not the authoritative
        # producer prefix.  The host worker may not publish that gauge until
        # it gets scheduled, even though completed D2H regions are already a
        # safe one-window-lag producer for future PFS admission.  Account for
        # the block-aligned host-only metadata prefix as well, then subtract
        # every PFS byte already admitted from that prefix.  The C++ ready
        # predicate remains the final physical gate, so a controller ceiling
        # can never issue an unready request.
        static_host_ready = max(0, pfs.total_bytes - d2h.total_bytes)
        derived_host_ready = max(
            0,
            static_host_ready + d2h.completed_bytes - pfs.admitted_bytes,
        )
        host_ready_bytes = min(
            pfs.unadmitted_bytes,
            max(int(relative["pfs"]["ready_bytes"]), derived_host_ready),
        )
        return self.v4.RankProgress(
            rank=int(packet["rank"]),
            now_ns=int(packet["now_corrected_ns"]),
            deadline_ns=int(packet["deadline_corrected_ns"]),
            d2h=d2h,
            pfs=pfs,
            d2h_rate_bytes_per_second=round(
                self.args.tempo_v4_d2h_floor_gbps * 1e9
            ),
            pfs_rate_bytes_per_second=round(
                self.args.tempo_v4_pfs_floor_gbps * 1e9
            ),
            finalization_reserve_ns=int(self.args.tempo_v4_finalization_reserve_ms * 1e6),
            pipeline_reserve_ns=int(self.args.tempo_v4_pipeline_reserve_ms * 1e6),
            host_ready_bytes=host_ready_bytes,
            watchdog_fail_open=bool(relative.get("watchdog_fail_open", False)),
            progress_stalled=progress_stalled,
        )

    def _latch_common_control_terminal(
        self,
        *,
        checkpoint_origin: int,
        step: int,
        mode: str,
        reason: str,
    ) -> None:
        """Stop future gathers only after every rank consumed one common result."""

        if mode not in ("FINALIZE", "DRAIN"):
            raise ValueError(f"invalid common control terminal mode {mode}")
        with self.control_lock:
            previous_origin = self.control_common_terminal_origin
            previous_mode = self.control_common_terminal_mode
            if previous_origin is not None:
                if previous_origin != checkpoint_origin or previous_mode != mode:
                    raise RuntimeError(
                        "conflicting common control terminal decisions: "
                        f"{previous_origin}:{previous_mode} vs {checkpoint_origin}:{mode}"
                    )
                return
            self.control_common_terminal_origin = checkpoint_origin
            self.control_common_terminal_mode = mode
            self.control_common_terminal_reason = reason
            self.metrics.v4_control_common_terminal_mode = mode
        # The durability worker may retire callback slots only after this
        # group-derived latch.  It is deliberately never set from rank-local
        # done/error state.
        self.control_finalize_event.set()
        self._emit(
            "control_terminal",
            urgent=True,
            controller_step=step,
            checkpoint_origin=checkpoint_origin,
            terminal_mode=mode,
            reason=reason,
            control_gather_calls=self.control_gather_calls,
            controller_packet_encoded_bytes=int(
                getattr(self, "_controller_packet_last_encoded_bytes", 0)
            ),
            controller_packet_max_encoded_bytes=int(
                self.metrics.v4_controller_packet_bytes_max
            ),
            controller_packet_frame_bytes=V4ControllerPacketCodec.MAX_PACKET_BYTES,
            controller_packet_failures=int(
                self.metrics.v4_controller_packet_failures
            ),
        )

    def _terminal_control_decision(
        self,
        *,
        step: int,
        checkpoint_origin: int,
        packets: list[dict[str, Any]],
    ) -> None:
        """Resolve the bounded window from one final common status snapshot."""

        errors: list[str] = []
        ranks = [int(packet.get("rank", -1)) for packet in packets]
        if sorted(ranks) != list(range(self.world_size)):
            errors.append(f"rank set mismatch {sorted(ranks)}")
        checkpoint_ids = {str(packet.get("checkpoint_id", "")) for packet in packets}
        expected_checkpoint_id = f"step-{checkpoint_origin}"
        if checkpoint_ids != {expected_checkpoint_id}:
            errors.append(f"checkpoint ids {sorted(checkpoint_ids)}")
        if not all(bool(packet.get("active", False)) for packet in packets):
            errors.append("one or more ranks reported an inactive checkpoint")
        external = self._external_fail_open_reason(packets)
        if external:
            errors.append(external)

        progress: list[Any] = []
        for packet in sorted(packets, key=lambda value: int(value.get("rank", -1))):
            if not isinstance(packet.get("event_relative_stats"), dict):
                errors.append(f"rank {packet.get('rank')} lacks event-relative stats")
                continue
            try:
                progress.append(self._rank_progress(packet))
            except BaseException as exc:
                errors.append(f"rank {packet.get('rank')} progress error: {exc!r}")
        complete = bool(
            not errors
            and len(progress) == self.world_size
            and all(rank_progress.finished for rank_progress in progress)
        )
        if complete:
            terminal_plan_begin = time.perf_counter()
            snapshot = self.v4.PlannerInput(
                checkpoint_id=expected_checkpoint_id,
                generation=self.event_generation,
                step=step,
                ranks=tuple(progress),
                windows=(),
                active=True,
                signatures_valid=False,
                observer_healthy=True,
                external_fail_open_reason="",
            )
            try:
                plan = self.controller.plan(snapshot)
                self.v4.validate_plan(snapshot, plan, self.config)
                if plan.mode is not self.v4.AdmissionMode.FINALIZE:
                    raise RuntimeError(f"terminal complete snapshot produced {plan.mode}")
            except BaseException as exc:
                errors.append(f"FINALIZE validation failed: {exc!r}")
            else:
                self.current_controller_plan = plan
                self.current_rank_plan = plan.for_rank(self.rank)
                self.current_plan_step = step
                self.current_expected_signatures = []
                self.current_seen_phases = 0
                self.metrics.v4_mode = str(plan.mode.value)
                self.metrics.v4_global_slack_ns = int(plan.global_slack_ns)
                self.metrics.v4_projected_completion_ns = int(
                    plan.projected_completion_ns
                )
                self.metrics.v4_deadline_feasible = bool(plan.deadline_feasible)
                self.metrics.v4_plan_count += 1
                # The terminal planner result is a real controller plan and
                # must be present in the common packet/plan ledger.  Without
                # this record, the runtime can correctly finish the event
                # while the analyzer cannot bind the terminal gather to its
                # FINALIZE plan and must reject the event as incomplete.
                terminal_elapsed_ms = (
                    time.perf_counter() - terminal_plan_begin
                ) * 1000.0
                self.metrics.v4_controller_ms += terminal_elapsed_ms
                terminal_packet = next(
                    packet
                    for packet in packets
                    if int(packet.get("rank", -1)) == self.rank
                )
                self._emit(
                    "plan",
                    urgent=True,
                    controller_step=step,
                    generation=self.event_generation,
                    controller_ms=terminal_elapsed_ms,
                    controller_packet_encoded_bytes=int(
                        getattr(self, "_controller_packet_last_encoded_bytes", 0)
                    ),
                    controller_packet_max_encoded_bytes=int(
                        self.metrics.v4_controller_packet_bytes_max
                    ),
                    controller_packet_frame_bytes=(
                        V4ControllerPacketCodec.MAX_PACKET_BYTES
                    ),
                    controller_packet_failures=int(
                        self.metrics.v4_controller_packet_failures
                    ),
                    trigger_unix_ns=self.trigger_ns,
                    trigger_corrected_ns=(self.trigger_ns or 0)
                    + int(self.args.clock_offset_ns),
                    deadline_corrected_ns=(self.trigger_ns or 0)
                    + int(self.args.clock_offset_ns)
                    + int(self.args.deadline_seconds * 1e9),
                    input_digest=plan.input_digest,
                    controller_plan_version=int(plan.plan_version),
                    runtime_plan_version=int(self.current_runtime_plan_version),
                    controller_plan_reused=False,
                    split_guard_recovery_plan_step=int(
                        getattr(self, "split_guard_recovery_plan_step", -1)
                    ),
                    mode=str(plan.mode.value),
                    global_slack_ns=int(plan.global_slack_ns),
                    projected_completion_ns=int(plan.projected_completion_ns),
                    deadline_feasible=bool(plan.deadline_feasible),
                    force_drain=bool(plan.force_drain),
                    reason=str(plan.reason),
                    signatures_valid=False,
                    local_rank_plan=asdict(self.current_rank_plan),
                    local_raw_stats=terminal_packet.get("raw_stats"),
                    local_event_relative_stats=terminal_packet.get(
                        "event_relative_stats"
                    ),
                    local_envelope_breach=terminal_packet.get(
                        "envelope_breach", ""
                    ),
                    projected_group_intersections={},
                    prior_projection_vs_actual={},
                )
                self.event_generation += 1
                self._latch_common_control_terminal(
                    checkpoint_origin=checkpoint_origin,
                    step=step,
                    mode="FINALIZE",
                    reason="all ranks complete at terminal control status gather",
                )
                return

        # SplitGuard intentionally keeps the PFS lane work-conserving after
        # the bounded observation window.  At this terminal gather, the
        # stream-ordered D2H schedule is already closed, while outstanding
        # host-ready PFS requests may still be completing.  Treat that normal
        # persistence tail as FINALIZE; wait_durable() performs the bounded
        # force-drain/fsync join.  The scheduled controller retains the strict
        # incomplete=>DRAIN behavior above because it has no independent
        # continuous PFS lane.
        if (
            getattr(self, "split_guard_mode", False)
            and not errors
            and len(progress) == self.world_size
        ):
            self.metrics.v4_mode = "FINALIZE"
            self.metrics.v4_control_window_exhausted = False
            self._latch_common_control_terminal(
                checkpoint_origin=checkpoint_origin,
                step=step,
                mode="FINALIZE",
                reason=(
                    "split_guard terminal: D2H schedule closed; "
                    "continuous PFS tail joins in wait_durable"
                ),
            )
            return

        unfinished = [
            rank_progress.rank for rank_progress in progress if not rank_progress.finished
        ]
        details = "; ".join(sorted(set(errors)))
        reason = (
            "control_window_exhausted: terminal common status found incomplete "
            f"ranks={unfinished}"
            + (f"; {details}" if details else "")
        )
        self.metrics.v4_control_window_exhausted = True
        self._force_drain(reason)
        self._latch_common_control_terminal(
            checkpoint_origin=checkpoint_origin,
            step=step,
            mode="DRAIN",
            reason=reason,
        )

    def _force_drain(self, reason: str) -> None:
        lowered = reason.lower()
        if "control_window_exhausted" in lowered:
            reason_class = "control_window_exhausted"
            nonfailure_drain = False
        elif "global slack" in lowered and "fail-open watermark" in lowered:
            reason_class = "deadline_feasibility"
            nonfailure_drain = True
        elif "watchdog" in lowered or "stalled" in lowered:
            reason_class = "watchdog_or_zero_progress"
            nonfailure_drain = False
        elif "signature" in lowered or "phase order" in lowered:
            reason_class = "collective_signature"
            nonfailure_drain = False
        elif "controller" in lowered or "plan" in lowered:
            reason_class = "controller_failure"
            nonfailure_drain = False
        else:
            reason_class = "runtime_error"
            nonfailure_drain = False
        with self.control_lock:
            if self.fail_open_reason:
                return
            self.fail_open_reason = reason
            try:
                raw = self._read_stage_stats()
                if not bool(raw.get("enabled", False)):
                    # A first fail-open must still arm the decoupled 1 MiB
                    # D2H / 4 MiB payload-PFS data path.
                    self._install_synchronous_bootstrap(
                        logical_phase_id=max(0, self.installed_logical_phase_id),
                        d2h_budget_bytes=UINT64_MAX,
                        pfs_budget_bytes=UINT64_MAX,
                    )
                self.ckpt_engine.force_drain()
            except BaseException as exc:
                reason = f"{reason}; force_drain error={exc!r}"
                self.fail_open_reason = reason
            self.metrics.v4_force_drain = True
            self.metrics.v4_force_drain_reason = reason
            self.metrics.v4_force_drain_reason_class = reason_class
            self.metrics.v4_mode = "DRAIN"
        self._emit(
            "fail_open",
            urgent=True,
            controller_step=self.current_plan_step,
            generation=self.event_generation,
            reason=reason,
            reason_class=reason_class,
            nonfailure_drain=nonfailure_drain,
        )

    def on_observer_error(self, reason: str) -> None:
        self.observer_error = reason
        if self._event_live():
            self._force_drain(reason)

    def _reuse_plan_without_gather(self, step: int) -> float:
        """Advance a work-conserving event using its last group plan.

        The plan was produced by a prior rank-symmetric gather.  This path
        intentionally performs no new process-group collective; it only
        rebinds the logical controller step and creates the next immutable
        stream transition slots.  Terminal status still uses the scheduled
        common gather, and the default stride is one, so this cannot affect
        normal/split-guard measurements unless explicitly requested.
        """
        if self.current_controller_plan is None or self.current_rank_plan is None:
            return 0.0
        begin = time.perf_counter()
        self.current_plan_step = int(step)
        self.current_seen_phases = 0
        self.current_expected_signatures = [
            str(window.signature)
            for window in self.current_rank_plan.windows
            if str(window.kind.value) == "collective"
        ]
        if not self._prepare_plan_transitions():
            self._force_drain("failed to prepare reused work-conserving transitions")
            return (time.perf_counter() - begin) * 1000.0
        first = self.current_rank_plan.windows[0]
        stream = torch.cuda.current_stream(self.device)
        stream_ptr = (
            self.observer.register_control_stream(stream)
            if self.observer is not None
            else int(stream.cuda_stream)
        )
        self._enqueue_phase_transition(int(first.phase_id), stream_ptr)
        return (time.perf_counter() - begin) * 1000.0

    def on_step_begin(self, step: int) -> float:
        # Reassert the policy-specific allocator mode at the point where the
        # live controller is about to plan.  This guards against backend
        # wrappers/reloads that may replace the controller instance after
        # construction, while keeping v4_open as the unrestricted reference.
        if hasattr(self, "controller"):
            # Keep the mode contract identical to __init__: the strict
            # scheduled/split-guard path is compute-only, while the explicit
            # work-conserving experiment may place a bounded one-request D2H
            # residual at a collective boundary.  Reasserting the raw policy
            # predicate here silently turned work_conserving back into the
            # strict compute-only planner on every training step.
            self.controller.compute_only_d2h = (
                self.args.policy == "tempo_v4"
                and not getattr(self, "work_conserving_mode", False)
            )
            self.controller.policy_name = str(self.args.policy)
        schedule_entry = self.control_gather_schedule.get(step)
        if schedule_entry is None:
            self.control_gather_skips += 1
            return 0.0
        gather_kind, checkpoint_origin = schedule_entry
        if (
            checkpoint_origin is not None
            and self.control_common_terminal_origin == checkpoint_origin
        ):
            # This latch was derived from a prior common gather.  It is the only
            # event-dependent state allowed to shorten the static maximum.
            self.control_gather_skips += 1
            return 0.0
        if (
            gather_kind == "controlled"
            and checkpoint_origin is not None
            and int(getattr(self, "control_reuse_stride", 1)) > 1
            and self.current_controller_plan is not None
            and self.current_plan_step >= checkpoint_origin + 1
            and (int(step) - int(checkpoint_origin) - 1)
            % int(getattr(self, "control_reuse_stride", 1))
            != 0
        ):
            return self._reuse_plan_without_gather(step)

        begin = time.perf_counter()
        observer_idle = bool(
            self.observer is not None
            and self.observer.wait_idle_bounded(self.args.tempo_v4_controller_timeout_ms / 1000.0)
        )
        if not observer_idle:
            self.observer_error = "observer completion queue exceeded controller timeout"
        local = self._local_packet(step, observer_idle)
        try:
            self.current_event_host_ready_bytes = int(
                self._rank_progress(local).host_ready_bytes
            )
        except BaseException:
            self.current_event_host_ready_bytes = 0
        self.control_gather_calls += 1
        if checkpoint_origin is not None and self.checkpoint_step == checkpoint_origin:
            self.metrics.v4_control_gather_count += 1
        if gather_kind == "terminal":
            self.control_terminal_gather_calls += 1
            if self.checkpoint_step == checkpoint_origin:
                self.metrics.v4_control_terminal_gather_count += 1
        try:
            packets = self._gather_packets(local, step)
        except BaseException as exc:
            self.controller_group_failed = True
            if self._event_live():
                self._force_drain(f"controller all_gather failed: {exc!r}")
                # No common decision is possible after process-group failure;
                # unblock bounded teardown and fail the rank immediately.
                self.control_finalize_event.set()
            raise RuntimeError("controller all_gather failed during scheduled rendezvous") from exc

        # The first producer-lead prefix must be group-symmetric.  Keep the
        # minimum derived host-ready inventory from the common packet set so
        # each rank clamps its local plan to the same safe frontier.  Later
        # rolling leases may advance monotonically from this common base.
        try:
            self.current_group_host_ready_bytes = min(
                int(self._rank_progress(packet).host_ready_bytes)
                for packet in packets
            )
            self.current_group_host_ready_valid = True
        except (TypeError, KeyError, ValueError, RuntimeError):
            self.current_group_host_ready_bytes = 0
            self.current_group_host_ready_valid = False

        if gather_kind == "terminal":
            assert checkpoint_origin is not None
            self._terminal_control_decision(
                step=step,
                checkpoint_origin=checkpoint_origin,
                packets=packets,
            )
            elapsed_ms = (time.perf_counter() - begin) * 1000
            self.metrics.v4_controller_ms += elapsed_ms
            return elapsed_ms

        self._record_compute_intersection_realization(packets)
        self._update_tail_feedback(packets)
        if gather_kind == "warmup" or not self._event_live():
            elapsed_ms = (time.perf_counter() - begin) * 1000
            return elapsed_ms

        # Do not force a split-guard event to remain live until the static
        # cp+20 terminal rendezvous after every rank has already completed its
        # D2H and PFS extents.  That deterministic terminal is a safety cap,
        # not a required minimum duration.  A normal scheduled gather already
        # contains the same rank-ordered packet set, so this common completion
        # test is deterministic and lets wait_durable/fsync start immediately.
        # It is deliberately limited to split_guard; the legacy scheduled
        # path keeps its strict terminal-window semantics.
        if (
            getattr(self, "split_guard_mode", False)
            and checkpoint_origin is not None
        ):
            try:
                progress = [
                    self._rank_progress(packet)
                    for packet in sorted(packets, key=lambda value: int(value["rank"]))
                ]
            except BaseException:
                progress = []
            if len(progress) == self.world_size and all(
                rank_progress.finished for rank_progress in progress
            ):
                self._terminal_control_decision(
                    step=step,
                    checkpoint_origin=checkpoint_origin,
                    packets=packets,
                )
                elapsed_ms = (time.perf_counter() - begin) * 1000
                self.metrics.v4_controller_ms += elapsed_ms
                return elapsed_ms

        windows = self._windows_from_packets(packets)
        actual_intersections = {
            str(item["signature"]): int(item["projected_intersection_ns"])
            for item in self.last_intersection_diagnostics
        }
        prior_projection_evaluation = [
            {
                "signature": signature,
                "projected_ns": int(projected_ns),
                "actual_ns": int(actual_intersections.get(signature, 0)),
                "shortfall_ns": max(
                    0, int(projected_ns) - int(actual_intersections.get(signature, 0))
                ),
            }
            for signature, projected_ns in sorted(
                self.current_projected_intersections.items()
            )
        ]
        signatures_valid = bool(windows) and self._profiles_match(packets)
        external_reasons = self._external_fail_open_reason(packets)
        if not signatures_valid:
            external_reasons = "; ".join(
                value
                for value in (
                    external_reasons,
                    "collective signature mismatch or incomplete completion profile",
                )
                if value
            )
        snapshot = self.v4.PlannerInput(
            checkpoint_id=self.checkpoint_id,
            generation=self.event_generation,
            step=step,
            ranks=tuple(
                self._rank_progress(packet)
                for packet in sorted(packets, key=lambda value: int(value["rank"]))
            ),
            windows=windows,
            active=True,
            signatures_valid=signatures_valid,
            observer_healthy=all(bool(packet.get("observer_healthy", False)) for packet in packets),
            external_fail_open_reason=external_reasons,
        )
        # The controller's staleness ledger advances only when ``plan`` is
        # called.  A reused transition therefore cannot be counted as a
        # fresh step: even a one-step reuse makes the next real call appear
        # two steps old and violates max_plan_staleness_steps=1.  Keep the
        # recovery latch as a mode decision, but disable plan reuse until an
        # explicit heartbeat/ledger API exists.  This costs a small amount of
        # CPU and prevents the live deadline/watchdog failure mode.
        reused_recovery_plan = False
        if reused_recovery_plan:
            # The recovery plan is already validated and its finite causal
            # lane is deliberately monotone across runtime plan versions.
            # Keep the common gather and per-step stream transitions, but do
            # not spend hundreds of milliseconds solving the same plan again.
            plan = self.current_controller_plan
            self.split_guard_recovery_replans += 1
        else:
            plan = self.controller.plan(snapshot)
            self.split_guard_last_full_plan_step = int(step)
            try:
                self.v4.validate_plan(snapshot, plan, self.config)
            except BaseException as exc:
                self._force_drain(f"controller plan invariant failed: {exc!r}")
                plan = self.controller.plan(
                    self.v4.PlannerInput(
                        checkpoint_id=self.checkpoint_id,
                        generation=self.event_generation + 1,
                        step=step,
                        ranks=snapshot.ranks,
                        windows=snapshot.windows,
                        active=True,
                        signatures_valid=False,
                        observer_healthy=False,
                        external_fail_open_reason=f"controller plan invariant failed: {exc!r}",
                    )
                )
                self.event_generation += 1

        self.current_controller_plan = plan
        self.current_rank_plan = plan.for_rank(self.rank)
        # These prior-step intersections are capacity projections for this
        # step, never a current-step wall-clock activation gate.
        self.current_projected_intersections = dict(actual_intersections)
        self.current_installable_phases = {
            int(window.phase_id)
            for window in windows
            if max(
                int(window.hard_d2h_capacity_ppm),
                int(window.hard_pfs_capacity_ppm),
            )
            > 0
        }
        self.current_plan_step = step
        self.current_expected_signatures = [
            str(window.signature)
            for window in self.current_rank_plan.windows
            if str(window.kind.value) == "collective"
        ]
        self.current_seen_phases = 0
        self.metrics.v4_mode = str(plan.mode.value)
        self.metrics.v4_global_slack_ns = int(plan.global_slack_ns)
        self.metrics.v4_projected_completion_ns = int(plan.projected_completion_ns)
        self.metrics.v4_deadline_feasible = bool(plan.deadline_feasible)
        self.metrics.v4_plan_count += 1
        # A split-guard plan can become BALANCED with a negative projected
        # slack before the controller raises its explicit force_drain bit.
        # Enter the event-local recovery cache at that first irreversible
        # low-slack point; waiting for force_drain wastes several more full
        # planner calls and is exactly the CPU/deadline failure seen in live
        # event-76 telemetry.  The current plan is still installed normally;
        # only subsequent common gathers reuse it.
        if (
            getattr(self, "split_guard_mode", False)
            and not reused_recovery_plan
            and str(plan.mode.value) == "BALANCED"
            and int(plan.global_slack_ns) <= 0
        ):
            self.split_guard_recovery_plan_latched = True
            self.split_guard_recovery_plan_step = int(step)
        if any(packet.get("envelope_breach") for packet in packets):
            self.metrics.v4_envelope_breach = True
            self.metrics.v4_envelope_breach_reason = "; ".join(
                str(packet["envelope_breach"])
                for packet in packets
                if packet.get("envelope_breach")
            )

        if plan.mode is self.v4.AdmissionMode.FINALIZE:
            # FINALIZE means both byte-moving stages are already complete. It
            # is a normal state, not fail-open, and needs no new phase budget.
            assert checkpoint_origin is not None
            self._latch_common_control_terminal(
                checkpoint_origin=checkpoint_origin,
                step=step,
                mode="FINALIZE",
                reason=str(plan.reason),
            )
        elif plan.force_drain:
            # Only the controller's bounded low-slack projection may enter
            # the finite recovery lane.  Structural failures (staleness,
            # duplicate generation, signature/observer mismatch, watchdog,
            # or a genuinely exhausted horizon) must remain DRAIN; converting
            # those to BALANCED hides the exact error and can issue work from
            # an invalid snapshot.
            plan_reason_lower = str(plan.reason).lower()
            allow_bounded_projection_recovery = bool(
                getattr(self, "split_guard_mode", False)
                and "global slack" in plan_reason_lower
                and (
                    "bounded recovery" in plan_reason_lower
                    or "recovery watermark" in plan_reason_lower
                )
                and "no bounded phase horizon" not in plan_reason_lower
            )
            if allow_bounded_projection_recovery:
                # A low-rate projection is not a structural failure for the
                # work-conserving split lane.  Keep the plan's window geometry,
                # clear the irreversible DRAIN bit, and let the stream-ordered
                # compute/PFS ceilings carry the event to wait_durable().
                plan = replace(
                    plan,
                    mode=self.v4.AdmissionMode.BALANCED,
                    force_drain=False,
                    reason=(
                        "split_guard retained finite causal lane after deadline "
                        f"projection: {plan.reason}"
                    ),
                )
                self.current_controller_plan = plan
                self.current_rank_plan = plan.for_rank(self.rank)
                self.split_guard_recovery_plan_latched = True
                self.split_guard_recovery_plan_step = int(step)
                self.metrics.v4_mode = str(plan.mode.value)
                self.metrics.v4_deadline_feasible = bool(plan.deadline_feasible)
                if not self._prepare_plan_transitions():
                    self._force_drain(
                        "DataStates failed to prepare split_guard recovery transitions"
                    )
                else:
                    first = self.current_rank_plan.windows[0]
                    stream = torch.cuda.current_stream(self.device)
                    stream_ptr = (
                        self.observer.register_control_stream(stream)
                        if self.observer is not None
                        else int(stream.cuda_stream)
                    )
                    self._enqueue_phase_transition(int(first.phase_id), stream_ptr)
            else:
                self._force_drain(str(plan.reason))
                assert checkpoint_origin is not None
                self._latch_common_control_terminal(
                    checkpoint_origin=checkpoint_origin,
                    step=step,
                    mode="DRAIN",
                    reason=str(plan.reason),
                )
        elif not self.fail_open_reason:
            if not self._prepare_plan_transitions():
                self._force_drain("DataStates failed to prepare the step phase schedule")
            else:
                first = self.current_rank_plan.windows[0]
                stream = torch.cuda.current_stream(self.device)
                stream_ptr = (
                    self.observer.register_control_stream(stream)
                    if self.observer is not None
                    else int(stream.cuda_stream)
                )
                # This callback sits behind prior queued GPU work and before
                # newly enqueued forward work; phase 0 is never CPU-opened.
                self._enqueue_phase_transition(int(first.phase_id), stream_ptr)

        local_plan = asdict(self.current_rank_plan)
        if getattr(self, "pfs_future_lease", False):
            local_plan["pfs_future_lease_bytes"] = int(
                getattr(self, "current_pfs_future_lease_bytes", 0)
            )
            local_plan["pfs_ready_budget_bytes"] = int(
                getattr(self, "current_pfs_ready_budget_bytes", 0)
            )
        elapsed_ms = (time.perf_counter() - begin) * 1000
        self.metrics.v4_controller_ms += elapsed_ms
        self._emit(
            "plan",
            controller_step=step,
            generation=self.event_generation,
            controller_ms=elapsed_ms,
            controller_packet_encoded_bytes=int(
                getattr(self, "_controller_packet_last_encoded_bytes", 0)
            ),
            controller_packet_max_encoded_bytes=int(
                self.metrics.v4_controller_packet_bytes_max
            ),
            controller_packet_frame_bytes=V4ControllerPacketCodec.MAX_PACKET_BYTES,
            controller_packet_failures=int(
                self.metrics.v4_controller_packet_failures
            ),
            trigger_unix_ns=self.trigger_ns,
            trigger_corrected_ns=(self.trigger_ns or 0) + int(self.args.clock_offset_ns),
            deadline_corrected_ns=(self.trigger_ns or 0)
            + int(self.args.clock_offset_ns)
            + int(self.args.deadline_seconds * 1e9),
            input_digest=plan.input_digest,
            controller_plan_version=int(plan.plan_version),
            runtime_plan_version=self.current_runtime_plan_version,
            controller_plan_reused=bool(reused_recovery_plan),
            split_guard_recovery_plan_step=int(
                getattr(self, "split_guard_recovery_plan_step", -1)
            ),
            mode=str(plan.mode.value),
            global_slack_ns=int(plan.global_slack_ns),
            projected_completion_ns=int(plan.projected_completion_ns),
            deadline_feasible=bool(plan.deadline_feasible),
            force_drain=bool(plan.force_drain),
            reason=str(plan.reason),
            signatures_valid=signatures_valid,
            local_rank_plan=local_plan,
            local_raw_stats=local.get("raw_stats"),
            local_event_relative_stats=local.get("event_relative_stats"),
            local_envelope_breach=local.get("envelope_breach", ""),
            projected_group_intersections=self.last_intersection_diagnostics,
            prior_projection_vs_actual=prior_projection_evaluation,
        )
        self.event_generation += 1
        return elapsed_ms

    def on_collective_phase(
        self,
        *,
        step: int,
        phase_index: int,
        sequence: int,
        signature: str,
        ready_unix_ns: int,
        cuda_stream_ptr: int,
    ) -> dict[str, Any]:
        del sequence, ready_unix_ns
        metadata = {
            "checkpoint_active_at_ready": self._phase_schedule_active(),
            "controlled_at_ready": False,
            "controller_plan_version": 0,
            "logical_phase_id": phase_index,
            "runtime_phase_id": self.installed_runtime_phase_id,
            "credit_accepted": self.installed_credit_accepted,
            "arrival_plan_source": "stream_ordered_lead_in",
            "drain_active_at_ready": False,
            "finalize_at_ready": False,
            "completion_callback_lag": False,
        }
        if not self._event_live():
            return metadata
        if (
            self.current_controller_plan is not None
            and self.current_controller_plan.mode is self.v4.AdmissionMode.FINALIZE
        ):
            # The event remains unrecorded until its background fsync/commit
            # worker is joined, but no D2H or PFS data admission remains.
            metadata.update(
                {
                    "checkpoint_active_at_ready": False,
                    "finalize_at_ready": True,
                    "arrival_plan_source": "finalize_no_data_work",
                }
            )
            return metadata
        if (
            getattr(self, "split_guard_mode", False)
            and self.control_common_terminal_mode == "FINALIZE"
            and self.control_common_terminal_origin == self.checkpoint_step
        ):
            # The status-only terminal gather is scheduled at the end of the
            # observation window and can share its training step with one
            # trailing FSDP collective.  That collective is outside the
            # controller window; do not compare it with the previous plan
            # step or turn an expected tail into a signature DRAIN.
            metadata.update(
                {
                    "checkpoint_active_at_ready": False,
                    "finalize_at_ready": True,
                    "arrival_plan_source": "finalize_no_data_work",
                    "logical_phase_id": phase_index,
                }
            )
            return metadata
        if not self._phase_schedule_active() and not self.fail_open_reason:
            return metadata
        if self.fail_open_reason:
            # An irreversible DRAIN deliberately gives up phase control to
            # preserve deadline/liveness. Keep measuring this real overlap,
            # but do not run the normal phase-order machine against a latched
            # open engine or manufacture accepted-control coverage.
            metadata.update(
                {
                    "checkpoint_active_at_ready": self._control_live(),
                    "controller_plan_version": int(
                        self.current_controller_plan.plan_version
                        if self.current_controller_plan is not None
                        else 0
                    ),
                    "logical_phase_id": 2 * phase_index + 1,
                    "runtime_phase_id": self.installed_runtime_phase_id,
                    "credit_accepted": False,
                    "arrival_runtime_phase_id": self.installed_runtime_phase_id,
                    "arrival_credit_accepted": False,
                    "execution_credit_accepted": False,
                    "completion_callback_lag": False,
                    "arrival_plan_source": "deadline_drain",
                    "drain_active_at_ready": self._control_live(),
                    "phase_install_ms": 0.0,
                }
            )
            return metadata
        if step != self.current_plan_step or phase_index >= len(self.current_expected_signatures):
            self.metrics.v4_signature_mismatch_count += 1
            self._force_drain(
                f"unexpected FSDP phase step={step} index={phase_index} for plan step={self.current_plan_step}"
            )
            return metadata
        expected = self.current_expected_signatures[phase_index]
        if signature != expected:
            self.metrics.v4_signature_mismatch_count += 1
            self._force_drain(
                f"FSDP signature mismatch index={phase_index} expected={expected} actual={signature}"
            )
            return metadata
        arrival_logical_phase = 2 * phase_index
        execution_logical_phase = arrival_logical_phase + 1
        with self.control_lock:
            arrival_slot = self.current_phase_slots.get(arrival_logical_phase)
            arrival_runtime_phase = (
                int(arrival_slot["phase_id"])
                if arrival_slot is not None
                else self.installed_runtime_phase_id
            )
            arrival_credit_accepted = bool(
                self.installed_logical_phase_id == arrival_logical_phase
                and self.installed_credit_accepted
            )
            arrival_installable = bool(
                arrival_slot is not None and arrival_slot["installable"]
            )
        install_begin = time.perf_counter()
        execution_accepted = self._enqueue_phase_transition(
            execution_logical_phase,
            cuda_stream_ptr,
        )
        install_ms = (time.perf_counter() - install_begin) * 1000
        self.current_seen_phases += 1
        arrival_plan_source = (
            "stream_ordered_lead_in"
            if arrival_installable
            else "stream_ordered_zero_compute"
        )
        metadata.update(
            {
                "controlled_at_ready": bool(
                    not self.fail_open_reason
                    and arrival_credit_accepted
                    and execution_accepted
                ),
                "controller_plan_version": int(self.current_controller_plan.plan_version),
                "logical_phase_id": execution_logical_phase,
                "runtime_phase_id": self.installed_runtime_phase_id,
                "credit_accepted": execution_accepted,
                "arrival_runtime_phase_id": arrival_runtime_phase,
                "arrival_credit_accepted": arrival_credit_accepted,
                "arrival_plan_source": arrival_plan_source,
                "execution_credit_accepted": execution_accepted,
                "completion_callback_lag": False,
                "phase_install_ms": install_ms,
                "phase_close_enqueue_ms": install_ms,
                "phase_open_enqueue_ms": 0.0,
            }
        )
        self._emit(
            "phase_ready",
            controller_step=step,
            generation=self.event_generation,
            phase_index=phase_index,
            signature=signature,
            arrival_logical_phase_id=arrival_logical_phase,
            arrival_plan_runtime_phase_id=arrival_runtime_phase,
            arrival_credit_accepted=arrival_credit_accepted,
            arrival_plan_source=arrival_plan_source,
            execution_logical_phase_id=execution_logical_phase,
            execution_plan_runtime_phase_id=self.installed_runtime_phase_id,
            execution_credit_accepted=execution_accepted,
            completion_callback_lag=False,
            phase_install_ms=install_ms,
            phase_close_enqueue_ms=install_ms,
            controlled_at_ready=metadata["controlled_at_ready"],
        )
        return metadata

    def on_collective_enqueued(
        self,
        *,
        step: int,
        phase_index: int,
        sequence: int,
        signature: str,
        cuda_stream_ptr: int,
        phase_metadata: dict[str, Any],
    ) -> None:
        """Enqueue the next compute delta after NCCL and its timing event."""

        del sequence
        if (
            not self._event_live()
            or not self._phase_schedule_active()
            or self.fail_open_reason
            or self.control_retiring
            or step != self.current_plan_step
            or phase_index >= len(self.current_expected_signatures)
            or signature != self.current_expected_signatures[phase_index]
        ):
            return
        next_phase = 2 * phase_index + 2
        if self.current_rank_plan is None or next_phase >= len(
            self.current_rank_plan.windows
        ):
            return
        install_begin = time.perf_counter()
        accepted = self._enqueue_phase_transition(
            next_phase,
            cuda_stream_ptr,
        )
        install_ms = (time.perf_counter() - install_begin) * 1000
        slot = self.current_phase_slots.get(next_phase)
        zeroed_noninstallable = bool(
            self.scheduled and slot is not None and not slot["installable"]
        )
        phase_metadata["phase_open_enqueue_ms"] = install_ms
        phase_metadata["phase_install_ms"] = float(
            phase_metadata.get("phase_close_enqueue_ms", 0.0)
        ) + install_ms
        phase_metadata["next_compute_credit_accepted"] = accepted
        phase_metadata["next_compute_zeroed_noninstallable"] = zeroed_noninstallable
        self._emit(
            "phase_open",
            controller_step=step,
            generation=self.event_generation,
            phase_index=phase_index,
            signature=signature,
            next_logical_phase_id=next_phase,
            next_runtime_phase_id=(0 if slot is None else int(slot["phase_id"])),
            credit_accepted=accepted,
            zeroed_noninstallable=zeroed_noninstallable,
            phase_open_enqueue_ms=install_ms,
        )

    def close_step_credit_before_probe(self, step: int) -> float:
        """Enqueue the terminal CLOSE before the probe collective.

        The caller has synchronized training compute and marked the observer
        profile complete. Enqueuing on that same current stream therefore puts
        this CLOSE after the final step-exit work and before the subsequently
        submitted probe, without synchronizing DataStates' private streams.
        """

        if not self._phase_schedule_active() or step != self.current_plan_step:
            return 0.0
        terminal_phase = int(self.current_terminal_logical_phase_id)
        slot = self.current_phase_slots.get(terminal_phase)
        terminal_pfs_release = (
            0 if slot is None else int(slot.get("terminal_pfs_release_bytes", 0))
        )
        if (
            terminal_phase < 0
            or slot is None
            or not bool(slot.get("terminal_close", False))
            or int(slot.get("d2h_active_budget_bytes", -1)) != 0
            or int(slot.get("pfs_active_budget_bytes", -1))
            != terminal_pfs_release
            or (
                terminal_pfs_release
                and not getattr(self, "split_guard_mode", False)
            )
        ):
            self._force_drain("missing terminal step credit CLOSE")
            return 0.0
        if self.current_seen_phases != len(self.current_expected_signatures):
            self.metrics.v4_signature_mismatch_count += 1
            self._force_drain(
                "terminal step credit CLOSE observed an incomplete FSDP phase set: "
                f"expected={len(self.current_expected_signatures)} "
                f"observed={self.current_seen_phases}"
            )
            return 0.0
        stream = torch.cuda.current_stream(self.device)
        callback_stream = stream
        close_done: Any | None = None
        if self.observer is not None:
            callback_stream = self.observer.get_transition_stream()
            stream_ready = torch.cuda.Event(enable_timing=False)
            close_done = torch.cuda.Event(enable_timing=False)
            stream_ready.record(stream)
            callback_stream.wait_event(stream_ready)
            stream_ptr = self.observer.register_control_stream(callback_stream)
        else:
            stream_ptr = int(stream.cuda_stream)
        begin = time.perf_counter()
        accepted = self._enqueue_phase_transition(terminal_phase, stream_ptr)
        if close_done is not None:
            close_done.record(callback_stream)
            # The following synthetic probe must observe the terminal CLOSE.
            stream.wait_event(close_done)
        enqueue_ms = (time.perf_counter() - begin) * 1000.0
        self.current_terminal_credit_closed = bool(accepted)
        self._emit(
            "step_credit_close",
            controller_step=step,
            generation=self.event_generation,
            terminal_logical_phase_id=terminal_phase,
            terminal_runtime_plan_version=int(slot["plan_version"]),
            terminal_runtime_phase_id=int(slot["phase_id"]),
            d2h_active_budget_bytes=int(slot["d2h_active_budget_bytes"]),
            pfs_active_budget_bytes=int(slot["pfs_active_budget_bytes"]),
            terminal_pfs_release_bytes=terminal_pfs_release,
            credit_accepted=bool(accepted),
            stream_ordered=True,
            ordering="after_step_exit_before_synthetic_probe",
            terminal_close_enqueue_ms=enqueue_ms,
        )
        return enqueue_ms

    def on_collective_complete(
        self,
        *,
        step: int,
        phase_index: int,
        sequence: int,
        signature: str,
        completion_unix_ns: int,
    ) -> None:
        del sequence
        # CUDA-event completion can be arbitrarily delayed by this observer
        # thread.  All control transitions were already enqueued synchronously
        # in the wrapped collective call, so this callback is telemetry-only.
        self._emit(
            "phase_complete",
            controller_step=step,
            generation=self.event_generation,
            completed_logical_phase_id=2 * phase_index + 1,
            completed_signature=signature,
            completion_unix_ns=completion_unix_ns,
            observer_telemetry_only=True,
            control_mutation=False,
        )

    def on_step_end(self, step: int) -> None:
        if self._event_live() and step == self.current_plan_step:
            phase_validation_active = bool(
                self._control_live()
                and not self.fail_open_reason
                and self.current_controller_plan is not None
                and self.current_controller_plan.mode
                is not self.v4.AdmissionMode.FINALIZE
            )
            if (
                phase_validation_active
                and self.current_seen_phases != len(self.current_expected_signatures)
            ):
                self.metrics.v4_signature_mismatch_count += 1
                self._force_drain(
                    f"FSDP phase count mismatch expected={len(self.current_expected_signatures)} "
                    f"observed={self.current_seen_phases}"
                )
            if (
                phase_validation_active
                and self.current_terminal_logical_phase_id >= 0
                and not self.current_terminal_credit_closed
            ):
                self._force_drain(
                    "step ended without applying its terminal zero-credit CLOSE"
                )
            try:
                raw = self._read_stage_stats()
                if phase_validation_active:
                    terminal_slot = self.current_phase_slots.get(
                        self.current_terminal_logical_phase_id
                    )
                    if not self._terminal_credit_stats_ok(raw, terminal_slot):
                        self._force_drain(
                            "terminal step credit CLOSE was not active before probe completion"
                        )
                relative = self._event_relative_stats(raw)
                self.metrics.v4_watchdog_trip_count = int(relative["watchdog_trip_count"])
                self.metrics.v4_rejected_plan_count = int(relative["rejected_plan_count"])
                invalid = int(relative["invalid_plan_count"])
                invariant = int(relative["invariant_violation_count"])
                if invalid or invariant:
                    self._force_drain(
                        f"DataStates control invariant counters nonzero invalid={invalid} invariant={invariant}"
                    )
                self._emit(
                    "step",
                    controller_step=step,
                    generation=self.event_generation,
                    raw_stats=raw,
                    event_relative_stats=relative,
                    fail_open_reason=self.fail_open_reason,
                )
            except BaseException as exc:
                self._force_drain(f"end-of-step telemetry failed: {exc!r}")
        self._flush_telemetry()

    def record_step(self, row: dict[str, Any]) -> None:
        self._emit("training_step", controller_step=int(row["step"]), row=row)
        self._flush_telemetry()

    def _terminal_credit_stats_ok(
        self, raw: dict[str, Any], terminal_slot: dict[str, Any] | None
    ) -> bool:
        """Validate the stream-applied terminal CLOSE.

        A SplitGuard terminal CLOSE has zero *delta* credit, but it retains
        the cumulative PFS lease so already-admitted O_DIRECT writes cannot
        be invalidated between controller replans.  The ordinary path must
        observe zero instantaneous budgets; the SplitGuard path instead
        checks that the carried PFS cumulative ceiling is unchanged.
        """

        if terminal_slot is None:
            return False
        if (
            int(raw["plan_version"]) != int(terminal_slot["plan_version"])
            or int(raw["phase_id"]) != int(terminal_slot["phase_id"])
            or int(raw["d2h_budget_bytes"]) != 0
        ):
            return False
        if int(raw["pfs_budget_bytes"]) == 0:
            return True
        return bool(
            self.split_guard_mode
            and int(raw.get("pfs_cumulative_ceiling_bytes", -1))
            == int(terminal_slot["pfs_cumulative_ceiling_bytes"])
            and int(raw["pfs_budget_bytes"]) >= 0
        )

    def _local_durability_evidence(self) -> dict[str, Any]:
        assert self.path is not None
        finalize_timeout = max(
            0.25,
            float(self.args.deadline_seconds)
            + float(self.args.tempo_v4_controller_timeout_ms) / 1000.0,
        )
        if not self.control_finalize_event.wait(finalize_timeout):
            self._force_drain(
                "durability completed before a common phase-schedule finalize boundary"
            )
            raise RuntimeError("timed out waiting for common phase-schedule finalize")
        # Freeze future control enqueues, synchronize every stream that carried
        # a transition, and retire the callback self-hold before consuming the
        # terminal trace.  Completion callbacks remain telemetry-only.
        self._retire_credit_transitions()
        raw = self._read_stage_stats()
        self._assert_structural_stats(raw)
        relative = self._event_relative_stats(raw)
        admission_trace = self._take_admission_trace()
        self._validate_admission_trace(admission_trace, relative)
        checkpoint_stat = self.path.stat()
        checkpoint_bytes = checkpoint_stat.st_size
        self._validate_checkpoint_file_extent(checkpoint_bytes)
        pfs = relative["pfs"]
        d2h = relative["d2h"]
        fsync_ns = int(relative["pfs_fsync_monotonic_ns"])
        valid = bool(
            relative.get("pfs_fsync_complete", False)
            and fsync_ns > self.event_start_monotonic_ns
            and int(pfs["completed_bytes"]) >= checkpoint_bytes
            and checkpoint_bytes == self.event_expected_pfs_bytes
            and int(pfs["inflight_bytes"]) == 0
            and int(pfs["inflight_requests"]) == 0
            and int(d2h["completed_bytes"]) >= self.event_expected_state_bytes
            and int(d2h["inflight_bytes"]) == 0
            and int(relative["invalid_plan_count"]) == 0
            and int(relative["invariant_violation_count"]) == 0
            and bool(raw.get("pfs_odirect_required", False))
            and bool(raw.get("pfs_odirect_verified", False))
            and int(relative.get("pfs_odirect_open_count", 0)) > 0
            # The configured quantum is an upper bound.  A small tensor or
            # aligned tail may legitimately issue only a smaller request;
            # requiring equality would make the durability proof depend on
            # an arbitrary smoke-model tensor layout.
            and int(d2h["max_request_bytes"])
            <= self.args.tempo_v4_d2h_chunk_mb * MIB
            and int(d2h["peak_inflight_bytes"])
            <= self.args.tempo_v4_d2h_chunk_mb * MIB
            and int(d2h["peak_inflight_requests"]) <= 1
            and int(pfs["max_request_bytes"])
            <= self.args.tempo_v4_pfs_chunk_mb * MIB
            and int(pfs["peak_inflight_bytes"])
            <= self.args.tempo_v4_max_pfs_inflight_mb * MIB
            and int(pfs["peak_inflight_requests"])
            <= self.args.tempo_v4_max_pfs_inflight_mb
            // self.args.tempo_v4_pfs_chunk_mb
            and int(raw["max_pfs_inflight_requests"])
            == self.args.tempo_v4_max_pfs_inflight_mb // self.args.tempo_v4_pfs_chunk_mb
        )
        if not valid:
            raise RuntimeError(
                "v4 durability evidence invalid: "
                + json.dumps(
                    {
                        "checkpoint_bytes": checkpoint_bytes,
                        "event_start_monotonic_ns": self.event_start_monotonic_ns,
                        "relative": relative,
                    },
                    sort_keys=True,
                )
            )
        self.metrics.fsync_evidence_valid = True
        full_evidence = {
            "kind": "v4_stage_counters_and_fsync",
            "checkpoint_file_bytes": checkpoint_bytes,
            "logical_layout": self.event_logical_layout,
            "event_start_monotonic_ns": self.event_start_monotonic_ns,
            "pfs_fsync_monotonic_ns": fsync_ns,
            "event_relative_stats": relative,
            "admission_trace": admission_trace,
        }
        # The exact admission trace can contain more than one thousand stream
        # transitions.  It belongs in the rank-local telemetry journal, not in
        # every rank's globally gathered commit manifest.  Commit the full
        # evidence by digest and carry only the small durability facts needed
        # to establish that the marker was published after this rank's fsync.
        self.final_durability_evidence = full_evidence
        return self._commit_durability_evidence(full_evidence)

    def wait_durable(self) -> None:
        if not self._event_live():
            return
        finalize_begin_ns = time.perf_counter_ns()
        # The deterministic control window must already have produced one
        # group-derived FINALIZE or DRAIN latch.  Never manufacture that latch
        # here from rank-local teardown timing: doing so could hide a skipped
        # controller rendezvous and release callback retirement asymmetrically.
        if not self.control_finalize_event.is_set():
            raise RuntimeError(
                "checkpoint reached wait_durable without a common control terminal"
            )
        # Training has no more tail-sensitive work for this event. FINALIZE
        # opens the stage ceilings but retains the 16 MiB PFS safety cap.
        try:
            self.ckpt_engine.force_drain()
            self._emit(
                "finalize",
                urgent=True,
                controller_step=self.current_plan_step,
                reason="training interval ended; bounded final drain",
            )
        except BaseException as exc:
            self._force_drain(f"finalize force_drain failed: {exc!r}")
        assert self.trigger_ns is not None
        deadline_remaining = max(
            0.05,
            self.args.deadline_seconds - (time.time_ns() - self.trigger_ns) / 1e9,
        )
        completed = self.done.wait(deadline_remaining)
        if not completed:
            self._force_drain("durability deadline expired during bounded wait")
            completed = self.done.wait(max(0.25, 2 * self.args.tempo_v4_watchdog_ms / 1000.0))
        if not completed:
            self._emit(
                "finish_timeout",
                urgent=True,
                deadline_remaining_seconds=deadline_remaining,
                fail_open_reason=self.fail_open_reason,
            )
            raise TimeoutError("TEMPO v4 DataStates worker exceeded bounded final drain")
        try:
            super().wait_durable()
        except BaseException as exc:
            # Preserve the final stage/commit failure even when the common
            # durability path raises before it can append the aggregate
            # checkpoint record.
            self._emit(
                "finish_error",
                urgent=True,
                error=repr(exc),
                fail_open_reason=self.fail_open_reason,
            )
            raise
        raw = self._read_stage_stats()
        relative = self._event_relative_stats(raw)
        finalization_elapsed_ms = (time.perf_counter_ns() - finalize_begin_ns) / 1e6
        reserved_residual_ms = (
            self.args.tempo_v4_finalization_reserve_ms
            + self.args.tempo_v4_pipeline_reserve_ms
        )
        # The controller projects the slowest rank's stage completion.  A fast
        # rank waiting inside the all-rank commit is therefore not additional
        # global finalization work; charging that wait again double-counts
        # cross-rank completion skew.  Keep it as diagnostic telemetry, while
        # bounding the actual post-training drain+commit elapsed time here.
        barrier_reserve_breached = False
        residual_reserve_breached = bool(
            finalization_elapsed_ms > reserved_residual_ms
            or barrier_reserve_breached
        )
        if residual_reserve_breached:
            self.metrics.v4_envelope_breach = True
            residual_reason = (
                "actual final drain+commit exceeded configured residual reserve: "
                f"actual={finalization_elapsed_ms:.3f}ms reserved={reserved_residual_ms:.3f}ms "
                f"barrier={self.metrics.durability_barrier_ms:.3f}ms "
                f"barrier_reserved={self.args.tempo_v4_finalization_reserve_ms:.3f}ms"
            )
            self.metrics.v4_envelope_breach_reason = "; ".join(
                value
                for value in (self.metrics.v4_envelope_breach_reason, residual_reason)
                if value
            )
        self.metrics.v4_watchdog_trip_count = int(relative["watchdog_trip_count"])
        self.metrics.v4_rejected_plan_count = int(relative["rejected_plan_count"])
        self.metrics.v4_mode = "FINALIZE" if not self.fail_open_reason else "DRAIN"
        final_durability_evidence = getattr(
            self, "final_durability_evidence", None
        )
        self._emit(
            "finish",
            urgent=True,
            mode=self.metrics.v4_mode,
            durable_ms=self.metrics.durable_ms,
            deadline_met=self.metrics.deadline_met,
            finalization_elapsed_ms=finalization_elapsed_ms,
            configured_finalization_reserve_ms=self.args.tempo_v4_finalization_reserve_ms,
            configured_pipeline_reserve_ms=self.args.tempo_v4_pipeline_reserve_ms,
            residual_reserve_breached=residual_reserve_breached,
            barrier_reserve_breached=barrier_reserve_breached,
            durability_barrier_ms=self.metrics.durability_barrier_ms,
            fsync_evidence_valid=self.metrics.fsync_evidence_valid,
            commit_validated=self.metrics.commit_validated,
            commit_manifest_sha256=self.metrics.commit_manifest_sha256,
            durability_evidence=final_durability_evidence,
            durability_evidence_sha256=(
                canonical_sha256(final_durability_evidence)
                if final_durability_evidence is not None
                else ""
            ),
            raw_stats=raw,
            event_relative_stats=relative,
            fail_open_reason=self.fail_open_reason,
        )
        self.controller.close_event(self.checkpoint_id)

    def finish_event(self) -> None:
        previous_count = len(self.checkpoint_events)
        super().finish_event()
        if len(self.checkpoint_events) != previous_count:
            # Preserve each completed event even if a later event or cleanup
            # fails; the final aggregate files are still rewritten at success.
            atomic_json(
                self.output_dir / f"checkpoint_events_rank{self.rank}.json",
                self.checkpoint_events,
            )
            atomic_json(
                self.output_dir / f"checkpoint_rank{self.rank}.json",
                asdict(self.metrics),
            )

    def close(self) -> None:
        if self._event_live():
            try:
                self.ckpt_engine.force_drain()
            except BaseException:
                pass
        super().close()
        if not self.telemetry_enabled:
            return
        self._emit(
            "journal_close",
            records_before_close=self.telemetry_records_emitted,
            local_write_calls_before_close=self.telemetry_write_calls,
            shared_write_calls_during_measurement=(
                self.telemetry_shared_write_calls_during_measurement
            ),
            normal_publication_after_backend_close=True,
            publisher_sha256=self.telemetry_publisher_sha256,
            preflight=self.telemetry_preflight_evidence,
        )
        self._close_local_telemetry()
        self._publish_local_telemetry()


def make_backend(**kwargs: Any) -> CheckpointBackend | None:
    policy = kwargs["args"].policy
    if policy == "none":
        return None
    if policy == "torch_async":
        return DCPBackend(**kwargs)
    if policy == "torchsnapshot":
        return TorchSnapshotBackend(**kwargs)
    if policy == "datastates":
        return DataStatesBackend(**kwargs)
    if policy == "tempo":
        return TempoBackend(**kwargs)
    if policy == "tempo_v2":
        return TempoV2Backend(**kwargs)
    if policy == "tempo_v3":
        return TempoV3Backend(**kwargs)
    if policy in ("v4_open", "tempo_v4"):
        return TempoV4Backend(**kwargs)
    raise AssertionError(policy)


def materialize_optimizer_state(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Create optimizer tensors so a fresh process has restore destinations."""
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 10_000)
    value = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(value).float().square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.current_stream(device).synchronize()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_root = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    model = MiniGPT(args.layers, args.hidden_size, args.ffn_size, args.heads).to(device)
    global_model_parameters = sum(parameter.numel() for parameter in model.parameters())
    mixed = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({DecoderBlock}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed,
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    # Process-group formation is outside the one-second persistence
    # deadline.  On a cold four-node restore launch, a rank can arrive after
    # the default five-second store timeout even though the job is healthy;
    # that invalidates the whole matched comparison before restore starts.
    # Keep the measured deadline unchanged while allowing bounded startup
    # jitter to converge.
    durability_timeout_seconds = max(30.0, args.deadline_seconds + 10.0)
    control_group = dist.new_group(
        backend="gloo", timeout=dt.timedelta(seconds=durability_timeout_seconds)
    )
    controller_group = dist.new_group(
        backend="gloo",
        timeout=dt.timedelta(milliseconds=args.tempo_v4_controller_timeout_ms),
    )
    dcp_group = dist.new_group(backend="gloo")
    warm = torch.ones(1, dtype=torch.int64)
    dist.all_reduce(warm, group=control_group)
    dist.all_reduce(warm, group=controller_group)
    dist.all_reduce(warm, group=dcp_group)
    if args.restore_only:
        clock_offset_ns, clock_rtt_ns = 0, 0
    else:
        clock_offset_ns, clock_rtt_ns = calibrate_wall_clock(
            rank, world_size, control_group, args.clock_calibration_samples
        )
    args.clock_offset_ns = clock_offset_ns
    args.clock_calibration_rtt_ns = clock_rtt_ns
    if not args.restore_only:
        atomic_json(
            output_dir / f"clock_calibration_rank{rank}.json",
            {
                "rank": rank,
                "offset_to_rank0_ns": clock_offset_ns,
                "minimum_round_trip_ns": clock_rtt_ns,
                "samples": args.clock_calibration_samples,
            },
        )
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed + rank)
    backend = make_backend(
        args=args,
        model=model,
        optimizer=optimizer,
        rng=rng,
        device=device,
        rank=rank,
        world_size=world_size,
        control_group=control_group,
        controller_group=controller_group,
        dcp_group=dcp_group,
    )
    observer = CudaCollectiveObserver(
        device=device,
        rank=rank,
        clock_offset_ns=clock_offset_ns,
        backend=backend if isinstance(backend, TempoV3Backend) else None,
        activity_backend=backend,
        phase_listener=backend if isinstance(backend, TempoV4Backend) else None,
    )
    if isinstance(backend, TempoV3Backend):
        backend.attach_gate(observer)
    if backend is not None:
        backend.attach_observer(observer)

    cleanup_complete = False

    def cleanup_runtime(raise_errors: bool = True) -> None:
        nonlocal cleanup_complete
        if cleanup_complete:
            return
        cleanup_complete = True
        error: BaseException | None = None
        try:
            observer.close()
        except BaseException as exc:
            error = exc
        try:
            if backend:
                backend.close()
        except BaseException as exc:
            if error is None:
                error = exc
        if error is not None and raise_errors:
            raise RuntimeError("runtime cleanup failed") from error

    atexit.register(cleanup_runtime, False)

    if args.restore_only:
        if backend is None:
            raise RuntimeError("--restore-only requires a checkpoint policy")
        materialize_optimizer_state(model, optimizer, args, device)
        expected_path = output_dir / f"expected_rank{rank}.pt"
        expected = torch.load(expected_path, map_location="cpu", weights_only=True)
        backend.expected_checksum = float(expected["model_checksum"])
        backend.expected_optimizer_checksum = float(expected["optimizer_checksum"])
        backend.expected_rng = expected["rng"]
        backend.select_restore_step(int(expected["step"]))
        observer.set_step(-2)
        restore = backend.restore()
        restore.update(
            {
                "rank": rank,
                "world_size": world_size,
                "policy": args.policy,
                "tier_mode": str(getattr(args, "tier_mode", "")),
            }
        )
        atomic_json(output_dir / f"fresh_restore_rank{rank}.json", restore)
        cleanup_runtime()
        atexit.unregister(cleanup_runtime)
        if observer.rows:
            atomic_csv(output_dir / f"fresh_collectives_rank{rank}.csv", sorted(observer.rows, key=lambda row: int(row["sequence"])))
        dist.barrier()
        if rank == 0:
            print("FRESH_RESTORE_COMPLETE " + json.dumps(restore, sort_keys=True), flush=True)
        dist.destroy_process_group()
        return

    probe_elements = args.probe_mb * 1024 * 1024 // 2
    probe = torch.ones(probe_elements, dtype=torch.bfloat16, device=device)
    for _ in range(3):
        dist.all_reduce(probe)
    torch.cuda.synchronize()

    rows: list[dict[str, Any]] = []
    checkpoint_step_set = set(args.checkpoint_steps)
    for step in range(args.steps):
        step_begin = time.perf_counter()
        controller_ms = backend.on_step_begin(step) if backend else 0.0
        # The controller rendezvous remains charged to step_ms, but it is not
        # usable compute capacity: the causal profile starts only after the
        # phase plan for this step has been installed.
        observer.set_step(step)
        if backend:
            backend.set_compute()
        value = torch.randn(
            args.batch_size,
            args.sequence_length,
            args.hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=rng,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(value)
            loss = output.float().square().mean()
        loss.backward()
        consistency_block_ms = backend.before_optimizer_step() if backend else 0.0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if backend:
            consistency_block_ms += backend.after_optimizer_step()

        if backend:
            backend.set_collective()
        # A device-wide synchronize also drains DataStates' private D2H stream
        # and would turn this asynchronous experiment into a synchronous one.
        torch.cuda.current_stream(device).synchronize()
        training_finish_ns = time.time_ns()
        observer.finish_step(step, training_finish_ns)
        terminal_close_enqueue_ms = (
            backend.close_step_credit_before_probe(step) if backend else 0.0
        )
        # The probe's host-side arrival is after the terminal CLOSE enqueue;
        # its CUDA work is submitted behind that callback on the same stream.
        arrival_ns = time.time_ns()
        probe_start = torch.cuda.Event(enable_timing=True)
        probe_end = torch.cuda.Event(enable_timing=True)
        probe_start.record()
        dist.all_reduce(probe)
        probe_end.record()
        probe_end.synchronize()
        probe_ms = probe_start.elapsed_time(probe_end)
        completion_ns = time.time_ns()
        completion_mono_ns = time.perf_counter_ns()
        if backend:
            backend.observe_collective(probe_ms, baseline_sample=step < args.warmup_steps)
            backend.set_compute()

        # Correctness checks and checkpoint API staging are not part of the
        # training kernel/collective time.  They are reported independently as
        # validation_ms and trigger_ms.
        training_step_ms = (time.perf_counter() - step_begin) * 1000
        if backend:
            backend.on_step_end(step)
        checkpoint_triggered = False
        trigger_ms = 0.0
        checkpoint_backpressure_ms = 0.0
        if backend and step in checkpoint_step_set:
            # Finalize the previous event before reusing the checkpoint engine.
            # With the default spacing this is normally a non-blocking join; if
            # persistence misses that spacing, the debt remains in the next
            # logical-iteration interval instead of being hidden.
            backpressure_begin = time.perf_counter()
            backend.finish_event()
            checkpoint_backpressure_ms = (time.perf_counter() - backpressure_begin) * 1000
            backend.start(step)
            trigger_ms = backend.metrics.trigger_ms
            checkpoint_triggered = True

        window_origin = next(
            (
                checkpoint_step
                for checkpoint_step in reversed(args.checkpoint_steps)
                if checkpoint_step < step <= checkpoint_step + args.window_steps
            ),
            None,
        )
        window = window_origin is not None
        rows.append(
            {
                "rank": rank,
                "hostname": socket.gethostname(),
                "policy": args.policy,
                "step": step,
                "measured": int(step >= args.warmup_steps),
                "checkpoint_window": int(window),
                "checkpoint_origin_step": -1 if window_origin is None else window_origin,
                "checkpoint_active": int(bool(backend and backend.active())),
                "checkpoint_d2h_active": int(bool(backend and backend.d2h_active())),
                "checkpoint_triggered": int(checkpoint_triggered),
                "loss": float(loss.item()),
                "step_ms": training_step_ms,
                "probe_ms": probe_ms,
                "training_finish_unix_ns": training_finish_ns,
                "probe_arrival_unix_ns": arrival_ns,
                "probe_completion_unix_ns": completion_ns,
                "probe_completion_mono_ns": completion_mono_ns,
                "consistency_block_ms": consistency_block_ms,
                "trigger_ms": trigger_ms,
                "checkpoint_backpressure_ms": checkpoint_backpressure_ms,
                "controller_ms": controller_ms,
                "terminal_close_enqueue_ms": terminal_close_enqueue_ms,
            }
        )
        if backend:
            backend.record_step(rows[-1])
        if backend and step == 0:
            backend.prepare()
        if rank == 0:
            print(
                f"policy={args.policy} step={step:03d} loss={loss.item():.5f} "
                f"step_ms={rows[-1]['step_ms']:.1f} probe_ms={probe_ms:.3f} "
                f"window={int(window)} active={rows[-1]['checkpoint_active']}",
                flush=True,
            )

    observer.set_step(args.steps)
    if backend:
        backend.finish_event()
        assert backend.checkpoint_step is not None
        assert backend.expected_checksum is not None
        assert backend.expected_optimizer_checksum is not None
        assert backend.expected_rng is not None
        atomic_torch_save(
            output_dir / f"expected_rank{rank}.pt",
            {
                "step": backend.checkpoint_step,
                "model_checksum": backend.expected_checksum,
                "optimizer_checksum": backend.expected_optimizer_checksum,
                "rng": backend.expected_rng,
            },
        )
        atomic_json(output_dir / f"checkpoint_rank{rank}.json", asdict(backend.metrics))
        atomic_json(
            output_dir / f"checkpoint_events_rank{rank}.json",
            backend.checkpoint_events,
        )

    cleanup_runtime()
    atexit.unregister(cleanup_runtime)
    if observer.phase_fabric_counters_enabled:
        atomic_json(
            output_dir / f"fabric_phase_counters_rank{rank}.json",
            {
                "schema_version": "tempo-rd-node-slice-counter-1",
                "scope": "node_slice",
                "rank": rank,
                "world_size": world_size,
                "counter_source": "sysfs:/sys/class/net/hsn*/statistics;host_device_sum",
                "records": sorted(
                    observer.phase_fabric_counters,
                    key=lambda row: int(row["sequence"]),
                ),
            },
        )
    if observer.rows:
        atomic_csv(output_dir / f"collectives_rank{rank}.csv", sorted(observer.rows, key=lambda row: int(row["sequence"])))

    atomic_csv(output_dir / f"steps_rank{rank}.csv", rows)
    measured = [row for row in rows if row["measured"]]
    window_rows = [row for row in rows if row["checkpoint_window"]]
    summary = {
        "policy": args.policy,
        "tier_mode": str(getattr(args, "tier_mode", "")),
        "tier_endpoint": str(getattr(backend.metrics, "tier_endpoint", "")) if backend else "",
        "tier_host_preloaded": bool(getattr(backend.metrics, "tier_host_preloaded", False)) if backend else False,
        "tier_gpu_transfer": bool(getattr(backend.metrics, "tier_gpu_transfer", False)) if backend else False,
        "rank": rank,
        "world_size": world_size,
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "source_sha256": source_sha256(),
        "runtime_python_modules_schema": RUNTIME_PYTHON_MODULES_SCHEMA,
        "runtime_python_modules": runtime_python_module_provenance(),
        "model_parameters": global_model_parameters,
        "clock_offset_ns": clock_offset_ns,
        "clock_calibration_rtt_ns": clock_rtt_ns,
        "v4_controller_sha256": _V4_CONTROLLER_SHA256
        if args.policy in ("v4_open", "tempo_v4")
        else "",
        "v4_controller_timeout_ms": args.tempo_v4_controller_timeout_ms,
        "c0_d2h_rate_bps": _v4_open_c0_rate_bps(args),
        "c0_enabled": bool(_v4_open_c0_rate_bps(args)),
        "c0_max_inflight_bytes": V4_D2H_REQUEST_MIB * MIB,
        "v4_telemetry_mode": str(args.tempo_v4_telemetry),
        "step_p99_ms": percentile([row["step_ms"] for row in measured], 99),
        "window_step_p99_ms": percentile([row["step_ms"] for row in window_rows], 99),
        "window_probe_p99_ms": percentile([row["probe_ms"] for row in window_rows], 99),
        "fresh_restore_required": (
            args.policy != "none" and str(getattr(args, "tier_mode", "")) != "d2h_only"
        ),
    }
    if isinstance(backend, TempoV4Backend):
        summary.update(
            {
                "v4_control_gather_calls": backend.control_gather_calls,
                "v4_control_terminal_gather_calls": (
                    backend.control_terminal_gather_calls
                ),
                "v4_control_gather_max_calls": len(
                    backend.control_gather_schedule
                ),
                "v4_control_gather_schedule_sha256": canonical_sha256(
                    sorted(
                        (step, kind, origin)
                        for step, (
                            kind,
                            origin,
                        ) in backend.control_gather_schedule.items()
                    )
                ),
            }
        )
    atomic_json(output_dir / f"summary_rank{rank}.json", summary)
    dist.barrier()
    if rank == 0:
        print("RUN_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # A graceful DataStates shutdown may wait on a distributed durability
        # barrier whose peer has already failed.  Exit this rank immediately so
        # srun --kill-on-bad-exit can terminate the step instead of burning the
        # remainder of the allocation in interpreter/destructor cleanup.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
