#!/usr/bin/env python3
"""Screen a compiled TEMPO epoch calendar on official LMCache/NIXL.

Ranks 0..3 move KV chunks to ranks 4..7 with the pinned official LMCache
``NixlChannel`` while every rank executes the same NCCL decoder.  The calendar
is compiled before distributed startup.  Its token-time action is a local
descriptor lookup and queue submission; enqueue, actual NIXL start, finish,
deadline, and queue lag are all retained so a queued plan cannot masquerade as
executed interconnect admission.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import queue
import socket
import statistics
import threading
import time
from types import MethodType
from typing import Any, Iterable

import numpy as np

from eval.sota_4node import run_inference_interconnect_2node as foreground
from eval.sota_4node import run_lmcache_nixl_contention_2node as official
from tempo.inference_epoch import EpochPlan, EpochProfile, load_epoch_artifact


WORLD_SIZE = 8
NODES = 2
RANKS_PER_NODE = 4
PAIR_COUNT = 4
CHUNKS_PER_REQUEST = 4
MIB = 1 << 20
MODE_ORDER = (
    "fg_only",
    "lmcache_greedy",
    "lmcache_static_serial",
    "tempo_epoch",
)
LATIN_ROWS = tuple(
    tuple(MODE_ORDER[(column + row) % len(MODE_ORDER)] for column in range(len(MODE_ORDER)))
    for row in range(len(MODE_ORDER))
)
BLOCK_MODES = tuple(mode for row in LATIN_ROWS for mode in row)
CANONICAL_QUANTA = tuple(
    (pair, chunk)
    for chunk in range(CHUNKS_PER_REQUEST)
    for pair in range(PAIR_COUNT)
)


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def quantum_indices_for_token(
    plan: EpochPlan,
    mode: str,
    token_index: int,
) -> tuple[int, ...]:
    if mode not in MODE_ORDER:
        raise ValueError(f"unknown mode: {mode}")
    if isinstance(token_index, bool) or not isinstance(token_index, int) or token_index < 0:
        raise ValueError("token_index must be a non-negative int")
    if not plan.feasible:
        raise ValueError("cannot schedule an infeasible plan")
    if mode == "fg_only":
        return ()
    if mode == "lmcache_greedy":
        return tuple(range(len(CANONICAL_QUANTA))) if token_index == 0 else ()
    if mode == "lmcache_static_serial":
        return (token_index,) if token_index < len(CANONICAL_QUANTA) else ()
    if token_index >= len(plan.quantum_indices_by_token):
        return ()
    return plan.quantum_indices_by_token[token_index]


def object_indices_for_rank(
    plan: EpochPlan,
    mode: str,
    token_index: int,
    *,
    pair_index: int,
    requests: int,
) -> tuple[int, ...]:
    if isinstance(pair_index, bool) or not isinstance(pair_index, int):
        raise ValueError("pair_index must be an int")
    if not 0 <= pair_index < PAIR_COUNT:
        raise ValueError("pair_index must be in 0..3")
    if isinstance(requests, bool) or not isinstance(requests, int) or requests <= 0:
        raise ValueError("requests must be a positive int")
    chunks = tuple(
        CANONICAL_QUANTA[index][1]
        for index in quantum_indices_for_token(plan, mode, token_index)
        if CANONICAL_QUANTA[index][0] == pair_index
    )
    return tuple(
        request * CHUNKS_PER_REQUEST + chunk
        for chunk in chunks
        for request in range(requests)
    )


def _set_rank_environment() -> None:
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]


def _load_plan() -> tuple[EpochProfile, EpochPlan, dict[str, Any], str]:
    raw_path = os.environ.get("TEMPO_EPOCH_PLAN")
    if not raw_path:
        raise SystemExit("TEMPO_EPOCH_PLAN must name a compiled artifact")
    repo_root = Path(__file__).resolve().parents[2]
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve()
    if resolved != repo_root and repo_root not in resolved.parents:
        raise SystemExit("TEMPO_EPOCH_PLAN must resolve inside the repository")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("artifact must contain an object")
        profile, plan = load_epoch_artifact(payload)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid TEMPO_EPOCH_PLAN: {exc}") from exc
    if not plan.feasible or profile.total_quanta != len(CANONICAL_QUANTA):
        raise SystemExit("TEMPO_EPOCH_PLAN is infeasible or has wrong geometry")
    if profile.max_width > PAIR_COUNT:
        raise SystemExit("TEMPO_EPOCH_PLAN width exceeds independent rank pairs")
    for assignments in plan.quantum_indices_by_token:
        pairs = [CANONICAL_QUANTA[index][0] for index in assignments]
        if len(pairs) != len(set(pairs)):
            raise SystemExit("TEMPO_EPOCH_PLAN assigns one pair twice in a token")
    return profile, plan, payload, str(resolved.relative_to(repo_root))


def _resolve_output_dir(raw_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = raw_path if raw_path.is_absolute() else repo_root / raw_path
    resolved = candidate.resolve()
    if resolved == repo_root or repo_root not in resolved.parents:
        raise SystemExit("output-dir must resolve below the repository root")
    return resolved


def _validate_topology(dist: Any, rank: int, local_rank: int) -> list[str]:
    layouts: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(layouts, (socket.gethostname(), local_rank))
    hosts = [str(item[0]) for item in layouts]
    local_ranks = [int(item[1]) for item in layouts]
    valid = (
        len(set(hosts)) == NODES
        and len(set(hosts[:RANKS_PER_NODE])) == 1
        and len(set(hosts[RANKS_PER_NODE:])) == 1
        and hosts[0] != hosts[RANKS_PER_NODE]
        and local_ranks[:RANKS_PER_NODE] == list(range(RANKS_PER_NODE))
        and local_ranks[RANKS_PER_NODE:] == list(range(RANKS_PER_NODE))
        and rank == (rank // RANKS_PER_NODE) * RANKS_PER_NODE + local_rank
    )
    if not valid:
        raise RuntimeError("requires node-major ranks 0..3 and 4..7")
    return hosts


def _make_chunk_memory(
    torch: Any,
    TensorMemoryObj: Any,
    MemoryObjMetadata: Any,
    MemoryFormat: Any,
    *,
    requests: int,
    chunk_bytes: int,
) -> tuple[Any, Any, list[Any], dict[int, int]]:
    object_count = requests * CHUNKS_PER_REQUEST
    total_bytes = object_count * chunk_bytes
    backing = torch.empty(total_bytes + chunk_bytes - 1, dtype=torch.uint8, device="cuda")
    offset = (-backing.data_ptr()) % chunk_bytes
    buffer = backing[offset:offset + total_bytes]
    objects: list[Any] = []
    index_by_address: dict[int, int] = {}
    for index in range(object_count):
        raw = buffer[index * chunk_bytes:(index + 1) * chunk_bytes]
        shape = torch.Size([chunk_bytes])
        objects.append(
            TensorMemoryObj(
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
        )
        index_by_address[raw.data_ptr()] = index
    return backing, buffer, objects, index_by_address


def _install_descriptor_index_shim(channel: Any, index_by_address: dict[int, int]) -> None:
    def descriptor_indices(_channel: Any, batch: list[Any]) -> np.ndarray:
        try:
            indices = [index_by_address[item.meta.address] for item in batch]
        except KeyError as exc:
            raise RuntimeError("LMCache object is outside the prepared descriptor list") from exc
        return np.asarray(indices, dtype=np.uint64)

    channel.get_local_mem_indices = MethodType(descriptor_indices, channel)


def _expected_byte(block: int, request: int, chunk: int, pair: int) -> int:
    return 1 + ((block * 37 + request * 11 + chunk * 5 + pair * 3) % 251)


def _warm_subset_transfer(
    torch: Any,
    dist: Any,
    *,
    channel: Any,
    objects: list[Any],
    rank: int,
    pair_index: int,
    device_index: int,
) -> dict[str, Any]:
    selected = (0, len(objects) - 1)
    expected = (211 + pair_index, 71 + pair_index)
    is_source = rank < RANKS_PER_NODE
    if is_source:
        for index, value in zip(selected, expected, strict=True):
            objects[index].raw_data.fill_(value)
    else:
        for index in selected:
            objects[index].raw_data.zero_()
    torch.cuda.synchronize()
    dist.barrier()
    local_ok = True
    error = None
    if is_source:
        try:
            count = int(
                channel.batched_write(
                    objects=[objects[index] for index in selected],
                    transfer_spec={
                        "receiver_id": f"rank-{rank + RANKS_PER_NODE}",
                        "remote_indexes": np.asarray(selected, dtype=np.uint64),
                    },
                )
            )
            local_ok = count == len(selected)
        except BaseException as exc:
            local_ok = False
            error = f"{type(exc).__name__}: {exc}"
    dist.barrier()
    if not is_source:
        local_ok = all(
            bool(torch.all(objects[index].raw_data == value).item())
            for index, value in zip(selected, expected, strict=True)
        )
    status = torch.tensor([1 if local_ok else 0], dtype=torch.int32, device=device_index)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if int(status.item()) != 1:
        raise RuntimeError(f"LMCache/NIXL noncontiguous subset warmup failed: {error}")
    dist.barrier()
    return {
        "object_indices": list(selected),
        "noncontiguous": True,
        "verified": True,
    }


def _run_block(
    torch: Any,
    dist: Any,
    *,
    plan: EpochPlan,
    channel: Any,
    objects: list[Any],
    rank: int,
    device_index: int,
    pair_index: int,
    block_index: int,
    mode: str,
    decoder: list[dict[str, Any]],
    requests: int,
    tokens: int,
    hidden: int,
    chunk_bytes: int,
) -> dict[str, Any]:
    is_source = rank < RANKS_PER_NODE
    is_receiver = not is_source
    for request in range(requests):
        for chunk in range(CHUNKS_PER_REQUEST):
            index = request * CHUNKS_PER_REQUEST + chunk
            if is_source:
                objects[index].raw_data.fill_(
                    _expected_byte(block_index, request, chunk, pair_index)
                )
            else:
                objects[index].raw_data.zero_()
    torch.cuda.synchronize()

    batches: queue.SimpleQueue[Any] = queue.SimpleQueue()
    sentinel = object()
    transfer_records: list[dict[str, Any]] = []
    pending_lock = threading.Lock()
    pending_batches = 0
    peak_pending_batches = 0

    def transfer_worker() -> None:
        nonlocal pending_batches
        torch.cuda.set_device(device_index)
        while True:
            item = batches.get()
            if item is sentinel:
                return
            token_index, indices, enqueue_ns = item
            started_ns = time.perf_counter_ns()
            error = None
            completed = 0
            try:
                completed = int(
                    channel.batched_write(
                        objects=[objects[index] for index in indices],
                        transfer_spec={
                            "receiver_id": f"rank-{rank + RANKS_PER_NODE}",
                            "remote_indexes": np.asarray(indices, dtype=np.uint64),
                        },
                    )
                )
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
            finished_ns = time.perf_counter_ns()
            with pending_lock:
                pending_batches -= 1
            transfer_records.append(
                {
                    "scheduled_token": token_index,
                    "object_indices": list(indices),
                    "completed_objects": completed,
                    "enqueue_ns": enqueue_ns,
                    "started_ns": started_ns,
                    "finished_ns": finished_ns,
                    "start_lag_ms": (started_ns - enqueue_ns) / 1_000_000.0,
                    "elapsed_ms": (finished_ns - started_ns) / 1_000_000.0,
                    "error": error,
                }
            )

    worker = None
    if is_source and mode != "fg_only":
        worker = threading.Thread(target=transfer_worker, name=f"lmcache-epoch-{rank}")
        worker.start()

    x = torch.linspace(-0.5, 0.5, hidden, dtype=torch.float16, device="cuda").view(1, hidden)
    token_latency_ms: list[float] = []
    decoder_latency_ms: list[float] = []
    token_end_ns: list[int] = []
    dist.barrier()
    block_start_ns = time.perf_counter_ns()
    first_enqueue_ns: int | None = None
    for token_index in range(tokens):
        token_started_ns = time.perf_counter_ns()
        selected = object_indices_for_rank(
            plan,
            mode,
            token_index,
            pair_index=pair_index,
            requests=requests,
        )
        if is_source and selected:
            enqueue_ns = time.perf_counter_ns()
            if first_enqueue_ns is None:
                first_enqueue_ns = enqueue_ns
            with pending_lock:
                pending_batches += 1
                peak_pending_batches = max(peak_pending_batches, pending_batches)
            batches.put((token_index, selected, enqueue_ns))
        decoder_started_ns = time.perf_counter_ns()
        x = foreground._decoder_token(torch, dist, x, decoder)
        torch.cuda.synchronize()
        token_finished_ns = time.perf_counter_ns()
        decoder_latency_ms.append((token_finished_ns - decoder_started_ns) / 1_000_000.0)
        token_latency_ms.append((token_finished_ns - token_started_ns) / 1_000_000.0)
        token_end_ns.append(token_finished_ns)

    foreground_end_ns = time.perf_counter_ns()
    if worker is not None:
        batches.put(sentinel)
        worker.join()
    background_end_ns = time.perf_counter_ns()
    for record in transfer_records:
        scheduled_token = int(record["scheduled_token"])
        record["started_within_scheduled_token"] = (
            int(record["started_ns"]) <= token_end_ns[scheduled_token]
        )
        record["finished_by_plan_deadline"] = (
            int(record["finished_ns"])
            <= token_end_ns[int(plan.completion_token_exclusive) - 1]
        )
    dist.barrier()

    verified_objects = 0
    if is_receiver and mode != "fg_only":
        for request in range(requests):
            for chunk in range(CHUNKS_PER_REQUEST):
                index = request * CHUNKS_PER_REQUEST + chunk
                expected = _expected_byte(block_index, request, chunk, pair_index)
                if bool(torch.all(objects[index].raw_data == expected).item()):
                    verified_objects += 1
    dist.barrier()

    expected_indices = tuple(range(requests * CHUNKS_PER_REQUEST)) if mode != "fg_only" else ()
    actual_indices = tuple(
        index for record in transfer_records for index in record["object_indices"]
    )
    completed_objects = sum(int(record["completed_objects"]) for record in transfer_records)
    errors = [record["error"] for record in transfer_records if record["error"] is not None]
    expected_objects = len(expected_indices)
    last_finished_ns = max(
        (int(record["finished_ns"]) for record in transfer_records),
        default=block_start_ns,
    )
    expected_batch_calls = sum(
        1
        for token in range(tokens)
        if object_indices_for_rank(
            plan, mode, token, pair_index=pair_index, requests=requests
        )
    ) if is_source else 0
    schedule_adherence = all(
        bool(record["started_within_scheduled_token"]) for record in transfer_records
    )
    deadline_met = all(
        bool(record["finished_by_plan_deadline"]) for record in transfer_records
    )
    finite = bool(torch.isfinite(x).all().item())
    local_correct = finite and not errors
    if is_source:
        local_correct = (
            local_correct
            and completed_objects == expected_objects
            and sorted(actual_indices) == list(expected_indices)
            and len(transfer_records) == expected_batch_calls
        )
    else:
        local_correct = local_correct and verified_objects == expected_objects
    return {
        "block_index": block_index,
        "mode": mode,
        "token_latency_ms": token_latency_ms,
        "decoder_latency_ms": decoder_latency_ms,
        "first_token_step_ms": token_latency_ms[0],
        "foreground_checksum": float(x.float().sum().item()),
        "foreground_finite": finite,
        "background_batch_calls": len(transfer_records),
        "expected_background_batch_calls": expected_batch_calls,
        "background_completed_bytes": completed_objects * chunk_bytes,
        "receiver_verified_bytes": verified_objects * chunk_bytes,
        "expected_source_bytes": expected_objects * chunk_bytes if is_source else 0,
        "expected_receive_bytes": expected_objects * chunk_bytes if is_receiver else 0,
        "first_enqueue_from_block_start_ms": (
            0.0 if first_enqueue_ns is None else (first_enqueue_ns - block_start_ns) / 1_000_000.0
        ),
        "background_finish_from_block_start_ms": (
            0.0 if first_enqueue_ns is None else (last_finished_ns - block_start_ns) / 1_000_000.0
        ),
        "background_completion_from_first_enqueue_ms": (
            0.0 if first_enqueue_ns is None else (last_finished_ns - first_enqueue_ns) / 1_000_000.0
        ),
        "post_foreground_drain_ms": max(
            0.0, (last_finished_ns - foreground_end_ns) / 1_000_000.0
        ) if first_enqueue_ns is not None else 0.0,
        "peak_pending_batches": peak_pending_batches,
        "max_descriptor_start_lag_ms": max(
            (float(record["start_lag_ms"]) for record in transfer_records),
            default=0.0,
        ),
        "schedule_start_adherence_met": schedule_adherence,
        "plan_deadline_met": deadline_met,
        "block_elapsed_ms": (background_end_ns - block_start_ns) / 1_000_000.0,
        "transfer_errors": errors,
        "transfer_records": transfer_records,
        "correctness_met": local_correct,
        "execution": "local_descriptor_epoch_queue_no_hot_path_global_control",
    }


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != WORLD_SIZE or sorted(item["rank"] for item in records) != list(range(WORLD_SIZE)):
        raise ValueError("records must contain exact ranks 0..7")
    ordered = sorted(records, key=lambda item: item["rank"])
    config = ordered[0]["config"]
    if any(item["config"] != config for item in ordered):
        raise ValueError("rank configs differ")
    for item in ordered:
        if tuple(block["mode"] for block in item["blocks"]) != BLOCK_MODES:
            raise ValueError("rank block sequences differ")
        if tuple(int(block["block_index"]) for block in item["blocks"]) != tuple(range(len(BLOCK_MODES))):
            raise ValueError("rank block indices differ")

    mode_samples: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "token_tail_ms": [],
            "decoder_tail_ms": [],
            "finish_ms": [],
            "drain_ms": [],
            "start_lag_ms": [],
            "completed_bytes": 0,
            "verified_bytes": 0,
            "correctness_met": True,
            "schedule_start_adherence_met": True,
            "plan_deadline_met": True,
            "replicates": 0,
        }
    )
    blocks: list[dict[str, Any]] = []
    local_expected = int(config["requests"]) * int(config["kv_bytes"])
    for block_index, mode in enumerate(BLOCK_MODES):
        rank_blocks = [item["blocks"][block_index] for item in ordered]
        token_tail = [
            max(float(block["token_latency_ms"][token]) for block in rank_blocks)
            for token in range(int(config["tokens"]))
        ]
        decoder_tail = [
            max(float(block["decoder_latency_ms"][token]) for block in rank_blocks)
            for token in range(int(config["tokens"]))
        ]
        completed = sum(int(block["background_completed_bytes"]) for block in rank_blocks)
        verified = sum(int(block["receiver_verified_bytes"]) for block in rank_blocks)
        expected = 0 if mode == "fg_only" else PAIR_COUNT * local_expected
        source_blocks = rank_blocks[:RANKS_PER_NODE]
        receiver_blocks = rank_blocks[RANKS_PER_NODE:]
        distribution_ok = (
            all(int(block["background_completed_bytes"]) == (local_expected if mode != "fg_only" else 0) for block in source_blocks)
            and all(int(block["receiver_verified_bytes"]) == (local_expected if mode != "fg_only" else 0) for block in receiver_blocks)
            and all(int(block["receiver_verified_bytes"]) == 0 for block in source_blocks)
            and all(int(block["background_completed_bytes"]) == 0 for block in receiver_blocks)
        )
        finish_ms = max(float(block["background_finish_from_block_start_ms"]) for block in source_blocks)
        drain_ms = max(float(block["post_foreground_drain_ms"]) for block in source_blocks)
        start_lag_ms = max(float(block["max_descriptor_start_lag_ms"]) for block in source_blocks)
        errors = [error for block in rank_blocks for error in block["transfer_errors"]]
        adherence = all(bool(block["schedule_start_adherence_met"]) for block in source_blocks)
        deadline_met = all(bool(block["plan_deadline_met"]) for block in source_blocks)
        correct = (
            all(block["correctness_met"] is True for block in rank_blocks)
            and not errors
            and distribution_ok
            and completed == expected
            and verified == expected
        )
        result = {
            "block_index": block_index,
            "mode": mode,
            "global_first_token_step_ms": token_tail[0],
            "global_token_tail_p50_ms": statistics.median(token_tail),
            "global_token_tail_p99_ms": percentile(token_tail, 0.99),
            "global_token_tail_max_ms": max(token_tail),
            "global_decoder_tail_p99_ms": percentile(decoder_tail, 0.99),
            "background_finish_from_block_start_ms": finish_ms,
            "post_foreground_drain_ms": drain_ms,
            "max_descriptor_start_lag_ms": start_lag_ms,
            "schedule_start_adherence_met": adherence,
            "plan_deadline_met": deadline_met,
            "expected_background_bytes": expected,
            "background_completed_bytes": completed,
            "receiver_verified_bytes": verified,
            "background_batch_calls": sum(int(block["background_batch_calls"]) for block in source_blocks),
            "effective_background_gbps": 0.0 if finish_ms <= 0 else expected / (finish_ms / 1000.0) / 1e9,
            "transfer_errors": errors,
            "correctness_met": correct,
        }
        blocks.append(result)
        bucket = mode_samples[mode]
        bucket["token_tail_ms"].extend(token_tail)
        bucket["decoder_tail_ms"].extend(decoder_tail)
        bucket["finish_ms"].append(finish_ms)
        bucket["drain_ms"].append(drain_ms)
        bucket["start_lag_ms"].append(start_lag_ms)
        bucket["completed_bytes"] += completed
        bucket["verified_bytes"] += verified
        bucket["correctness_met"] = bool(bucket["correctness_met"]) and correct
        bucket["schedule_start_adherence_met"] = bool(bucket["schedule_start_adherence_met"]) and adherence
        bucket["plan_deadline_met"] = bool(bucket["plan_deadline_met"]) and deadline_met
        bucket["replicates"] += 1

    modes: dict[str, dict[str, Any]] = {}
    for mode in MODE_ORDER:
        bucket = mode_samples[mode]
        tails = bucket.pop("token_tail_ms")
        decoder_tails = bucket.pop("decoder_tail_ms")
        finishes = bucket.pop("finish_ms")
        drains = bucket.pop("drain_ms")
        start_lags = bucket.pop("start_lag_ms")
        modes[mode] = {
            **bucket,
            "global_token_tail_p50_ms": statistics.median(tails),
            "global_token_tail_p99_ms": percentile(tails, 0.99),
            "global_decoder_tail_p99_ms": percentile(decoder_tails, 0.99),
            "background_finish_p50_ms": statistics.median(finishes),
            "background_finish_p99_ms": percentile(finishes, 0.99),
            "post_foreground_drain_p50_ms": statistics.median(drains),
            "post_foreground_drain_p99_ms": percentile(drains, 0.99),
            "descriptor_start_lag_p99_ms": percentile(start_lags, 0.99),
        }
    overall_correct = all(block["correctness_met"] for block in blocks)
    tempo_executed = (
        bool(modes["tempo_epoch"]["schedule_start_adherence_met"])
        and bool(modes["tempo_epoch"]["plan_deadline_met"])
    )
    if not overall_correct:
        screen_outcome = "invalid_correctness"
    elif not tempo_executed:
        screen_outcome = "kill_descriptor_calendar_service_mismatch"
    else:
        screen_outcome = "valid_measurement_requires_performance_comparison"
    return {
        "schema_version": "tempo-lmcache-epoch-2node-2",
        "evidence_state": "live_official_component_with_compatibility_shim",
        "claim_scope": "research_scheduler_screen_not_sota_promotion",
        "world_size": WORLD_SIZE,
        "nodes": NODES,
        "block_sequence": list(BLOCK_MODES),
        "config": config,
        "baseline": {
            "name": "LMCache NixlChannel",
            "commit": official.LMCACHE_COMMIT,
            "component": "lazy_init_peer_connection + batched_write",
            "proxy": False,
            "descriptor_compatibility_shim": "physical-address objects mapped to prepared-descriptor uint64 indices",
        },
        "scheduler_semantics": {
            "name": "TEMPO epoch descriptor calendar",
            "hot_path_global_control": False,
            "adaptive": False,
            "actual_start_admission_claim_requires_schedule_start_adherence": True,
        },
        "blocks": blocks,
        "modes": modes,
        "tempo_epoch_execution_valid": tempo_executed,
        "screen_outcome": screen_outcome,
        "overall_correctness_met": overall_correct,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--kv-mib", type=int, default=32)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--port-base", type=int, default=30100)
    args = parser.parse_args()
    numeric = (
        args.requests,
        args.kv_mib,
        args.chunk_mib,
        args.tokens,
        args.layers,
        args.hidden_size,
        args.context,
    )
    if any(isinstance(value, bool) or value <= 0 for value in numeric):
        parser.error("all workload values must be positive")
    if args.tokens < len(CANONICAL_QUANTA):
        parser.error("tokens must be at least 16")
    if args.kv_mib != CHUNKS_PER_REQUEST * args.chunk_mib:
        parser.error("kv-mib must equal four chunk-mib chunks")
    if args.hidden_size % 8:
        parser.error("hidden-size must be divisible by 8")
    if not 1024 <= args.port_base <= 65535 - PAIR_COUNT:
        parser.error("port-base must leave four valid ports")
    return args


def main() -> None:
    args = _parse_args()
    args.output_dir = _resolve_output_dir(args.output_dir)
    profile, plan, epoch_artifact, epoch_path = _load_plan()
    if plan.completion_token_exclusive is None or plan.completion_token_exclusive > args.tokens:
        raise SystemExit("epoch plan completion exceeds runtime token horizon")
    _set_rank_environment()
    try:
        import torch
        import torch.distributed as dist
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch with CUDA/NCCL is required") from exc
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise SystemExit("CUDA and NCCL are required")
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise SystemExit("WORLD_SIZE must be exactly 8")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device_index = 0 if torch.cuda.device_count() == 1 else local_rank
    torch.cuda.set_device(device_index)
    dist.init_process_group("nccl")
    try:
        hosts = _validate_topology(dist, rank, local_rank)
        NixlChannel, TensorMemoryObj, MemoryObjMetadata, MemoryFormat = (
            official._load_official_lmcache(Path(__file__).resolve().parents[2])
        )
        chunk_bytes = args.chunk_mib * MIB
        kv_bytes = args.kv_mib * MIB
        backing, buffer, objects, index_by_address = _make_chunk_memory(
            torch,
            TensorMemoryObj,
            MemoryObjMetadata,
            MemoryFormat,
            requests=args.requests,
            chunk_bytes=chunk_bytes,
        )
        pair_index = rank % RANKS_PER_NODE
        is_source = rank < RANKS_PER_NODE
        channel = NixlChannel(
            async_mode=False,
            role="sender" if is_source else "receiver",
            buffer_ptr=buffer.data_ptr(),
            buffer_size=buffer.numel(),
            align_bytes=chunk_bytes,
            tp_rank=local_rank,
            peer_init_url=None if is_source else f"*:{args.port_base + pair_index}",
            backends=["UCX"],
            device=f"cuda:{device_index}",
        )
        _install_descriptor_index_shim(channel, index_by_address)
        peer_rank = rank + RANKS_PER_NODE if is_source else rank - RANKS_PER_NODE
        dist.barrier()
        if is_source:
            channel.lazy_init_peer_connection(
                local_id=f"rank-{rank}",
                peer_id=f"rank-{peer_rank}",
                peer_init_url=f"{hosts[peer_rank]}:{args.port_base + pair_index}",
            )
        dist.barrier()
        if not channel.remote_xfer_handler_exists(f"rank-{peer_rank}"):
            raise RuntimeError("LMCache/NIXL peer handshake did not install a handler")

        warmup = _warm_subset_transfer(
            torch,
            dist,
            channel=channel,
            objects=objects,
            rank=rank,
            pair_index=pair_index,
            device_index=device_index,
        )
        decoder = foreground._make_decoder(
            torch,
            rank=rank,
            layers=args.layers,
            hidden=args.hidden_size,
            context=args.context,
        )
        warm = torch.zeros(1, args.hidden_size, dtype=torch.float16, device="cuda")
        warm = foreground._decoder_token(torch, dist, warm, decoder)
        del warm
        torch.cuda.synchronize()
        dist.barrier()

        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        blocks = [
            _run_block(
                torch,
                dist,
                plan=plan,
                channel=channel,
                objects=objects,
                rank=rank,
                device_index=device_index,
                pair_index=pair_index,
                block_index=block_index,
                mode=mode,
                decoder=decoder,
                requests=args.requests,
                tokens=args.tokens,
                hidden=args.hidden_size,
                chunk_bytes=chunk_bytes,
            )
            for block_index, mode in enumerate(BLOCK_MODES)
        ]
        config = {
            "requests": args.requests,
            "kv_bytes": kv_bytes,
            "chunk_bytes": chunk_bytes,
            "chunks_per_request": CHUNKS_PER_REQUEST,
            "tokens": args.tokens,
            "layers": args.layers,
            "hidden_size": args.hidden_size,
            "context": args.context,
            "port_base": args.port_base,
            "replicates_per_mode": len(MODE_ORDER),
            "epoch_plan_path": epoch_path,
            "epoch_plan_signature": plan.signature,
            "epoch_width_by_token": list(plan.width_by_token),
            "epoch_completion_token_exclusive": plan.completion_token_exclusive,
            "hot_path_global_control": False,
            "lmcache_commit": official.LMCACHE_COMMIT,
            "nixl_version": importlib.metadata.version("nixl"),
            "nixl_backend": "UCX",
            "nixl_transport_claim": "not_independently_attributed",
            "foreground_nccl_net": os.environ.get("NCCL_NET", "module_default"),
            "foreground_nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
            "ucx_tls": os.environ.get("UCX_TLS"),
            "epoch_artifact": epoch_artifact,
            "unmeasured_nixl_warmup": warmup,
        }
        rank_record = {
            "schema_version": "tempo-lmcache-epoch-rank-2",
            "rank": rank,
            "local_rank": local_rank,
            "device_index": device_index,
            "hostname": hosts[rank],
            "world_size": WORLD_SIZE,
            "nodes": NODES,
            "config": config,
            "blocks": blocks,
        }
        gathered = [None] * WORLD_SIZE if rank == 0 else None
        dist.gather_object(rank_record, gathered, dst=0)
        status: list[Any] = [None]
        if rank == 0:
            try:
                assert gathered is not None
                for item in gathered:
                    rank_path = args.output_dir / f"rank_{int(item['rank'])}.json"
                    rank_path.write_text(
                        json.dumps(item, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                result = aggregate_rank_records(gathered)
                (args.output_dir / "result.json").write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                status[0] = {
                    "ok": bool(result["overall_correctness_met"]),
                    "screen_outcome": result["screen_outcome"],
                }
            except BaseException as exc:
                status[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(status, src=0)
        dist.barrier()
        if not isinstance(status[0], dict) or status[0].get("ok") is not True:
            raise RuntimeError(f"LMCache epoch run failed: {status[0]}")
        if rank == 0:
            print(json.dumps({"output": str(args.output_dir / "result.json"), **status[0]}, sort_keys=True))
        del backing
        # Do not call receiver channel.close(): this pinned LMCache commit can
        # join its blocking listener before terminating the ZMQ context.
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
