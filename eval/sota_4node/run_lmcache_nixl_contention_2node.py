#!/usr/bin/env python3
"""Run the official LMCache NixlChannel beside WORLD NCCL collectives.

Ranks 0..3 issue one-sided NIXL/UCX writes using either matched 1:1 receivers
or a 4:1 receiver-incast pattern while every rank runs the same NCCL
all-reduce token loop.  This imports the checked-out LMCache implementation
directly; it is not an algorithm proxy.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import threading
import time
from typing import Any, Iterable

import numpy as np

from eval.sota_4node.cojob_phase_gate import wait_for_start_file
from tempo.cross_layer_observer import (
    NCCLObserverSnapshot,
    publish_observer_snapshot,
)


WORLD_SIZE = 8
NODES = 2
RANKS_PER_NODE = 4
PAIR_COUNT = 4
TRAFFIC_PAIRED = "paired_1to1"
TRAFFIC_INCAST = "incast_4to1"
MIB = 1024 * 1024
LMCACHE_COMMIT = "227d13f5c9fdb52ddb933641d34331f678de03a0"


def percentile(values: Iterable[float], fraction: float) -> float:
    """Return a nearest-rank percentile."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one sample")
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate eight bounded rank records into the experiment result."""

    if len(records) != WORLD_SIZE or sorted(item["rank"] for item in records) != list(
        range(WORLD_SIZE)
    ):
        raise ValueError("records must contain ranks 0..7 exactly once")
    records = sorted(records, key=lambda item: item["rank"])
    config = records[0]["config"]
    if any(item["config"] != config for item in records):
        raise ValueError("rank configs differ")

    expected_bytes = (
        0
        if config.get("background_mode") == "nccl_only"
        else PAIR_COUNT * config["requests"] * config["kv_bytes"]
    )
    blocks: list[dict[str, Any]] = []
    pooled_tail: list[float] = []
    completion_samples: list[float] = []
    for block_index in range(config["blocks"]):
        rank_blocks = [item["blocks"][block_index] for item in records]
        tails = [
            max(float(block["token_latency_ms"][token]) for block in rank_blocks)
            for token in range(config["token_iters"])
        ]
        completed = sum(int(block["source_completed_bytes"]) for block in rank_blocks)
        verified = sum(int(block["receiver_verified_bytes"]) for block in rank_blocks)
        completion = max(float(block["transfer_elapsed_ms"]) for block in rank_blocks)
        drain = max(float(block["post_foreground_drain_ms"]) for block in rank_blocks)
        correct = (
            completed == expected_bytes
            and verified == expected_bytes
            and all(block["foreground_correct"] for block in rank_blocks)
            and all(block["transfer_error"] is None for block in rank_blocks)
        )
        blocks.append(
            {
                "block_index": block_index,
                "global_token_tail_p50_ms": statistics.median(tails),
                "global_token_tail_p99_ms": percentile(tails, 0.99),
                "global_token_tail_max_ms": max(tails),
                "background_completion_ms": completion,
                "post_foreground_drain_ms": drain,
                "expected_background_bytes": expected_bytes,
                "source_completed_bytes": completed,
                "receiver_verified_bytes": verified,
                "full_bytes_completed": completed == expected_bytes,
                "full_bytes_verified": verified == expected_bytes,
                "correctness_met": correct,
            }
        )
        pooled_tail.extend(tails)
        completion_samples.append(completion)

    active_elapsed = [
        float(item["active_loop_elapsed_ms"])
        for item in records
        if item.get("active_loop_elapsed_ms") is not None
    ]
    minimum_requested_ms = (
        float(config.get("minimum_active_duration_s", 0.0)) * 1000.0
    )
    active_loop = None
    if active_elapsed:
        if len(active_elapsed) != WORLD_SIZE:
            raise ValueError("active-loop receipts must be present for every rank")
        active_loop = {
            "rank_min_elapsed_ms": min(active_elapsed),
            "rank_max_elapsed_ms": max(active_elapsed),
            "minimum_requested_ms": minimum_requested_ms,
            "horizon_met": min(active_elapsed) >= minimum_requested_ms,
        }

    return {
        "schema_version": "tempo-lmcache-nixl-contention-2node-1",
        "evidence_state": "live_official_component",
        "baseline": {
            "name": "LMCache NixlChannel",
            "commit": LMCACHE_COMMIT,
            "component": "NixlChannel.lazy_init_peer_connection + batched_write",
            "backend": "NIXL UCX",
            "proxy": False,
            "cleanup_note": (
                "receiver close() is not called because this LMCache commit joins "
                "its blocking ZMQ listener before terminating the context; process "
                "exit performs cleanup"
            ),
        },
        "world_size": WORLD_SIZE,
        "nodes": NODES,
        "pair_count": PAIR_COUNT,
        "pairing": (
            [[rank, rank + RANKS_PER_NODE] for rank in range(PAIR_COUNT)]
            if config.get("traffic_pattern", TRAFFIC_PAIRED) == TRAFFIC_PAIRED
            else [[rank, RANKS_PER_NODE] for rank in range(PAIR_COUNT)]
        ),
        "config": config,
        "blocks": blocks,
        "summary": {
            "global_token_tail_p50_ms": statistics.median(pooled_tail),
            "global_token_tail_p99_ms": percentile(pooled_tail, 0.99),
            "background_completion_p50_ms": statistics.median(completion_samples),
            "background_completion_p99_ms": percentile(completion_samples, 0.99),
        },
        "active_loop": active_loop,
        "rank_diagnostics": [
            {
                "rank": item["rank"],
                "hostname": item.get("hostname"),
                "is_source": item["rank"] < RANKS_PER_NODE,
                "is_receiver": (
                    item["rank"] >= RANKS_PER_NODE
                    if config.get("traffic_pattern", TRAFFIC_PAIRED)
                    == TRAFFIC_PAIRED
                    else item["rank"] == RANKS_PER_NODE
                ),
                "blocks": [
                    {
                        "block_index": block.get("block_index", index),
                        "attempted_objects": block.get("transfer_attempted_objects"),
                        "returned_objects": block.get("transfer_returned_objects"),
                        "started": block.get("transfer_started"),
                        "finished": block.get("transfer_finished"),
                        "worker_alive_after_join": block.get("worker_alive_after_join"),
                        "elapsed_ms": block.get("transfer_elapsed_ms", 0.0),
                        "error": block.get("transfer_error"),
                        "debug": block.get("transfer_debug"),
                    }
                    for index, block in enumerate(item["blocks"])
                ],
            }
            for item in records
        ],
        "overall_correctness_met": all(block["correctness_met"] for block in blocks),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--kv-mib", type=int, default=32)
    parser.add_argument("--token-iters", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument(
        "--minimum-active-duration-s",
        type=float,
        default=0.0,
        help=(
            "keep issuing back-to-back measured blocks until this rank-0 "
            "active-loop horizon is reached; zero preserves the fixed-block "
            "legacy contract"
        ),
    )
    parser.add_argument(
        "--maximum-blocks",
        type=int,
        default=None,
        help=(
            "hard bound for a minimum-duration run; defaults to --blocks and "
            "must be explicitly raised when a positive duration is requested"
        ),
    )
    parser.add_argument("--foreground-mib", type=int, default=4)
    parser.add_argument(
        "--block-delay-s",
        type=float,
        default=0.0,
        help="bounded idle interval after each completed contention block",
    )
    parser.add_argument("--port-base", type=int, default=29940)
    parser.add_argument(
        "--no-background-transfer",
        action="store_true",
        help="run the matched NCCL-only control without LMCache/NIXL traffic",
    )
    parser.add_argument(
        "--traffic-pattern",
        choices=(TRAFFIC_PAIRED, TRAFFIC_INCAST),
        default=TRAFFIC_PAIRED,
        help=(
            "use four independent 1:1 P/D transfers or four source GPUs "
            "writing disjoint descriptors into one receiver GPU"
        ),
    )
    parser.add_argument(
        "--observer-output",
        type=Path,
        default=None,
        help=(
            "atomically publish tempo-nccl-observer-v1 after each completed "
            "block for an allocation-scoped TEMPO router"
        ),
    )
    parser.add_argument(
        "--observer-history-dir",
        type=Path,
        default=None,
        help=(
            "also write immutable per-sequence observer snapshots below "
            "this directory; the live pointer remains observer-output"
        ),
    )
    parser.add_argument(
        "--observer-history-stride",
        type=int,
        default=16,
        help=(
            "write one immutable history snapshot every N blocks while keeping "
            "the live observer pointer updated after every block"
        ),
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help=(
            "keep producing bounded contention blocks until this shared file "
            "appears, then publish a final complete observer receipt"
        ),
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        default=None,
        help="rank 0 writes this marker after all NIXL peer handlers are ready",
    )
    parser.add_argument(
        "--start-delay-s",
        type=float,
        default=0.0,
        help="hold after readiness so a co-located inference service can start",
    )
    parser.add_argument(
        "--start-file",
        type=Path,
        default=None,
        help=(
            "after NIXL readiness, wait for this shared phase marker before "
            "issuing the first contention block"
        ),
    )
    parser.add_argument(
        "--start-file-timeout-s",
        type=float,
        default=1800.0,
        help="bounded wait for --start-file after NIXL readiness",
    )
    parser.add_argument(
        "--process-group-timeout-s",
        type=float,
        default=60.0,
        help=(
            "bounded NCCL process-group timeout; a stalled collective is an "
            "overload receipt, not a reason to hold the allocation for 10 minutes"
        ),
    )
    parser.add_argument(
        "--nixl-transfer-timeout-s",
        type=float,
        default=30.0,
        help=(
            "bounded wait for one official LMCache/NIXL write; a stuck transfer "
            "is recorded as a fabric-contention failure instead of wedging the "
            "next NCCL collective"
        ),
    )
    args = parser.parse_args()
    if min(
        args.requests,
        args.kv_mib,
        args.token_iters,
        args.blocks,
        args.foreground_mib,
    ) <= 0:
        parser.error("requests, sizes, token-iters, and blocks must be positive")
    if (
        not math.isfinite(args.process_group_timeout_s)
        or args.process_group_timeout_s < 5.0
        or args.process_group_timeout_s > 3600.0
    ):
        parser.error("process-group-timeout-s must be in [5, 3600]")
    if (
        not math.isfinite(args.nixl_transfer_timeout_s)
        or args.nixl_transfer_timeout_s < 1.0
        or args.nixl_transfer_timeout_s > 3600.0
    ):
        parser.error("nixl-transfer-timeout-s must be in [1, 3600]")
    if not math.isfinite(args.start_delay_s) or args.start_delay_s < 0.0:
        parser.error("start-delay-s must be finite and non-negative")
    if args.start_file is not None and not args.start_file.is_absolute():
        parser.error("start-file must be absolute")
    if (
        not math.isfinite(args.start_file_timeout_s)
        or not 1.0 <= args.start_file_timeout_s <= 3600.0
    ):
        parser.error("start-file-timeout-s must be in [1, 3600]")
    if not math.isfinite(args.block_delay_s) or not 0.0 <= args.block_delay_s <= 60.0:
        parser.error("block-delay-s must be finite and in [0, 60]")
    if (
        not math.isfinite(args.minimum_active_duration_s)
        or not 0.0 <= args.minimum_active_duration_s <= 3600.0
    ):
        parser.error("minimum-active-duration-s must be in [0, 3600]")
    if args.maximum_blocks is None:
        args.maximum_blocks = args.blocks
    if not args.blocks <= args.maximum_blocks <= 100000:
        parser.error("maximum-blocks must be in [blocks, 100000]")
    if args.minimum_active_duration_s > 0.0 and args.maximum_blocks == args.blocks:
        parser.error(
            "a positive minimum-active-duration-s requires maximum-blocks > blocks"
        )
    if not 1 <= args.observer_history_stride <= 10000:
        parser.error("observer-history-stride must be in [1, 10000]")
    if not 1024 <= args.port_base <= 65535 - PAIR_COUNT:
        parser.error("port-base must leave four valid TCP ports")
    return args


def _publish_observer(
    *,
    output: Path | None,
    history_dir: Path | None,
    observer: NCCLObserverSnapshot,
    history_stride: int = 1,
) -> None:
    """Publish live telemetry and a bounded-rate immutable audit copy."""

    if output is None and history_dir is None:
        return
    if history_dir is not None and (
        observer.producer_state == "complete"
        or observer.sequence == 1
        or observer.sequence % history_stride == 0
    ):
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / (
            f"observer-seq-{observer.sequence:06d}-{observer.producer_state}.json"
        )
        publish_observer_snapshot(history_path, observer)
    if output is not None:
        publish_observer_snapshot(output, observer)


def _observer_snapshot(
    *,
    rank_blocks: list[dict[str, Any]],
    config: dict[str, Any],
    hosts: list[str],
    sequence: int,
    producer_state: str,
) -> NCCLObserverSnapshot:
    """Build one rank-aggregated window without cross-host clock arithmetic."""

    tails = [
        max(float(block["token_latency_ms"][token]) for block in rank_blocks)
        for token in range(config["token_iters"])
    ]
    transfer_samples = [
        float(block["transfer_elapsed_ms"])
        for block in rank_blocks
        if float(block["transfer_elapsed_ms"]) > 0.0
    ]
    window_ms = max(
        1.0,
        sum(float(value) for value in tails),
        max(transfer_samples, default=0.0),
    )
    topology_material = "tempo-nixl-nccl-2node-v1|" + "|".join(sorted(set(hosts)))
    topology_fingerprint = hashlib.sha256(
        topology_material.encode("utf-8")
    ).hexdigest()
    return NCCLObserverSnapshot(
        source_epoch=os.environ.get(
            "TEMPO_GO_CROSS_LAYER_EPOCH",
            f"slurm-{os.environ.get('SLURM_JOB_ID', 'local')}",
        ),
        sequence=sequence,
        sampled_unix_ns=time.time_ns(),
        window_ms=window_ms,
        communicator_id=os.environ.get(
            "TEMPO_GO_NCCL_COMMUNICATOR_ID", "nixl-nccl-2node-world"),
        topology_fingerprint_sha256=topology_fingerprint,
        nccl_collective_p99_ms=percentile(tails, 0.99),
        # Arrival spread would require a clock-synchronized rank observer;
        # do not manufacture it from host perf_counter_ns values.
        nccl_arrival_spread_ms=None,
        lmcache_transfer_p99_ms=(
            percentile(transfer_samples, 0.99)
            if transfer_samples else None
        ),
        uncertainty_ms=0.0,
        rank_count=WORLD_SIZE,
        background_mode=str(config["background_mode"]),
        producer_state=producer_state,
        correctness_met=all(
            bool(block["foreground_correct"])
            and block["transfer_error"] is None
            for block in rank_blocks
        ),
    )


def _set_rank_environment() -> None:
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    if "MASTER_ADDR" not in os.environ:
        launch_address = os.environ.get("SLURM_LAUNCH_NODE_IPADDR")
        if launch_address:
            os.environ["MASTER_ADDR"] = launch_address


def _validate_topology(torch: Any, dist: Any, rank: int, local_rank: int) -> list[str]:
    hosts: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(hosts, socket.gethostname())
    if (
        len(set(hosts)) != NODES
        or len(set(hosts[:RANKS_PER_NODE])) != 1
        or len(set(hosts[RANKS_PER_NODE:])) != 1
        or hosts[0] == hosts[RANKS_PER_NODE]
        or rank != (0 if rank < RANKS_PER_NODE else 1) * RANKS_PER_NODE + local_rank
    ):
        raise RuntimeError("requires node-major ranks: node0=0..3, node1=4..7")
    return [str(host) for host in hosts]


def _cuda_index(torch: Any, local_rank: int) -> int:
    """Map the Slurm local rank to an ordinal in the task's visible GPU set."""

    visible_count = int(torch.cuda.device_count())
    if visible_count == 1:
        # ``--gpu-bind=single:1`` exposes the task's assigned GPU as ordinal 0.
        return 0
    if 0 <= local_rank < visible_count:
        return local_rank
    raise RuntimeError(
        f"local rank {local_rank} is not a visible CUDA ordinal "
        f"(device_count={visible_count}, CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})"
    )


def _load_official_lmcache(repo_root: Path) -> tuple[Any, Any, Any, Any]:
    checkout = (repo_root / "third_party" / "lmcache").resolve()
    sys.path.insert(0, str(checkout))
    try:
        from lmcache.v1.memory_management import (
            MemoryFormat,
            MemoryObjMetadata,
            TensorMemoryObj,
        )
        from lmcache.v1.transfer_channel.nixl_channel import NixlChannel
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "official third_party/lmcache plus its msgspec, pyzmq, and nixl "
            "runtime dependencies are required"
        ) from exc
    module_path = Path(sys.modules[NixlChannel.__module__].__file__).resolve()
    if checkout not in module_path.parents:
        raise RuntimeError(f"NixlChannel was not imported from {checkout}")
    return NixlChannel, TensorMemoryObj, MemoryObjMetadata, MemoryFormat


def _make_memory(
    torch: Any,
    TensorMemoryObj: Any,
    MemoryObjMetadata: Any,
    MemoryFormat: Any,
    *,
    requests: int,
    kv_bytes: int,
) -> tuple[Any, Any, list[Any]]:
    total_bytes = requests * kv_bytes
    backing = torch.empty(total_bytes + kv_bytes - 1, dtype=torch.uint8, device="cuda")
    offset = (-backing.data_ptr()) % kv_bytes
    buffer = backing[offset : offset + total_bytes]
    objects = []
    for request in range(requests):
        raw = buffer[request * kv_bytes : (request + 1) * kv_bytes]
        shape = torch.Size([kv_bytes])
        objects.append(
            TensorMemoryObj(
                raw_data=raw,
                metadata=MemoryObjMetadata(
                    shape=shape,
                    dtype=torch.uint8,
                    address=raw.data_ptr(),
                    phy_size=kv_bytes,
                    ref_count=1,
                    pin_count=0,
                    fmt=MemoryFormat.BINARY,
                    shapes=[shape],
                    dtypes=[torch.uint8],
                ),
                parent_allocator=None,
            )
        )
    return backing, buffer, objects


def _expected_byte(block_index: int, request: int, pair: int) -> int:
    return 1 + ((block_index * 37 + request * 11 + pair * 3) % 251)


def _receiver_rank(source_rank: int, traffic_pattern: str) -> int:
    if not 0 <= source_rank < RANKS_PER_NODE:
        raise ValueError("source rank must be in 0..3")
    if traffic_pattern == TRAFFIC_PAIRED:
        return source_rank + RANKS_PER_NODE
    if traffic_pattern == TRAFFIC_INCAST:
        return RANKS_PER_NODE
    raise ValueError("traffic pattern is invalid")


def _receiver_source_pairs(rank: int, traffic_pattern: str) -> tuple[int, ...]:
    if traffic_pattern == TRAFFIC_PAIRED:
        return (rank - RANKS_PER_NODE,) if rank >= RANKS_PER_NODE else ()
    if traffic_pattern == TRAFFIC_INCAST:
        return tuple(range(RANKS_PER_NODE)) if rank == RANKS_PER_NODE else ()
    raise ValueError("traffic pattern is invalid")


def _memory_object_count(rank: int, traffic_pattern: str, requests: int) -> int:
    if rank < RANKS_PER_NODE:
        return requests
    return len(_receiver_source_pairs(rank, traffic_pattern)) * requests


def _remote_descriptor_indices(
    source_rank: int, traffic_pattern: str, requests: int,
) -> list[int]:
    offset = source_rank * requests if traffic_pattern == TRAFFIC_INCAST else 0
    return list(range(offset, offset + requests))


def _run_block(
    torch: Any,
    dist: Any,
    *,
    channel: Any,
    objects: list[Any],
    remote_addresses: list[int],
    rank: int,
    local_rank: int,
    cuda_index: int,
    background_transfer: bool,
    traffic_pattern: str,
    pair: int,
    block_index: int,
    requests: int,
    kv_bytes: int,
    token_iters: int,
    foreground_elements: int,
    nixl_transfer_timeout_s: float,
) -> dict[str, Any]:
    is_source = rank < RANKS_PER_NODE
    receiver_pairs = _receiver_source_pairs(rank, traffic_pattern)
    for request, obj in enumerate(objects):
        if is_source:
            obj.raw_data.fill_(_expected_byte(block_index, request, pair))
        else:
            obj.raw_data.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    start_transfer = threading.Event()
    transfer: dict[str, Any] = {
        "attempted_objects": len(objects) if is_source and background_transfer else 0,
        "count": 0,
        "elapsed_ms": 0.0,
        "error": None,
        "debug": None,
        "started": False,
        "finished": False,
    }

    def source_write() -> None:
        torch.cuda.set_device(cuda_index)
        start_transfer.wait()
        transfer["started"] = True
        started = time.perf_counter_ns()
        try:
            local_indices = channel.get_local_mem_indices(objects)
            remote_indices = np.asarray(remote_addresses, dtype=np.uint64)
            receiver_rank = _receiver_rank(rank, traffic_pattern)
            remote_handler = channel.remote_xfer_handlers_dict.get(
                f"rank-{receiver_rank}"
            )
            transfer["debug"] = {
                "local_indices_type": type(local_indices).__name__,
                "local_index_type": type(local_indices[0]).__name__,
                "remote_indices_type": type(remote_indices).__name__,
                "remote_indices_dtype": str(remote_indices.dtype),
                "remote_index_type": type(remote_indices[0]).__name__,
                "local_handler_type": type(channel.nixl_wrapper.xfer_handler).__name__,
                "remote_handler_type": type(remote_handler).__name__,
            }
            transfer["count"] = channel.batched_write(
                objects=objects,
                transfer_spec={
                    "receiver_id": f"rank-{receiver_rank}",
                    "remote_indexes": remote_indices,
                },
            )
        except BaseException as exc:  # preserve worker failure for rank JSON
            transfer["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            transfer["elapsed_ms"] = (time.perf_counter_ns() - started) / 1_000_000.0
            transfer["finished"] = True

    worker = None
    if is_source and background_transfer:
        worker = threading.Thread(
            target=source_write,
            name=f"nixl-source-{rank}",
            daemon=True,
        )
        worker.start()

    foreground = torch.empty(foreground_elements, dtype=torch.float32, device="cuda")
    token_latency_ms: list[float] = []
    foreground_correct = True
    dist.barrier()
    if is_source and background_transfer:
        start_transfer.set()
    for token in range(token_iters):
        expected = float(sum(range(1, WORLD_SIZE + 1)) + WORLD_SIZE * (block_index + token))
        foreground.fill_(rank + 1 + block_index + token)
        started = time.perf_counter_ns()
        dist.all_reduce(foreground, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        token_latency_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        foreground_correct = foreground_correct and bool(
            torch.all(foreground == expected).item()
        )

    drain_started = time.perf_counter_ns()
    local_block_failure = not foreground_correct
    if worker is not None:
        worker.join(timeout=nixl_transfer_timeout_s)
        if worker.is_alive():
            transfer["error"] = (
                "TimeoutError: official LMCache/NIXL batched_write exceeded "
                f"{nixl_transfer_timeout_s:.3f}s"
            )
            transfer["finished"] = False
            local_block_failure = True
    if transfer["error"] is not None:
        local_block_failure = True

    # A source-side NIXL timeout used to raise here while the other ranks
    # entered the following NCCL barrier.  That turns an application/data-plane
    # failure into a misleading ProcessGroupNCCL timeout on Perlmutter.  Make
    # the failure itself a collective, bounded decision: every rank reaches
    # this one status reduction, and then every rank exits the block together.
    # The reduction uses the already-initialized NCCL communicator so it does
    # not create a hidden control-plane transport or a privileged network
    # dependency.  A real fabric/GPU stall can still time out this reduction,
    # but the receipt and stack location then identify the synchronized
    # failure boundary rather than a later unconditional barrier.
    block_failure = torch.tensor(
        [1 if local_block_failure else 0],
        dtype=torch.int32,
        device=f"cuda:{cuda_index}",
    )
    dist.all_reduce(block_failure, op=dist.ReduceOp.MAX)
    if bool(block_failure.item()):
        local_reason = (
            transfer["error"]
            or ("foreground_collective_or_data_check_failed" if not foreground_correct else "peer_rank_failed")
        )
        raise RuntimeError(
            "synchronized co-job block failure: "
            f"block={block_index} rank={rank} reason={local_reason}"
        )
    drain_ms = (time.perf_counter_ns() - drain_started) / 1_000_000.0
    dist.barrier()

    verified_bytes = 0
    if background_transfer:
        for receiver_slot, source_pair in enumerate(receiver_pairs):
            for request in range(requests):
                obj = objects[receiver_slot * requests + request]
                expected = _expected_byte(block_index, request, source_pair)
                if bool(torch.all(obj.raw_data == expected).item()):
                    verified_bytes += kv_bytes
    dist.barrier()
    return {
        "block_index": block_index,
        "token_latency_ms": token_latency_ms,
        "foreground_correct": foreground_correct,
        "source_completed_bytes": int(transfer["count"]) * kv_bytes,
        "receiver_verified_bytes": verified_bytes,
        "transfer_elapsed_ms": float(transfer["elapsed_ms"]),
        "transfer_attempted_objects": int(transfer["attempted_objects"]),
        "transfer_returned_objects": int(transfer["count"]),
        "transfer_started": bool(transfer["started"]),
        "transfer_finished": bool(transfer["finished"]),
        "worker_alive_after_join": bool(worker.is_alive()) if worker is not None else False,
        "transfer_debug": transfer["debug"],
        "post_foreground_drain_ms": drain_ms if is_source else 0.0,
        "transfer_error": transfer["error"],
    }


def main() -> None:
    args = _parse_args()
    _set_rank_environment()
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise SystemExit("PyTorch with CUDA/NCCL is required") from exc
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise SystemExit("CUDA and NCCL are required")
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise SystemExit("WORLD_SIZE must be exactly 8")
    if not os.environ.get("MASTER_ADDR") or not os.environ.get("MASTER_PORT"):
        raise SystemExit("MASTER_ADDR and MASTER_PORT are required")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    cuda_index = _cuda_index(torch, local_rank)
    torch.cuda.set_device(cuda_index)
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=args.process_group_timeout_s)
    )
    try:
        hosts = _validate_topology(torch, dist, rank, local_rank)
        background_transfer = not args.no_background_transfer
        kv_bytes = args.kv_mib * MIB
        channel = None
        objects: list[Any] = []
        object_count = _memory_object_count(
            rank, args.traffic_pattern, args.requests,
        )
        if background_transfer and object_count:
            repo_root = Path(__file__).resolve().parents[2]
            NixlChannel, TensorMemoryObj, MemoryObjMetadata, MemoryFormat = (
                _load_official_lmcache(repo_root)
            )
            backing, buffer, objects = _make_memory(
                torch,
                TensorMemoryObj,
                MemoryObjMetadata,
                MemoryFormat,
                requests=object_count,
                kv_bytes=kv_bytes,
            )
            del backing  # the view retains its backing storage
        pair = rank % RANKS_PER_NODE
        is_source = rank < RANKS_PER_NODE
        remote_addresses: list[int] = []
        if background_transfer and object_count:
            if is_source:
                local_endpoint = None
            elif args.traffic_pattern == TRAFFIC_PAIRED:
                local_endpoint = f"*:{args.port_base + pair}"
            else:
                local_endpoint = f"*:{args.port_base}"
            channel = NixlChannel(
                async_mode=False,
                role="sender" if is_source else "receiver",
                buffer_ptr=buffer.data_ptr(),
                buffer_size=buffer.numel(),
                align_bytes=kv_bytes,
                tp_rank=local_rank,
                peer_init_url=local_endpoint,
                backends=["UCX"],
                device=f"cuda:{cuda_index}",
            )
            # Current NIXL bindings index the receiver's prepared descriptor
            # list.  Incast sources own disjoint ranges in rank-4's descriptor
            # array, preventing concurrent writers from racing on one slot.
            if is_source:
                remote_addresses = _remote_descriptor_indices(
                    rank, args.traffic_pattern, args.requests,
                )
        if background_transfer:
            dist.barrier()
            if is_source:
                peer_rank = _receiver_rank(rank, args.traffic_pattern)
                channel.lazy_init_peer_connection(
                    local_id=f"rank-{rank}",
                    peer_id=f"rank-{peer_rank}",
                    peer_init_url=(
                        f"{hosts[peer_rank]}:{args.port_base + pair}"
                        if args.traffic_pattern == TRAFFIC_PAIRED
                        else f"{hosts[peer_rank]}:{args.port_base}"
                    ),
                )
            dist.barrier()
            expected_peers = (
                (f"rank-{_receiver_rank(rank, args.traffic_pattern)}",)
                if is_source
                else tuple(
                    f"rank-{source_pair}"
                    for source_pair in _receiver_source_pairs(
                        rank, args.traffic_pattern
                    )
                )
            )
            if any(
                not channel.remote_xfer_handler_exists(peer_id)
                for peer_id in expected_peers
            ):
                raise RuntimeError(
                    "NIXL peer handshake did not install every transfer handler"
                )
        dist.barrier()
        if rank == 0 and args.ready_file is not None:
            args.ready_file.parent.mkdir(parents=True, exist_ok=True)
            args.ready_file.write_text("ready\n", encoding="utf-8")
        if args.start_file is not None:
            wait_for_start_file(
                args.start_file,
                timeout_s=args.start_file_timeout_s,
                stop_file=args.stop_file,
            )
            # Ranks can observe a shared filesystem marker in adjacent polling
            # intervals.  Align the first measured collective/transfer block.
            dist.barrier()
        if args.start_delay_s:
            time.sleep(args.start_delay_s)

        config = {
            "requests": args.requests,
            "kv_bytes": kv_bytes,
            "token_iters": args.token_iters,
            "blocks": args.blocks,
            "minimum_active_duration_s": args.minimum_active_duration_s,
            "maximum_blocks": args.maximum_blocks,
            "foreground_bytes": args.foreground_mib * MIB,
            "block_delay_s": args.block_delay_s,
            "start_delay_s": args.start_delay_s,
            "start_file": (
                str(args.start_file.resolve())
                if args.start_file is not None
                else None
            ),
            "start_file_timeout_s": args.start_file_timeout_s,
            "process_group_timeout_s": args.process_group_timeout_s,
            "nixl_transfer_timeout_s": args.nixl_transfer_timeout_s,
            "observer_history_stride": args.observer_history_stride,
            "port_base": args.port_base,
            "traffic_pattern": args.traffic_pattern,
            "background_mode": "nixl_ucx" if background_transfer else "nccl_only",
        }
        blocks: list[dict[str, Any]] = []
        rank_blocks_for_observer: list[list[dict[str, Any]]] | None = (
            [[] for _ in range(WORLD_SIZE)] if rank == 0 else None
        )
        block_index = 0
        active_started_ns = time.perf_counter_ns()
        while block_index < args.maximum_blocks:
            block_result = _run_block(
                torch,
                dist,
                channel=channel,
                objects=objects,
                remote_addresses=remote_addresses,
                rank=rank,
                local_rank=local_rank,
                cuda_index=cuda_index,
                background_transfer=background_transfer,
                traffic_pattern=args.traffic_pattern,
                pair=pair,
                block_index=block_index,
                requests=args.requests,
                kv_bytes=kv_bytes,
                token_iters=args.token_iters,
                foreground_elements=(args.foreground_mib * MIB) // 4,
                nixl_transfer_timeout_s=args.nixl_transfer_timeout_s,
            )
            blocks.append(block_result)
            gathered_block = [None] * WORLD_SIZE if rank == 0 else None
            dist.gather_object(
                {"rank": rank, "block": block_result},
                gathered_block,
                dst=0,
            )
            if rank == 0:
                assert gathered_block is not None
                assert rank_blocks_for_observer is not None
                for item in gathered_block:
                    assert isinstance(item, dict)
                    rank_blocks_for_observer[int(item["rank"])].append(
                        item["block"])
                if (
                    args.observer_output is not None
                    or args.observer_history_dir is not None
                ):
                    observer = _observer_snapshot(
                        rank_blocks=[
                            item[block_index]
                            for item in rank_blocks_for_observer
                        ],
                        config=config,
                        hosts=hosts,
                        sequence=block_index + 1,
                        producer_state=(
                            "active"
                            if all(
                                bool(item[block_index]["foreground_correct"])
                                and item[block_index]["transfer_error"] is None
                                for item in rank_blocks_for_observer
                            )
                            else "complete"
                        ),
                    )
                    _publish_observer(
                        output=args.observer_output,
                        history_dir=args.observer_history_dir,
                        observer=observer,
                        history_stride=args.observer_history_stride,
                    )
            dist.barrier()
            if args.block_delay_s:
                time.sleep(args.block_delay_s)
            block_index += 1
            control: list[Any] = [None]
            if rank == 0:
                stop_requested = bool(
                    args.stop_file is not None and args.stop_file.is_file()
                )
                rank0_elapsed_s = (
                    time.perf_counter_ns() - active_started_ns
                ) / 1_000_000_000.0
                continue_running = (
                    not stop_requested
                    and block_index < args.maximum_blocks
                    and (
                        block_index < args.blocks
                        or rank0_elapsed_s < args.minimum_active_duration_s
                    )
                )
                control[0] = {
                    "stop_requested": stop_requested,
                    "continue_running": continue_running,
                }
            dist.broadcast_object_list(control, src=0)
            if not bool(control[0]["continue_running"]):
                break
        active_loop_elapsed_ms = (
            time.perf_counter_ns() - active_started_ns
        ) / 1_000_000.0
        elapsed_records: list[Any] | None = (
            [None] * WORLD_SIZE if rank == 0 else None
        )
        dist.gather_object(
            {
                "rank": rank,
                "active_loop_elapsed_ms": active_loop_elapsed_ms,
            },
            elapsed_records,
            dst=0,
        )
        if rank == 0:
            assert rank_blocks_for_observer is not None
            assert elapsed_records is not None
            actual_blocks = len(rank_blocks_for_observer[0])
            if actual_blocks <= 0 or any(
                len(item) != actual_blocks for item in rank_blocks_for_observer
            ):
                raise RuntimeError("rank block counts differ at co-job shutdown")
            elapsed_by_rank = {
                int(item["rank"]): float(item["active_loop_elapsed_ms"])
                for item in elapsed_records
            }
            if set(elapsed_by_rank) != set(range(WORLD_SIZE)):
                raise RuntimeError("active-loop rank receipts differ")
            config["blocks"] = actual_blocks
            rank_records = []
            for item_rank in range(WORLD_SIZE):
                rank_records.append({
                    "rank": item_rank,
                    "local_rank": item_rank % RANKS_PER_NODE,
                    "hostname": hosts[item_rank],
                    "config": config,
                    "blocks": rank_blocks_for_observer[item_rank],
                    "active_loop_elapsed_ms": elapsed_by_rank[item_rank],
                })
            result = aggregate_rank_records(rank_records)
            if (
                args.observer_output is not None
                or args.observer_history_dir is not None
            ):
                final_observer = _observer_snapshot(
                    rank_blocks=[
                        item[actual_blocks - 1]
                        for item in rank_blocks_for_observer
                    ],
                    config=config,
                    hosts=hosts,
                    sequence=actual_blocks + 1,
                    producer_state="complete",
                )
                _publish_observer(
                    output=args.observer_output,
                    history_dir=args.observer_history_dir,
                    observer=final_observer,
                    history_stride=args.observer_history_stride,
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "correctness": result["overall_correctness_met"],
                    },
                    sort_keys=True,
                )
            )
        dist.barrier()
        # Do not call receiver channel.close(): at this commit it joins the
        # blocking ZMQ listener before context termination and can hang.
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
