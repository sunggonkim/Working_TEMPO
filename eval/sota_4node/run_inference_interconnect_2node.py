#!/usr/bin/env python3
"""Minimal two-node LLM-interconnect contention experiment.

The foreground is a small CUDA decoder loop with tensor-parallel NCCL
all-reduces.  The background moves real KV-sized CUDA pages between matched
ranks on two nodes.  This is a research harness, not a serving framework.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import socket
import statistics
import time
from typing import Any, Iterable

from tempo.domain_admission import DomainAdmissionController, DomainBudget, DomainRequest
from tempo.resource_domain import ResourceDomain


WORLD_SIZE = 8
NODES = 2
RANKS_PER_NODE = 4
PAIR_COUNT = 4
CHUNKS_PER_REQUEST = 4
MODE_ORDER = (
    "fg_only",
    "uncontrolled",
    "local",
    "global_static",
    "aot_uniform2",
    "aot_ramp2",
    "aot_ramp4",
    "aot_uniform2_coalesced",
    "aot_ramp2_coalesced",
    "aot_ramp4_coalesced",
    "tempo",
)
LATIN_ROWS = tuple(
    tuple(MODE_ORDER[(column + row) % len(MODE_ORDER)] for column in range(len(MODE_ORDER)))
    for row in range(len(MODE_ORDER))
)
BLOCK_MODES = tuple(mode for row in LATIN_ROWS for mode in row)
PAIR_WARMUP_SOURCE_NODES = (0, 1)
FABRIC_ROUTE = (ResourceDomain.NIC_FABRIC, ResourceDomain.SLINGSHOT_FABRIC)
AOT_PAIR_CONCURRENCY_BY_MODE = {
    "aot_uniform2": (2,) * 8,
    "aot_ramp2": (1,) * 4 + (2,) * 6,
    "aot_ramp4": (1,) * 4 + (4,) * 3,
    "aot_uniform2_coalesced": (2,) * 8,
    "aot_ramp2_coalesced": (1,) * 4 + (2,) * 6,
    "aot_ramp4_coalesced": (1,) * 4 + (4,) * 3,
}
COALESCED_AOT_MODES = frozenset({
    "aot_uniform2_coalesced",
    "aot_ramp2_coalesced",
    "aot_ramp4_coalesced",
})
AOT_PAIR_CHUNKS = tuple(
    (pair, chunk)
    for chunk in range(CHUNKS_PER_REQUEST)
    for pair in range(PAIR_COUNT)
)
if any(
    sum(widths) != len(AOT_PAIR_CHUNKS)
    or any(width not in (1, 2, 4) for width in widths)
    for widths in AOT_PAIR_CONCURRENCY_BY_MODE.values()
):
    raise RuntimeError("each AOT schedule must issue every pair/chunk exactly once")


def schedule_entries(
    mode: str,
    token_index: int,
    *,
    requests_per_block: int = 1,
) -> tuple[tuple[int, int, int], ...]:
    """Return ``(request, pair, chunk)`` transfers issued at one token."""

    if mode not in MODE_ORDER:
        raise ValueError(f"unknown mode: {mode}")
    if type(token_index) is not int or token_index < 0:
        raise ValueError("token_index must be a non-negative int")
    if type(requests_per_block) is not int or requests_per_block <= 0:
        raise ValueError("requests_per_block must be a positive int")
    if mode == "fg_only":
        return ()
    if mode == "uncontrolled":
        if token_index != 0:
            return ()
        return tuple(
            (request, pair, chunk)
            for request in range(requests_per_block)
            for pair in range(PAIR_COUNT)
            for chunk in range(CHUNKS_PER_REQUEST)
        )
    if mode == "local":
        if token_index % PAIR_COUNT or token_index // PAIR_COUNT >= CHUNKS_PER_REQUEST:
            return ()
        chunk = token_index // PAIR_COUNT
        return tuple(
            (request, pair, chunk)
            for request in range(requests_per_block)
            for pair in range(PAIR_COUNT)
        )
    if mode in AOT_PAIR_CONCURRENCY_BY_MODE:
        widths = AOT_PAIR_CONCURRENCY_BY_MODE[mode]
        if token_index >= len(widths):
            return ()
        first = sum(widths[:token_index])
        selected = AOT_PAIR_CHUNKS[first:first + widths[token_index]]
        return tuple(
            (request, pair, chunk)
            for pair, chunk in selected
            for request in range(requests_per_block)
        )
    if token_index >= PAIR_COUNT * CHUNKS_PER_REQUEST:
        return ()
    pair = token_index % PAIR_COUNT
    chunk = token_index // PAIR_COUNT
    return tuple((request, pair, chunk) for request in range(requests_per_block))


def schedule_summary(mode: str, *, requests_per_block: int = 1) -> dict[str, int]:
    entries = [
        schedule_entries(mode, token, requests_per_block=requests_per_block)
        for token in range(PAIR_COUNT * CHUNKS_PER_REQUEST)
    ]
    return {
        "chunks": sum(len(item) for item in entries),
        "max_active_pairs": max((len({entry[1] for entry in item}) for item in entries), default=0),
        "max_chunks_at_token": max((len(item) for item in entries), default=0),
    }


def source_node_for(block_index: int, request_index: int) -> int:
    """Alternate migration direction across both blocks and requests."""

    if min(block_index, request_index) < 0:
        raise ValueError("block and request indexes must be non-negative")
    return (block_index + request_index) % NODES


def coalesced_transfer_groups(
    entries: Iterable[tuple[int, int, int]],
    *,
    block_index: int,
    pair_index: int,
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Group one pair's logical requests by direction and chunk."""

    grouped: dict[tuple[int, int], list[int]] = {}
    for request, scheduled_pair, chunk in entries:
        if scheduled_pair != pair_index:
            continue
        key = (source_node_for(block_index, request), chunk)
        grouped.setdefault(key, []).append(request)
    return tuple(
        (source_node, chunk, tuple(requests))
        for (source_node, chunk), requests in sorted(grouped.items())
    )


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires samples")
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the eight small rank records without hardware dependencies."""

    if len(records) != WORLD_SIZE or sorted(item.get("rank") for item in records) != list(range(WORLD_SIZE)):
        raise ValueError("aggregate requires exact ranks 0..7")
    ordered = sorted(records, key=lambda item: item["rank"])
    if any(item.get("world_size") != WORLD_SIZE or item.get("nodes") != NODES for item in ordered):
        raise ValueError("rank topology does not match 2 nodes / 8 ranks")
    config = ordered[0].get("config")
    if any(item.get("config") != config for item in ordered):
        raise ValueError("rank configs differ")
    if any(tuple(block["mode"] for block in item.get("blocks", [])) != BLOCK_MODES for item in ordered):
        raise ValueError("balanced block sequence is not exact")
    pair_warmups = [item.get("pair_warmup") for item in ordered]
    control_warmups = [item.get("control_warmup") for item in ordered]
    pair_warmup_correct = all(
        isinstance(item, dict)
        and item.get("correctness_met") is True
        and item.get("background_stream") == "dedicated_cuda"
        and item.get("source_nodes") == list(PAIR_WARMUP_SOURCE_NODES)
        and type(item.get("bytes_per_direction")) is int
        and item["bytes_per_direction"] > 0
        for item in pair_warmups
    )
    control_warmup_correct = all(
        isinstance(item, dict)
        and item.get("correctness_met") is True
        and item.get("source_rank") == 0
        and item.get("value") == 137
        for item in control_warmups
    )
    warmup_correct = pair_warmup_correct and control_warmup_correct

    reference = {item["rank"]: float(item["blocks"][0]["foreground_checksum"]) for item in ordered}
    block_results: list[dict[str, Any]] = []
    mode_samples: dict[str, dict[str, list[float] | int | bool]] = defaultdict(
        lambda: {
            "rank_token_latency_ms": [],
            "rank_first_token_step_ms": [],
            "global_token_tail_ms": [],
            "global_first_token_step_ms": [],
            "global_post_foreground_drain_ms": [],
            "global_background_completion_upper_bound_ms": [],
            "global_block_data_plane_ms": [],
            "background_completed_bytes": 0,
            "expected_background_bytes": 0,
            "background_collective_participations": 0,
            "correctness_met": True,
            "replicates": 0,
        }
    )
    token_count = int(config["tokens"])
    for block_index, mode in enumerate(BLOCK_MODES):
        rank_blocks = [item["blocks"][block_index] for item in ordered]
        if any(len(block["token_latency_ms"]) != token_count for block in rank_blocks):
            raise ValueError("rank token counts differ from config")
        rank_token_samples = [
            float(value)
            for block in rank_blocks
            for value in block["token_latency_ms"]
        ]
        rank_first_token_samples = [
            float(block["first_token_step_ms"]) for block in rank_blocks
        ]
        global_token_tail = [
            max(float(block["token_latency_ms"][token]) for block in rank_blocks)
            for token in range(token_count)
        ]
        global_first_token = global_token_tail[0]
        global_drain = max(float(block["post_foreground_drain_ms"]) for block in rank_blocks)
        global_background_elapsed = max(
            float(block["background_completion_upper_bound_ms"]) for block in rank_blocks
        )
        global_block_data_plane = max(float(block["block_data_plane_ms"]) for block in rank_blocks)
        completed = sum(int(block["background_completed_bytes"]) for block in rank_blocks)
        expected = sum(int(block["expected_receive_bytes"]) for block in rank_blocks)
        collective_participations = sum(
            int(block["background_operations_participated"]) for block in rank_blocks
        )
        checksum_ok = all(
            math.isclose(
                float(block["foreground_checksum"]),
                reference[rank],
                rel_tol=1e-3,
                abs_tol=1e-3,
            )
            for rank, block in enumerate(rank_blocks)
        )
        correct = (
            all(block["correctness_met"] is True for block in rank_blocks)
            and all(block.get("controller_released") is True for block in rank_blocks)
            and all(block.get("background_stream") == "dedicated_cuda" for block in rank_blocks)
            and checksum_ok
            and completed == expected
        )
        admissions = rank_blocks[0].get("admissions", [])
        block_result = {
            "block_index": block_index,
            "mode": mode,
            "global_first_token_step_ms": global_first_token,
            "global_token_latency_p50_ms": statistics.median(global_token_tail),
            "global_token_latency_p99_ms": _percentile(global_token_tail, 0.99),
            "rank_first_token_step_p50_ms": statistics.median(rank_first_token_samples),
            "rank_first_token_step_p99_ms": _percentile(rank_first_token_samples, 0.99),
            "rank_token_latency_p50_ms": statistics.median(rank_token_samples),
            "rank_token_latency_p99_ms": _percentile(rank_token_samples, 0.99),
            "global_post_foreground_drain_ms": global_drain,
            "global_background_completion_upper_bound_ms": global_background_elapsed,
            "global_block_data_plane_ms": global_block_data_plane,
            "background_completed_bytes": completed,
            "expected_background_bytes": expected,
            "background_collective_participations": collective_participations,
            "background_completion_met": completed == expected,
            "correctness_met": correct,
            "schedule": schedule_summary(
                mode, requests_per_block=int(config["requests_per_block"])
            ),
            "admissions": admissions,
        }
        block_results.append(block_result)
        bucket = mode_samples[mode]
        bucket["rank_token_latency_ms"].extend(rank_token_samples)  # type: ignore[union-attr]
        bucket["rank_first_token_step_ms"].extend(rank_first_token_samples)  # type: ignore[union-attr]
        bucket["global_token_tail_ms"].extend(global_token_tail)  # type: ignore[union-attr]
        bucket["global_first_token_step_ms"].append(global_first_token)  # type: ignore[union-attr]
        bucket["global_post_foreground_drain_ms"].append(global_drain)  # type: ignore[union-attr]
        bucket["global_background_completion_upper_bound_ms"].append(global_background_elapsed)  # type: ignore[union-attr]
        bucket["global_block_data_plane_ms"].append(global_block_data_plane)  # type: ignore[union-attr]
        bucket["background_completed_bytes"] = int(bucket["background_completed_bytes"]) + completed
        bucket["expected_background_bytes"] = int(bucket["expected_background_bytes"]) + expected
        bucket["background_collective_participations"] = (
            int(bucket["background_collective_participations"]) + collective_participations
        )
        bucket["correctness_met"] = bool(bucket["correctness_met"]) and correct
        bucket["replicates"] = int(bucket["replicates"]) + 1

    modes: dict[str, dict[str, Any]] = {}
    for mode in MODE_ORDER:
        bucket = mode_samples[mode]
        rank_token_samples = bucket.pop("rank_token_latency_ms")
        rank_first_token_samples = bucket.pop("rank_first_token_step_ms")
        global_token_tail = bucket.pop("global_token_tail_ms")
        global_first_token_samples = bucket.pop("global_first_token_step_ms")
        global_drain_samples = bucket.pop("global_post_foreground_drain_ms")
        global_background_elapsed_samples = bucket.pop("global_background_completion_upper_bound_ms")
        global_block_data_plane_samples = bucket.pop("global_block_data_plane_ms")
        modes[mode] = {
            **bucket,
            "global_first_token_step_p50_ms": statistics.median(global_first_token_samples),
            "global_first_token_step_p99_ms": _percentile(global_first_token_samples, 0.99),
            "global_token_latency_p50_ms": statistics.median(global_token_tail),
            "global_token_latency_p99_ms": _percentile(global_token_tail, 0.99),
            "rank_first_token_step_p50_ms": statistics.median(rank_first_token_samples),
            "rank_first_token_step_p99_ms": _percentile(rank_first_token_samples, 0.99),
            "rank_token_latency_p50_ms": statistics.median(rank_token_samples),
            "rank_token_latency_p99_ms": _percentile(rank_token_samples, 0.99),
            "global_post_foreground_drain_p50_ms": statistics.median(global_drain_samples),
            "global_post_foreground_drain_p99_ms": _percentile(global_drain_samples, 0.99),
            "global_background_completion_upper_bound_p50_ms": statistics.median(global_background_elapsed_samples),
            "global_background_completion_upper_bound_p99_ms": _percentile(global_background_elapsed_samples, 0.99),
            "global_block_data_plane_p50_ms": statistics.median(global_block_data_plane_samples),
            "global_block_data_plane_p99_ms": _percentile(global_block_data_plane_samples, 0.99),
        }
    return {
        "schema_version": "tempo-inference-interconnect-2node-5",
        "evidence_state": "live_two_node",
        "world_size": WORLD_SIZE,
        "nodes": NODES,
        "ranks_per_node": RANKS_PER_NODE,
        "block_sequence": list(BLOCK_MODES),
        "config": config,
        "pair_warmup": {
            "source_nodes": list(PAIR_WARMUP_SOURCE_NODES),
            "bytes_per_direction": pair_warmups[0]["bytes_per_direction"] if pair_warmup_correct else 0,
            "correctness_met": pair_warmup_correct,
        },
        "control_warmup": {
            "source_rank": 0,
            "value": 137,
            "correctness_met": control_warmup_correct,
        },
        "blocks": block_results,
        "modes": modes,
        "overall_correctness_met": warmup_correct and all(
            block["correctness_met"] for block in block_results
        ),
    }


def _set_slurm_rank_environment() -> None:
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]


def _validate_topology(torch: Any, dist: Any, rank: int, local_rank: int) -> tuple[int, str]:
    hostname = socket.gethostname()
    hosts: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(hosts, hostname)
    local_tensor = torch.tensor([local_rank], dtype=torch.int64, device="cuda")
    gathered = [torch.empty_like(local_tensor) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, local_tensor)
    local_ranks = [int(item.item()) for item in gathered]
    if len(set(hosts)) != NODES or len(set(hosts[:4])) != 1 or len(set(hosts[4:])) != 1 or hosts[0] == hosts[4]:
        raise RuntimeError("ranks must be node-major across exactly two hosts")
    if local_ranks[:4] != list(range(4)) or local_ranks[4:] != list(range(4)):
        raise RuntimeError("each node must map four ranks to local CUDA devices 0..3")
    return (0 if rank < RANKS_PER_NODE else 1), hostname


def _make_decoder(torch: Any, *, rank: int, layers: int, hidden: int, context: int) -> list[dict[str, Any]]:
    heads = 8
    head_dim = hidden // heads
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260813 + rank)
    scale = hidden ** -0.5
    result = []
    for _ in range(layers):
        result.append({
            "q_weight": torch.randn(
                hidden, hidden, device="cuda", dtype=torch.float16, generator=generator
            ) * scale,
            "out_weight": torch.randn(
                hidden, hidden, device="cuda", dtype=torch.float16, generator=generator
            ) * scale,
            "keys": torch.randn(
                heads, context, head_dim, device="cuda", dtype=torch.float16, generator=generator
            ) * 0.125,
            "values": torch.randn(
                heads, context, head_dim, device="cuda", dtype=torch.float16, generator=generator
            ) * 0.125,
        })
    return result


def _decoder_token(torch: Any, dist: Any, x: Any, decoder: list[dict[str, Any]]) -> Any:
    heads = decoder[0]["keys"].shape[0]
    head_dim = x.shape[-1] // heads
    for layer in decoder:
        query = (x @ layer["q_weight"]).view(heads, 1, head_dim)
        scores = torch.matmul(query, layer["keys"].transpose(-1, -2)) / math.sqrt(head_dim)
        probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=torch.float16)
        attended = torch.matmul(probabilities, layer["values"]).reshape(1, -1)
        projected = attended @ layer["out_weight"]
        dist.all_reduce(projected, op=dist.ReduceOp.SUM)
        x = torch.tanh(x + projected / WORLD_SIZE)
    return x


def _make_controller(total_bytes: int) -> DomainAdmissionController:
    rate = 25_000_000_000
    return DomainAdmissionController(
        {
            domain: DomainBudget(domain, rate, total_bytes, rate)
            for domain in FABRIC_ROUTE
        },
        catch_up_slack_ns=0,
    )


def _admit_tempo_token(
    controller: DomainAdmissionController,
    *,
    block_index: int,
    token_index: int,
    requests_per_block: int,
    chunk_bytes: int,
) -> tuple[int, list[dict[str, Any]], list[str]]:
    entries = schedule_entries("tempo", token_index, requests_per_block=requests_per_block)
    if not entries:
        return -1, [], []
    pair = entries[0][1]
    now_ns = time.monotonic_ns()
    records: list[dict[str, Any]] = []
    active: list[str] = []
    all_admitted = True
    for request, entry_pair, chunk in entries:
        if entry_pair != pair:
            raise RuntimeError("tempo schedule selected more than one pair")
        request_id = f"b{block_index}-r{request}-p{pair}-c{chunk}"
        decision = controller.admit(DomainRequest(
            request_id=request_id,
            flow_id=f"kv-block-{block_index}-request-{request}",
            bytes=chunk_bytes,
            route=FABRIC_ROUTE,
            now_ns=now_ns,
            deadline_ns=now_ns + 60_000_000_000,
            nonpreemptible_residual_bytes=chunk_bytes,
            foreground_domains=FABRIC_ROUTE,
        ))
        records.append({
            "request_id": request_id,
            "request_index": request,
            "pair_index": pair,
            "chunk_index": chunk,
            "admitted": decision.admitted,
            "admitted_bytes": decision.admitted_bytes,
            "reason": decision.reason,
            "launched": False,
        })
        if decision.admitted:
            active.append(request_id)
        else:
            all_admitted = False
    if not all_admitted:
        for request_id in active:
            controller.cancel(request_id)
        return -1, records, []
    for record in records:
        record["launched"] = True
    return pair, records, active


def _expected_byte(block_index: int, request_index: int, pair_index: int) -> int:
    return 1 + ((block_index * 31 + request_index * 11 + pair_index * 3) % 251)


def _warm_communicators(
    torch: Any,
    dist: Any,
    *,
    rank: int,
    node_index: int,
    pair_index: int,
    pair_group: Any,
    background_stream: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pair_buffer = torch.empty(1, dtype=torch.uint8, device="cuda")
    pair_correct = True
    for source_node in PAIR_WARMUP_SOURCE_NODES:
        expected = 41 + pair_index * 2 + source_node
        if node_index == source_node:
            pair_buffer.fill_(expected)
        else:
            pair_buffer.zero_()
        torch.cuda.synchronize()
        source_rank = pair_index + source_node * RANKS_PER_NODE
        with torch.cuda.stream(background_stream):
            work = dist.broadcast(
                pair_buffer,
                src=source_rank,
                group=pair_group,
                async_op=True,
            )
        work.wait()
        background_stream.synchronize()
        pair_correct = pair_correct and int(pair_buffer.item()) == expected
    dist.barrier()

    control = torch.empty(1, dtype=torch.int32, device="cuda")
    if rank == 0:
        control.fill_(137)
    else:
        control.zero_()
    dist.broadcast(control, src=0)
    control_correct = int(control.item()) == 137
    dist.barrier()
    return (
        {
            "source_nodes": list(PAIR_WARMUP_SOURCE_NODES),
            "bytes_per_direction": 1,
            "background_stream": "dedicated_cuda",
            "correctness_met": pair_correct,
        },
        {
            "source_rank": 0,
            "value": 137,
            "correctness_met": control_correct,
        },
    )


def _run_block(
    torch: Any,
    dist: Any,
    *,
    rank: int,
    node_index: int,
    pair_index: int,
    pair_group: Any,
    background_stream: Any,
    block_index: int,
    mode: str,
    decoder: list[dict[str, Any]],
    tokens: int,
    hidden: int,
    requests_per_block: int,
    kv_bytes: int,
    chunk_bytes: int,
) -> dict[str, Any]:
    coalesced = mode in COALESCED_AOT_MODES
    pages: list[Any] = []
    coalesced_pages: dict[tuple[int, int], Any] = {}
    source_requests: dict[int, tuple[int, ...]] = {
        source_node: tuple(
            request
            for request in range(requests_per_block)
            if source_node_for(block_index, request) == source_node
        )
        for source_node in range(NODES)
    }
    request_offsets = {
        request: offset
        for requests in source_requests.values()
        for offset, request in enumerate(requests)
    }
    if mode != "fg_only":
        if coalesced:
            for source_node, requests in source_requests.items():
                for chunk in range(CHUNKS_PER_REQUEST):
                    page = torch.empty(
                        len(requests) * chunk_bytes,
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    if node_index == source_node:
                        for offset, request in enumerate(requests):
                            first = offset * chunk_bytes
                            page[first:first + chunk_bytes].fill_(
                                _expected_byte(block_index, request, pair_index)
                            )
                    else:
                        page.zero_()
                    coalesced_pages[(source_node, chunk)] = page
        else:
            for request in range(requests_per_block):
                page = torch.empty(kv_bytes, dtype=torch.uint8, device="cuda")
                if node_index == source_node_for(block_index, request):
                    page.fill_(_expected_byte(block_index, request, pair_index))
                else:
                    page.zero_()
                pages.append(page)
    torch.cuda.synchronize()
    dist.barrier()

    total_background_bytes = requests_per_block * PAIR_COUNT * kv_bytes
    controller = _make_controller(total_background_bytes) if rank == 0 and mode == "tempo" else None
    control = torch.empty(1, dtype=torch.int32, device="cuda") if mode == "tempo" else None
    works: list[Any] = []
    receiver_work_count = 0
    first_background_issue_ns: int | None = None
    admissions: list[dict[str, Any]] = []
    active_admissions: list[str] = []
    x = torch.linspace(-0.5, 0.5, hidden, dtype=torch.float16, device="cuda").view(1, hidden)
    token_latency_ms: list[float] = []
    block_start_ns = time.perf_counter_ns()

    for token_index in range(tokens):
        start_ns = time.perf_counter_ns()
        entries = schedule_entries(mode, token_index, requests_per_block=requests_per_block)
        if mode == "tempo":
            if rank == 0:
                assert controller is not None and control is not None
                selected, admission_records, active = _admit_tempo_token(
                    controller,
                    block_index=block_index,
                    token_index=token_index,
                    requests_per_block=requests_per_block,
                    chunk_bytes=chunk_bytes,
                )
                control.fill_(selected)
                admissions.extend(admission_records)
                active_admissions.extend(active)
            assert control is not None
            dist.broadcast(control, src=0)
            selected = int(control.item())
            entries = tuple(entry for entry in entries if entry[1] == selected)

        if coalesced:
            groups = coalesced_transfer_groups(
                entries,
                block_index=block_index,
                pair_index=pair_index,
            )
            for source_node, chunk, requests in groups:
                source_rank = pair_index + source_node * RANKS_PER_NODE
                if first_background_issue_ns is None:
                    first_background_issue_ns = time.perf_counter_ns()
                if node_index != source_node:
                    receiver_work_count += len(requests)
                with torch.cuda.stream(background_stream):
                    works.append(dist.broadcast(
                        coalesced_pages[(source_node, chunk)],
                        src=source_rank,
                        group=pair_group,
                        async_op=True,
                    ))
        else:
            for request, scheduled_pair, chunk in entries:
                if scheduled_pair != pair_index:
                    continue
                source_node = source_node_for(block_index, request)
                source_rank = pair_index + source_node * RANKS_PER_NODE
                first = chunk * chunk_bytes
                last = first + chunk_bytes
                if first_background_issue_ns is None:
                    first_background_issue_ns = time.perf_counter_ns()
                if node_index != source_node:
                    receiver_work_count += 1
                with torch.cuda.stream(background_stream):
                    works.append(dist.broadcast(
                        pages[request][first:last],
                        src=source_rank,
                        group=pair_group,
                        async_op=True,
                    ))

        x = _decoder_token(torch, dist, x, decoder)
        foreground_done = torch.cuda.Event()
        foreground_done.record()
        foreground_done.synchronize()
        token_latency_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)

    foreground_end_ns = time.perf_counter_ns()
    for work in works:
        work.wait()
    background_stream.synchronize()
    torch.cuda.synchronize()
    background_end_ns = time.perf_counter_ns()
    background_completed_bytes = receiver_work_count * chunk_bytes
    post_foreground_drain_ms = (
        0.0 if first_background_issue_ns is None
        else (background_end_ns - foreground_end_ns) / 1_000_000.0
    )
    background_completion_upper_bound_ms = (
        0.0 if first_background_issue_ns is None
        else (background_end_ns - first_background_issue_ns) / 1_000_000.0
    )
    page_correct = True
    expected_receive_bytes = 0
    if mode != "fg_only":
        for request in range(requests_per_block):
            expected = _expected_byte(block_index, request, pair_index)
            source_node = source_node_for(block_index, request)
            if coalesced:
                offset = request_offsets[request]
                first = offset * chunk_bytes
                last = first + chunk_bytes
                for chunk in range(CHUNKS_PER_REQUEST):
                    page = coalesced_pages[(source_node, chunk)]
                    page_correct = page_correct and bool(torch.all(
                        page[first:last] == expected
                    ).item())
            else:
                page_correct = page_correct and bool(torch.all(pages[request] == expected).item())
            if node_index != source_node:
                expected_receive_bytes += kv_bytes
    finite = bool(torch.isfinite(x).all().item())
    checksum = float(x.float().sum().item())
    dist.barrier()
    controller_released = True
    if rank == 0 and controller is not None:
        for request_id in active_admissions:
            controller.complete(request_id, chunk_bytes)
        controller_released = not any(controller.inflight_bytes.values())
        if not controller_released:
            raise RuntimeError("TEMPO admission reservations were not released")
    return {
        "block_index": block_index,
        "mode": mode,
        "source_nodes": [source_node_for(block_index, request) for request in range(requests_per_block)],
        "token_latency_ms": token_latency_ms,
        "first_token_step_ms": token_latency_ms[0],
        "foreground_checksum": checksum,
        "background_operations_participated": len(works),
        "background_layout": "prepacked_source_direction_chunk" if coalesced else "per_request",
        "expected_receive_bytes": expected_receive_bytes,
        "background_completed_bytes": background_completed_bytes,
        "post_foreground_drain_ms": post_foreground_drain_ms,
        "background_completion_upper_bound_ms": background_completion_upper_bound_ms,
        "block_data_plane_ms": (background_end_ns - block_start_ns) / 1_000_000.0,
        "correctness_met": page_correct and finite and background_completed_bytes == expected_receive_bytes,
        "controller_released": controller_released,
        "background_stream": "dedicated_cuda",
        "admissions": admissions,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests-per-block", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--kv-mib", type=int, default=64)
    parser.add_argument("--chunk-mib", type=int, default=16)
    args = parser.parse_args()
    if args.requests_per_block <= 0 or args.layers <= 0 or args.context <= 0:
        parser.error("requests/layers/context must be positive")
    if args.tokens < PAIR_COUNT * CHUNKS_PER_REQUEST:
        parser.error("tokens must be at least 16 so every schedule completes")
    if args.hidden_size <= 0 or args.hidden_size % 8:
        parser.error("hidden-size must be positive and divisible by 8")
    if args.kv_mib <= 0 or args.chunk_mib <= 0 or args.kv_mib != CHUNKS_PER_REQUEST * args.chunk_mib:
        parser.error("kv-mib must be exactly four chunk-mib chunks")
    return args


def main() -> None:
    args = _parse_args()
    _set_slurm_rank_environment()
    try:
        import torch
        import torch.distributed as dist
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch with NCCL support is required") from exc
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise SystemExit("CUDA and NCCL are required")
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise SystemExit("WORLD_SIZE must be exactly 8")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not 0 <= local_rank < RANKS_PER_NODE:
        raise SystemExit("LOCAL_RANK must be in 0..3")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        if dist.get_world_size() != WORLD_SIZE:
            raise RuntimeError("the experiment requires exactly eight ranks")
        node_index, hostname = _validate_topology(torch, dist, rank, local_rank)
        pair_groups = [dist.new_group([pair, pair + RANKS_PER_NODE], backend="nccl") for pair in range(PAIR_COUNT)]
        pair_index = rank % RANKS_PER_NODE
        decoder = _make_decoder(
            torch,
            rank=rank,
            layers=args.layers,
            hidden=args.hidden_size,
            context=args.context,
        )
        warm = torch.zeros(1, args.hidden_size, dtype=torch.float16, device="cuda")
        warm = _decoder_token(torch, dist, warm, decoder)
        del warm
        torch.cuda.synchronize()
        background_stream = torch.cuda.Stream()
        pair_warmup, control_warmup = _warm_communicators(
            torch,
            dist,
            rank=rank,
            node_index=node_index,
            pair_index=pair_index,
            pair_group=pair_groups[pair_index],
            background_stream=background_stream,
        )
        if not pair_warmup["correctness_met"]:
            raise RuntimeError("pair communicator warmup correctness failed")
        if not control_warmup["correctness_met"]:
            raise RuntimeError("world control communicator warmup correctness failed")

        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        kv_bytes = args.kv_mib * 1024 * 1024
        chunk_bytes = args.chunk_mib * 1024 * 1024
        blocks = [
            _run_block(
                torch,
                dist,
                rank=rank,
                node_index=node_index,
                pair_index=pair_index,
                pair_group=pair_groups[pair_index],
                background_stream=background_stream,
                block_index=block_index,
                mode=mode,
                decoder=decoder,
                tokens=args.tokens,
                hidden=args.hidden_size,
                requests_per_block=args.requests_per_block,
                kv_bytes=kv_bytes,
                chunk_bytes=chunk_bytes,
            )
            for block_index, mode in enumerate(BLOCK_MODES)
        ]
        config = {
            "requests_per_block": args.requests_per_block,
            "tokens": args.tokens,
            "layers": args.layers,
            "hidden_size": args.hidden_size,
            "context": args.context,
            "kv_bytes": kv_bytes,
            "chunk_bytes": chunk_bytes,
            "chunks_per_request": CHUNKS_PER_REQUEST,
            "replicates_per_mode": len(MODE_ORDER),
            "pair_warmup_directions": len(PAIR_WARMUP_SOURCE_NODES),
            "world_control_warmup": True,
            "aot_pair_concurrency_by_mode": {
                mode: list(widths)
                for mode, widths in AOT_PAIR_CONCURRENCY_BY_MODE.items()
            },
            "aot_semantics": "fixed_ahead_of_time_width_sweep_not_adaptive",
            "coalesced_modes": sorted(COALESCED_AOT_MODES),
            "coalesced_layout": "prepacked_source_direction_chunk",
        }
        rank_record = {
            "schema_version": "tempo-inference-interconnect-rank-4",
            "rank": rank,
            "local_rank": local_rank,
            "node_index": node_index,
            "hostname": hostname,
            "world_size": WORLD_SIZE,
            "nodes": NODES,
            "config": config,
            "pair_warmup": pair_warmup,
            "control_warmup": control_warmup,
            "blocks": blocks,
        }
        rank_path = args.output_dir / f"rank_{rank}.json"
        rank_path.write_text(json.dumps(rank_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        dist.barrier()
        if rank == 0:
            records = [
                json.loads((args.output_dir / f"rank_{item_rank}.json").read_text(encoding="utf-8"))
                for item_rank in range(WORLD_SIZE)
            ]
            result = aggregate_rank_records(records)
            (args.output_dir / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not result["overall_correctness_met"]:
                raise RuntimeError("two-node inference correctness failed")
            print(json.dumps({
                "output": str(args.output_dir / "result.json"),
                "correctness": True,
                "blocks": len(BLOCK_MODES),
            }, sort_keys=True))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
