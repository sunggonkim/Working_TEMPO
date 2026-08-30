#!/usr/bin/env python3
"""Screen frozen LMCache/NIXL admission beside a real vLLM TP8 decode.

This process is the bounded sidecar half of the experiment.  Launch it as
eight node-major ranks (four ranks per node) while one vLLM 0.26 native-MP
TP8 server spans the same two nodes.  Rank zero drives the server through the
streaming OpenAI completions endpoint.  A small Gloo control message is sent
for every returned token in every mode, while only the selected modes issue
official LMCache ``NixlChannel`` writes.

The foreground and data plane are real, but the boundary is intentional:
the registered sidecar buffers have the frozen KV traffic geometry and are
not vLLM-owned KV-cache tensors.  A single TP8 engine has no remote
producer/consumer engine for the LMCache vLLM connector to move KV between;
using the connector alone would therefore not create the required competing
inter-node NIXL traffic.  Results describe this as a component-level sidecar
screen, never as end-to-end connector integration.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import http.client
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
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from eval.sota_4node import compile_lmcache_active_pulse_group2_plan as compiled
from eval.sota_4node import run_lmcache_active_pulse_group2_2node as group2
from eval.sota_4node import run_lmcache_epoch_2node as epoch
from eval.sota_4node import run_lmcache_microburst_2node as microburst
from eval.sota_4node import run_lmcache_nixl_contention_2node as official


WORLD_SIZE = 8
NODES = 2
RANKS_PER_NODE = 4
PAIR_COUNT = 4
REQUESTS = 2
TOKENS = 64
CHUNKS_PER_REQUEST = 16
CHUNK_BYTES = 512 * (1 << 10)
KV_BYTES_PER_RANK = CHUNKS_PER_REQUEST * CHUNK_BYTES
ABSOLUTE_DEADLINE_NS = 91_257_744
START_LAG_CAP_NS = 2_272_580
EXPECTED_PLAN_SIGNATURE = (
    "757b9ddca7f727e7fce2647af4edf83b7771a70749633f15fae2f08477b9e555"
)
EXPECTED_ARTIFACT_SIGNATURE = (
    "38095fcb5b0ae7f4d9d2e2c4085b7dae5e7b373d62aef967062c38641f7a0d43"
)
PULSE_TOKENS = (4, 7, 10, 13, 17, 20, 23, 26)
MODES = ("fg_only", "lmcache_greedy", "tempo_group2")
LATIN_ROWS = tuple(
    tuple(MODES[(column + row) % len(MODES)] for column in range(len(MODES)))
    for row in range(len(MODES))
)
BLOCK_SPECS = tuple(
    (prompt_index, position, mode)
    for prompt_index, row in enumerate(LATIN_ROWS)
    for position, mode in enumerate(row)
)

# Three equal-shape prompts make prefix-cache exposure a Latin-square nuisance
# factor: every mode appears once cold, once second, and once third.
PROMPTS = (
    """A distributed language-model service shares an interconnect between
tensor-parallel decode collectives and background KV movement. Explain how a
research prototype should measure interference without assuming that average
latency represents tail latency. Discuss token timing, transfer completion,
correctness, and an absolute service deadline. Give a concrete, concise
analysis suitable for an experimental notebook.""",
    """A multi-node inference engine runs tensor parallel generation while a
cache layer moves attention state over the same fabric. Explain how a research
prototype can isolate contention without treating throughput alone as proof.
Discuss first-token latency, per-output-token timing, byte correctness, and a
fixed completion boundary. Give a concrete, concise analysis suitable for an
experimental notebook.""",
    """An accelerator cluster executes collective communication for decoding
at the same time that a storage component transfers model state between
nodes. Explain how a research prototype can compare admission policies without
hiding failed work after generation. Discuss TTFT, TPOT tails, exact output,
and deadline completion. Give a concrete, concise analysis suitable for an
experimental notebook.""",
)
WARMUP_PROMPT = (
    "Warm up a deterministic distributed inference request in one short sentence."
)


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def schedule_object_indices(
    mode: str,
    token_index: int,
    *,
    pair_index: int,
) -> tuple[int, ...]:
    """Return rank-local registered-object indices for one token trigger."""

    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if isinstance(token_index, bool) or not isinstance(token_index, int):
        raise ValueError("token_index must be an int")
    if not 0 <= token_index < TOKENS:
        raise ValueError(f"token_index must be in 0..{TOKENS - 1}")
    if isinstance(pair_index, bool) or not isinstance(pair_index, int):
        raise ValueError("pair_index must be an int")
    if not 0 <= pair_index < PAIR_COUNT:
        raise ValueError("pair_index must be in 0..3")
    if mode == "fg_only":
        return ()
    if mode == "lmcache_greedy":
        return tuple(range(REQUESTS * CHUNKS_PER_REQUEST)) if token_index == 0 else ()
    if token_index not in PULSE_TOKENS:
        return ()
    group_index = PULSE_TOKENS.index(token_index)
    chunks = (2 * group_index, 2 * group_index + 1)
    return tuple(
        request * CHUNKS_PER_REQUEST + chunk
        for chunk in chunks
        for request in range(REQUESTS)
    )


def validate_frozen_schedule() -> None:
    if tuple(compiled.EXPECTED_PULSE_TOKENS) != PULSE_TOKENS:
        raise RuntimeError("compiled group2 pulse tokens changed")
    if compiled.pilot.DEADLINE_NS != ABSOLUTE_DEADLINE_NS:
        raise RuntimeError("compiled absolute deadline changed")
    if compiled.pilot.START_LAG_CAP_NS != START_LAG_CAP_NS:
        raise RuntimeError("compiled start-lag cap changed")
    for pair in range(PAIR_COUNT):
        flattened = tuple(
            index
            for token in range(TOKENS)
            for index in schedule_object_indices(
                "tempo_group2", token, pair_index=pair
            )
        )
        if sorted(flattened) != list(range(REQUESTS * CHUNKS_PER_REQUEST)):
            raise RuntimeError(f"group2 rank-local object coverage changed for pair {pair}")


def load_frozen_plan(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("artifact must contain an object")
        active_profile, active_plan = compiled.load_group2_experiment_artifact(payload)
        _, runtime_plan = group2._adapt_group2_plan(active_profile, active_plan)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid frozen group2 artifact: {exc}") from exc
    if active_plan.signature != EXPECTED_PLAN_SIGNATURE:
        raise ValueError("frozen group2 plan signature changed")
    if runtime_plan.signature != EXPECTED_PLAN_SIGNATURE:
        raise ValueError("runtime group2 plan signature changed")
    if payload.get("artifact_signature_sha256") != EXPECTED_ARTIFACT_SIGNATURE:
        raise ValueError("frozen group2 artifact signature changed")
    if tuple(payload.get("expected_width4_pulse_tokens", ())) != PULSE_TOKENS:
        raise ValueError("frozen group2 artifact pulse tokens changed")
    return payload, active_plan.signature


def iter_sse_chunks(
    lines: Iterable[bytes | str],
    *,
    now_ns: Callable[[], int] = time.perf_counter_ns,
) -> Iterator[tuple[list[int], str, int]]:
    """Yield token-id deltas, text deltas, and client arrival timestamps."""

    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        payload = json.loads(data)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        token_ids = choice.get("token_ids") or []
        if not isinstance(token_ids, list):
            raise ValueError("stream token_ids must be a list")
        yield [int(token_id) for token_id in token_ids], str(choice.get("text", "")), now_ns()


def request_completion(
    *,
    api_host: str,
    api_port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
    on_started: Callable[[int], None] | None = None,
    on_token: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    connection = http.client.HTTPConnection(api_host, api_port, timeout=timeout_s)
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 0,
            "ignore_eos": True,
            "stream": True,
            "return_token_ids": True,
        }
    )
    started_ns = time.perf_counter_ns()
    if on_started is not None:
        on_started(started_ns)
    try:
        connection.request(
            "POST",
            "/v1/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {response.status}: {detail}")
        token_ids: list[int] = []
        token_arrival_ns: list[int] = []
        text_parts: list[str] = []
        for delta_ids, delta_text, arrived_ns in iter_sse_chunks(response):
            text_parts.append(delta_text)
            for token_id in delta_ids:
                token_ids.append(token_id)
                token_arrival_ns.append(arrived_ns)
                if on_token is not None:
                    on_token(token_id, arrived_ns)
        finished_ns = time.perf_counter_ns()
    finally:
        connection.close()
    if len(token_ids) != max_tokens:
        raise RuntimeError(
            f"vLLM returned {len(token_ids)} token ids, expected {max_tokens}"
        )
    digest = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "request_started_ns": started_ns,
        "token_arrival_ns": token_arrival_ns,
        "finished_ns": finished_ns,
        "token_ids": token_ids,
        "output_token_sha256": digest,
        "output_text_sha256": hashlib.sha256(
            "".join(text_parts).encode("utf-8")
        ).hexdigest(),
    }


def _set_rank_environment() -> None:
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]


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
        raise RuntimeError("requires node-major sidecar ranks 0..3 and 4..7")
    return hosts


def _expected_byte(block: int, request: int, chunk: int, pair: int) -> int:
    return 1 + ((block * 37 + request * 11 + chunk * 5 + pair * 3) % 251)


def _warm_channel(
    torch: Any,
    dist: Any,
    *,
    channel: Any,
    objects: list[Any],
    rank: int,
    pair_index: int,
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
    error = None
    local_ok = True
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
    statuses: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(statuses, {"ok": local_ok, "error": error})
    if not all(bool(item["ok"]) for item in statuses):
        raise RuntimeError(f"LMCache/NIXL sidecar warmup failed: {statuses}")
    dist.barrier()
    return {"object_indices": list(selected), "noncontiguous": True, "verified": True}


def _next_control_event(
    events: queue.Queue[tuple[str, Any]],
    *,
    timeout_s: float,
) -> tuple[str, Any]:
    try:
        return events.get(timeout=timeout_s)
    except queue.Empty as exc:
        raise RuntimeError("timed out waiting for the streaming vLLM client") from exc


def _run_block(
    torch: Any,
    dist: Any,
    *,
    channel: Any,
    objects: list[Any],
    rank: int,
    device_index: int,
    pair_index: int,
    block_index: int,
    prompt_index: int,
    latin_position: int,
    mode: str,
    api_host: str,
    api_port: int,
    model: str,
    request_timeout_s: float,
) -> dict[str, Any]:
    is_source = rank < RANKS_PER_NODE
    is_receiver = not is_source
    for request in range(REQUESTS):
        for chunk in range(CHUNKS_PER_REQUEST):
            index = request * CHUNKS_PER_REQUEST + chunk
            if is_source:
                objects[index].raw_data.fill_(
                    _expected_byte(block_index, request, chunk, pair_index)
                )
            else:
                objects[index].raw_data.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    transfer_queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
    sentinel = object()
    transfer_records: list[dict[str, Any]] = []
    pending_lock = threading.Lock()
    pending_batches = 0
    peak_pending_batches = 0

    def transfer_worker() -> None:
        nonlocal pending_batches
        torch.cuda.set_device(device_index)
        while True:
            item = transfer_queue.get()
            if item is sentinel:
                return
            token_index, indices, trigger_ns, enqueue_ns = item
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
                    "trigger_ns": trigger_ns,
                    "enqueue_ns": enqueue_ns,
                    "started_ns": started_ns,
                    "finished_ns": finished_ns,
                    "control_delivery_lag_ms": (enqueue_ns - trigger_ns) / 1_000_000.0,
                    "descriptor_start_lag_ms": (started_ns - enqueue_ns) / 1_000_000.0,
                    "elapsed_ms": (finished_ns - started_ns) / 1_000_000.0,
                    "error": error,
                }
            )

    worker = None
    if is_source and mode != "fg_only":
        worker = threading.Thread(
            target=transfer_worker,
            name=f"vllm-lmcache-sidecar-{rank}",
        )
        worker.start()

    def enqueue(token_index: int, trigger_ns: int) -> None:
        nonlocal pending_batches, peak_pending_batches
        indices = schedule_object_indices(mode, token_index, pair_index=pair_index)
        if not is_source or not indices:
            return
        enqueue_ns = time.perf_counter_ns()
        with pending_lock:
            pending_batches += 1
            peak_pending_batches = max(peak_pending_batches, pending_batches)
        transfer_queue.put((token_index, indices, trigger_ns, enqueue_ns))

    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    client_thread = None
    if rank == 0:
        def client_target() -> None:
            try:
                result = request_completion(
                    api_host=api_host,
                    api_port=api_port,
                    model=model,
                    prompt=PROMPTS[prompt_index],
                    max_tokens=TOKENS,
                    timeout_s=request_timeout_s,
                    on_started=lambda value: events.put(("started", value)),
                    on_token=lambda token_id, arrived_ns: events.put(
                        ("token", {"token_id": token_id, "arrival_ns": arrived_ns})
                    ),
                )
                events.put(("finished", result))
            except BaseException as exc:
                events.put(("error", f"{type(exc).__name__}: {exc}"))

        client_thread = threading.Thread(
            target=client_target,
            name=f"vllm-stream-client-{block_index}",
        )
        client_thread.start()

    client_result: dict[str, Any] | None = None
    token_arrivals: list[int] = []
    request_started_ns = 0
    try:
        control: list[Any] = [None]
        if rank == 0:
            kind, value = _next_control_event(events, timeout_s=request_timeout_s)
            control[0] = {"kind": kind, "value": value}
        dist.broadcast_object_list(control, src=0)
        if control[0]["kind"] != "started":
            raise RuntimeError(f"vLLM client failed before request start: {control[0]}")
        request_started_ns = int(control[0]["value"])
        enqueue(0, request_started_ns)

        for token_index in range(TOKENS):
            control = [None]
            if rank == 0:
                kind, value = _next_control_event(events, timeout_s=request_timeout_s)
                control[0] = {"kind": kind, "value": value}
            dist.broadcast_object_list(control, src=0)
            if control[0]["kind"] != "token":
                raise RuntimeError(
                    f"vLLM client failed before token {token_index}: {control[0]}"
                )
            arrived_ns = int(control[0]["value"]["arrival_ns"])
            token_arrivals.append(arrived_ns)
            if mode == "tempo_group2":
                enqueue(token_index, arrived_ns)

        control = [None]
        if rank == 0:
            kind, value = _next_control_event(events, timeout_s=request_timeout_s)
            control[0] = {"kind": kind, "value": value}
        dist.broadcast_object_list(control, src=0)
        if control[0]["kind"] != "finished":
            raise RuntimeError(f"vLLM client did not finish cleanly: {control[0]}")
        client_result = control[0]["value"] if rank == 0 else None
    finally:
        if worker is not None:
            transfer_queue.put(sentinel)
            worker.join()
        if client_thread is not None:
            client_thread.join()

    background_end_ns = max(
        (int(record["finished_ns"]) for record in transfer_records),
        default=request_started_ns,
    )
    foreground_end_ns = token_arrivals[-1]
    absolute_deadline_ns = request_started_ns + ABSOLUTE_DEADLINE_NS
    for record in transfer_records:
        token = int(record["scheduled_token"])
        if mode == "lmcache_greedy":
            window_end_ns = token_arrivals[0]
        else:
            window_end_ns = (
                token_arrivals[token + 1]
                if token + 1 < len(token_arrivals)
                else foreground_end_ns
            )
        record["started_within_trigger_window"] = (
            int(record["started_ns"]) >= int(record["trigger_ns"])
            and int(record["started_ns"]) <= window_end_ns
        )
        record["finished_by_absolute_deadline"] = (
            int(record["finished_ns"]) <= absolute_deadline_ns
        )

    dist.barrier()
    verified_objects = 0
    receiver_zero_objects = 0
    if is_receiver:
        for request in range(REQUESTS):
            for chunk in range(CHUNKS_PER_REQUEST):
                index = request * CHUNKS_PER_REQUEST + chunk
                if mode == "fg_only":
                    if bool(torch.all(objects[index].raw_data == 0).item()):
                        receiver_zero_objects += 1
                else:
                    expected = _expected_byte(
                        block_index, request, chunk, pair_index
                    )
                    if bool(torch.all(objects[index].raw_data == expected).item()):
                        verified_objects += 1
    dist.barrier()

    expected_indices = tuple(
        index
        for token in range(TOKENS)
        for index in schedule_object_indices(mode, token, pair_index=pair_index)
    )
    actual_indices = tuple(
        index for record in transfer_records for index in record["object_indices"]
    )
    errors = [record["error"] for record in transfer_records if record["error"]]
    completed_objects = sum(int(record["completed_objects"]) for record in transfer_records)
    expected_objects = len(expected_indices)
    expected_batch_calls = sum(
        bool(schedule_object_indices(mode, token, pair_index=pair_index))
        for token in range(TOKENS)
    )
    schedule_adherence = all(
        bool(record["started_within_trigger_window"]) for record in transfer_records
    )
    deadline_met = all(
        bool(record["finished_by_absolute_deadline"]) for record in transfer_records
    )
    lag_cap_met = all(
        round(float(record["descriptor_start_lag_ms"]) * 1_000_000)
        <= START_LAG_CAP_NS
        for record in transfer_records
    )
    local_correct = not errors
    if is_source:
        local_correct = (
            local_correct
            and completed_objects == expected_objects
            and sorted(actual_indices) == sorted(expected_indices)
            and len(transfer_records) == expected_batch_calls
        )
    elif mode == "fg_only":
        local_correct = local_correct and receiver_zero_objects == REQUESTS * CHUNKS_PER_REQUEST
    else:
        local_correct = local_correct and verified_objects == expected_objects

    if rank == 0:
        assert client_result is not None
        arrivals = [int(value) for value in client_result["token_arrival_ns"]]
        inter_token_ms = [
            (arrivals[index] - arrivals[index - 1]) / 1_000_000.0
            for index in range(1, len(arrivals))
        ]
        client_metrics: dict[str, Any] | None = {
            "request_started_ns": int(client_result["request_started_ns"]),
            "finished_ns": int(client_result["finished_ns"]),
            "token_arrival_ns": arrivals,
            "token_ids": [int(value) for value in client_result["token_ids"]],
            "output_token_sha256": client_result["output_token_sha256"],
            "output_text_sha256": client_result["output_text_sha256"],
            "ttft_ms": (arrivals[0] - request_started_ns) / 1_000_000.0,
            "tpot_p50_ms": statistics.median(inter_token_ms),
            "tpot_p99_ms": percentile(inter_token_ms, 0.99),
            "tpot_max_ms": max(inter_token_ms),
            "request_e2e_ms": (
                int(client_result["finished_ns"]) - request_started_ns
            ) / 1_000_000.0,
            "generated_tokens": len(arrivals),
        }
    else:
        client_metrics = None

    return {
        "block_index": block_index,
        "prompt_index": prompt_index,
        "latin_position": latin_position,
        "mode": mode,
        "client": client_metrics,
        "background_batch_calls": len(transfer_records),
        "expected_background_batch_calls": expected_batch_calls if is_source else 0,
        "background_completed_bytes": completed_objects * CHUNK_BYTES,
        "receiver_verified_bytes": verified_objects * CHUNK_BYTES,
        "expected_source_bytes": expected_objects * CHUNK_BYTES if is_source else 0,
        "expected_receive_bytes": expected_objects * CHUNK_BYTES if is_receiver else 0,
        "background_finish_from_request_start_ms": (
            background_end_ns - request_started_ns
        ) / 1_000_000.0,
        "post_foreground_drain_ms": max(
            0.0, (background_end_ns - foreground_end_ns) / 1_000_000.0
        ),
        "peak_pending_batches": peak_pending_batches,
        "max_control_delivery_lag_ms": max(
            (float(record["control_delivery_lag_ms"]) for record in transfer_records),
            default=0.0,
        ),
        "max_descriptor_start_lag_ms": max(
            (float(record["descriptor_start_lag_ms"]) for record in transfer_records),
            default=0.0,
        ),
        "schedule_start_adherence_met": schedule_adherence,
        "absolute_service_deadline_ns": ABSOLUTE_DEADLINE_NS,
        "absolute_service_deadline_met": deadline_met,
        "start_lag_cap_ns": START_LAG_CAP_NS,
        "start_lag_cap_met": lag_cap_met,
        "transfer_errors": errors,
        "transfer_records": transfer_records,
        "correctness_met": local_correct,
    }


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != WORLD_SIZE or sorted(item["rank"] for item in records) != list(
        range(WORLD_SIZE)
    ):
        raise ValueError("records must contain exact ranks 0..7")
    ordered = sorted(records, key=lambda item: int(item["rank"]))
    config = ordered[0]["config"]
    if any(item["config"] != config for item in ordered):
        raise ValueError("rank configs differ")
    expected_sequence = [mode for _, _, mode in BLOCK_SPECS]
    for item in ordered:
        if [block["mode"] for block in item["blocks"]] != expected_sequence:
            raise ValueError("rank block sequences differ")

    blocks: list[dict[str, Any]] = []
    mode_samples: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"ttft": [], "tpot_p50": [], "tpot_p99": [], "e2e": [], "finish": []}
    )
    for block_index, (prompt_index, latin_position, mode) in enumerate(BLOCK_SPECS):
        rank_blocks = [item["blocks"][block_index] for item in ordered]
        source_blocks = rank_blocks[:RANKS_PER_NODE]
        receiver_blocks = rank_blocks[RANKS_PER_NODE:]
        client = rank_blocks[0]["client"]
        if not isinstance(client, dict):
            raise ValueError("rank zero client metrics are missing")
        expected_bytes = (
            0
            if mode == "fg_only"
            else PAIR_COUNT * REQUESTS * CHUNKS_PER_REQUEST * CHUNK_BYTES
        )
        completed = sum(int(block["background_completed_bytes"]) for block in source_blocks)
        verified = sum(int(block["receiver_verified_bytes"]) for block in receiver_blocks)
        errors = [error for block in rank_blocks for error in block["transfer_errors"]]
        distribution_ok = (
            all(int(block["background_completed_bytes"]) == (0 if mode == "fg_only" else REQUESTS * KV_BYTES_PER_RANK) for block in source_blocks)
            and all(int(block["receiver_verified_bytes"]) == (0 if mode == "fg_only" else REQUESTS * KV_BYTES_PER_RANK) for block in receiver_blocks)
        )
        correct = (
            all(bool(block["correctness_met"]) for block in rank_blocks)
            and not errors
            and distribution_ok
            and completed == expected_bytes
            and verified == expected_bytes
            and int(client["generated_tokens"]) == TOKENS
        )
        adherence = all(
            bool(block["schedule_start_adherence_met"]) for block in source_blocks
        )
        deadline = all(
            bool(block["absolute_service_deadline_met"]) for block in source_blocks
        )
        lag_cap = all(bool(block["start_lag_cap_met"]) for block in source_blocks)
        drain_ms = max(float(block["post_foreground_drain_ms"]) for block in source_blocks)
        finish_ms = max(
            float(block["background_finish_from_request_start_ms"])
            for block in source_blocks
        )
        block_result = {
            "block_index": block_index,
            "prompt_index": prompt_index,
            "latin_position": latin_position,
            "mode": mode,
            "ttft_ms": float(client["ttft_ms"]),
            "tpot_p50_ms": float(client["tpot_p50_ms"]),
            "tpot_p99_ms": float(client["tpot_p99_ms"]),
            "tpot_max_ms": float(client["tpot_max_ms"]),
            "request_e2e_ms": float(client["request_e2e_ms"]),
            "generated_tokens": int(client["generated_tokens"]),
            "output_token_sha256": client["output_token_sha256"],
            "expected_background_bytes": expected_bytes,
            "background_completed_bytes": completed,
            "receiver_verified_bytes": verified,
            "background_finish_from_request_start_ms": finish_ms,
            "post_foreground_drain_ms": drain_ms,
            "schedule_start_adherence_met": adherence,
            "absolute_service_deadline_met": deadline,
            "start_lag_cap_met": lag_cap,
            "max_control_delivery_lag_ms": max(
                float(block["max_control_delivery_lag_ms"]) for block in source_blocks
            ),
            "max_descriptor_start_lag_ms": max(
                float(block["max_descriptor_start_lag_ms"]) for block in source_blocks
            ),
            "transfer_errors": errors,
            "correctness_met": correct,
        }
        blocks.append(block_result)
        bucket = mode_samples[mode]
        bucket["ttft"].append(block_result["ttft_ms"])
        bucket["tpot_p50"].append(block_result["tpot_p50_ms"])
        bucket["tpot_p99"].append(block_result["tpot_p99_ms"])
        bucket["e2e"].append(block_result["request_e2e_ms"])
        bucket["finish"].append(block_result["background_finish_from_request_start_ms"])

    prompt_equivalence: dict[str, bool] = {}
    for prompt_index in range(len(PROMPTS)):
        hashes = {
            block["output_token_sha256"]
            for block in blocks
            if block["prompt_index"] == prompt_index
        }
        prompt_equivalence[str(prompt_index)] = len(hashes) == 1
    output_equivalence = all(prompt_equivalence.values())
    modes: dict[str, Any] = {}
    for mode in MODES:
        bucket = mode_samples[mode]
        modes[mode] = {
            "replicates": len(bucket["ttft"]),
            "ttft_p50_ms": statistics.median(bucket["ttft"]),
            "ttft_p99_ms": percentile(bucket["ttft"], 0.99),
            "tpot_p50_of_blocks_ms": statistics.median(bucket["tpot_p50"]),
            "tpot_p99_of_blocks_ms": percentile(bucket["tpot_p99"], 0.99),
            "request_e2e_p50_ms": statistics.median(bucket["e2e"]),
            "background_finish_p50_ms": statistics.median(bucket["finish"]),
            "correctness_met": all(
                block["correctness_met"] for block in blocks if block["mode"] == mode
            ),
        }

    overall_correct = output_equivalence and all(
        bool(block["correctness_met"]) for block in blocks
    )
    candidate_blocks = [block for block in blocks if block["mode"] == "tempo_group2"]
    candidate_adherence = all(
        bool(block["schedule_start_adherence_met"]) for block in candidate_blocks
    )
    candidate_deadline = all(
        bool(block["absolute_service_deadline_met"]) for block in candidate_blocks
    )
    candidate_no_drain = all(
        float(block["post_foreground_drain_ms"]) == 0.0 for block in candidate_blocks
    )
    candidate_lag_cap = all(bool(block["start_lag_cap_met"]) for block in candidate_blocks)
    if not overall_correct:
        outcome = "invalid_output_or_transfer_correctness"
    elif not candidate_adherence:
        outcome = "kill_external_token_trigger_adherence_miss"
    elif not candidate_deadline:
        outcome = "kill_absolute_service_deadline_miss"
    elif not candidate_no_drain:
        outcome = "kill_post_foreground_drain"
    elif not candidate_lag_cap:
        outcome = "valid_service_but_frozen_lag_cap_not_met"
    else:
        outcome = "valid_component_screen_requires_performance_comparison"
    return {
        "schema_version": "tempo-vllm-tp8-lmcache-sidecar-screen-1",
        "evidence_state": "real_vllm_foreground_official_lmcache_component_sidecar",
        "claim_scope": "research_component_screen_not_end_to_end_kv_connector",
        "honesty_boundary": (
            "NixlChannel moves registered GPU buffers with the frozen KV traffic "
            "geometry; those buffers are not vLLM-owned live KV-cache tensors"
        ),
        "foreground": {
            "runtime": "vLLM 0.26 native multi-node multiprocess executor",
            "expected_tensor_parallel_size": 8,
            "measurement": "streaming /v1/completions client token arrivals",
            "control_plane": "Gloo token broadcast applied identically in every mode",
        },
        "background": {
            "name": "LMCache NixlChannel",
            "commit": official.LMCACHE_COMMIT,
            "component": "lazy_init_peer_connection + batched_write",
            "backend": "NIXL UCX",
            "proxy": False,
            "connector_drop_in_used": False,
            "connector_drop_in_reason": (
                "one TP8 engine has no distinct remote producer/consumer engine, "
                "so its connector cannot create competing inter-node KV movement"
            ),
        },
        "world_size": WORLD_SIZE,
        "nodes": NODES,
        "pairing": [[rank, rank + RANKS_PER_NODE] for rank in range(PAIR_COUNT)],
        "block_sequence": [mode for _, _, mode in BLOCK_SPECS],
        "latin_rows": [list(row) for row in LATIN_ROWS],
        "config": config,
        "frozen_group2": {
            "plan_signature": EXPECTED_PLAN_SIGNATURE,
            "artifact_signature_sha256": EXPECTED_ARTIFACT_SIGNATURE,
            "pulse_tokens": list(PULSE_TOKENS),
            "runtime_width": 8,
            "absolute_deadline_ns": ABSOLUTE_DEADLINE_NS,
            "start_lag_cap_ns": START_LAG_CAP_NS,
            "retuned": False,
        },
        "blocks": blocks,
        "modes": modes,
        "output_equivalence_by_prompt": prompt_equivalence,
        "output_equivalence_met": output_equivalence,
        "candidate_schedule_adherence_met": candidate_adherence,
        "candidate_absolute_deadline_met": candidate_deadline,
        "candidate_no_post_foreground_drain_met": candidate_no_drain,
        "candidate_start_lag_cap_met": candidate_lag_cap,
        "overall_correctness_met": overall_correct,
        "screen_outcome": outcome,
    }


def _resolve_below_repo(path: Path, repo_root: Path, *, label: str) -> Path:
    candidate = path if path.is_absolute() else repo_root / path
    resolved = candidate.resolve()
    if resolved == repo_root or repo_root not in resolved.parents:
        raise SystemExit(f"{label} must resolve below the repository root")
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "results/lmcache_active_pulse_group2_job_56929977/"
            "active_pulse_group2_plan.json"
        ),
    )
    parser.add_argument("--api-host", required=True)
    parser.add_argument("--api-port", type=int, required=True)
    parser.add_argument(
        "--model",
        default="models/TinyLlama-1.1B-Chat-v1.0",
    )
    parser.add_argument("--nixl-port-base", type=int, default=35100)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    if not 1024 <= args.api_port <= 65535:
        parser.error("api-port must be a valid TCP port")
    if not 1024 <= args.nixl_port_base <= 65535 - PAIR_COUNT:
        parser.error("nixl-port-base must leave four valid TCP ports")
    if args.request_timeout_s <= 0:
        parser.error("request-timeout-s must be positive")
    return args


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    args.output_dir = _resolve_below_repo(args.output_dir, repo_root, label="output-dir")
    args.plan = _resolve_below_repo(args.plan, repo_root, label="plan")
    validate_frozen_schedule()
    _, plan_signature = load_frozen_plan(args.plan)
    if plan_signature != EXPECTED_PLAN_SIGNATURE:
        raise SystemExit("frozen plan signature mismatch")

    _set_rank_environment()
    try:
        import torch
        import torch.distributed as dist
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch with CUDA and Gloo is required") from exc
    if not torch.cuda.is_available() or not dist.is_gloo_available():
        raise SystemExit("CUDA and Gloo are required")
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise SystemExit("WORLD_SIZE must be exactly 8")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    visible_devices = torch.cuda.device_count()
    if visible_devices not in (1, RANKS_PER_NODE):
        raise SystemExit("sidecar rank must see one GPU or all four local GPUs")
    device_index = 0 if visible_devices == 1 else local_rank
    torch.cuda.set_device(device_index)
    dist.init_process_group("gloo")
    try:
        hosts = _validate_topology(dist, rank, local_rank)
        NixlChannel, TensorMemoryObj, MemoryObjMetadata, MemoryFormat = (
            official._load_official_lmcache(repo_root)
        )
        microburst.install_microburst_geometry()
        backing, buffer, objects, index_by_address = epoch._make_chunk_memory(
            torch,
            TensorMemoryObj,
            MemoryObjMetadata,
            MemoryFormat,
            requests=REQUESTS,
            chunk_bytes=CHUNK_BYTES,
        )
        pair_index = rank % RANKS_PER_NODE
        is_source = rank < RANKS_PER_NODE
        channel = NixlChannel(
            async_mode=False,
            role="sender" if is_source else "receiver",
            buffer_ptr=buffer.data_ptr(),
            buffer_size=buffer.numel(),
            align_bytes=CHUNK_BYTES,
            tp_rank=local_rank,
            peer_init_url=(
                None if is_source else f"*:{args.nixl_port_base + pair_index}"
            ),
            backends=["UCX"],
            device=f"cuda:{device_index}",
        )
        epoch._install_descriptor_index_shim(channel, index_by_address)
        peer_rank = rank + RANKS_PER_NODE if is_source else rank - RANKS_PER_NODE
        dist.barrier()
        if is_source:
            channel.lazy_init_peer_connection(
                local_id=f"rank-{rank}",
                peer_id=f"rank-{peer_rank}",
                peer_init_url=f"{hosts[peer_rank]}:{args.nixl_port_base + pair_index}",
            )
        dist.barrier()
        if not channel.remote_xfer_handler_exists(f"rank-{peer_rank}"):
            raise RuntimeError("LMCache/NIXL peer handshake did not install a handler")
        warmup = _warm_channel(
            torch,
            dist,
            channel=channel,
            objects=objects,
            rank=rank,
            pair_index=pair_index,
        )

        warmup_status: list[Any] = [None]
        if rank == 0:
            try:
                warm = request_completion(
                    api_host=args.api_host,
                    api_port=args.api_port,
                    model=args.model,
                    prompt=WARMUP_PROMPT,
                    max_tokens=8,
                    timeout_s=args.request_timeout_s,
                )
                warmup_status[0] = {
                    "ok": True,
                    "generated_tokens": len(warm["token_ids"]),
                }
            except BaseException as exc:
                warmup_status[0] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        dist.broadcast_object_list(warmup_status, src=0)
        if not warmup_status[0].get("ok"):
            raise RuntimeError(f"vLLM warmup failed: {warmup_status[0]}")
        dist.barrier()

        blocks = [
            _run_block(
                torch,
                dist,
                channel=channel,
                objects=objects,
                rank=rank,
                device_index=device_index,
                pair_index=pair_index,
                block_index=block_index,
                prompt_index=prompt_index,
                latin_position=latin_position,
                mode=mode,
                api_host=args.api_host,
                api_port=args.api_port,
                model=args.model,
                request_timeout_s=args.request_timeout_s,
            )
            for block_index, (prompt_index, latin_position, mode) in enumerate(
                BLOCK_SPECS
            )
        ]
        config = {
            "model": args.model,
            "api_host": args.api_host,
            "api_port": args.api_port,
            "requests": REQUESTS,
            "tokens_per_request": TOKENS,
            "chunks_per_rank_request": CHUNKS_PER_REQUEST,
            "chunk_bytes": CHUNK_BYTES,
            "kv_bytes_per_rank_request": KV_BYTES_PER_RANK,
            "nixl_port_base": args.nixl_port_base,
            "replicates_per_mode": len(LATIN_ROWS),
            "plan_path": str(args.plan.relative_to(repo_root)),
            "plan_signature": plan_signature,
            "lmcache_commit": official.LMCACHE_COMMIT,
            "nixl_version": importlib.metadata.version("nixl"),
            "nixl_backend": "UCX",
            "control_process_group": "Gloo",
            "foreground_expected_tensor_parallel_size": 8,
            "prefix_cache_exposure_balanced_by_latin_prompt_rows": True,
            "unmeasured_nixl_warmup": warmup,
            "unmeasured_vllm_warmup": warmup_status[0],
        }
        rank_record = {
            "schema_version": "tempo-vllm-tp8-lmcache-sidecar-rank-1",
            "rank": rank,
            "local_rank": local_rank,
            "device_index": device_index,
            "hostname": hosts[rank],
            "config": config,
            "blocks": blocks,
        }
        gathered = [None] * WORLD_SIZE if rank == 0 else None
        dist.gather_object(rank_record, gathered, dst=0)
        final_status: list[Any] = [None]
        if rank == 0:
            try:
                assert gathered is not None
                args.output_dir.mkdir(parents=True, exist_ok=True)
                for item in gathered:
                    (args.output_dir / f"rank_{int(item['rank'])}.json").write_text(
                        json.dumps(item, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                result = aggregate_rank_records(gathered)
                result_path = args.output_dir / "result.json"
                result_path.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                final_status[0] = {
                    "ok": bool(result["overall_correctness_met"]),
                    "output": str(result_path),
                    "screen_outcome": result["screen_outcome"],
                }
            except BaseException as exc:
                final_status[0] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        dist.broadcast_object_list(final_status, src=0)
        dist.barrier()
        if not isinstance(final_status[0], dict) or not final_status[0].get("ok"):
            raise RuntimeError(f"vLLM/LMCache sidecar screen failed: {final_status[0]}")
        if rank == 0:
            print(json.dumps(final_status[0], sort_keys=True))
        del backing
        # Do not call receiver close(): this pinned commit can join its blocking
        # listener before the ZMQ context is terminated. Process exit cleans up.
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
