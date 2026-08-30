#!/usr/bin/env python3
"""Fail-closed revision of the TP16 single-flight hybrid boost campaign.

This add-only revision keeps v5's data plane and audited hook, but orders all
source worker entries before the HTTP request, measures the gate through the
next engine step, validates rank/block identities and duplicate trace events,
and requires a paired LMCache service-makespan win.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np

from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old


CONTRACT_ID = "tp16-single-flight-hybrid-boost-v6"
RESULT_SCHEMA = "tempo-vllm-tp16-single-flight-hybrid-result-6"


def _expected_contract() -> dict[str, Any]:
    payload = old._expected_contract()
    payload["schema_version"] = "tempo-tp16-single-flight-hybrid-contract-6"
    payload["contract_id"] = CONTRACT_ID
    payload["transfer"]["worker_entry_precedes_http_request"] = True
    payload["boost"]["promotion_gate_clock"] = (
        "hook_output_enqueued_to_next_engine_step_enter"
    )
    payload["campaign"]["paired_lmcache_win_required"] = True
    payload["campaign"]["rank_block_identity_fail_closed"] = True
    payload["campaign"]["duplicate_trace_ids_fail_closed"] = True
    return payload


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_contract():
        raise ValueError("TP16 hybrid boost v6 contract changed")
    return payload


def _transfer_worker(
    *,
    channel: Any,
    obj: Any,
    receiver_id: str,
    mode: str,
    boost: threading.Event,
    entered: threading.Event,
    done: threading.Event,
    state: dict[str, Any],
) -> None:
    state["started_ns"] = time.perf_counter_ns()
    entered.set()
    try:
        spec = {
            "receiver_id": receiver_id,
            "remote_indexes": np.asarray([0], dtype=np.uint64),
        }
        if mode == old.LMCACHE:
            state["completed"] = int(
                channel.batched_write(objects=[obj], transfer_spec=spec)
            )
        elif mode == old.TEMPO:
            state.update(channel.tempo_adaptive_write([obj], spec, boost))
        else:
            raise RuntimeError(f"worker received invalid mode {mode}")
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        state["finished_ns"] = time.perf_counter_ns()
        done.set()


def _run_block(
    torch: Any,
    dist: Any,
    *,
    channel: Any,
    obj: Any,
    rank: int,
    pair: int,
    block_index: int,
    prompt_index: int,
    mode: str,
    args: Any,
) -> dict[str, Any]:
    old.protocol.install_generic_release_protocol()
    source = rank < old.SOURCE_COUNT
    receiver = not source
    expected = 1 + ((block_index * 37 + pair * 3) % 251)
    obj.raw_data.fill_(expected if source and mode != old.FG else 0)
    torch.cuda.synchronize()
    dist.barrier()

    marked = mode == old.TEMPO
    prefix = "tempo-scout" if marked else "control"
    caller_id = (
        f"{prefix}-{args.allocation_id}-c{args.campaign_index}-b{block_index}-{mode}"
    )
    events: queue.Queue[tuple[bool, Any]] = queue.Queue()
    listener = connection = ready = client = None
    if rank == 0 and marked:
        listener = old.gate.GateListener(
            old.gate.GateConfig(
                args.quiescence_socket, args.quiescence_trace, timeout_s=30.0
            )
        )
        listener.open()

    start_signal = torch.tensor([block_index], dtype=torch.int64, device="cpu")
    dist.broadcast(start_signal, src=0)
    local_origin_ns = time.perf_counter_ns()
    if int(start_signal.item()) != block_index:
        raise RuntimeError("hybrid v6 request-start control changed")

    boost = threading.Event()
    entered = threading.Event()
    done = threading.Event()
    state: dict[str, Any] = {
        "started_ns": 0,
        "finished_ns": 0,
        "completed": 0,
        "polls": 0,
        "low_priority_sleeps": 0,
        "boost_polls": 0,
        "yields": 0,
        "error": None,
    }
    worker = None
    if source and mode != old.FG:
        worker = threading.Thread(
            target=_transfer_worker,
            kwargs={
                "channel": channel,
                "obj": obj,
                "receiver_id": f"rank-{rank + old.RECEIVER_OFFSET}",
                "mode": mode,
                "boost": boost,
                "entered": entered,
                "done": done,
                "state": state,
            },
            name=f"hybrid-v6-transfer-rank{rank}-block{block_index}",
            daemon=True,
        )
        worker.start()

    if source and mode != old.FG and not entered.wait(5.0):
        raise RuntimeError("hybrid v6 source worker did not enter transfer call")
    entered_status = torch.tensor(
        [1 if not source or mode == old.FG or entered.is_set() else 0],
        dtype=torch.int64,
        device="cpu",
    )
    dist.all_reduce(entered_status, op=dist.ReduceOp.MIN)
    if int(entered_status.item()) != 1:
        raise RuntimeError("hybrid v6 prelaunch worker-entry handshake failed")

    client_started_from_origin_ns = 0
    client_finished_from_origin_ns = 0
    if rank == 0:
        client_started_from_origin_ns = time.perf_counter_ns() - local_origin_ns
        client = threading.Thread(
            target=old.bulk._request_thread,
            kwargs={
                "events": events,
                "args": args,
                "prompt": old.base.PROMPTS[prompt_index],
                "caller_id": caller_id,
                "tokens": old.TOKENS,
            },
            name=f"hybrid-v6-http-{block_index}",
        )
        client.start()

    boost_wait_timed_out = False
    release_error = None
    release_payload = None
    boost_hold_ns = 0
    if marked:
        if rank == 0:
            connection = listener.accept()
            ready = connection.event
            if caller_id != ready.request_id.removeprefix("cmpl-").split("-0-")[0]:
                if caller_id not in ready.request_id:
                    raise RuntimeError("hybrid v6 gate ready identity mismatch")
        gate_signal = torch.tensor(
            [int(ready.event_id) if rank == 0 else -1],
            dtype=torch.int64,
            device="cpu",
        )
        hold_started_ns = time.perf_counter_ns() if rank == 0 else 0
        dist.broadcast(gate_signal, src=0)
        if int(gate_signal.item()) < 0:
            raise RuntimeError("hybrid v6 gate signal changed")
        if source:
            boost.set()
            boost_wait_timed_out = not done.wait(old.BOOST_WAIT_CAP_MS / 1000.0)
        local_done = torch.tensor(
            [0 if source and boost_wait_timed_out else 1],
            dtype=torch.int64,
            device="cpu",
        )
        dist.all_reduce(local_done, op=dist.ReduceOp.MIN)
        all_sources_done = bool(int(local_done.item()))
        if all_sources_done:
            dist.barrier()
            verified_at_gate = (
                old.BYTES_PER_SOURCE
                if receiver and bool(torch.all(obj.raw_data == expected).item())
                else 0
            )
        else:
            verified_at_gate = 0
        gate_status = torch.tensor(
            [
                1 if source and done.is_set() else 0,
                int(state["completed"]) if source else 0,
                old._descriptor_count(channel) if source else 0,
                old.BYTES_PER_SOURCE
                if source and state["completed"] == 1
                else 0,
                max(0, int(state["finished_ns"]) - int(state["started_ns"]))
                if source and done.is_set()
                else 0,
                1 if source and state["error"] is not None else 0,
                verified_at_gate,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        gathered = (
            [torch.zeros_like(gate_status) for _ in range(old.WORLD_SIZE)]
            if rank == 0
            else None
        )
        dist.gather(gate_status, gather_list=gathered, dst=0)
        if rank == 0:
            try:
                rows = [tensor.tolist() for tensor in gathered]
                sources = rows[: old.SOURCE_COUNT]
                receivers = rows[old.SOURCE_COUNT :]
                structural = all_sources_done and all(
                    row[0] == 1
                    and row[1] == 1
                    and row[2] == 1
                    and row[3] == old.BYTES_PER_SOURCE
                    and row[5] == 0
                    for row in sources
                ) and all(row[6] == old.BYTES_PER_SOURCE for row in receivers)
                boost_hold_ns = time.perf_counter_ns() - hold_started_ns
                if structural:
                    frame = old.protocol.ReleaseFrame.wave(
                        ready,
                        mode=old.TEMPO,
                        completed_bytes=old.GLOBAL_BYTES,
                        source_elapsed_ns=tuple(int(row[4]) for row in sources),
                        wave_elapsed_ns=boost_hold_ns,
                    )
                else:
                    frame = old.protocol.ReleaseFrame.noop(ready)
                    release_error = (
                        "hybrid v6 boost did not finish exact transfer within cap"
                    )
                connection.release(frame)
                release_payload = frame.to_payload()
            finally:
                if connection is not None:
                    connection.close()
                if listener is not None:
                    listener.close()

    client_control: list[Any] = [None]
    if rank == 0:
        ok, value = events.get(timeout=args.request_timeout_s)
        client_finished_from_origin_ns = time.perf_counter_ns() - local_origin_ns
        client.join(timeout=1.0)
        client_control[0] = {"ok": ok, "value": value}
    dist.broadcast_object_list(client_control, src=0)
    local_foreground_done_ns = time.perf_counter_ns()
    if not client_control[0]["ok"]:
        raise RuntimeError(f"vLLM request failed: {client_control[0]['value']}")

    if source and mode != old.FG:
        if not done.wait(60.0):
            raise RuntimeError("hybrid v6 source transfer did not terminate in 60 seconds")
        worker.join(timeout=1.0)
        if worker.is_alive():
            raise RuntimeError("hybrid v6 source worker remained alive")
    dist.barrier()
    verified = 0
    zero_ok = True
    if receiver:
        if mode == old.FG:
            zero_ok = bool(torch.all(obj.raw_data == 0).item())
        else:
            verified = (
                old.BYTES_PER_SOURCE
                if bool(torch.all(obj.raw_data == expected).item())
                else 0
            )
    dist.barrier()

    elapsed_ns = (
        max(0, int(state["finished_ns"]) - int(state["started_ns"]))
        if source and mode != old.FG
        else 0
    )
    completion_from_origin_ns = (
        max(0, int(state["finished_ns"]) - local_origin_ns)
        if source and mode != old.FG
        else 0
    )
    post_foreground_drain_ns = (
        max(0, int(state["finished_ns"]) - local_foreground_done_ns)
        if source and mode != old.FG
        else 0
    )
    local = {
        "rank": rank,
        "source": source,
        "calls": 1 if source and mode != old.FG else 0,
        "completed": int(state["completed"]) if source else 0,
        "descriptors": old._descriptor_count(channel)
        if source and mode != old.FG
        else 0,
        "bytes": old.BYTES_PER_SOURCE
        if source and state["completed"] == 1
        else 0,
        "elapsed_ns": elapsed_ns,
        "completion_from_origin_ns": completion_from_origin_ns,
        "post_foreground_drain_ns": post_foreground_drain_ns,
        "start_lag_ns": max(0, int(state["started_ns"]) - local_origin_ns)
        if source and mode != old.FG
        else 0,
        "polls": int(state["polls"]),
        "low_priority_sleeps": int(state["low_priority_sleeps"]),
        "boost_polls": int(state["boost_polls"]),
        "yields": int(state["yields"]),
        "boost_wait_timed_out": boost_wait_timed_out,
        "error": state["error"],
    }
    return {
        "block_index": block_index,
        "prompt_index": prompt_index,
        "mode": mode,
        "client": old.scout._client_metrics(client_control[0]["value"])
        if rank == 0
        else None,
        "client_started_from_origin_ns": client_started_from_origin_ns,
        "client_finished_from_origin_ns": client_finished_from_origin_ns,
        "gate_ready": ready.to_payload() if rank == 0 and marked else None,
        "gate_release": {"payload": release_payload, "error": release_error}
        if rank == 0 and marked
        else None,
        "boost_hold_ns": boost_hold_ns,
        "source_call": local,
        "receiver_verified_bytes": verified,
        "receiver_zero_ok": zero_ok,
        "correctness_met": (
            (not marked or release_error is None)
            and (
                not source
                or mode == old.FG
                or state["completed"] == 1
                and state["error"] is None
            )
            and (
                not receiver
                or (zero_ok if mode == old.FG else verified == old.BYTES_PER_SOURCE)
            )
        ),
    }


def _validate_trace(path: Path, expected: list[tuple[str, str]]) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for kind in ("ready", "release", "next_engine_step_enter"):
        kind_rows = [row for row in rows if row.get("kind") == kind]
        ids = [int(row["event_id"]) for row in kind_rows]
        if len(ids) != len(expected) or len(ids) != len(set(ids)):
            raise ValueError(f"hybrid v6 duplicate or missing {kind} event ids")
    validated = old._validate_trace(path, expected)
    readies = {int(row["event_id"]): row for row in rows if row.get("kind") == "ready"}
    releases = {
        int(row["event_id"]): row for row in rows if row.get("kind") == "release"
    }
    nexts = {
        int(row["event_id"]): row
        for row in rows
        if row.get("kind") == "next_engine_step_enter"
    }
    events = []
    for event_id, (_caller, mode) in enumerate(expected):
        ready, release, nxt = readies[event_id], releases[event_id], nexts[event_id]
        events.append(
            {
                "event_id": event_id,
                "request_id": str(ready["request_id"]),
                "mode": mode,
                "fence_ms": (
                    int(ready["fence_finished_ns"]) - int(ready["fence_started_ns"])
                )
                / 1e6,
                "ready_to_release_ms": (
                    int(release["released_ns"]) - int(ready["ready_ns"])
                )
                / 1e6,
                "release_to_next_step_ms": (
                    int(nxt["entered_ns"]) - int(release["released_ns"])
                )
                / 1e6,
                "total_gate_bubble_ms": (
                    int(nxt["entered_ns"]) - int(ready["output_enqueued_ns"])
                )
                / 1e6,
            }
        )
    return {**validated, "events": events}


def _aggregate(records: list[dict[str, Any]], trace: dict[str, Any], args: Any):
    ordered = sorted(records, key=lambda item: int(item["rank"]))
    if len(ordered) != old.WORLD_SIZE:
        raise ValueError("hybrid v6 rank records are incomplete")
    for index, (prompt, mode) in enumerate(old.BLOCKS):
        for rank, item in enumerate(ordered):
            block = item["blocks"][index]
            if (
                int(block.get("block_index", -1)) != index
                or int(block.get("prompt_index", -1)) != prompt
                or block.get("mode") != mode
                or int(block.get("source_call", {}).get("rank", -1)) != rank
            ):
                raise ValueError(
                    f"hybrid v6 block identity mismatch at block {index}, rank {rank}"
                )
    result = old._aggregate(records, trace, args)
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    result["config"]["prelaunch_order"] = "all_source_workers_entered_before_http"
    result["config"]["promotion_gate_clock"] = (
        "hook_output_enqueued_to_next_engine_step_enter"
    )

    trace_by_block = {}
    for event in trace.get("events", []):
        for index, (_prompt, mode) in enumerate(old.BLOCKS):
            if mode == old.TEMPO and f"-b{index}-{old.TEMPO}" in event["request_id"]:
                trace_by_block[index] = event
    for index, block in enumerate(result["blocks"]):
        rank_zero = ordered[0]["blocks"][index]
        foreground_ms = float(rank_zero["client_finished_from_origin_ns"]) / 1e6
        block["foreground_completion_from_start_ms"] = foreground_ms
        block["service_makespan_ms"] = max(
            foreground_ms, block["background_completion_from_start_ms"]
        )
        event = trace_by_block.get(index)
        block["gate_fence_ms"] = event["fence_ms"] if event else 0.0
        block["gate_ready_to_release_ms"] = (
            event["ready_to_release_ms"] if event else 0.0
        )
        block["gate_release_to_next_step_ms"] = (
            event["release_to_next_step_ms"] if event else 0.0
        )
        block["total_gate_bubble_ms"] = (
            event["total_gate_bubble_ms"] if event else 0.0
        )

    by_mode = {
        mode: [block for block in result["blocks"] if block["mode"] == mode]
        for mode in old.MODES
    }
    for mode, blocks in by_mode.items():
        result["mode_metrics"][mode]["service_makespan_p50_ms"] = statistics.median(
            block["service_makespan_ms"] for block in blocks
        )
        result["mode_metrics"][mode]["total_gate_bubble_p50_ms"] = (
            statistics.median(block["total_gate_bubble_ms"] for block in blocks)
        )
        result["mode_metrics"][mode]["total_gate_bubble_max_ms"] = max(
            block["total_gate_bubble_ms"] for block in blocks
        )

    paired = []
    for prompt in range(3):
        fg = next(block for block in by_mode[old.FG] if block["prompt_index"] == prompt)
        baseline = next(
            block for block in by_mode[old.LMCACHE] if block["prompt_index"] == prompt
        )
        tempo = next(
            block for block in by_mode[old.TEMPO] if block["prompt_index"] == prompt
        )
        original = next(row for row in result["paired"] if row["prompt_index"] == prompt)
        paired.append(
            {
                **original,
                "tempo_minus_lmcache_service_makespan_ms": (
                    tempo["service_makespan_ms"] - baseline["service_makespan_ms"]
                ),
                "tempo_minus_fg_service_makespan_ms": (
                    tempo["service_makespan_ms"] - fg["service_makespan_ms"]
                ),
            }
        )
    result["paired"] = paired
    deltas = [row["tempo_minus_lmcache_service_makespan_ms"] for row in paired]
    tempo_bubbles = [block["total_gate_bubble_ms"] for block in by_mode[old.TEMPO]]
    gates = dict(result["candidate_gates"])
    gates.pop("boost_hold_median_le_25ms", None)
    gates.pop("boost_hold_max_le_30ms", None)
    gates.pop("tempo_service_makespan_beats_lmcache", None)
    gates.update(
        {
            "total_gate_bubble_median_le_25ms": statistics.median(tempo_bubbles)
            <= old.BOOST_PROMOTION_MEDIAN_MS,
            "total_gate_bubble_max_le_30ms": max(tempo_bubbles)
            <= old.BOOST_PROMOTION_MAX_MS,
            "tempo_service_makespan_beats_lmcache_paired": (
                statistics.median(deltas) < 0.0
                and sum(delta < 0.0 for delta in deltas) >= 2
            ),
        }
    )
    result["candidate_gates"] = gates
    result["screen_outcome"] = (
        "invalid_correctness_output_or_trace"
        if not result["overall_correctness_met"]
        else "hybrid_candidate_pass"
        if all(gates.values())
        else "hybrid_candidate_revise_or_stop"
    )
    return result


def main() -> None:
    old.CONTRACT_ID = CONTRACT_ID
    old.RESULT_SCHEMA = RESULT_SCHEMA
    old._load_contract = _load_contract
    old._run_block = _run_block
    old._validate_trace = _validate_trace
    old._aggregate = _aggregate
    old.main()


if __name__ == "__main__":
    main()
