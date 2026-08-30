#!/usr/bin/env python3
"""Recursion-safe token-31 nonblocking promotion candidate.

The measured transfer hot path is v6's single prepared 16 MiB descriptor per
source.  At READY this revision only proves that all eight source workers have
observed promotion, then truthfully releases zero completed transfer geometry
so decoding can resume while the promoted workers continue making progress.
"""

from __future__ import annotations

import json
from pathlib import Path
import queue
import statistics
import threading
import time
from typing import Any

from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as async_protocol


CONTRACT_ID = "tp16-single-flight-async-promotion-v8"
RESULT_SCHEMA = "tempo-vllm-tp16-single-flight-async-result-8"
GATE_BUBBLE_MEDIAN_MS = 4.0
GATE_BUBBLE_MAX_MS = 6.0
COMPLETION_REDUCTION_MIN = 0.20
TPOT_P99_RATIO_MAX = 1.10

_ORIGINAL_VALIDATE_TRACE = old._validate_trace
_ORIGINAL_AGGREGATE = old._aggregate
_V6_VALIDATE_TRACE = fixed._validate_trace
_V6_AGGREGATE = fixed._aggregate


def _expected_contract() -> dict[str, Any]:
    payload = fixed._expected_contract()
    payload["schema_version"] = "tempo-tp16-single-flight-async-contract-8"
    payload["contract_id"] = CONTRACT_ID
    payload["boost"].update(
        {
            "target_output_token_index_zero_based": 30,
            "generated_token_count_one_based": 31,
            "completion_wait_inside_gate": False,
            "promotion_armed_sources": 8,
            "release_mode": async_protocol.ASYNC_MODE,
        }
    )
    payload["release"] = {
        "mode": async_protocol.ASYNC_MODE,
        "completed_sources": 0,
        "physical_descriptors": 0,
        "completed_bytes": 0,
        "promotion_armed_sources": 8,
        "decode_resumes_before_transfer_join": True,
    }
    payload["campaign"]["candidate_gates"] = {
        "gate_bubble_p50_ms_le": GATE_BUBBLE_MEDIAN_MS,
        "gate_bubble_max_ms_le": GATE_BUBBLE_MAX_MS,
        "all_tempo_post_foreground_drain_zero": True,
        "paired_service_win_min_prompts": 2,
        "paired_service_delta_median_lt_ms": 0.0,
        "completion_reduction_median_ge_fraction": COMPLETION_REDUCTION_MIN,
        "tempo_tpot_p99_le_lmcache_ratio": TPOT_P99_RATIO_MAX,
    }
    return payload


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_contract():
        raise ValueError("TP16 async promotion v8 contract changed")
    return payload


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
    async_protocol.install_async_release_protocol()
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
        raise RuntimeError("async v8 request-start control changed")

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
            target=fixed._transfer_worker,
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
            name=f"async-v8-transfer-rank{rank}-block{block_index}",
            daemon=True,
        )
        worker.start()

    if source and mode != old.FG and not entered.wait(5.0):
        raise RuntimeError("async v8 source worker did not enter transfer call")
    entered_status = torch.tensor(
        [1 if not source or mode == old.FG or entered.is_set() else 0],
        dtype=torch.int64,
        device="cpu",
    )
    dist.all_reduce(entered_status, op=dist.ReduceOp.MIN)
    if int(entered_status.item()) != 1:
        raise RuntimeError("async v8 prelaunch worker-entry handshake failed")

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
            name=f"async-v8-http-{block_index}",
        )
        client.start()

    release_error = None
    release_payload = None
    promotion_hold_ns = 0
    promotion_armed_sources = 0
    if marked:
        if rank == 0:
            connection = listener.accept()
            ready = connection.event
            if caller_id not in ready.request_id:
                raise RuntimeError("async v8 gate ready identity mismatch")
        gate_signal = torch.tensor(
            [int(ready.event_id) if rank == 0 else -1],
            dtype=torch.int64,
            device="cpu",
        )
        hold_started_ns = time.perf_counter_ns() if rank == 0 else 0
        dist.broadcast(gate_signal, src=0)
        if int(gate_signal.item()) < 0:
            raise RuntimeError("async v8 gate signal changed")
        if source:
            boost.set()
        armed = torch.tensor(
            [1 if source and boost.is_set() else 0], dtype=torch.int64, device="cpu"
        )
        dist.all_reduce(armed, op=dist.ReduceOp.SUM)
        promotion_armed_sources = int(armed.item())
        if rank == 0:
            try:
                promotion_hold_ns = time.perf_counter_ns() - hold_started_ns
                if promotion_armed_sources == old.SOURCE_COUNT:
                    frame = async_protocol.ReleaseFrame.promotion(
                        ready, promotion_armed_sources=promotion_armed_sources
                    )
                else:
                    frame = async_protocol.ReleaseFrame.noop(ready)
                    release_error = "async v8 did not arm exactly eight sources"
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
            raise RuntimeError("async v8 source transfer did not terminate in 60 seconds")
        worker.join(timeout=1.0)
        if worker.is_alive():
            raise RuntimeError("async v8 source worker remained alive")
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
        if source and mode != old.FG else 0
    )
    completion_from_origin_ns = (
        max(0, int(state["finished_ns"]) - local_origin_ns)
        if source and mode != old.FG else 0
    )
    post_foreground_drain_ns = (
        max(0, int(state["finished_ns"]) - local_foreground_done_ns)
        if source and mode != old.FG else 0
    )
    local = {
        "rank": rank,
        "source": source,
        "calls": 1 if source and mode != old.FG else 0,
        "completed": int(state["completed"]) if source else 0,
        "descriptors": old._descriptor_count(channel) if source and mode != old.FG else 0,
        "bytes": old.BYTES_PER_SOURCE if source and state["completed"] == 1 else 0,
        "elapsed_ns": elapsed_ns,
        "completion_from_origin_ns": completion_from_origin_ns,
        "post_foreground_drain_ns": post_foreground_drain_ns,
        "start_lag_ns": max(0, int(state["started_ns"]) - local_origin_ns)
        if source and mode != old.FG else 0,
        "polls": int(state["polls"]),
        "low_priority_sleeps": int(state["low_priority_sleeps"]),
        "boost_polls": int(state["boost_polls"]),
        "yields": int(state["yields"]),
        "boost_wait_timed_out": False,
        "error": state["error"],
    }
    return {
        "block_index": block_index,
        "prompt_index": prompt_index,
        "mode": mode,
        "client": old.scout._client_metrics(client_control[0]["value"])
        if rank == 0 else None,
        "client_started_from_origin_ns": client_started_from_origin_ns,
        "client_finished_from_origin_ns": client_finished_from_origin_ns,
        "gate_ready": ready.to_payload() if rank == 0 and marked else None,
        "gate_release": {"payload": release_payload, "error": release_error}
        if rank == 0 and marked else None,
        "boost_hold_ns": promotion_hold_ns,
        "promotion_armed_sources": promotion_armed_sources if marked else 0,
        "source_call": local,
        "receiver_verified_bytes": verified,
        "receiver_zero_ok": zero_ok,
        "correctness_met": (
            (not marked or release_error is None)
            and (not source or mode == old.FG or state["completed"] == 1
                 and state["error"] is None)
            and (not receiver or (zero_ok if mode == old.FG
                                  else verified == old.BYTES_PER_SOURCE))
        ),
    }


def _validate_trace(path: Path, expected: list[tuple[str, str]]) -> dict[str, Any]:
    mapped = [
        (caller, async_protocol.ASYNC_MODE if mode == old.TEMPO else mode)
        for caller, mode in expected
    ]
    installed_validate = old._validate_trace
    installed_bytes = old.GLOBAL_BYTES
    old._validate_trace = _ORIGINAL_VALIDATE_TRACE
    old.GLOBAL_BYTES = 0
    try:
        result = _V6_VALIDATE_TRACE(path, mapped)
    finally:
        old.GLOBAL_BYTES = installed_bytes
        old._validate_trace = installed_validate
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    releases = [row for row in rows if row.get("mode") == async_protocol.ASYNC_MODE]
    expected_async = sum(mode == old.TEMPO for _caller, mode in expected)
    if len(releases) != expected_async or any(
        row.get("completed_sources") != 0
        or row.get("physical_descriptors") != 0
        or row.get("completed_bytes") != 0
        or row.get("source_elapsed_ns") != []
        or row.get("wave_elapsed_ns") != 0
        or row.get("promotion_armed_sources") != old.SOURCE_COUNT
        for row in releases
    ):
        raise ValueError("async v8 trace made a false completion or arm claim")
    return {**result, "release_mode": async_protocol.ASYNC_MODE,
            "promotion_releases": len(releases)}


def _aggregate(records: list[dict[str, Any]], trace: dict[str, Any], args: Any):
    installed = old._aggregate
    old._aggregate = _ORIGINAL_AGGREGATE
    try:
        result = _V6_AGGREGATE(records, trace, args)
    finally:
        old._aggregate = installed
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    result["config"].update(
        {
            "boost_token_index_zero_based": 30,
            "boost_generated_token_count_one_based": 31,
            "promotion_release_mode": async_protocol.ASYNC_MODE,
            "promotion_release_completed_bytes": 0,
            "promotion_armed_sources": old.SOURCE_COUNT,
            "completion_wait_inside_gate": False,
        }
    )
    by_mode = {
        mode: [block for block in result["blocks"] if block["mode"] == mode]
        for mode in old.MODES
    }
    reductions = []
    paired = []
    for row in result["paired"]:
        prompt = row["prompt_index"]
        baseline = next(b for b in by_mode[old.LMCACHE] if b["prompt_index"] == prompt)
        tempo = next(b for b in by_mode[old.TEMPO] if b["prompt_index"] == prompt)
        baseline_completion = baseline["background_completion_from_start_ms"]
        reduction = (
            (baseline_completion - tempo["background_completion_from_start_ms"])
            / baseline_completion
        )
        reductions.append(reduction)
        paired.append({**row, "tempo_completion_reduction_vs_lmcache_fraction": reduction})
    result["paired"] = paired
    tempo_bubbles = [block["total_gate_bubble_ms"] for block in by_mode[old.TEMPO]]
    service_deltas = [row["tempo_minus_lmcache_service_makespan_ms"] for row in paired]
    tempo_tpot_p99 = result["mode_metrics"][old.TEMPO]["tpot_p99_max_ms"]
    lmcache_tpot_p99 = result["mode_metrics"][old.LMCACHE]["tpot_p99_max_ms"]
    gates = {
        "correctness_output_trace": bool(result["overall_correctness_met"]),
        "total_gate_bubble_median_le_4ms": statistics.median(tempo_bubbles)
        <= GATE_BUBBLE_MEDIAN_MS,
        "total_gate_bubble_max_le_6ms": max(tempo_bubbles) <= GATE_BUBBLE_MAX_MS,
        "all_tempo_post_foreground_drain_zero": all(
            block["post_foreground_drain_ms"] == 0.0 for block in by_mode[old.TEMPO]
        ),
        "tempo_service_makespan_beats_lmcache_paired": (
            statistics.median(service_deltas) < 0.0
            and sum(delta < 0.0 for delta in service_deltas) >= 2
        ),
        "tempo_completion_reduction_median_ge_20pct": (
            statistics.median(reductions) >= COMPLETION_REDUCTION_MIN
        ),
        "tempo_tpot_p99_le_1_10x_lmcache": (
            tempo_tpot_p99 <= TPOT_P99_RATIO_MAX * lmcache_tpot_p99
        ),
    }
    result["candidate_gates"] = gates
    result["screen_outcome"] = (
        "invalid_correctness_output_or_trace"
        if not result["overall_correctness_met"]
        else "async_candidate_pass" if all(gates.values())
        else "async_candidate_revise_or_stop"
    )
    return result


def main() -> None:
    async_protocol.install_async_release_protocol()
    old.protocol.ReleaseFrame = async_protocol.ReleaseFrame
    old.protocol.install_generic_release_protocol = (
        async_protocol.install_async_release_protocol
    )
    old.bulk.protocol.ReleaseFrame = async_protocol.ReleaseFrame
    old.bulk.protocol.install_generic_release_protocol = (
        async_protocol.install_async_release_protocol
    )
    fixed.CONTRACT_ID = CONTRACT_ID
    fixed.RESULT_SCHEMA = RESULT_SCHEMA
    fixed._load_contract = _load_contract
    fixed._run_block = _run_block
    fixed._validate_trace = _validate_trace
    fixed._aggregate = _aggregate
    fixed.main()


if __name__ == "__main__":
    main()
