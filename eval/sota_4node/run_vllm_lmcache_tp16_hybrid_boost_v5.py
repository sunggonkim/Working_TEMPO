#!/usr/bin/env python3
"""TP16 single-flight prelaunch with a token-31 quiescent completion boost.

The campaign compares three prompt-balanced modes in one vLLM lifecycle:

* ``fg_only``: no background movement;
* ``lmcache_prelaunch_no_gate``: the pinned official LMCache NixlChannel starts
  one contiguous 16 MiB descriptor per source at request start and is allowed
  to contend with decode until it finishes;
* ``tempo_prelaunch_quiescent_boost``: the exact same 128 MiB geometry is
  prelaunched once, but a prepared NIXL handle uses low-rate progress during
  decode and is promoted to spin/yield completion while TP16 is fenced after
  generated token 31.

There is no per-token queue and no second transfer wave.  A marked no-op
request warms the all-rank accelerator fence outside measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import queue
import statistics
import sys
import threading
import time
from typing import Any, Iterable

import numpy as np

from eval.sota_4node import run_vllm_lmcache_tp8_sidecar as base
from eval.sota_4node import run_vllm_lmcache_tp16_pair_stagger_coalesced_v1 as tp16
from eval.sota_4node import run_vllm_lmcache_tp16_quiescence_scout_v1 as scout
from eval.sota_4node import run_vllm_lmcache_tp16_quiescence_bulk_v4 as bulk
from eval.sota_4node import vllm_decode_quiescence_gate_launch_v3 as gate
from eval.sota_4node import vllm_quiescence_wave_protocol_v5 as protocol


WORLD_SIZE = 16
SOURCE_COUNT = 8
RECEIVER_OFFSET = 8
BYTES_PER_SOURCE = 16 << 20
GLOBAL_BYTES = SOURCE_COUNT * BYTES_PER_SOURCE
TOKENS = 64
FG = "fg_only"
LMCACHE = "lmcache_prelaunch_no_gate"
TEMPO = protocol.HYBRID_MODE
MODES = (FG, LMCACHE, TEMPO)
BLOCKS = (
    (0, FG),
    (0, LMCACHE),
    (0, TEMPO),
    (1, TEMPO),
    (1, FG),
    (1, LMCACHE),
    (2, LMCACHE),
    (2, TEMPO),
    (2, FG),
)
CONTRACT_ID = "tp16-single-flight-hybrid-boost-v5"
RESULT_SCHEMA = "tempo-vllm-tp16-single-flight-hybrid-result-5"
BOOST_WAIT_CAP_MS = 35.0
BOOST_PROMOTION_MEDIAN_MS = 25.0
BOOST_PROMOTION_MAX_MS = 30.0


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires samples")
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--api-host", required=True)
    parser.add_argument("--api-port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--nixl-port-base", type=int, required=True)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--campaign-index", type=int, choices=range(3), required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--quiescence-socket", type=Path, required=True)
    parser.add_argument("--quiescence-trace", type=Path, required=True)
    args = parser.parse_args()
    backend = os.environ.get("TEMPO_NIXL_BACKEND", "UCX")
    if backend not in {"UCX", "LIBFABRIC"}:
        parser.error("TEMPO_NIXL_BACKEND must be UCX or LIBFABRIC")
    args.nixl_backend = backend
    if not 1024 <= args.api_port <= 65535:
        parser.error("api-port is invalid")
    if not 1024 <= args.nixl_port_base <= 65528:
        parser.error("nixl-port-base is invalid")
    if args.request_timeout_s <= 0:
        parser.error("request timeout must be positive")
    return args


def _expected_contract() -> dict[str, Any]:
    return {
        "schema_version": "tempo-tp16-single-flight-hybrid-contract-5",
        "contract_id": CONTRACT_ID,
        "topology": {
            "nodes": 4,
            "world_size": WORLD_SIZE,
            "source_ranks": list(range(SOURCE_COUNT)),
            "receiver_ranks": list(range(RECEIVER_OFFSET, WORLD_SIZE)),
            "pairing": [[rank, rank + RECEIVER_OFFSET] for rank in range(SOURCE_COUNT)],
        },
        "transfer": {
            "bytes_per_source": BYTES_PER_SOURCE,
            "global_bytes": GLOBAL_BYTES,
            "calls_global": SOURCE_COUNT,
            "physical_descriptors_global": SOURCE_COUNT,
            "single_flight_per_source": True,
            "prelaunch_at_request_start": True,
        },
        "boost": {
            "target_output_token_index_zero_based": 30,
            "generated_token_count_one_based": 31,
            "wait_cap_ms": BOOST_WAIT_CAP_MS,
            "promotion_median_gate_ms": BOOST_PROMOTION_MEDIAN_MS,
            "promotion_max_gate_ms": BOOST_PROMOTION_MAX_MS,
            "prepared_handle_repost": True,
            "decode_progress_sleep_ms": 1.0,
            "boost_progress": "spin_yield_64",
        },
        "campaign": {
            "modes": list(MODES),
            "blocks": len(BLOCKS),
            "replicates_per_mode": 3,
            "unmeasured_fence_prewarm": True,
        },
    }


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_contract():
        raise ValueError("TP16 hybrid boost contract changed")
    return payload


def _make_memory(torch: Any, TensorMemoryObj: Any, MemoryObjMetadata: Any, MemoryFormat: Any):
    backing = torch.empty(
        2 * BYTES_PER_SOURCE - 1, dtype=torch.uint8, device="cuda"
    )
    offset = (-backing.data_ptr()) % BYTES_PER_SOURCE
    buffer = backing[offset : offset + BYTES_PER_SOURCE]
    if buffer.numel() != BYTES_PER_SOURCE or buffer.data_ptr() % BYTES_PER_SOURCE:
        raise RuntimeError("hybrid buffer alignment changed")
    shape = torch.Size([BYTES_PER_SOURCE])
    obj = TensorMemoryObj(
        raw_data=buffer,
        metadata=MemoryObjMetadata(
            shape=shape,
            dtype=torch.uint8,
            address=buffer.data_ptr(),
            phy_size=BYTES_PER_SOURCE,
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.BINARY,
            shapes=[shape],
            dtypes=[torch.uint8],
        ),
        parent_allocator=None,
    )
    return backing, buffer, [obj], {buffer.data_ptr(): 0}


def _descriptor_count(channel: Any) -> int:
    descriptors = channel.nixl_wrapper.xfer_descs
    if hasattr(descriptors, "__len__"):
        return int(len(descriptors))
    method = getattr(descriptors, "descCount", None)
    if not callable(method):
        raise RuntimeError("NIXL descriptor list cardinality is unavailable")
    return int(method())


def _hybrid_channel_class(base_channel: Any) -> Any:
    class HybridChannel(base_channel):
        def tempo_prepare(
            self, objects: list[Any], transfer_spec: dict[str, Any]
        ) -> Any:
            key = (
                str(transfer_spec["receiver_id"]),
                tuple(int(value) for value in transfer_spec["remote_indexes"]),
                tuple(int(value) for value in self.get_local_mem_indices(objects)),
            )
            handles = getattr(self, "_tempo_prepared_handles", None)
            if handles is None:
                handles = {}
                self._tempo_prepared_handles = handles
            if key not in handles:
                handles[key] = self.nixl_agent.make_prepped_xfer(
                    "WRITE",
                    self.nixl_wrapper.xfer_handler,
                    list(key[2]),
                    self.remote_xfer_handlers_dict[key[0]],
                    np.asarray(key[1], dtype=np.uint64),
                )
            return handles[key]

        def tempo_adaptive_write(
            self,
            objects: list[Any],
            transfer_spec: dict[str, Any],
            boost: threading.Event,
        ) -> dict[str, int]:
            handle = self.tempo_prepare(objects, transfer_spec)
            posted = self.nixl_agent.transfer(handle)
            if posted == "ERR":
                raise RuntimeError("TEMPO failed to post prepared NIXL handle")
            polls = low_priority_sleeps = boost_polls = yields = 0
            while True:
                status = self.nixl_agent.check_xfer_state(handle)
                polls += 1
                if status == "ERR":
                    raise RuntimeError("TEMPO prepared NIXL transfer failed")
                if status == "DONE":
                    return {
                        "completed": len(objects),
                        "polls": polls,
                        "low_priority_sleeps": low_priority_sleeps,
                        "boost_polls": boost_polls,
                        "yields": yields,
                    }
                if status != "PROC":
                    raise RuntimeError(f"unexpected NIXL state {status}")
                if boost.is_set():
                    boost_polls += 1
                    if boost_polls % 64 == 0:
                        time.sleep(0)
                        yields += 1
                else:
                    time.sleep(0.001)
                    low_priority_sleeps += 1

    HybridChannel.__name__ = "HybridChannel"
    return HybridChannel


def _load_channel(repo_root: Path) -> tuple[Any, Any, Any, Any]:
    channel, tensor, metadata, memory_format = base.official._load_official_lmcache(
        repo_root
    )
    return _hybrid_channel_class(channel), tensor, metadata, memory_format


def _warm(torch: Any, dist: Any, channel: Any, obj: Any, rank: int, pair: int) -> None:
    source = rank < SOURCE_COUNT
    expected = 223 + pair
    obj.raw_data.fill_(expected if source else 0)
    torch.cuda.synchronize()
    dist.barrier()
    local = {"ok": True, "count": 0, "error": None}
    if source:
        try:
            local["count"] = int(
                channel.batched_write(
                    objects=[obj],
                    transfer_spec={
                        "receiver_id": f"rank-{rank + RECEIVER_OFFSET}",
                        "remote_indexes": np.asarray([0], dtype=np.uint64),
                    },
                )
            )
            local["ok"] = local["count"] == 1
        except BaseException as exc:
            local.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    dist.barrier()
    if not source:
        local["ok"] = bool(torch.all(obj.raw_data == expected).item())
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, local)
    if not all(bool(row["ok"]) for row in rows):
        raise RuntimeError(f"hybrid NIXL warmup failed: {rows}")
    dist.barrier()


def _transfer_worker(
    *,
    channel: Any,
    obj: Any,
    receiver_id: str,
    mode: str,
    boost: threading.Event,
    done: threading.Event,
    state: dict[str, Any],
) -> None:
    state["started_ns"] = time.perf_counter_ns()
    try:
        spec = {
            "receiver_id": receiver_id,
            "remote_indexes": np.asarray([0], dtype=np.uint64),
        }
        if mode == LMCACHE:
            state["completed"] = int(
                channel.batched_write(objects=[obj], transfer_spec=spec)
            )
        elif mode == TEMPO:
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
    args: argparse.Namespace,
) -> dict[str, Any]:
    protocol.install_generic_release_protocol()
    source = rank < SOURCE_COUNT
    receiver = not source
    expected = 1 + ((block_index * 37 + pair * 3) % 251)
    obj.raw_data.fill_(expected if source and mode != FG else 0)
    torch.cuda.synchronize()
    dist.barrier()

    marked = mode == TEMPO
    prefix = "tempo-scout" if marked else "control"
    caller_id = (
        f"{prefix}-{args.allocation_id}-c{args.campaign_index}-b{block_index}-{mode}"
    )
    events: queue.Queue[tuple[bool, Any]] = queue.Queue()
    listener = connection = ready = client = None
    if rank == 0:
        if marked:
            listener = gate.GateListener(
                gate.GateConfig(
                    args.quiescence_socket, args.quiescence_trace, timeout_s=30.0
                )
            )
            listener.open()
        client = threading.Thread(
            target=bulk._request_thread,
            kwargs={
                "events": events,
                "args": args,
                "prompt": base.PROMPTS[prompt_index],
                "caller_id": caller_id,
                "tokens": TOKENS,
            },
            name=f"hybrid-http-{block_index}",
        )
        client.start()

    start_signal = torch.tensor([block_index], dtype=torch.int64, device="cpu")
    dist.broadcast(start_signal, src=0)
    local_origin_ns = time.perf_counter_ns()
    if int(start_signal.item()) != block_index:
        raise RuntimeError("hybrid request-start control changed")

    boost = threading.Event()
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
    if source and mode != FG:
        worker = threading.Thread(
            target=_transfer_worker,
            kwargs={
                "channel": channel,
                "obj": obj,
                "receiver_id": f"rank-{rank + RECEIVER_OFFSET}",
                "mode": mode,
                "boost": boost,
                "done": done,
                "state": state,
            },
            name=f"hybrid-transfer-rank{rank}-block{block_index}",
            daemon=True,
        )
        worker.start()

    boost_wait_timed_out = False
    release_error = None
    release_payload = None
    boost_hold_ns = 0
    if marked:
        if rank == 0:
            connection = listener.accept()
            ready = connection.event
            if caller_id not in ready.request_id:
                raise RuntimeError("hybrid gate ready identity mismatch")
        gate_signal = torch.tensor(
            [int(ready.event_id) if rank == 0 else -1],
            dtype=torch.int64,
            device="cpu",
        )
        hold_started_ns = time.perf_counter_ns() if rank == 0 else 0
        dist.broadcast(gate_signal, src=0)
        if int(gate_signal.item()) < 0:
            raise RuntimeError("hybrid gate signal changed")
        if source:
            boost.set()
            boost_wait_timed_out = not done.wait(BOOST_WAIT_CAP_MS / 1000.0)
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
                BYTES_PER_SOURCE
                if receiver and bool(torch.all(obj.raw_data == expected).item())
                else 0
            )
        else:
            verified_at_gate = 0
        gate_status = torch.tensor(
            [
                1 if source and done.is_set() else 0,
                int(state["completed"]) if source else 0,
                _descriptor_count(channel) if source else 0,
                BYTES_PER_SOURCE if source and state["completed"] == 1 else 0,
                max(0, int(state["finished_ns"]) - int(state["started_ns"]))
                if source and done.is_set()
                else 0,
                1 if source and state["error"] is not None else 0,
                verified_at_gate,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        gathered = [torch.zeros_like(gate_status) for _ in range(WORLD_SIZE)] if rank == 0 else None
        dist.gather(gate_status, gather_list=gathered, dst=0)
        if rank == 0:
            try:
                rows = [tensor.tolist() for tensor in gathered]
                sources = rows[:SOURCE_COUNT]
                receivers = rows[SOURCE_COUNT:]
                structural = all_sources_done and all(
                    row[0] == 1
                    and row[1] == 1
                    and row[2] == 1
                    and row[3] == BYTES_PER_SOURCE
                    and row[5] == 0
                    for row in sources
                ) and all(row[6] == BYTES_PER_SOURCE for row in receivers)
                boost_hold_ns = time.perf_counter_ns() - hold_started_ns
                if structural:
                    frame = protocol.ReleaseFrame.wave(
                        ready,
                        mode=TEMPO,
                        completed_bytes=GLOBAL_BYTES,
                        source_elapsed_ns=tuple(int(row[4]) for row in sources),
                        wave_elapsed_ns=boost_hold_ns,
                    )
                else:
                    frame = protocol.ReleaseFrame.noop(ready)
                    release_error = "hybrid boost did not finish exact transfer within cap"
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
        client.join(timeout=1.0)
        client_control[0] = {"ok": ok, "value": value}
    dist.broadcast_object_list(client_control, src=0)
    local_foreground_done_ns = time.perf_counter_ns()
    if not client_control[0]["ok"]:
        raise RuntimeError(f"vLLM request failed: {client_control[0]['value']}")

    if source and mode != FG:
        if not done.wait(60.0):
            raise RuntimeError("hybrid source transfer did not terminate in 60 seconds")
        worker.join(timeout=1.0)
        if worker.is_alive():
            raise RuntimeError("hybrid source worker remained alive after completion")
    dist.barrier()
    verified = 0
    zero_ok = True
    if receiver:
        if mode == FG:
            zero_ok = bool(torch.all(obj.raw_data == 0).item())
        else:
            verified = (
                BYTES_PER_SOURCE if bool(torch.all(obj.raw_data == expected).item()) else 0
            )
    dist.barrier()

    elapsed_ns = (
        max(0, int(state["finished_ns"]) - int(state["started_ns"]))
        if source and mode != FG
        else 0
    )
    completion_from_origin_ns = (
        max(0, int(state["finished_ns"]) - local_origin_ns)
        if source and mode != FG
        else 0
    )
    post_foreground_drain_ns = (
        max(0, int(state["finished_ns"]) - local_foreground_done_ns)
        if source and mode != FG
        else 0
    )
    local = {
        "rank": rank,
        "source": source,
        "calls": 1 if source and mode != FG else 0,
        "completed": int(state["completed"]) if source else 0,
        "descriptors": _descriptor_count(channel) if source and mode != FG else 0,
        "bytes": BYTES_PER_SOURCE if source and state["completed"] == 1 else 0,
        "elapsed_ns": elapsed_ns,
        "completion_from_origin_ns": completion_from_origin_ns,
        "post_foreground_drain_ns": post_foreground_drain_ns,
        "start_lag_ns": max(0, int(state["started_ns"]) - local_origin_ns)
        if source and mode != FG
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
        "client": scout._client_metrics(client_control[0]["value"]) if rank == 0 else None,
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
                or mode == FG
                or state["completed"] == 1
                and state["error"] is None
            )
            and (
                not receiver
                or (zero_ok if mode == FG else verified == BYTES_PER_SOURCE)
            )
        ),
    }


def _validate_trace(path: Path, expected: list[tuple[str, str]]) -> dict[str, Any]:
    if not path.is_file() or not 0 < path.stat().st_size <= (1 << 20):
        raise ValueError("hybrid trace is missing or unbounded")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) > 256:
        raise ValueError("hybrid trace has too many rows")
    provenance = [row for row in rows if row.get("kind") == "provenance"]
    if len(provenance) != 1:
        raise ValueError("hybrid trace must contain one provenance row")
    required = {
        "protocol": gate.PROTOCOL,
        "world_size": WORLD_SIZE,
        "tensor_parallel_size": WORLD_SIZE,
        "async_scheduling": False,
        "speculative_decoding": False,
        "vllm_version": "0.26.0+cu129",
        "engine_core_process_step_sha256": (
            "41295db73bb85ebda9cee7c4f32d944e5f973b6bcc0433ff6b152a9368b175b9"
        ),
    }
    if any(provenance[0].get(key) != value for key, value in required.items()):
        raise ValueError("hybrid trace provenance mismatch")
    if any(row.get("kind") == "error" for row in rows):
        raise ValueError("hybrid trace contains a hook error")
    readies = {int(row["event_id"]): row for row in rows if row.get("kind") == "ready"}
    releases = {int(row["event_id"]): row for row in rows if row.get("kind") == "release"}
    nexts = {
        int(row["event_id"]): row
        for row in rows
        if row.get("kind") == "next_engine_step_enter"
    }
    ids = set(range(len(expected)))
    if set(readies) != ids or set(releases) != ids or set(nexts) != ids:
        raise ValueError("hybrid trace event coverage mismatch")
    for event_id, (caller, mode) in enumerate(expected):
        ready, release, nxt = readies[event_id], releases[event_id], nexts[event_id]
        if caller not in ready.get("request_id", "") or release.get("mode") != mode:
            raise ValueError("hybrid trace request/mode mismatch")
        fence = ready.get("fence_rows")
        if not isinstance(fence, list) or sorted(int(row[0]) for row in fence) != list(
            range(WORLD_SIZE)
        ):
            raise ValueError("hybrid trace fence rank coverage mismatch")
        timeline = [
            int(ready["output_enqueued_ns"]),
            int(ready["fence_started_ns"]),
            int(ready["fence_finished_ns"]),
            int(ready["ready_ns"]),
            int(release["released_ns"]),
            int(release["gate_returned_ns"]),
            int(nxt["entered_ns"]),
        ]
        if timeline != sorted(timeline):
            raise ValueError("hybrid trace timeline ordering mismatch")
        expected_bytes = 0 if mode == protocol.NOOP_MODE else GLOBAL_BYTES
        if int(release.get("completed_bytes", -1)) != expected_bytes:
            raise ValueError("hybrid trace byte geometry mismatch")
    return {"validated": True, "event_count": len(expected), "record_count": len(rows)}


def _aggregate(records: list[dict[str, Any]], trace: dict[str, Any], args: argparse.Namespace):
    ordered = sorted(records, key=lambda item: int(item["rank"]))
    if len(ordered) != WORLD_SIZE or [item["rank"] for item in ordered] != list(
        range(WORLD_SIZE)
    ):
        raise ValueError("hybrid rank records are incomplete")
    blocks = []
    for index, (prompt, mode) in enumerate(BLOCKS):
        rank_blocks = [item["blocks"][index] for item in ordered]
        source_blocks = rank_blocks[:SOURCE_COUNT]
        receiver_blocks = rank_blocks[SOURCE_COUNT:]
        completed = sum(int(block["source_call"]["bytes"]) for block in source_blocks)
        verified = sum(int(block["receiver_verified_bytes"]) for block in receiver_blocks)
        calls = sum(int(block["source_call"]["calls"]) for block in source_blocks)
        descriptors = sum(
            int(block["source_call"]["descriptors"]) for block in source_blocks
        )
        expected = 0 if mode == FG else GLOBAL_BYTES
        correct = (
            all(bool(block["correctness_met"]) for block in rank_blocks)
            and completed == verified == expected
            and calls == descriptors == (0 if mode == FG else SOURCE_COUNT)
        )
        client = rank_blocks[0]["client"]
        max_completion_ms = max(
            float(block["source_call"]["completion_from_origin_ns"]) / 1e6
            for block in source_blocks
        )
        max_drain_ms = max(
            float(block["source_call"]["post_foreground_drain_ns"]) / 1e6
            for block in source_blocks
        )
        blocks.append(
            {
                "block_index": index,
                "prompt_index": prompt,
                "mode": mode,
                **client,
                "background_completed_bytes": completed,
                "receiver_verified_bytes": verified,
                "source_calls": calls,
                "physical_descriptors": descriptors,
                "background_completion_from_start_ms": max_completion_ms,
                "post_foreground_drain_ms": max_drain_ms,
                "service_makespan_ms": max(client["request_e2e_ms"], max_completion_ms),
                "boost_hold_ms": rank_blocks[0]["boost_hold_ns"] / 1e6,
                "max_source_elapsed_ms": max(
                    float(block["source_call"]["elapsed_ns"]) / 1e6
                    for block in source_blocks
                ),
                "max_source_start_lag_ms": max(
                    float(block["source_call"]["start_lag_ns"]) / 1e6
                    for block in source_blocks
                ),
                "correctness_met": correct,
            }
        )
    output_equal = all(
        len(
            {
                block["output_token_sha256"]
                for block in blocks
                if block["prompt_index"] == prompt
            }
        )
        == 1
        for prompt in range(3)
    )
    overall = (
        trace.get("validated") is True
        and output_equal
        and all(block["correctness_met"] for block in blocks)
    )
    by_mode = {mode: [block for block in blocks if block["mode"] == mode] for mode in MODES}
    metrics = {}
    for mode, values in by_mode.items():
        metrics[mode] = {
            "replicates": len(values),
            "ttft_p50_ms": statistics.median(block["ttft_ms"] for block in values),
            "tpot_p50_ms": statistics.median(block["tpot_p50_ms"] for block in values),
            "tpot_p99_max_ms": max(block["tpot_p99_ms"] for block in values),
            "e2e_p50_ms": statistics.median(block["request_e2e_ms"] for block in values),
            "service_makespan_p50_ms": statistics.median(
                block["service_makespan_ms"] for block in values
            ),
            "post_foreground_drain_max_ms": max(
                block["post_foreground_drain_ms"] for block in values
            ),
            "boost_hold_p50_ms": statistics.median(
                block["boost_hold_ms"] for block in values
            ),
            "boost_hold_max_ms": max(block["boost_hold_ms"] for block in values),
        }
    paired = []
    for prompt in range(3):
        fg = next(block for block in by_mode[FG] if block["prompt_index"] == prompt)
        baseline = next(
            block for block in by_mode[LMCACHE] if block["prompt_index"] == prompt
        )
        tempo = next(block for block in by_mode[TEMPO] if block["prompt_index"] == prompt)
        paired.append(
            {
                "prompt_index": prompt,
                "tempo_minus_lmcache_service_makespan_ms": (
                    tempo["service_makespan_ms"] - baseline["service_makespan_ms"]
                ),
                "tempo_minus_lmcache_drain_ms": (
                    tempo["post_foreground_drain_ms"]
                    - baseline["post_foreground_drain_ms"]
                ),
                "tempo_minus_lmcache_e2e_ms": (
                    tempo["request_e2e_ms"] - baseline["request_e2e_ms"]
                ),
                "tempo_minus_fg_e2e_ms": tempo["request_e2e_ms"] - fg["request_e2e_ms"],
                "tempo_minus_lmcache_tpot_p99_ms": (
                    tempo["tpot_p99_ms"] - baseline["tpot_p99_ms"]
                ),
            }
        )
    tempo_values = by_mode[TEMPO]
    tempo_holds = [block["boost_hold_ms"] for block in tempo_values]
    fg_e2e = metrics[FG]["e2e_p50_ms"]
    gates = {
        "correctness_output_trace": overall,
        "all_tempo_post_foreground_drain_zero": all(
            block["post_foreground_drain_ms"] == 0.0 for block in tempo_values
        ),
        "boost_hold_median_le_25ms": statistics.median(tempo_holds)
        <= BOOST_PROMOTION_MEDIAN_MS,
        "boost_hold_max_le_30ms": max(tempo_holds) <= BOOST_PROMOTION_MAX_MS,
        "tempo_e2e_p50_le_1_05x_fg": metrics[TEMPO]["e2e_p50_ms"] <= 1.05 * fg_e2e,
        "tempo_service_makespan_beats_lmcache": (
            metrics[TEMPO]["service_makespan_p50_ms"]
            < metrics[LMCACHE]["service_makespan_p50_ms"]
        ),
    }
    return {
        "schema_version": RESULT_SCHEMA,
        "contract_id": CONTRACT_ID,
        "allocation_id": args.allocation_id,
        "campaign_index": args.campaign_index,
        "evidence_scope": "single-allocation real-vllm tp16 hybrid campaign",
        "promotion_valid": False,
        "config": {
            "nodes": 4,
            "world_size": WORLD_SIZE,
            "tokens": TOKENS,
            "nixl_backend": args.nixl_backend,
            "bytes_per_source": BYTES_PER_SOURCE,
            "global_bytes": GLOBAL_BYTES,
            "boost_token_index_zero_based": 30,
            "boost_wait_cap_ms": BOOST_WAIT_CAP_MS,
        },
        "hook_trace": trace,
        "hook_trace_validated": trace.get("validated") is True,
        "output_equivalence_met": output_equal,
        "overall_correctness_met": overall,
        "blocks": blocks,
        "mode_metrics": metrics,
        "paired": paired,
        "candidate_gates": gates,
        "screen_outcome": (
            "invalid_correctness_output_or_trace"
            if not overall
            else "hybrid_candidate_pass"
            if all(gates.values())
            else "hybrid_candidate_revise_or_stop"
        ),
    }


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    args.output_dir = base._resolve_below_repo(args.output_dir, repo_root, label="output-dir")
    args.plan = base._resolve_below_repo(args.plan, repo_root, label="plan")
    args.model = str(Path(args.model).resolve())
    _load_contract(args.plan)
    protocol.install_generic_release_protocol()
    base._set_rank_environment()
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise SystemExit("CUDA and WORLD_SIZE=16 are required")
    import nixl

    if nixl._api.__name__ != "nixl_cu12._api":
        raise RuntimeError(f"hybrid campaign requires nixl_cu12._api, got {nixl._api.__name__}")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    visible = torch.cuda.device_count()
    device = 0 if visible == 1 else local_rank
    torch.cuda.set_device(device)
    dist.init_process_group("gloo")
    try:
        hosts = tp16._validate_topology(dist, rank, local_rank)
        Channel, TensorMemoryObj, MemoryObjMetadata, MemoryFormat = _load_channel(repo_root)
        backing, buffer, objects, index_by_address = _make_memory(
            torch, TensorMemoryObj, MemoryObjMetadata, MemoryFormat
        )
        pair = rank % SOURCE_COUNT
        source = rank < SOURCE_COUNT
        peer = rank + RECEIVER_OFFSET if source else rank - RECEIVER_OFFSET
        channel = Channel(
            async_mode=False,
            role="sender" if source else "receiver",
            buffer_ptr=buffer.data_ptr(),
            buffer_size=buffer.numel(),
            align_bytes=BYTES_PER_SOURCE,
            tp_rank=local_rank,
            peer_init_url=None if source else f"*:{args.nixl_port_base + pair}",
            backends=[args.nixl_backend],
            device=f"cuda:{device}",
        )
        base.epoch._install_descriptor_index_shim(channel, index_by_address)
        if _descriptor_count(channel) != 1:
            raise RuntimeError("hybrid channel did not create exactly one descriptor")
        dist.barrier()
        if source:
            channel.lazy_init_peer_connection(
                local_id=f"rank-{rank}",
                peer_id=f"rank-{peer}",
                peer_init_url=f"{hosts[peer]}:{args.nixl_port_base + pair}",
            )
        dist.barrier()
        if not channel.remote_xfer_handler_exists(f"rank-{peer}"):
            raise RuntimeError("hybrid LMCache peer handler is missing")
        if source:
            channel.tempo_prepare(
                [objects[0]],
                {
                    "receiver_id": f"rank-{peer}",
                    "remote_indexes": np.asarray([0], dtype=np.uint64),
                },
            )
        dist.barrier()
        _warm(torch, dist, channel, objects[0], rank, pair)

        prewarm = bulk._run_epoch(
            torch,
            dist,
            channel=channel,
            obj=objects[0],
            rank=rank,
            pair=pair,
            block_index=-1,
            prompt_index=0,
            mode=protocol.NOOP_MODE,
            args=args,
            measured=False,
        )
        if not prewarm["correctness_met"]:
            raise RuntimeError("hybrid accelerator-fence prewarm failed")

        blocks = [
            _run_block(
                torch,
                dist,
                channel=channel,
                obj=objects[0],
                rank=rank,
                pair=pair,
                block_index=index,
                prompt_index=prompt,
                mode=mode,
                args=args,
            )
            for index, (prompt, mode) in enumerate(BLOCKS)
        ]
        expected_trace = [
            (
                f"tempo-scout-{args.allocation_id}-c{args.campaign_index}-fence-prewarm-{protocol.NOOP_MODE}",
                protocol.NOOP_MODE,
            )
        ] + [
            (
                f"tempo-scout-{args.allocation_id}-c{args.campaign_index}-b{i}-{mode}",
                mode,
            )
            for i, (_prompt, mode) in enumerate(BLOCKS)
            if mode == TEMPO
        ]
        trace_control: list[Any] = [None]
        if rank == 0:
            try:
                trace_control[0] = _validate_trace(args.quiescence_trace, expected_trace)
            except BaseException as exc:
                trace_control[0] = {
                    "validated": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        dist.broadcast_object_list(trace_control, src=0)

        record = {
            "rank": rank,
            "local_rank": local_rank,
            "hostname": hosts[rank],
            "nixl_api": nixl._api.__name__,
            "prewarm": prewarm,
            "blocks": blocks,
        }
        gathered = [None] * WORLD_SIZE if rank == 0 else None
        dist.gather_object(record, gathered, dst=0)
        final: list[Any] = [None]
        if rank == 0:
            try:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                for item in gathered:
                    (args.output_dir / f"rank_{item['rank']}.json").write_text(
                        json.dumps(item, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                result = _aggregate(gathered, trace_control[0], args)
                path = args.output_dir / "result.json"
                path.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                final[0] = {
                    "ok": result["overall_correctness_met"],
                    "output": str(path),
                    "outcome": result["screen_outcome"],
                }
            except BaseException as exc:
                final[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(final, src=0)
        dist.barrier()
        if not final[0]["ok"]:
            raise RuntimeError(f"hybrid campaign failed: {final[0]}")
        if rank == 0:
            print(json.dumps(final[0], sort_keys=True))
        del backing
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
