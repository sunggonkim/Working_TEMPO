#!/usr/bin/env python3
"""TP16 32 MiB quiescent epoch: official LMCache versus TEMPO polling.

Each measured data block moves one contiguous 4 MiB descriptor from each of
eight source ranks to its paired receiver (32 MiB globally).  The vLLM engine
is fenced after generated token 31.  Three prompt-balanced repetitions compare
the pinned LMCache ``batched_write`` completion loop with an otherwise
identical NIXL operation whose 1 ms polling sleep is replaced by bounded
spin/yield progress.  A marked, unmeasured no-op request warms the all-rank
accelerator fence before any measured block.
"""

from __future__ import annotations

import argparse
import hashlib
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
from eval.sota_4node import vllm_decode_quiescence_gate_launch_v3 as gate
from eval.sota_4node import vllm_quiescence_wave_protocol_v4 as protocol


WORLD_SIZE = 16
SOURCE_COUNT = 8
RECEIVER_OFFSET = 8
BYTES_PER_SOURCE = 4 << 20
GLOBAL_BYTES = SOURCE_COUNT * BYTES_PER_SOURCE
TOKENS = 64
NOOP = protocol.NOOP_MODE
LMCACHE = "quiescent_lmcache_bulk"
TEMPO = "quiescent_tempo_bulk"
MODES = (NOOP, LMCACHE, TEMPO)
BLOCKS = (
    (0, NOOP),
    (0, LMCACHE),
    (0, TEMPO),
    (1, TEMPO),
    (1, NOOP),
    (1, LMCACHE),
    (2, LMCACHE),
    (2, TEMPO),
    (2, NOOP),
)
CONTRACT_ID = "tp16-quiescence-bulk-v4"
RESULT_SCHEMA = "tempo-vllm-tp16-quiescence-bulk-result-4"
MAX_MEDIAN_WAVE_MS = 10.0
MAX_INDIVIDUAL_WAVE_MS = 12.0


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
        "schema_version": "tempo-tp16-quiescence-bulk-contract-4",
        "contract_id": CONTRACT_ID,
        "topology": {
            "nodes": 4,
            "world_size": WORLD_SIZE,
            "source_ranks": list(range(SOURCE_COUNT)),
            "receiver_ranks": list(range(RECEIVER_OFFSET, WORLD_SIZE)),
            "pairing": [[rank, rank + RECEIVER_OFFSET] for rank in range(SOURCE_COUNT)],
        },
        "epoch": {
            "target_output_token_index_zero_based": 30,
            "generated_token_count_one_based": 31,
            "bytes_per_source": BYTES_PER_SOURCE,
            "global_bytes": GLOBAL_BYTES,
            "source_calls": SOURCE_COUNT,
            "physical_descriptors": SOURCE_COUNT,
            "contiguous_descriptors_per_source": 1,
        },
        "campaign": {
            "modes": list(MODES),
            "blocks": len(BLOCKS),
            "replicates_per_mode": 3,
            "unmeasured_fence_prewarm": True,
        },
        "gates": {
            "tempo_median_wave_ms": MAX_MEDIAN_WAVE_MS,
            "tempo_each_wave_ms": MAX_INDIVIDUAL_WAVE_MS,
            "exact_bytes_calls_descriptors": True,
            "output_equivalence": True,
            "hook_trace": True,
        },
    }


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_contract():
        raise ValueError("TP16 quiescence bulk contract changed")
    return payload


def _make_memory(torch: Any, TensorMemoryObj: Any, MemoryObjMetadata: Any, MemoryFormat: Any):
    backing = torch.empty(
        2 * BYTES_PER_SOURCE - 1, dtype=torch.uint8, device="cuda"
    )
    offset = (-backing.data_ptr()) % BYTES_PER_SOURCE
    buffer = backing[offset : offset + BYTES_PER_SOURCE]
    if buffer.numel() != BYTES_PER_SOURCE or buffer.data_ptr() % BYTES_PER_SOURCE:
        raise RuntimeError("bulk buffer alignment changed")
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


def _tempo_channel_class(base_channel: Any) -> Any:
    class TempoCompletionChannel(base_channel):
        def tempo_write(
            self, objects: list[Any], transfer_spec: dict[str, Any]
        ) -> tuple[int, int, int]:
            handle = self.nixl_agent.make_prepped_xfer(
                "WRITE",
                self.nixl_wrapper.xfer_handler,
                self.get_local_mem_indices(objects),
                self.remote_xfer_handlers_dict[transfer_spec["receiver_id"]],
                transfer_spec["remote_indexes"],
            )
            self.nixl_agent.transfer(handle)
            polls = 0
            yields = 0
            while True:
                status = self.nixl_agent.check_xfer_state(handle)
                polls += 1
                if status == "ERR":
                    raise RuntimeError("TEMPO NIXL write failed")
                if status == "DONE":
                    return len(objects), polls, yields
                if status != "PROC":
                    raise RuntimeError(f"unexpected NIXL transfer state {status}")
                # Preserve rapid completion progress without a fixed 1 ms hole,
                # but periodically yield the source CPU to Gloo/NIXL workers.
                if polls % 64 == 0:
                    time.sleep(0)
                    yields += 1

    TempoCompletionChannel.__name__ = "TempoCompletionChannel"
    return TempoCompletionChannel


def _load_channel(repo_root: Path) -> tuple[Any, Any, Any, Any]:
    channel, tensor, metadata, memory_format = base.official._load_official_lmcache(
        repo_root
    )
    return _tempo_channel_class(channel), tensor, metadata, memory_format


def _warm(torch: Any, dist: Any, channel: Any, obj: Any, rank: int, pair: int) -> None:
    source = rank < SOURCE_COUNT
    expected = 211 + pair
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
        raise RuntimeError(f"bulk NIXL warmup failed: {rows}")
    dist.barrier()


def _request_thread(
    events: queue.Queue[tuple[bool, Any]], *, args: argparse.Namespace,
    prompt: str, caller_id: str, tokens: int,
) -> None:
    try:
        events.put(
            (
                True,
                scout._request(
                    host=args.api_host,
                    port=args.api_port,
                    model=args.model,
                    prompt=prompt,
                    request_id=caller_id,
                    max_tokens=tokens,
                    timeout_s=args.request_timeout_s,
                ),
            )
        )
    except BaseException as exc:
        events.put((False, f"{type(exc).__name__}: {exc}"))


def _run_epoch(
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
    measured: bool = True,
) -> dict[str, Any]:
    protocol.install_generic_release_protocol()
    source = rank < SOURCE_COUNT
    receiver = not source
    expected = 1 + ((max(block_index, 0) * 37 + pair * 3) % 251)
    obj.raw_data.fill_(expected if source else 0)
    torch.cuda.synchronize()
    dist.barrier()

    label = f"b{block_index}" if measured else "fence-prewarm"
    caller_id = (
        f"tempo-scout-{args.allocation_id}-c{args.campaign_index}-{label}-{mode}"
    )
    events: queue.Queue[tuple[bool, Any]] = queue.Queue()
    listener = connection = ready = client = None
    if rank == 0:
        listener = gate.GateListener(
            gate.GateConfig(args.quiescence_socket, args.quiescence_trace, timeout_s=30.0)
        )
        listener.open()
        prompt = base.PROMPTS[prompt_index] if measured else base.WARMUP_PROMPT
        client = threading.Thread(
            target=_request_thread,
            kwargs={
                "events": events,
                "args": args,
                "prompt": prompt,
                "caller_id": caller_id,
                "tokens": TOKENS,
            },
            name=f"bulk-http-{label}",
        )
        client.start()
        connection = listener.accept()
        ready = connection.event
        if caller_id not in ready.request_id:
            raise RuntimeError("bulk gate ready request identity mismatch")

    gate_signal = torch.tensor(
        [int(ready.event_id) if rank == 0 else -1], dtype=torch.int64, device="cpu"
    )
    wave_started_ns = time.perf_counter_ns() if rank == 0 else 0
    dist.broadcast(gate_signal, src=0)
    if int(gate_signal.item()) < 0:
        raise RuntimeError("invalid bulk gate event id")

    calls = completed = descriptors = completed_bytes = elapsed_ns = 0
    polls = yields = error_flag = 0
    error = None
    if mode != NOOP and source:
        started_ns = time.perf_counter_ns()
        try:
            calls = 1
            descriptors = _descriptor_count(channel)
            spec = {
                "receiver_id": f"rank-{rank + RECEIVER_OFFSET}",
                "remote_indexes": np.asarray([0], dtype=np.uint64),
            }
            if mode == LMCACHE:
                completed = int(channel.batched_write(objects=[obj], transfer_spec=spec))
            elif mode == TEMPO:
                completed, polls, yields = channel.tempo_write([obj], spec)
            else:
                raise RuntimeError(f"unexpected bulk mode {mode}")
            completed_bytes = completed * BYTES_PER_SOURCE
        except BaseException as exc:
            error_flag = 1
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ns = time.perf_counter_ns() - started_ns

    status = torch.tensor(
        [
            calls,
            completed,
            descriptors,
            completed_bytes,
            elapsed_ns,
            error_flag,
            polls,
            yields,
        ],
        dtype=torch.int64,
        device="cpu",
    )
    gathered = [torch.zeros_like(status) for _ in range(WORLD_SIZE)] if rank == 0 else None
    dist.gather(status, gather_list=gathered, dst=0)
    wave_elapsed_ns = time.perf_counter_ns() - wave_started_ns if rank == 0 else 0

    release_error = None
    release_payload = None
    if rank == 0:
        try:
            rows = [tensor.tolist() for tensor in gathered]
            sources = rows[:SOURCE_COUNT]
            structural = mode == NOOP or all(
                row[0] == 1
                and row[1] == 1
                and row[2] == 1
                and row[3] == BYTES_PER_SOURCE
                and row[5] == 0
                for row in sources
            )
            if mode == NOOP:
                frame = protocol.ReleaseFrame.noop(ready)
            elif structural:
                frame = protocol.ReleaseFrame.wave(
                    ready,
                    mode=mode,
                    completed_bytes=GLOBAL_BYTES,
                    source_elapsed_ns=tuple(int(row[4]) for row in sources),
                    wave_elapsed_ns=wave_elapsed_ns,
                )
            else:
                frame = protocol.ReleaseFrame.noop(ready)
                release_error = "structural bulk transfer failure; emergency noop release"
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
    if not client_control[0]["ok"]:
        raise RuntimeError(f"vLLM request failed: {client_control[0]['value']}")

    dist.barrier()
    verified = 0
    zero_ok = True
    if receiver:
        if mode == NOOP:
            zero_ok = bool(torch.all(obj.raw_data == 0).item())
        else:
            verified = (
                BYTES_PER_SOURCE if bool(torch.all(obj.raw_data == expected).item()) else 0
            )
    dist.barrier()
    local = {
        "rank": rank,
        "source": source,
        "calls": calls,
        "completed": completed,
        "descriptors": descriptors,
        "bytes": completed_bytes,
        "elapsed_ns": elapsed_ns,
        "polls": polls,
        "yields": yields,
        "error": error,
    }
    return {
        "block_index": block_index,
        "prompt_index": prompt_index,
        "mode": mode,
        "measured": measured,
        "client": scout._client_metrics(client_control[0]["value"]) if rank == 0 else None,
        "gate_ready": ready.to_payload() if rank == 0 else None,
        "gate_release": {"payload": release_payload, "error": release_error}
        if rank == 0
        else None,
        "source_call": local,
        "receiver_verified_bytes": verified,
        "receiver_zero_ok": zero_ok,
        "wave_elapsed_ns": wave_elapsed_ns,
        "correctness_met": (
            release_error is None
            and (
                not source
                or mode == NOOP
                or calls == completed == descriptors == 1
                and error is None
            )
            and (
                not receiver
                or (zero_ok if mode == NOOP else verified == BYTES_PER_SOURCE)
            )
        ),
    }


def _validate_trace(path: Path, expected: list[tuple[str, str]]) -> dict[str, Any]:
    if not path.is_file() or not 0 < path.stat().st_size <= (1 << 20):
        raise ValueError("bulk quiescence trace is missing or unbounded")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) > 256:
        raise ValueError("bulk quiescence trace has too many rows")
    provenance = [row for row in rows if row.get("kind") == "provenance"]
    if len(provenance) != 1:
        raise ValueError("trace must contain one provenance row")
    required = {
        "protocol": gate.PROTOCOL,
        "world_size": 16,
        "tensor_parallel_size": 16,
        "async_scheduling": False,
        "speculative_decoding": False,
        "vllm_version": "0.26.0+cu129",
        "engine_core_process_step_sha256": (
            "41295db73bb85ebda9cee7c4f32d944e5f973b6bcc0433ff6b152a9368b175b9"
        ),
    }
    if any(provenance[0].get(key) != value for key, value in required.items()):
        raise ValueError("bulk trace provenance mismatch")
    if any(row.get("kind") == "error" for row in rows):
        raise ValueError("bulk trace contains a hook error")
    readies = {int(row["event_id"]): row for row in rows if row.get("kind") == "ready"}
    releases = {int(row["event_id"]): row for row in rows if row.get("kind") == "release"}
    nexts = {
        int(row["event_id"]): row
        for row in rows
        if row.get("kind") == "next_engine_step_enter"
    }
    ids = set(range(len(expected)))
    if set(readies) != ids or set(releases) != ids or set(nexts) != ids:
        raise ValueError("bulk trace event coverage mismatch")
    for event_id, (caller, mode) in enumerate(expected):
        ready, release, nxt = readies[event_id], releases[event_id], nexts[event_id]
        if caller not in ready.get("request_id", "") or release.get("mode") != mode:
            raise ValueError("bulk trace request/mode mismatch")
        fence = ready.get("fence_rows")
        if not isinstance(fence, list) or sorted(int(row[0]) for row in fence) != list(
            range(WORLD_SIZE)
        ):
            raise ValueError("bulk trace fence rank coverage mismatch")
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
            raise ValueError("bulk trace timeline ordering mismatch")
        expected_bytes = 0 if mode == NOOP else GLOBAL_BYTES
        if int(release.get("completed_bytes", -1)) != expected_bytes:
            raise ValueError("bulk trace byte geometry mismatch")
    return {"validated": True, "event_count": len(expected), "record_count": len(rows)}


def _aggregate(records: list[dict[str, Any]], trace: dict[str, Any], args: argparse.Namespace):
    ordered = sorted(records, key=lambda item: int(item["rank"]))
    if len(ordered) != WORLD_SIZE or [item["rank"] for item in ordered] != list(
        range(WORLD_SIZE)
    ):
        raise ValueError("bulk rank records are incomplete")
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
        expected = 0 if mode == NOOP else GLOBAL_BYTES
        correct = (
            all(bool(block["correctness_met"]) for block in rank_blocks)
            and completed == verified == expected
            and calls == descriptors == (0 if mode == NOOP else SOURCE_COUNT)
        )
        client = rank_blocks[0]["client"]
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
                "wave_elapsed_ms": rank_blocks[0]["wave_elapsed_ns"] / 1e6,
                "max_source_elapsed_ms": max(
                    float(block["source_call"]["elapsed_ns"]) / 1e6
                    for block in source_blocks
                ),
                "source_poll_count_max": max(
                    int(block["source_call"]["polls"]) for block in source_blocks
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
            "wave_p50_ms": statistics.median(block["wave_elapsed_ms"] for block in values),
            "wave_p99_ms": _percentile(
                (block["wave_elapsed_ms"] for block in values), 0.99
            ),
            "source_max_p50_ms": statistics.median(
                block["max_source_elapsed_ms"] for block in values
            ),
            "token31_to_32_p50_ms": statistics.median(
                block["token31_to_32_ms"] for block in values
            ),
            "e2e_p50_ms": statistics.median(block["request_e2e_ms"] for block in values),
        }
    paired = []
    for prompt in range(3):
        official = next(
            block for block in by_mode[LMCACHE] if block["prompt_index"] == prompt
        )
        tempo = next(block for block in by_mode[TEMPO] if block["prompt_index"] == prompt)
        paired.append(
            {
                "prompt_index": prompt,
                "wave_delta_ms": tempo["wave_elapsed_ms"] - official["wave_elapsed_ms"],
                "source_max_delta_ms": (
                    tempo["max_source_elapsed_ms"] - official["max_source_elapsed_ms"]
                ),
                "token31_to_32_delta_ms": (
                    tempo["token31_to_32_ms"] - official["token31_to_32_ms"]
                ),
                "e2e_delta_ms": tempo["request_e2e_ms"] - official["request_e2e_ms"],
            }
        )
    tempo_waves = [block["wave_elapsed_ms"] for block in by_mode[TEMPO]]
    gates = {
        "correctness_output_trace": overall,
        "tempo_median_wave_le_10ms": statistics.median(tempo_waves)
        <= MAX_MEDIAN_WAVE_MS,
        "every_tempo_wave_le_12ms": max(tempo_waves) <= MAX_INDIVIDUAL_WAVE_MS,
        "tempo_wave_median_beats_lmcache": (
            metrics[TEMPO]["wave_p50_ms"] < metrics[LMCACHE]["wave_p50_ms"]
        ),
    }
    return {
        "schema_version": RESULT_SCHEMA,
        "contract_id": CONTRACT_ID,
        "allocation_id": args.allocation_id,
        "campaign_index": args.campaign_index,
        "evidence_scope": "single-allocation real-vllm tp16 component campaign",
        "promotion_valid": False,
        "config": {
            "nodes": 4,
            "world_size": WORLD_SIZE,
            "tokens": TOKENS,
            "nixl_backend": args.nixl_backend,
            "bytes_per_source": BYTES_PER_SOURCE,
            "global_epoch_bytes": GLOBAL_BYTES,
            "completion_paths": {
                LMCACHE: "official LMCache batched_write 1ms polling",
                TEMPO: "same prepared NIXL WRITE with spin/yield completion",
            },
        },
        "hook_trace": trace,
        "hook_trace_validated": trace.get("validated") is True,
        "output_equivalence_met": output_equal,
        "overall_correctness_met": overall,
        "blocks": blocks,
        "mode_metrics": metrics,
        "paired_tempo_minus_lmcache": paired,
        "candidate_gates": gates,
        "screen_outcome": (
            "invalid_correctness_output_or_trace"
            if not overall
            else "bulk_epoch_candidate_pass"
            if all(gates.values())
            else "bulk_epoch_candidate_stop_or_revise"
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
        raise RuntimeError(f"bulk campaign requires nixl_cu12._api, got {nixl._api.__name__}")
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
            raise RuntimeError("bulk channel did not create exactly one descriptor")
        dist.barrier()
        if source:
            channel.lazy_init_peer_connection(
                local_id=f"rank-{rank}",
                peer_id=f"rank-{peer}",
                peer_init_url=f"{hosts[peer]}:{args.nixl_port_base + pair}",
            )
        dist.barrier()
        if not channel.remote_xfer_handler_exists(f"rank-{peer}"):
            raise RuntimeError("bulk LMCache peer handler is missing")
        _warm(torch, dist, channel, objects[0], rank, pair)

        prewarm = _run_epoch(
            torch,
            dist,
            channel=channel,
            obj=objects[0],
            rank=rank,
            pair=pair,
            block_index=-1,
            prompt_index=0,
            mode=NOOP,
            args=args,
            measured=False,
        )
        if not prewarm["correctness_met"]:
            raise RuntimeError("unmeasured accelerator-fence prewarm failed")

        blocks = [
            _run_epoch(
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
                f"tempo-scout-{args.allocation_id}-c{args.campaign_index}-fence-prewarm-{NOOP}",
                NOOP,
            )
        ] + [
            (
                f"tempo-scout-{args.allocation_id}-c{args.campaign_index}-b{i}-{mode}",
                mode,
            )
            for i, (_prompt, mode) in enumerate(BLOCKS)
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
            raise RuntimeError(f"bulk quiescence campaign failed: {final[0]}")
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
