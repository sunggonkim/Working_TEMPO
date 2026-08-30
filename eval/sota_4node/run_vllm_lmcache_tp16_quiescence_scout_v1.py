#!/usr/bin/env python3
"""Minimal TP16 quiescence A/B: three noop and three 8x512KiB waves."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import queue
import socket
import statistics
import sys
import threading
import time
from typing import Any

import numpy as np

from eval.sota_4node import run_vllm_lmcache_tp8_sidecar as base
from eval.sota_4node import run_vllm_lmcache_tp16_pair_stagger_coalesced_v1 as tp16
from eval.sota_4node import vllm_decode_quiescence_gate_launch_v3 as gate


WORLD_SIZE = 16
SOURCE_COUNT = 8
RECEIVER_OFFSET = 8
CHUNK_BYTES = 512 << 10
GLOBAL_BYTES = SOURCE_COUNT * CHUNK_BYTES
TOKENS = 64
MODES = ("quiescent_noop", "quiescent_512k")
# Each prompt sees both orders; each mode has three measured blocks.
BLOCKS = (
    (0, "quiescent_noop"),
    (0, "quiescent_512k"),
    (1, "quiescent_512k"),
    (1, "quiescent_noop"),
    (2, "quiescent_noop"),
    (2, "quiescent_512k"),
)


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
    if not 1024 <= args.api_port <= 65535:
        parser.error("api-port is invalid")
    if not 1024 <= args.nixl_port_base <= 65528:
        parser.error("nixl-port-base is invalid")
    if args.request_timeout_s <= 0:
        parser.error("request timeout must be positive")
    for value, prefix, suffix in (
        (args.quiescence_socket, "tempo-vllm-quiescence-", ".sock"),
        (args.quiescence_trace, "tempo-step-gate-", ".jsonl"),
    ):
        if not value.is_absolute() or value.parent != Path("/tmp"):
            parser.error("quiescence paths must be immediate /tmp children")
        if not value.name.startswith(prefix) or not value.name.endswith(suffix):
            parser.error("quiescence path name changed")
    return args


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract_id") != "tp16-quiescence-scout-v1":
        raise ValueError("quiescence scout contract changed")
    wave = payload.get("wave", {})
    if (
        wave.get("bytes_per_source") != CHUNK_BYTES
        or wave.get("global_bytes") != GLOBAL_BYTES
        or wave.get("source_calls") != SOURCE_COUNT
        or wave.get("physical_descriptors") != SOURCE_COUNT
    ):
        raise ValueError("quiescence scout wave contract changed")
    return payload


def _make_memory(torch: Any, TensorMemoryObj: Any, MemoryObjMetadata: Any, MemoryFormat: Any):
    backing = torch.empty(2 * CHUNK_BYTES - 1, dtype=torch.uint8, device="cuda")
    offset = (-backing.data_ptr()) % CHUNK_BYTES
    buffer = backing[offset : offset + CHUNK_BYTES]
    shape = torch.Size([CHUNK_BYTES])
    obj = TensorMemoryObj(
        raw_data=buffer,
        metadata=MemoryObjMetadata(
            shape=shape,
            dtype=torch.uint8,
            address=buffer.data_ptr(),
            phy_size=CHUNK_BYTES,
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
    method = getattr(channel.nixl_wrapper.xfer_descs, "descCount", None)
    if not callable(method):
        raise RuntimeError("NIXL descriptor list lacks descCount")
    return int(method())


def _request(
    *, host: str, port: int, model: str, prompt: str, request_id: str,
    max_tokens: int, timeout_s: float,
) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "request_id": request_id,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
    })
    started_ns = time.perf_counter_ns()
    try:
        connection.request("POST", "/v1/completions", body=body,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"vLLM HTTP {response.status}: {response.read(4096)!r}")
        token_ids: list[int] = []
        arrivals: list[int] = []
        text: list[str] = []
        for ids, delta, arrived_ns in base.iter_sse_chunks(response):
            text.append(delta)
            for token_id in ids:
                token_ids.append(int(token_id))
                arrivals.append(int(arrived_ns))
        finished_ns = time.perf_counter_ns()
    finally:
        connection.close()
    if len(token_ids) != max_tokens:
        raise RuntimeError(f"generated {len(token_ids)} tokens, expected {max_tokens}")
    encoded = json.dumps(token_ids, separators=(",", ":")).encode()
    return {
        "request_id": request_id,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "arrivals_ns": arrivals,
        "token_ids": token_ids,
        "output_token_sha256": hashlib.sha256(encoded).hexdigest(),
        "output_text_sha256": hashlib.sha256("".join(text).encode()).hexdigest(),
    }


def _warm(torch: Any, dist: Any, channel: Any, obj: Any, rank: int, pair: int) -> None:
    source = rank < SOURCE_COUNT
    expected = 181 + pair
    obj.raw_data.fill_(expected if source else 0)
    torch.cuda.synchronize()
    dist.barrier()
    local = {"ok": True, "count": 0, "error": None}
    if source:
        try:
            local["count"] = int(channel.batched_write(
                objects=[obj],
                transfer_spec={
                    "receiver_id": f"rank-{rank + RECEIVER_OFFSET}",
                    "remote_indexes": np.asarray([0], dtype=np.uint64),
                },
            ))
            local["ok"] = local["count"] == 1
        except BaseException as exc:
            local.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    dist.barrier()
    if not source:
        local["ok"] = bool(torch.all(obj.raw_data == expected).item())
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, local)
    if not all(bool(row["ok"]) for row in rows):
        raise RuntimeError(f"single-object NIXL warmup failed: {rows}")
    dist.barrier()


def _client_metrics(result: dict[str, Any]) -> dict[str, Any]:
    arrivals = result["arrivals_ns"]
    intervals = [(arrivals[i] - arrivals[i - 1]) / 1e6 for i in range(1, len(arrivals))]
    ordered = sorted(intervals)
    p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
    return {
        "request_id": result["request_id"],
        "generated_tokens": len(arrivals),
        "output_token_sha256": result["output_token_sha256"],
        "output_text_sha256": result["output_text_sha256"],
        "ttft_ms": (arrivals[0] - result["started_ns"]) / 1e6,
        "tpot_p50_ms": statistics.median(intervals),
        "tpot_p99_ms": p99,
        "request_e2e_ms": (result["finished_ns"] - result["started_ns"]) / 1e6,
        "token31_to_32_ms": (arrivals[31] - arrivals[30]) / 1e6,
    }


def _run_block(
    torch: Any, dist: Any, *, channel: Any, obj: Any, rank: int,
    pair: int, block_index: int, prompt_index: int, mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source = rank < SOURCE_COUNT
    receiver = not source
    expected = 1 + ((block_index * 37 + pair * 3) % 251)
    obj.raw_data.fill_(expected if source else 0)
    torch.cuda.synchronize()
    dist.barrier()

    caller_id = f"tempo-scout-{args.allocation_id}-c{args.campaign_index}-b{block_index}-{mode}"
    events: queue.Queue[tuple[bool, Any]] = queue.Queue()
    client = None
    if rank == 0:
        def target() -> None:
            try:
                events.put((True, _request(
                    host=args.api_host, port=args.api_port, model=args.model,
                    prompt=base.PROMPTS[prompt_index], request_id=caller_id,
                    max_tokens=TOKENS, timeout_s=args.request_timeout_s,
                )))
            except BaseException as exc:
                events.put((False, f"{type(exc).__name__}: {exc}"))
        client = threading.Thread(target=target, name=f"quiescence-http-{block_index}")

    listener = None
    connection = None
    ready = None
    if rank == 0:
        listener = gate.GateListener(gate.GateConfig(
            args.quiescence_socket, args.quiescence_trace, timeout_s=10.0
        ))
        listener.open()
        client.start()
        connection = listener.accept()
        ready = connection.event
        if caller_id not in ready.request_id:
            raise RuntimeError("gate ready request id does not match block")

    ready_control: list[Any] = [ready.to_payload() if rank == 0 else None]
    dist.broadcast_object_list(ready_control, src=0)
    decoded_ready = gate.ReadyEvent.from_payload(ready_control[0])
    if caller_id not in decoded_ready.request_id:
        raise RuntimeError("broadcast ready request id mismatch")

    local = {
        "rank": rank, "source": source, "calls": 0, "completed": 0,
        "descriptors": 0, "bytes": 0, "elapsed_ns": 0, "error": None,
    }
    wave_started_ns = time.perf_counter_ns()
    if mode == "quiescent_512k" and source:
        started = time.perf_counter_ns()
        try:
            local["calls"] = 1
            local["descriptors"] = _descriptor_count(channel)
            local["completed"] = int(channel.batched_write(
                objects=[obj],
                transfer_spec={
                    "receiver_id": f"rank-{rank + RECEIVER_OFFSET}",
                    "remote_indexes": np.asarray([0], dtype=np.uint64),
                },
            ))
            local["bytes"] = local["completed"] * CHUNK_BYTES
        except BaseException as exc:
            local["error"] = f"{type(exc).__name__}: {exc}"
        local["elapsed_ns"] = time.perf_counter_ns() - started
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, local)
    wave_elapsed_ns = time.perf_counter_ns() - wave_started_ns

    release_error = None
    release_payload = None
    if rank == 0:
        try:
            sources = rows[:SOURCE_COUNT]
            structural = (
                mode == "quiescent_noop" or all(
                    row["source"] and row["calls"] == 1 and row["completed"] == 1
                    and row["descriptors"] == 1 and row["bytes"] == CHUNK_BYTES
                    and row["error"] is None for row in sources
                )
            )
            if mode == "quiescent_512k" and structural:
                frame = gate.ReleaseFrame.wave512k(
                    ready,
                    source_elapsed_ns=tuple(int(row["elapsed_ns"]) for row in sources),
                    wave_elapsed_ns=wave_elapsed_ns,
                )
            else:
                frame = gate.ReleaseFrame.noop(ready)
                if mode == "quiescent_512k":
                    release_error = "structural transfer failure; emergency noop release"
            connection.release(frame)
            release_payload = frame.to_payload()
        finally:
            if connection is not None:
                connection.close()
            if listener is not None:
                listener.close()
    release_control: list[Any] = [{"payload": release_payload, "error": release_error} if rank == 0 else None]
    dist.broadcast_object_list(release_control, src=0)

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
        if mode == "quiescent_512k":
            verified = CHUNK_BYTES if bool(torch.all(obj.raw_data == expected).item()) else 0
        else:
            zero_ok = bool(torch.all(obj.raw_data == 0).item())
    dist.barrier()

    return {
        "block_index": block_index,
        "prompt_index": prompt_index,
        "mode": mode,
        "client": _client_metrics(client_control[0]["value"]) if rank == 0 else None,
        "gate_ready": ready_control[0] if rank == 0 else None,
        "gate_release": release_control[0] if rank == 0 else None,
        "source_call": local,
        "receiver_verified_bytes": verified,
        "receiver_zero_ok": zero_ok,
        "wave_elapsed_ns": wave_elapsed_ns if rank == 0 else 0,
        "correctness_met": (
            release_error is None
            and (not source or mode == "quiescent_noop" or (
                local["calls"] == 1 and local["completed"] == 1
                and local["descriptors"] == 1 and local["error"] is None
            ))
            and (not receiver or (verified == CHUNK_BYTES if mode == "quiescent_512k" else zero_ok))
        ),
    }


def _validate_trace(path: Path, expected: list[tuple[str, str]]) -> dict[str, Any]:
    if not path.is_file() or not 0 < path.stat().st_size <= (1 << 20):
        raise ValueError("quiescence trace is missing or unbounded")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) > 256:
        raise ValueError("quiescence trace has too many records")
    rows = [json.loads(line) for line in lines]
    provenance = [row for row in rows if row.get("kind") == "provenance"]
    if len(provenance) != 1:
        raise ValueError("trace must have one provenance record")
    p = provenance[0]
    required = {
        "protocol": gate.PROTOCOL,
        "world_size": 16,
        "tensor_parallel_size": 16,
        "async_scheduling": False,
        "speculative_decoding": False,
        "vllm_version": "0.26.0+cu129",
        "engine_core_process_step_sha256": "41295db73bb85ebda9cee7c4f32d944e5f973b6bcc0433ff6b152a9368b175b9",
    }
    if any(p.get(key) != value for key, value in required.items()):
        raise ValueError("trace provenance mismatch")
    if any(row.get("kind") == "error" for row in rows):
        raise ValueError("trace contains hook error")
    readies = {int(row["event_id"]): row for row in rows if row.get("kind") == "ready"}
    releases = {int(row["event_id"]): row for row in rows if row.get("kind") == "release"}
    nexts = {int(row["event_id"]): row for row in rows if row.get("kind") == "next_engine_step_enter"}
    if set(readies) != set(range(len(expected))) or set(releases) != set(readies) or set(nexts) != set(readies):
        raise ValueError("trace event-id coverage mismatch")
    for event_id, (caller, mode) in enumerate(expected):
        ready, release, nxt = readies[event_id], releases[event_id], nexts[event_id]
        if caller not in ready.get("request_id", "") or ready["request_id"] != release.get("request_id"):
            raise ValueError("trace request identity mismatch")
        if release.get("mode") != mode:
            raise ValueError("trace release mode mismatch")
        fence = ready.get("fence_rows")
        if not isinstance(fence, list) or sorted(int(row[0]) for row in fence) != list(range(16)):
            raise ValueError("trace fence rank coverage mismatch")
        timeline = (
            int(ready["output_enqueued_ns"]), int(ready["fence_started_ns"]),
            int(ready["fence_finished_ns"]), int(ready["ready_ns"]),
            int(release["released_ns"]), int(release["gate_returned_ns"]),
            int(nxt["entered_ns"]),
        )
        if list(timeline) != sorted(timeline):
            raise ValueError("trace timeline ordering mismatch")
    return {"validated": True, "event_count": len(expected), "record_count": len(rows)}


def _aggregate(records: list[dict[str, Any]], trace: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: int(item["rank"]))
    if len(ordered) != WORLD_SIZE or [item["rank"] for item in ordered] != list(range(WORLD_SIZE)):
        raise ValueError("rank records are incomplete")
    blocks = []
    for index, (prompt, mode) in enumerate(BLOCKS):
        rank_blocks = [item["blocks"][index] for item in ordered]
        source_blocks, receiver_blocks = rank_blocks[:8], rank_blocks[8:]
        client = rank_blocks[0]["client"]
        completed = sum(int(block["source_call"]["bytes"]) for block in source_blocks)
        calls = sum(int(block["source_call"]["calls"]) for block in source_blocks)
        descriptors = sum(int(block["source_call"]["descriptors"]) for block in source_blocks)
        verified = sum(int(block["receiver_verified_bytes"]) for block in receiver_blocks)
        expected_bytes = GLOBAL_BYTES if mode == "quiescent_512k" else 0
        correct = (
            all(bool(block["correctness_met"]) for block in rank_blocks)
            and completed == expected_bytes and verified == expected_bytes
            and calls == (8 if mode == "quiescent_512k" else 0)
            and descriptors == (8 if mode == "quiescent_512k" else 0)
        )
        blocks.append({
            "block_index": index, "prompt_index": prompt, "mode": mode,
            **client, "background_completed_bytes": completed,
            "receiver_verified_bytes": verified, "source_calls": calls,
            "physical_descriptors": descriptors,
            "wave_elapsed_ms": rank_blocks[0]["wave_elapsed_ns"] / 1e6,
            "max_source_elapsed_ms": max(float(block["source_call"]["elapsed_ns"]) / 1e6 for block in source_blocks),
            "correctness_met": correct,
        })
    output_equal = all(
        len({b["output_token_sha256"] for b in blocks if b["prompt_index"] == prompt}) == 1
        for prompt in range(3)
    )
    overall = trace.get("validated") is True and output_equal and all(b["correctness_met"] for b in blocks)
    transfer = [b for b in blocks if b["mode"] == "quiescent_512k"]
    noop = [b for b in blocks if b["mode"] == "quiescent_noop"]
    service_gate = all(b["max_source_elapsed_ms"] <= 4.0 and b["wave_elapsed_ms"] <= 5.0 for b in transfer)
    pairs = []
    for prompt in range(3):
        n = next(b for b in noop if b["prompt_index"] == prompt)
        t = next(b for b in transfer if b["prompt_index"] == prompt)
        pairs.append({
            "prompt_index": prompt,
            "token31_to_32_delta_ms": t["token31_to_32_ms"] - n["token31_to_32_ms"],
            "e2e_delta_ms": t["request_e2e_ms"] - n["request_e2e_ms"],
        })
    return {
        "schema_version": "tempo-vllm-tp16-quiescence-scout-result-1",
        "allocation_id": args.allocation_id,
        "campaign_index": args.campaign_index,
        "evidence_scope": "single-allocation oracle time-division component scout",
        "promotion_valid": False,
        "config": {"nodes": 4, "world_size": 16, "tokens": 64, "global_wave_bytes": GLOBAL_BYTES},
        "hook_trace": trace,
        "hook_trace_validated": trace.get("validated") is True,
        "output_equivalence_met": output_equal,
        "overall_correctness_met": overall,
        "service_gate_met": service_gate,
        "blocks": blocks,
        "paired_transfer_minus_noop": pairs,
        "screen_outcome": (
            "invalid_correctness_or_trace" if not overall else
            "stop_quiescent_service_gate" if not service_gate else
            "quiescent_wave_service_gate_pass"
        ),
    }


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    args.output_dir = base._resolve_below_repo(args.output_dir, repo_root, label="output-dir")
    args.plan = base._resolve_below_repo(args.plan, repo_root, label="plan")
    args.model = str(Path(args.model).resolve())
    _load_contract(args.plan)
    base._set_rank_environment()
    import torch
    import torch.distributed as dist
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise SystemExit("CUDA and WORLD_SIZE=16 are required")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    visible = torch.cuda.device_count()
    device = 0 if visible == 1 else local_rank
    torch.cuda.set_device(device)
    dist.init_process_group("gloo")
    try:
        hosts = tp16._validate_topology(dist, rank, local_rank)
        NixlChannel, TensorMemoryObj, MemoryObjMetadata, MemoryFormat = base.official._load_official_lmcache(repo_root)
        backing, buffer, objects, index_by_address = _make_memory(
            torch, TensorMemoryObj, MemoryObjMetadata, MemoryFormat
        )
        pair = rank % SOURCE_COUNT
        source = rank < SOURCE_COUNT
        peer = rank + RECEIVER_OFFSET if source else rank - RECEIVER_OFFSET
        channel = NixlChannel(
            async_mode=False, role="sender" if source else "receiver",
            buffer_ptr=buffer.data_ptr(), buffer_size=buffer.numel(),
            align_bytes=CHUNK_BYTES, tp_rank=local_rank,
            peer_init_url=None if source else f"*:{args.nixl_port_base + pair}",
            backends=["UCX"], device=f"cuda:{device}",
        )
        base.epoch._install_descriptor_index_shim(channel, index_by_address)
        if _descriptor_count(channel) != 1:
            raise RuntimeError("official LMCache did not create one descriptor")
        dist.barrier()
        if source:
            channel.lazy_init_peer_connection(
                local_id=f"rank-{rank}", peer_id=f"rank-{peer}",
                peer_init_url=f"{hosts[peer]}:{args.nixl_port_base + pair}",
            )
        dist.barrier()
        if not channel.remote_xfer_handler_exists(f"rank-{peer}"):
            raise RuntimeError("LMCache peer handler is missing")
        _warm(torch, dist, channel, objects[0], rank, pair)

        warm_control: list[Any] = [None]
        if rank == 0:
            try:
                warm = _request(
                    host=args.api_host, port=args.api_port, model=args.model,
                    prompt=base.WARMUP_PROMPT,
                    request_id=f"control-warmup-{args.allocation_id}",
                    max_tokens=8, timeout_s=args.request_timeout_s,
                )
                warm_control[0] = {"ok": True, "tokens": len(warm["token_ids"])}
            except BaseException as exc:
                warm_control[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(warm_control, src=0)
        if not warm_control[0]["ok"]:
            raise RuntimeError(f"vLLM warmup failed: {warm_control[0]}")

        blocks = [
            _run_block(
                torch, dist, channel=channel, obj=objects[0], rank=rank,
                pair=pair, block_index=index, prompt_index=prompt, mode=mode,
                args=args,
            )
            for index, (prompt, mode) in enumerate(BLOCKS)
        ]
        expected_trace = [
            (f"tempo-scout-{args.allocation_id}-c{args.campaign_index}-b{i}-{mode}", mode)
            for i, (_prompt, mode) in enumerate(BLOCKS)
        ]
        trace_control: list[Any] = [None]
        if rank == 0:
            try:
                trace_control[0] = _validate_trace(args.quiescence_trace, expected_trace)
            except BaseException as exc:
                trace_control[0] = {"validated": False, "error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(trace_control, src=0)

        record = {"rank": rank, "local_rank": local_rank, "hostname": hosts[rank], "blocks": blocks}
        gathered = [None] * WORLD_SIZE if rank == 0 else None
        dist.gather_object(record, gathered, dst=0)
        final: list[Any] = [None]
        if rank == 0:
            try:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                for item in gathered:
                    (args.output_dir / f"rank_{item['rank']}.json").write_text(
                        json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                result = _aggregate(gathered, trace_control[0], args)
                path = args.output_dir / "result.json"
                path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                final[0] = {"ok": result["overall_correctness_met"], "output": str(path), "outcome": result["screen_outcome"]}
            except BaseException as exc:
                final[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(final, src=0)
        dist.barrier()
        if not final[0]["ok"]:
            raise RuntimeError(f"quiescence scout failed: {final[0]}")
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
