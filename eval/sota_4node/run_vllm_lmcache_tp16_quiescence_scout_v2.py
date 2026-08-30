#!/usr/bin/env python3
"""Fixed-tensor control-plane variant of the TP16 quiescence scout."""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import Any

from eval.sota_4node import run_vllm_lmcache_tp16_quiescence_scout_v1 as v1
from eval.sota_4node import vllm_decode_quiescence_gate_launch_v3 as gate


def _run_block_fast(
    torch: Any, dist: Any, *, channel: Any, obj: Any, rank: int,
    pair: int, block_index: int, prompt_index: int, mode: str,
    args: Any,
) -> dict[str, Any]:
    source = rank < v1.SOURCE_COUNT
    receiver = not source
    expected = 1 + ((block_index * 37 + pair * 3) % 251)
    obj.raw_data.fill_(expected if source else 0)
    torch.cuda.synchronize()
    dist.barrier()

    caller_id = (
        f"tempo-scout-{args.allocation_id}-c{args.campaign_index}"
        f"-b{block_index}-{mode}"
    )
    events: queue.Queue[tuple[bool, Any]] = queue.Queue()
    client = None
    listener = None
    connection = None
    ready = None
    if rank == 0:
        def target() -> None:
            try:
                events.put((True, v1._request(
                    host=args.api_host, port=args.api_port, model=args.model,
                    prompt=v1.base.PROMPTS[prompt_index], request_id=caller_id,
                    max_tokens=v1.TOKENS, timeout_s=args.request_timeout_s,
                )))
            except BaseException as exc:
                events.put((False, f"{type(exc).__name__}: {exc}"))

        listener = gate.GateListener(gate.GateConfig(
            args.quiescence_socket, args.quiescence_trace, timeout_s=10.0
        ))
        listener.open()
        client = threading.Thread(target=target, name=f"fast-gate-http-{block_index}")
        client.start()
        connection = listener.accept()
        ready = connection.event
        if caller_id not in ready.request_id:
            raise RuntimeError("gate ready request id does not match block")

    # A fixed one-word tensor replaces JSON/object serialization on the hot path.
    gate_signal = torch.tensor(
        [int(ready.event_id) if rank == 0 else -1], dtype=torch.int64, device="cpu"
    )
    wave_started_ns = time.perf_counter_ns() if rank == 0 else 0
    dist.broadcast(gate_signal, src=0)
    if int(gate_signal.item()) < 0:
        raise RuntimeError("invalid fixed-tensor gate event id")

    calls = completed = descriptors = completed_bytes = elapsed_ns = error_flag = 0
    error = None
    if mode == "quiescent_512k" and source:
        started_ns = time.perf_counter_ns()
        try:
            calls = 1
            descriptors = v1._descriptor_count(channel)
            completed = int(channel.batched_write(
                objects=[obj],
                transfer_spec={
                    "receiver_id": f"rank-{rank + v1.RECEIVER_OFFSET}",
                    "remote_indexes": v1.np.asarray([0], dtype=v1.np.uint64),
                },
            ))
            completed_bytes = completed * v1.CHUNK_BYTES
        except BaseException as exc:
            error_flag = 1
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ns = time.perf_counter_ns() - started_ns

    status = torch.tensor(
        [calls, completed, descriptors, completed_bytes, elapsed_ns, error_flag],
        dtype=torch.int64,
        device="cpu",
    )
    gathered = [torch.zeros_like(status) for _ in range(v1.WORLD_SIZE)] if rank == 0 else None
    dist.gather(status, gather_list=gathered, dst=0)
    wave_elapsed_ns = time.perf_counter_ns() - wave_started_ns if rank == 0 else 0

    release_error = None
    release_payload = None
    if rank == 0:
        try:
            rows = [tensor.tolist() for tensor in gathered]
            sources = rows[:v1.SOURCE_COUNT]
            structural = mode == "quiescent_noop" or all(
                row[0] == 1 and row[1] == 1 and row[2] == 1
                and row[3] == v1.CHUNK_BYTES and row[5] == 0
                for row in sources
            )
            if mode == "quiescent_512k" and structural:
                frame = gate.ReleaseFrame.wave512k(
                    ready,
                    source_elapsed_ns=tuple(int(row[4]) for row in sources),
                    wave_elapsed_ns=wave_elapsed_ns,
                )
            else:
                frame = gate.ReleaseFrame.noop(ready)
                if mode == "quiescent_512k":
                    release_error = "structural transfer failure; emergency noop release"
            # Performance failure never suppresses release.
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
        if mode == "quiescent_512k":
            verified = v1.CHUNK_BYTES if bool(torch.all(obj.raw_data == expected).item()) else 0
        else:
            zero_ok = bool(torch.all(obj.raw_data == 0).item())
    dist.barrier()

    local = {
        "rank": rank,
        "source": source,
        "calls": calls,
        "completed": completed,
        "descriptors": descriptors,
        "bytes": completed_bytes,
        "elapsed_ns": elapsed_ns,
        "error": error,
    }
    return {
        "block_index": block_index,
        "prompt_index": prompt_index,
        "mode": mode,
        "client": v1._client_metrics(client_control[0]["value"]) if rank == 0 else None,
        "gate_ready": ready.to_payload() if rank == 0 else None,
        "gate_release": {"payload": release_payload, "error": release_error} if rank == 0 else None,
        "source_call": local,
        "receiver_verified_bytes": verified,
        "receiver_zero_ok": zero_ok,
        "wave_elapsed_ns": wave_elapsed_ns,
        "control_plane": "fixed_int64_broadcast_gather",
        "correctness_met": (
            release_error is None
            and (not source or mode == "quiescent_noop" or (
                calls == 1 and completed == 1 and descriptors == 1 and error is None
            ))
            and (not receiver or (
                verified == v1.CHUNK_BYTES if mode == "quiescent_512k" else zero_ok
            ))
        ),
    }


def main() -> None:
    v1._run_block = _run_block_fast
    v1.main()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
