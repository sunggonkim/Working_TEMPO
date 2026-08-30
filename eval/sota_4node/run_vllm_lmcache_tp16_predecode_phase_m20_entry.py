#!/usr/bin/env python3
"""M20: finish the prepared KV transfer before starting TP16 decode.

The candidate uses the same official LMCache NixlChannel, topology, buffers,
bytes, and one prepared descriptor per source as the prior campaigns.  Its
only algorithmic change is phase separation: all eight transfers complete
and receivers verify the payload before the HTTP request starts.  Admission
to-response time includes the transfer, so the comparison cannot hide the
serialization cost outside the measured service makespan.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import threading
import time
from typing import Any

from eval.sota_4node import run_vllm_lmcache_tp16_deadline_c9_entry as c9
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as v8
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol

CANDIDATE_MODE = "tempo_predecode_phase_separated"
CONTRACT_ID = "tp16-predecode-phase-separated-m20"
RESULT_SCHEMA = "tempo-vllm-tp16-predecode-phase-result-20"
BLOCKS = (
    (0, old.FG), (0, old.LMCACHE), (0, CANDIDATE_MODE),
    (1, CANDIDATE_MODE), (1, old.FG), (1, old.LMCACHE),
    (2, old.LMCACHE), (2, CANDIDATE_MODE), (2, old.FG),
)


def _expected_contract() -> dict[str, Any]:
    return {
        "schema_version": "tempo-tp16-predecode-phase-contract-20",
        "contract_id": CONTRACT_ID,
        "topology": {
            "nodes": 4, "world_size": 16,
            "source_ranks": list(range(8)),
            "receiver_ranks": list(range(8, 16)),
            "pairing": [[rank, rank + 8] for rank in range(8)],
        },
        "transfer": {
            "bytes_per_source": 16 << 20, "global_bytes": 128 << 20,
            "calls_global": 8, "physical_descriptors_global": 8,
            "prepared_handle_repost": True,
        },
        "algorithm": {
            "candidate_mode": CANDIDATE_MODE,
            "phase_order": ["remote_kv_transfer", "receiver_verify", "tp16_decode"],
            "transfer_decode_overlap": False,
            "decode_progress_sleep_ms": 1.0,
            "hook_events_per_measured_candidate": 0,
            "boost_enabled": False,
            "admission_to_response_includes_transfer": True,
        },
        "campaign": {
            "modes": [old.FG, old.LMCACHE, CANDIDATE_MODE],
            "blocks": 9, "replicates_per_mode": 3,
            "candidate_gates": {
                "exact_correctness": True,
                "all_transfers_complete_before_decode": True,
                "paired_service_delta_median_le_ms": -5.0,
                "meaningful_service_wins": 2,
                "request_e2e_le_fg_ratio": 1.03,
                "tpot_p99_le_lmcache_ratio": 1.10,
            },
        },
    }


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_contract():
        raise ValueError("M20 contract changed")
    return payload


def _run_candidate(
    torch: Any, dist: Any, *, channel: Any, obj: Any, rank: int, pair: int,
    block_index: int, prompt_index: int, args: Any,
) -> dict[str, Any]:
    source = rank < old.SOURCE_COUNT
    receiver = not source
    expected = 1 + ((block_index * 37 + pair * 3) % 251)
    obj.raw_data.fill_(expected if source else 0)
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.tensor([block_index], dtype=torch.int64, device="cpu")
    dist.broadcast(start, src=0)
    origin_ns = time.perf_counter_ns()
    if int(start.item()) != block_index:
        raise RuntimeError("M20 admission control changed")

    boost = threading.Event()
    entered = threading.Event()
    done = threading.Event()
    state: dict[str, Any] = {
        "started_ns": 0, "finished_ns": 0, "completed": 0, "polls": 0,
        "low_priority_sleeps": 0, "boost_polls": 0, "yields": 0, "error": None,
    }
    worker = None
    if source:
        worker = threading.Thread(
            target=fixed._transfer_worker,
            kwargs={
                "channel": channel, "obj": obj,
                "receiver_id": f"rank-{rank + old.RECEIVER_OFFSET}",
                "mode": CANDIDATE_MODE, "boost": boost,
                "entered": entered, "done": done, "state": state,
            },
            name=f"m20-transfer-r{rank}-b{block_index}", daemon=True,
        )
        worker.start()
    if source and not entered.wait(5.0):
        raise RuntimeError("M20 source worker did not enter")
    entered_ok = torch.tensor([int(not source or entered.is_set())], dtype=torch.int64)
    dist.all_reduce(entered_ok, op=dist.ReduceOp.MIN)
    if int(entered_ok.item()) != 1:
        raise RuntimeError("M20 worker-entry handshake failed")
    if source and not done.wait(60.0):
        raise RuntimeError("M20 transfer timed out")
    if worker is not None:
        worker.join()
    source_ok = torch.tensor(
        [int(not source or (state["completed"] == 1 and state["error"] is None))],
        dtype=torch.int64,
    )
    dist.all_reduce(source_ok, op=dist.ReduceOp.MIN)
    if int(source_ok.item()) != 1:
        raise RuntimeError("M20 source transfer failed")
    dist.barrier()
    verified = 0
    if receiver and bool(torch.all(obj.raw_data == expected).item()):
        verified = old.BYTES_PER_SOURCE
    verify_ok = torch.tensor(
        [int(not receiver or verified == old.BYTES_PER_SOURCE)], dtype=torch.int64
    )
    dist.all_reduce(verify_ok, op=dist.ReduceOp.MIN)
    if int(verify_ok.item()) != 1:
        raise RuntimeError("M20 receiver verification failed")
    transfer_phase_done_ns = time.perf_counter_ns()

    # Reuse the audited no-transfer request path after remote KV is resident.
    request = c9._run_block(
        torch, dist, channel=channel, obj=obj, rank=rank, pair=pair,
        block_index=block_index, prompt_index=prompt_index, mode=old.FG, args=args,
    )
    request_started_outer_ns = (
        transfer_phase_done_ns - origin_ns + int(request["client_started_from_origin_ns"])
    )
    request_finished_outer_ns = (
        transfer_phase_done_ns - origin_ns + int(request["client_finished_from_origin_ns"])
    )
    elapsed_ns = max(0, int(state["finished_ns"]) - int(state["started_ns"])) if source else 0
    completion_ns = max(0, int(state["finished_ns"]) - origin_ns) if source else 0
    call = {
        "rank": rank, "source": source, "calls": int(source),
        "completed": int(state["completed"]) if source else 0,
        "descriptors": old._descriptor_count(channel) if source else 0,
        "bytes": old.BYTES_PER_SOURCE if source and state["completed"] == 1 else 0,
        "elapsed_ns": elapsed_ns, "completion_from_origin_ns": completion_ns,
        "post_foreground_drain_ns": 0,
        "start_lag_ns": max(0, int(state["started_ns"]) - origin_ns) if source else 0,
        "polls": int(state["polls"]),
        "low_priority_sleeps": int(state["low_priority_sleeps"]),
        "boost_polls": int(state["boost_polls"]), "yields": int(state["yields"]),
        "boost_wait_timed_out": False, "error": state["error"],
    }
    return {
        **request,
        "mode": CANDIDATE_MODE,
        "client_started_from_origin_ns": request_started_outer_ns,
        "client_finished_from_origin_ns": request_finished_outer_ns,
        "controller_decision": "serialize_remote_before_decode",
        "transfer_completed_before_decode": True,
        "transfer_phase_elapsed_ns": transfer_phase_done_ns - origin_ns,
        "source_call": call,
        "receiver_verified_bytes": verified,
        "correctness_met": bool(request["correctness_met"]),
    }


def _run_block(*args: Any, mode: str, **kwargs: Any) -> dict[str, Any]:
    if mode == CANDIDATE_MODE:
        return _run_candidate(*args, mode=mode, **kwargs)
    return c9._run_block(*args, mode=mode, **kwargs)


def _aggregate(records: list[dict[str, Any]], trace: dict[str, Any], args: Any):
    result = c9._aggregate(records, trace, args)
    ordered = sorted(records, key=lambda row: int(row["rank"]))
    candidates = []
    for block in result["blocks"]:
        if block["mode"] != CANDIDATE_MODE:
            continue
        index = int(block["block_index"])
        raw = [row["blocks"][index] for row in ordered]
        outer_finish_ms = float(raw[0]["client_finished_from_origin_ns"]) / 1e6
        block["admission_to_response_ms"] = outer_finish_ms
        block["service_makespan_ms"] = max(
            outer_finish_ms, block["background_completion_from_start_ms"]
        )
        block["transfer_phase_elapsed_ms"] = max(
            float(row["transfer_phase_elapsed_ns"]) / 1e6 for row in raw
        )
        block["transfer_completed_before_decode"] = all(
            bool(row["transfer_completed_before_decode"]) for row in raw
        )
        candidates.append(block)

    by_mode = {
        mode: [row for row in result["blocks"] if row["mode"] == mode]
        for mode in (old.FG, old.LMCACHE, CANDIDATE_MODE)
    }
    result["mode_metrics"][CANDIDATE_MODE]["service_makespan_p50_ms"] = statistics.median(
        row["service_makespan_ms"] for row in candidates
    )
    deltas = []
    for paired in result["paired"]:
        prompt = int(paired["prompt_index"])
        candidate = next(row for row in candidates if row["prompt_index"] == prompt)
        baseline = next(row for row in by_mode[old.LMCACHE] if row["prompt_index"] == prompt)
        delta = candidate["service_makespan_ms"] - baseline["service_makespan_ms"]
        paired["tempo_minus_lmcache_service_makespan_ms"] = delta
        paired["tempo_admission_to_response_ms"] = candidate["admission_to_response_ms"]
        deltas.append(delta)
    fg_e2e = result["mode_metrics"][old.FG]["e2e_p50_ms"]
    cand_e2e = result["mode_metrics"][CANDIDATE_MODE]["e2e_p50_ms"]
    lm_tpot = result["mode_metrics"][old.LMCACHE]["tpot_p99_max_ms"]
    cand_tpot = result["mode_metrics"][CANDIDATE_MODE]["tpot_p99_max_ms"]
    gates = {
        "correctness_output_trace": bool(result["overall_correctness_met"]),
        "all_transfers_complete_before_decode": all(
            row["transfer_completed_before_decode"] for row in candidates
        ),
        "all_candidate_post_foreground_drain_zero": all(
            row["post_foreground_drain_ms"] == 0.0 for row in candidates
        ),
        "paired_service_median_le_minus_5ms": statistics.median(deltas) <= -5.0,
        "paired_service_meaningful_wins_ge_2": sum(delta <= -5.0 for delta in deltas) >= 2,
        "candidate_request_e2e_p50_le_1_03x_fg": cand_e2e <= 1.03 * fg_e2e,
        "candidate_tpot_p99_le_1_10x_lmcache": cand_tpot <= 1.10 * lm_tpot,
    }
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    result["config"].update(
        candidate_mode=CANDIDATE_MODE,
        phase_order=["remote_kv_transfer", "receiver_verify", "tp16_decode"],
        transfer_decode_overlap=False,
        admission_to_response_includes_transfer=True,
    )
    result["candidate_gates"] = gates
    result["screen_outcome"] = (
        "invalid_correctness_output_or_trace" if not result["overall_correctness_met"]
        else "predecode_phase_candidate_pass" if all(gates.values())
        else "predecode_phase_candidate_revise"
    )
    return result


def main() -> None:
    c9.CANDIDATE_MODE = CANDIDATE_MODE
    c9.CONTRACT_ID = CONTRACT_ID
    c9.RESULT_SCHEMA = RESULT_SCHEMA
    c9.BLOCKS = BLOCKS
    c9._install_candidate_mode()
    fixed._transfer_worker = old._transfer_worker
    protocol.install_async_release_protocol()
    old.protocol.ReleaseFrame = protocol.ReleaseFrame
    old.bulk.protocol.ReleaseFrame = protocol.ReleaseFrame
    v8.CONTRACT_ID = CONTRACT_ID
    v8.RESULT_SCHEMA = RESULT_SCHEMA
    v8._load_contract = _load_contract
    v8._run_block = _run_block
    v8._validate_trace = c9._validate_trace
    v8._aggregate = _aggregate
    v8.main()


if __name__ == "__main__":
    main()
