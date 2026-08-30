#!/usr/bin/env python3
"""Candidate C9: deadline-aware defer with no measured decode hook.

This keeps v8's prepared, single-flight 16 MiB/source transfer and worker-entry
prelaunch ordering.  Candidate B observed at least 386.655128 ms of completion
slack, above the predeclared 250 ms defer threshold, so C9 takes the controller's
``defer/no_rescue`` branch: candidate requests are not marked for the token-31
hook, the boost event remains unset, and the low-rate prepared worker is allowed
to finish naturally.  Exact transfer and receiver verification still happen
after the foreground response.
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
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as v8
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol


CANDIDATE_MODE = "tempo_prelaunch_deadline_defer"
CONTRACT_ID = "tp16-single-flight-deadline-defer-c9"
RESULT_SCHEMA = "tempo-vllm-tp16-single-flight-deadline-result-9"
CONTROLLER_DECISION = "defer/no_rescue"
PRIOR_OBSERVED_MIN_SLACK_MS = 386.655128
DEFER_THRESHOLD_MS = 250.0
E2E_RATIO_MAX = 1.05
TPOT_P99_RATIO_MAX = 1.10

BLOCKS = (
    (0, old.FG),
    (0, old.LMCACHE),
    (0, CANDIDATE_MODE),
    (1, CANDIDATE_MODE),
    (1, old.FG),
    (1, old.LMCACHE),
    (2, old.LMCACHE),
    (2, CANDIDATE_MODE),
    (2, old.FG),
)

_ORIGINAL_VALIDATE_TRACE = old._validate_trace
_V6_VALIDATE_TRACE = fixed._validate_trace
_V8_AGGREGATE = v8._aggregate


def _install_candidate_mode() -> None:
    old.TEMPO = CANDIDATE_MODE
    old.MODES = (old.FG, old.LMCACHE, CANDIDATE_MODE)
    old.BLOCKS = BLOCKS


def _expected_contract() -> dict[str, Any]:
    return {
        "schema_version": "tempo-tp16-single-flight-deadline-contract-9",
        "contract_id": CONTRACT_ID,
        "topology": {
            "nodes": 4,
            "world_size": 16,
            "source_ranks": list(range(8)),
            "receiver_ranks": list(range(8, 16)),
            "pairing": [[rank, rank + 8] for rank in range(8)],
        },
        "transfer": {
            "bytes_per_source": 16 << 20,
            "global_bytes": 128 << 20,
            "calls_global": 8,
            "physical_descriptors_global": 8,
            "single_flight_per_source": True,
            "prelaunch_at_request_start": True,
            "worker_entry_precedes_http_request": True,
            "prepared_handle_repost": True,
        },
        "deadline_controller": {
            "candidate_mode": CANDIDATE_MODE,
            "decision": CONTROLLER_DECISION,
            "decision_rule": (
                "defer_if_prior_observed_min_slack_ms_ge_threshold_ms"
            ),
            "basis_result": (
                "results/vllm_lmcache_tp16_hybrid_B_async_job_56946009/result.json"
            ),
            "prior_observed_min_slack_ms": PRIOR_OBSERVED_MIN_SLACK_MS,
            "defer_threshold_ms": DEFER_THRESHOLD_MS,
            "decode_progress_sleep_ms": 1.0,
            "measured_candidate_request_marked": False,
            "token31_hook_events_per_candidate": 0,
            "gate_collectives_per_candidate": 0,
            "rescue_armed_sources": 0,
            "permanent_boost": False,
            "runtime_rescue_enabled": False,
        },
        "campaign": {
            "modes": [old.FG, old.LMCACHE, CANDIDATE_MODE],
            "blocks": 9,
            "replicates_per_mode": 3,
            "unmeasured_fence_prewarm": True,
            "rank_block_identity_fail_closed": True,
            "duplicate_trace_ids_fail_closed": True,
            "candidate_gates": {
                "no_measured_candidate_hook_events": True,
                "all_candidate_gate_bubbles_zero": True,
                "all_candidate_boost_polls_zero": True,
                "all_candidate_post_foreground_drain_zero": True,
                "all_candidate_observed_deadlines_met": True,
                "paired_service_win_min_prompts": 2,
                "paired_service_delta_median_lt_ms": 0.0,
                "candidate_e2e_p50_le_fg_ratio": E2E_RATIO_MAX,
                "candidate_tpot_p99_le_lmcache_ratio": TPOT_P99_RATIO_MAX,
            },
        },
    }


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_contract():
        raise ValueError("TP16 deadline defer C9 contract changed")
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
    protocol.install_async_release_protocol()
    source = rank < old.SOURCE_COUNT
    receiver = not source
    candidate = mode == CANDIDATE_MODE
    expected = 1 + ((block_index * 37 + pair * 3) % 251)
    obj.raw_data.fill_(expected if source and mode != old.FG else 0)
    torch.cuda.synchronize()
    dist.barrier()

    # The C9 decision is made before the measured run from prior observed slack.
    # A control-prefixed request is deliberately invisible to the token-31 hook.
    caller_id = (
        f"control-{args.allocation_id}-c{args.campaign_index}-b{block_index}-{mode}"
    )
    events: queue.Queue[tuple[bool, Any]] = queue.Queue()

    start_signal = torch.tensor([block_index], dtype=torch.int64, device="cpu")
    dist.broadcast(start_signal, src=0)
    local_origin_ns = time.perf_counter_ns()
    if int(start_signal.item()) != block_index:
        raise RuntimeError("deadline C9 request-start control changed")

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
            name=f"deadline-c9-transfer-rank{rank}-block{block_index}",
            daemon=True,
        )
        worker.start()

    if source and mode != old.FG and not entered.wait(5.0):
        raise RuntimeError("deadline C9 source worker did not enter transfer call")
    entered_status = torch.tensor(
        [1 if not source or mode == old.FG or entered.is_set() else 0],
        dtype=torch.int64,
        device="cpu",
    )
    dist.all_reduce(entered_status, op=dist.ReduceOp.MIN)
    if int(entered_status.item()) != 1:
        raise RuntimeError("deadline C9 prelaunch worker-entry handshake failed")

    client_started_from_origin_ns = 0
    client_finished_from_origin_ns = 0
    client = None
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
            name=f"deadline-c9-http-{block_index}",
        )
        client.start()

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
            raise RuntimeError("deadline C9 source transfer did not terminate in 60 seconds")
        worker.join(timeout=1.0)
        if worker.is_alive():
            raise RuntimeError("deadline C9 source worker remained alive")
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
        "descriptors": (
            old._descriptor_count(channel) if source and mode != old.FG else 0
        ),
        "bytes": (
            old.BYTES_PER_SOURCE if source and state["completed"] == 1 else 0
        ),
        "elapsed_ns": elapsed_ns,
        "completion_from_origin_ns": completion_from_origin_ns,
        "post_foreground_drain_ns": post_foreground_drain_ns,
        "start_lag_ns": (
            max(0, int(state["started_ns"]) - local_origin_ns)
            if source and mode != old.FG
            else 0
        ),
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
        "client": (
            old.scout._client_metrics(client_control[0]["value"])
            if rank == 0
            else None
        ),
        "client_started_from_origin_ns": client_started_from_origin_ns,
        "client_finished_from_origin_ns": client_finished_from_origin_ns,
        "gate_ready": None,
        "gate_release": None,
        "boost_hold_ns": 0,
        "promotion_armed_sources": 0,
        "controller_decision": CONTROLLER_DECISION if candidate else "not_applicable",
        "candidate_hook_invocations": 0,
        "rescue_armed_sources": 0,
        "source_call": local,
        "receiver_verified_bytes": verified,
        "receiver_zero_ok": zero_ok,
        "correctness_met": (
            (not source or mode == old.FG or state["completed"] == 1
             and state["error"] is None)
            and (not receiver or (zero_ok if mode == old.FG
                                  else verified == old.BYTES_PER_SOURCE))
            and (not candidate or not boost.is_set())
        ),
    }


def _validate_trace(path: Path, expected: list[tuple[str, str]]) -> dict[str, Any]:
    # old.main predicts a marked event for each candidate from its mode table.
    # C9 intentionally sends control-prefixed requests, so only the unmeasured
    # fence prewarm remains in the real hook trace.
    filtered = [(caller, mode) for caller, mode in expected if mode != CANDIDATE_MODE]
    if len(filtered) != 1 or filtered[0][1] != protocol.NOOP_MODE:
        raise ValueError("deadline C9 expected trace did not reduce to one prewarm")
    installed_validate = old._validate_trace
    old._validate_trace = _ORIGINAL_VALIDATE_TRACE
    try:
        result = _V6_VALIDATE_TRACE(path, filtered)
    finally:
        old._validate_trace = installed_validate
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    candidate_rows = [
        row for row in rows if CANDIDATE_MODE in str(row.get("request_id", ""))
    ]
    if candidate_rows:
        raise ValueError("deadline C9 candidate unexpectedly entered token hook")
    return {
        **result,
        "candidate_hook_events": 0,
        "controller_decision": CONTROLLER_DECISION,
        "release_mode": "unmeasured_prewarm_only",
    }


def _aggregate(records: list[dict[str, Any]], trace: dict[str, Any], args: Any):
    result = _V8_AGGREGATE(records, trace, args)
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    for key in (
        "boost_token_index_zero_based",
        "boost_generated_token_count_one_based",
        "promotion_release_mode",
        "promotion_release_completed_bytes",
        "promotion_armed_sources",
        "completion_wait_inside_gate",
    ):
        result["config"].pop(key, None)
    result["config"].update(
        {
            "candidate_mode": CANDIDATE_MODE,
            "deadline_controller_decision": CONTROLLER_DECISION,
            "deadline_controller_basis": (
                "candidate_b_prior_observed_min_completion_slack"
            ),
            "prior_observed_min_slack_ms": PRIOR_OBSERVED_MIN_SLACK_MS,
            "defer_threshold_ms": DEFER_THRESHOLD_MS,
            "measured_candidate_hook_enabled": False,
            "candidate_gate_collectives": 0,
            "candidate_rescue_armed_sources": 0,
            "permanent_boost": False,
        }
    )

    ordered = sorted(records, key=lambda item: int(item["rank"]))
    by_mode = {
        mode: [block for block in result["blocks"] if block["mode"] == mode]
        for mode in old.MODES
    }
    progress_rows = []
    for index, block in enumerate(result["blocks"]):
        if block["mode"] != CANDIDATE_MODE:
            continue
        source_calls = [item["blocks"][index]["source_call"] for item in ordered[:8]]
        decisions = [item["blocks"][index]["controller_decision"] for item in ordered]
        progress = {
            "polls_sum": sum(int(call["polls"]) for call in source_calls),
            "polls_max": max(int(call["polls"]) for call in source_calls),
            "low_priority_sleeps_sum": sum(
                int(call["low_priority_sleeps"]) for call in source_calls
            ),
            "boost_polls_sum": sum(int(call["boost_polls"]) for call in source_calls),
            "yields_sum": sum(int(call["yields"]) for call in source_calls),
            "source_errors": sum(call["error"] is not None for call in source_calls),
        }
        block["controller_decision"] = CONTROLLER_DECISION
        block["candidate_hook_invocations"] = 0
        block["rescue_armed_sources"] = 0
        block["observed_completion_slack_ms"] = (
            block["foreground_completion_from_start_ms"]
            - block["background_completion_from_start_ms"]
        )
        block["worker_progress"] = progress
        progress_rows.append(
            {
                "block_index": index,
                "prompt_index": block["prompt_index"],
                "all_rank_decisions_defer": all(
                    value == CONTROLLER_DECISION for value in decisions
                ),
                "observed_completion_slack_ms": block["observed_completion_slack_ms"],
                **progress,
            }
        )

    candidate_blocks = by_mode[CANDIDATE_MODE]
    service_deltas = [
        row["tempo_minus_lmcache_service_makespan_ms"] for row in result["paired"]
    ]
    candidate_e2e = result["mode_metrics"][CANDIDATE_MODE]["e2e_p50_ms"]
    fg_e2e = result["mode_metrics"][old.FG]["e2e_p50_ms"]
    candidate_tpot = result["mode_metrics"][CANDIDATE_MODE]["tpot_p99_max_ms"]
    lmcache_tpot = result["mode_metrics"][old.LMCACHE]["tpot_p99_max_ms"]
    gates = {
        "correctness_output_trace": bool(result["overall_correctness_met"]),
        "no_measured_candidate_hook_events": trace.get("candidate_hook_events") == 0,
        "all_candidate_gate_bubbles_zero": all(
            block["total_gate_bubble_ms"] == 0.0 for block in candidate_blocks
        ),
        "all_candidate_boost_polls_zero": all(
            row["boost_polls_sum"] == 0 for row in progress_rows
        ),
        "all_candidate_decisions_defer": all(
            row["all_rank_decisions_defer"] for row in progress_rows
        ),
        "all_candidate_post_foreground_drain_zero": all(
            block["post_foreground_drain_ms"] == 0.0 for block in candidate_blocks
        ),
        "all_candidate_observed_deadlines_met": all(
            block["observed_completion_slack_ms"] >= 0.0
            for block in candidate_blocks
        ),
        "candidate_service_makespan_beats_lmcache_paired": (
            statistics.median(service_deltas) < 0.0
            and sum(delta < 0.0 for delta in service_deltas) >= 2
        ),
        "candidate_e2e_p50_le_1_05x_fg": candidate_e2e <= E2E_RATIO_MAX * fg_e2e,
        "candidate_tpot_p99_le_1_10x_lmcache": (
            candidate_tpot <= TPOT_P99_RATIO_MAX * lmcache_tpot
        ),
    }
    result["deadline_controller"] = {
        "decision": CONTROLLER_DECISION,
        "prior_observed_min_slack_ms": PRIOR_OBSERVED_MIN_SLACK_MS,
        "defer_threshold_ms": DEFER_THRESHOLD_MS,
        "candidate_hook_events": int(trace.get("candidate_hook_events", -1)),
        "blocks": progress_rows,
    }
    result["candidate_gates"] = gates
    result["screen_outcome"] = (
        "invalid_correctness_output_or_trace"
        if not result["overall_correctness_met"]
        else "deadline_defer_candidate_pass"
        if all(gates.values())
        else "deadline_defer_candidate_revise"
    )
    return result


def main() -> None:
    _install_candidate_mode()
    protocol.install_async_release_protocol()
    old.protocol.ReleaseFrame = protocol.ReleaseFrame
    old.protocol.install_generic_release_protocol = protocol.install_async_release_protocol
    old.bulk.protocol.ReleaseFrame = protocol.ReleaseFrame
    old.bulk.protocol.install_generic_release_protocol = (
        protocol.install_async_release_protocol
    )
    v8.CONTRACT_ID = CONTRACT_ID
    v8.RESULT_SCHEMA = RESULT_SCHEMA
    v8._load_contract = _load_contract
    v8._run_block = _run_block
    v8._validate_trace = _validate_trace
    v8._aggregate = _aggregate
    v8.main()


if __name__ == "__main__":
    main()
