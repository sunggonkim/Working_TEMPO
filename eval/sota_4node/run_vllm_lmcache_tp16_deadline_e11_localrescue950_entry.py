#!/usr/bin/env python3
"""E11 local late-rescue at 950 ms for only unfinished source transfers."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import threading
import time
from typing import Any

import numpy as np

from eval.sota_4node import run_vllm_lmcache_tp16_deadline_c9_entry as c9
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as v8
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol


CANDIDATE_MODE = "tempo_prelaunch_local_rescue_950ms"
CONTRACT_ID = "tp16-single-flight-local-rescue-950ms-e11"
RESULT_SCHEMA = "tempo-vllm-tp16-single-flight-local-rescue-result-11"
CONTROLLER_DECISION = "local_late_rescue"
LOCAL_RESCUE_TRIGGER_MS = 950.0
LOCAL_RESCUE_TRIGGER_S = 0.950
PRIOR_C9_MIN_SLACK_MS = 145.222977
PRIOR_C9_SERVICE_MEDIAN_DELTA_MS = -31.808033
OBSERVED_SLACK_MIN_MS = 200.0
SERVICE_DELTA_MAX_MS = -5.0
E2E_RATIO_MAX = 1.05
TPOT_P99_RATIO_MAX = 1.10
C9_RESULT = "results/vllm_lmcache_tp16_deadline_C9_job_56972950/result.json"

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

_C9_AGGREGATE = c9._aggregate
_RESCUE_RECORDS: dict[str, dict[str, Any]] = {}
_RESCUE_RECORDS_LOCK = threading.Lock()


def _install_candidate_mode() -> None:
    old.TEMPO = CANDIDATE_MODE
    old.MODES = (old.FG, old.LMCACHE, CANDIDATE_MODE)
    old.BLOCKS = BLOCKS


def _publish_rescue_record(worker_name: str, record: dict[str, Any]) -> None:
    with _RESCUE_RECORDS_LOCK:
        if worker_name in _RESCUE_RECORDS:
            raise RuntimeError(f"duplicate local-rescue record for {worker_name}")
        _RESCUE_RECORDS[worker_name] = dict(record)


def _take_rescue_record(worker_name: str) -> dict[str, Any] | None:
    with _RESCUE_RECORDS_LOCK:
        return _RESCUE_RECORDS.pop(worker_name, None)


def _local_rescue_transfer_worker(
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
    worker_name = threading.current_thread().name
    state["started_ns"] = time.perf_counter_ns()
    entered.set()
    candidate = mode == CANDIDATE_MODE
    completion_lock = threading.Lock()
    transfer_completed = False
    timer: threading.Timer | None = None
    rescue = {
        "configured_trigger_ms": LOCAL_RESCUE_TRIGGER_MS,
        "timer_started_ns": 0,
        "timer_callback_ns": 0,
        "trigger_callback_observed": False,
        "unfinished_at_trigger": False,
        "rescue_armed": False,
        "rescue_armed_ns": 0,
        "rescue_armed_from_worker_start_ms": 0.0,
        "completed_before_rescue": False,
        "timer_cancelled": False,
        "timer_joined": False,
    }

    def arm_if_unfinished() -> None:
        nonlocal transfer_completed
        callback_ns = time.perf_counter_ns()
        with completion_lock:
            rescue["timer_callback_ns"] = callback_ns
            rescue["trigger_callback_observed"] = True
            unfinished = not transfer_completed and not done.is_set()
            rescue["unfinished_at_trigger"] = unfinished
            if unfinished:
                boost.set()
                rescue["rescue_armed"] = True
                rescue["rescue_armed_ns"] = callback_ns
                rescue["rescue_armed_from_worker_start_ms"] = (
                    callback_ns - int(state["started_ns"])
                ) / 1e6

    try:
        spec = {
            "receiver_id": receiver_id,
            "remote_indexes": np.asarray([0], dtype=np.uint64),
        }
        if mode == old.LMCACHE:
            state["completed"] = int(
                channel.batched_write(objects=[obj], transfer_spec=spec)
            )
        elif candidate:
            rescue["timer_started_ns"] = time.perf_counter_ns()
            timer = threading.Timer(LOCAL_RESCUE_TRIGGER_S, arm_if_unfinished)
            timer.name = f"local-rescue-950-{worker_name}"
            timer.daemon = True
            timer.start()
            values = channel.tempo_adaptive_write([obj], spec, boost)
            with completion_lock:
                transfer_completed = True
                state.update(values)
                rescue["completed_before_rescue"] = not bool(rescue["rescue_armed"])
        else:
            raise RuntimeError(f"local-rescue worker received invalid mode {mode}")
    except BaseException as exc:
        with completion_lock:
            transfer_completed = True
        state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if timer is not None:
            timer.cancel()
            rescue["timer_cancelled"] = True
            timer.join(timeout=1.0)
            rescue["timer_joined"] = not timer.is_alive()
        if candidate:
            _publish_rescue_record(worker_name, rescue)
        state["finished_ns"] = time.perf_counter_ns()
        done.set()


def _expected_contract() -> dict[str, Any]:
    return {
        "schema_version": "tempo-tp16-single-flight-local-rescue-contract-11",
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
            "decode_progress_sleep_ms": 1.0,
        },
        "local_rescue": {
            "candidate_mode": CANDIDATE_MODE,
            "trigger_ms_from_source_worker_start": LOCAL_RESCUE_TRIGGER_MS,
            "scope": "per_source_local_timer",
            "arm_condition": "source_still_unfinished_at_trigger",
            "completed_sources_never_boosted": True,
            "global_rescue_collectives": 0,
            "measured_candidate_request_marked": False,
            "token31_hook_events_per_candidate": 0,
            "vllm_gate_events_per_candidate": 0,
            "basis_result": C9_RESULT,
            "prior_c9_min_slack_ms": PRIOR_C9_MIN_SLACK_MS,
            "prior_c9_service_median_delta_ms": PRIOR_C9_SERVICE_MEDIAN_DELTA_MS,
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
                "global_rescue_collectives": 0,
                "all_rescue_timers_joined": True,
                "rescue_only_unfinished_sources": True,
                "triggered_count_equals_unfinished_at_950ms": True,
                "at_least_one_rescue_observed": True,
                "all_candidate_post_foreground_drain_zero": True,
                "all_candidate_observed_slack_ge_ms": OBSERVED_SLACK_MIN_MS,
                "paired_service_delta_median_le_ms": SERVICE_DELTA_MAX_MS,
                "paired_service_win_max_delta_ms": SERVICE_DELTA_MAX_MS,
                "paired_service_win_min_prompts": 2,
                "candidate_e2e_p50_le_fg_ratio": E2E_RATIO_MAX,
                "candidate_tpot_p99_le_lmcache_ratio": TPOT_P99_RATIO_MAX,
            },
        },
    }


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_contract():
        raise ValueError("TP16 local-rescue950 E11 contract changed")
    return payload


def _run_block(*args: Any, **kwargs: Any) -> dict[str, Any]:
    rank = int(kwargs["rank"])
    block_index = int(kwargs["block_index"])
    mode = str(kwargs["mode"])
    worker_name = f"deadline-c9-transfer-rank{rank}-block{block_index}"
    result = c9._run_block(*args, **kwargs)
    if rank < old.SOURCE_COUNT and mode == CANDIDATE_MODE:
        rescue = _take_rescue_record(worker_name)
        if rescue is None:
            raise RuntimeError(f"missing local-rescue record for {worker_name}")
        result["source_call"].update(rescue)
        invariant = (
            bool(rescue["rescue_armed"]) == bool(rescue["unfinished_at_trigger"])
            and (bool(rescue["rescue_armed"]) or bool(rescue["completed_before_rescue"]))
            and bool(rescue["timer_joined"])
        )
        result["correctness_met"] = bool(result["correctness_met"] and invariant)
    return result


def _aggregate(records: list[dict[str, Any]], trace: dict[str, Any], args: Any):
    result = _C9_AGGREGATE(records, trace, args)
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    result["config"].update(
        {
            "candidate_mode": CANDIDATE_MODE,
            "controller": CONTROLLER_DECISION,
            "local_rescue_trigger_ms": LOCAL_RESCUE_TRIGGER_MS,
            "local_rescue_scope": "per_source",
            "global_rescue_collectives": 0,
            "decode_progress_sleep_ms": 1.0,
            "basis_result": C9_RESULT,
            "prior_c9_min_slack_ms": PRIOR_C9_MIN_SLACK_MS,
            "prior_c9_service_median_delta_ms": PRIOR_C9_SERVICE_MEDIAN_DELTA_MS,
        }
    )

    ordered = sorted(records, key=lambda item: int(item["rank"]))
    candidate_blocks = [
        block for block in result["blocks"] if block["mode"] == CANDIDATE_MODE
    ]
    rescue_rows = []
    for index, block in enumerate(result["blocks"]):
        if block["mode"] != CANDIDATE_MODE:
            continue
        calls = [item["blocks"][index]["source_call"] for item in ordered[:8]]
        armed = sum(bool(call["rescue_armed"]) for call in calls)
        unfinished = sum(bool(call["unfinished_at_trigger"]) for call in calls)
        completed_before = sum(bool(call["completed_before_rescue"]) for call in calls)
        row = {
            "block_index": index,
            "prompt_index": block["prompt_index"],
            "configured_trigger_ms": LOCAL_RESCUE_TRIGGER_MS,
            "rescue_armed_sources": armed,
            "unfinished_sources_at_950ms": unfinished,
            "completed_before_rescue_sources": completed_before,
            "triggered_count_matches_unfinished": armed == unfinished,
            "timers_joined": all(bool(call["timer_joined"]) for call in calls),
            "rescue_only_unfinished": all(
                bool(call["rescue_armed"]) == bool(call["unfinished_at_trigger"])
                for call in calls
            ),
            "completed_sources_never_boosted": all(
                not bool(call["rescue_armed"])
                for call in calls
                if bool(call["completed_before_rescue"])
            ),
            "armed_elapsed_ms": [
                float(call["rescue_armed_from_worker_start_ms"])
                for call in calls
                if bool(call["rescue_armed"])
            ],
        }
        block["local_rescue"] = row
        rescue_rows.append(row)

    service_deltas = [
        row["tempo_minus_lmcache_service_makespan_ms"] for row in result["paired"]
    ]
    candidate_e2e = result["mode_metrics"][CANDIDATE_MODE]["e2e_p50_ms"]
    fg_e2e = result["mode_metrics"][old.FG]["e2e_p50_ms"]
    candidate_tpot = result["mode_metrics"][CANDIDATE_MODE]["tpot_p99_max_ms"]
    lmcache_tpot = result["mode_metrics"][old.LMCACHE]["tpot_p99_max_ms"]
    total_armed = sum(row["rescue_armed_sources"] for row in rescue_rows)
    total_unfinished = sum(row["unfinished_sources_at_950ms"] for row in rescue_rows)
    gates = {
        "correctness_output_trace_exact_geometry": bool(result["overall_correctness_met"]),
        "no_measured_candidate_hook_events": trace.get("candidate_hook_events") == 0,
        "all_candidate_gate_bubbles_zero": all(
            block["total_gate_bubble_ms"] == 0.0 for block in candidate_blocks
        ),
        "global_rescue_collectives_zero": True,
        "all_rescue_timers_joined": all(row["timers_joined"] for row in rescue_rows),
        "rescue_only_unfinished_sources": all(
            row["rescue_only_unfinished"] and row["completed_sources_never_boosted"]
            for row in rescue_rows
        ),
        "triggered_count_equals_unfinished_at_950ms": (
            total_armed == total_unfinished
            and all(row["triggered_count_matches_unfinished"] for row in rescue_rows)
        ),
        "at_least_one_rescue_observed": total_armed > 0,
        "all_candidate_post_foreground_drain_zero": all(
            block["post_foreground_drain_ms"] == 0.0 for block in candidate_blocks
        ),
        "all_candidate_observed_slack_ge_200ms": all(
            block["observed_completion_slack_ms"] >= OBSERVED_SLACK_MIN_MS
            for block in candidate_blocks
        ),
        "candidate_service_median_le_minus_5ms_and_2_meaningful_wins": (
            statistics.median(service_deltas) <= SERVICE_DELTA_MAX_MS
            and sum(delta <= SERVICE_DELTA_MAX_MS for delta in service_deltas) >= 2
        ),
        "candidate_e2e_p50_le_1_05x_fg": candidate_e2e <= E2E_RATIO_MAX * fg_e2e,
        "candidate_tpot_p99_le_1_10x_lmcache": (
            candidate_tpot <= TPOT_P99_RATIO_MAX * lmcache_tpot
        ),
    }
    result["local_rescue"] = {
        "trigger_ms": LOCAL_RESCUE_TRIGGER_MS,
        "scope": "per_source",
        "global_rescue_collectives": 0,
        "total_rescue_armed_sources": total_armed,
        "total_unfinished_sources_at_950ms": total_unfinished,
        "observed_min_slack_ms": min(
            block["observed_completion_slack_ms"] for block in candidate_blocks
        ),
        "paired_service_delta_median_ms": statistics.median(service_deltas),
        "meaningful_service_wins_le_minus_5ms": sum(
            delta <= SERVICE_DELTA_MAX_MS for delta in service_deltas
        ),
        "blocks": rescue_rows,
    }
    result["candidate_gates"] = gates
    result["screen_outcome"] = (
        "invalid_correctness_output_or_trace"
        if not result["overall_correctness_met"]
        else "local_rescue950_candidate_pass"
        if all(gates.values())
        else "local_rescue950_candidate_revise"
    )
    return result


def main() -> None:
    _install_candidate_mode()
    c9.CANDIDATE_MODE = CANDIDATE_MODE
    c9.CONTRACT_ID = CONTRACT_ID
    c9.RESULT_SCHEMA = RESULT_SCHEMA
    c9.CONTROLLER_DECISION = CONTROLLER_DECISION
    c9.PRIOR_OBSERVED_MIN_SLACK_MS = PRIOR_C9_MIN_SLACK_MS
    c9.DEFER_THRESHOLD_MS = OBSERVED_SLACK_MIN_MS
    c9.BLOCKS = BLOCKS
    fixed._transfer_worker = _local_rescue_transfer_worker
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
    v8._validate_trace = c9._validate_trace
    v8._aggregate = _aggregate
    v8.main()


if __name__ == "__main__":
    main()
