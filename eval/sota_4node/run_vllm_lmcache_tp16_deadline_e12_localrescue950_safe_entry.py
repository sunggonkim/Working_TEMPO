#!/usr/bin/env python3
"""E12 race-free local rescue in the source transfer polling loop."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from eval.sota_4node import run_vllm_lmcache_tp16_deadline_c9_entry as c9
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_e11_localrescue950_entry as e11
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as v8
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol


CANDIDATE_MODE = "tempo_prelaunch_local_rescue_950ms_safe"
CONTRACT_ID = "tp16-single-flight-local-rescue-950ms-safe-e12"
RESULT_SCHEMA = "tempo-vllm-tp16-single-flight-local-rescue-safe-result-12"
CONTROLLER_DECISION = "local_late_rescue_poll_loop"
LOCAL_RESCUE_TRIGGER_MS = 950.0
LOCAL_RESCUE_TRIGGER_NS = 950_000_000
C9_RESULT = "results/vllm_lmcache_tp16_deadline_C9_job_56972950/result.json"

BLOCKS = (
    (0, old.FG), (0, old.LMCACHE), (0, CANDIDATE_MODE),
    (1, CANDIDATE_MODE), (1, old.FG), (1, old.LMCACHE),
    (2, old.LMCACHE), (2, CANDIDATE_MODE), (2, old.FG),
)

_E11_AGGREGATE = e11._aggregate


def _install_candidate_mode() -> None:
    old.TEMPO = CANDIDATE_MODE
    old.MODES = (old.FG, old.LMCACHE, CANDIDATE_MODE)
    old.BLOCKS = BLOCKS


def _safe_local_rescue_transfer_worker(
    *, channel: Any, obj: Any, receiver_id: str, mode: str,
    boost: threading.Event, entered: threading.Event, done: threading.Event,
    state: dict[str, Any],
) -> None:
    worker_name = threading.current_thread().name
    started_ns = time.perf_counter_ns()
    state["started_ns"] = started_ns
    entered.set()
    candidate = mode == CANDIDATE_MODE
    rescue = {
        "configured_trigger_ms": LOCAL_RESCUE_TRIGGER_MS,
        "timer_threads_created": 0,
        "timer_started_ns": 0,
        "timer_callback_ns": 0,
        "trigger_callback_observed": False,
        "unfinished_at_trigger": False,
        "status_at_arm": "",
        "rescue_armed": False,
        "rescue_armed_ns": 0,
        "rescue_armed_from_worker_start_ms": 0.0,
        "boost_observed_by_worker": False,
        "completed_before_rescue": False,
        "timer_cancelled": False,
        "timer_joined": True,
    }
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
            handle = channel.tempo_prepare([obj], spec)
            if channel.nixl_agent.transfer(handle) == "ERR":
                raise RuntimeError("TEMPO failed to post prepared NIXL handle")
            polls = low_priority_sleeps = boost_polls = yields = 0
            while True:
                status = channel.nixl_agent.check_xfer_state(handle)
                polls += 1
                if status == "ERR":
                    raise RuntimeError("TEMPO prepared NIXL transfer failed")
                if status == "DONE":
                    state.update(
                        completed=1,
                        polls=polls,
                        low_priority_sleeps=low_priority_sleeps,
                        boost_polls=boost_polls,
                        yields=yields,
                    )
                    rescue["completed_before_rescue"] = not bool(
                        rescue["rescue_armed"]
                    )
                    break
                if status != "PROC":
                    raise RuntimeError(f"unexpected NIXL state {status}")
                now_ns = time.perf_counter_ns()
                if (
                    not rescue["rescue_armed"]
                    and now_ns - started_ns >= LOCAL_RESCUE_TRIGGER_NS
                ):
                    # This decision is made only after this worker observed PROC.
                    boost.set()
                    rescue["trigger_callback_observed"] = True
                    rescue["unfinished_at_trigger"] = True
                    rescue["status_at_arm"] = "PROC"
                    rescue["rescue_armed"] = True
                    rescue["rescue_armed_ns"] = now_ns
                    rescue["rescue_armed_from_worker_start_ms"] = (
                        now_ns - started_ns
                    ) / 1e6
                if boost.is_set():
                    boost_polls += 1
                    rescue["boost_observed_by_worker"] = True
                    if boost_polls % 64 == 0:
                        time.sleep(0)
                        yields += 1
                else:
                    time.sleep(0.001)
                    low_priority_sleeps += 1
        else:
            raise RuntimeError(f"safe local-rescue worker received invalid mode {mode}")
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if candidate:
            e11._publish_rescue_record(worker_name, rescue)
        state["finished_ns"] = time.perf_counter_ns()
        done.set()


def _expected_contract() -> dict[str, Any]:
    payload = e11._expected_contract()
    payload["schema_version"] = "tempo-tp16-single-flight-local-rescue-safe-contract-12"
    payload["contract_id"] = CONTRACT_ID
    payload["campaign"]["modes"] = [old.FG, old.LMCACHE, CANDIDATE_MODE]
    rescue = payload["local_rescue"]
    rescue.update(
        {
            "candidate_mode": CANDIDATE_MODE,
            "scope": "per_source_same_polling_loop",
            "arm_condition": "observed_PROC_and_worker_elapsed_ge_950ms",
            "timer_threads_created": 0,
            "definitive_timer_join": "not_applicable_no_timer",
            "armed_implies_worker_boost_poll": True,
        }
    )
    gates = payload["campaign"]["candidate_gates"]
    gates.pop("all_rescue_timers_joined", None)
    gates.update(
        {
            "no_rescue_timer_threads": True,
            "all_armed_status_is_PROC": True,
            "all_armed_observed_by_worker": True,
            "all_candidate_poll_accounting_exact": True,
        }
    )
    return payload


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_contract():
        raise ValueError("TP16 safe local-rescue950 E12 contract changed")
    return payload


def _aggregate(records: list[dict[str, Any]], trace: dict[str, Any], args: Any):
    result = _E11_AGGREGATE(records, trace, args)
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    ordered = sorted(records, key=lambda item: int(item["rank"]))
    no_timers = status_exact = observed_exact = poll_exact = True
    for index, block in enumerate(result["blocks"]):
        if block["mode"] != CANDIDATE_MODE:
            continue
        calls = [item["blocks"][index]["source_call"] for item in ordered[:8]]
        no_timers &= all(int(call["timer_threads_created"]) == 0 for call in calls)
        status_exact &= all(
            not bool(call["rescue_armed"]) or call["status_at_arm"] == "PROC"
            for call in calls
        )
        observed_exact &= all(
            not bool(call["rescue_armed"])
            or bool(call["boost_observed_by_worker"])
            and int(call["boost_polls"]) > 0
            for call in calls
        )
        poll_exact &= all(
            int(call["polls"])
            == int(call["low_priority_sleeps"]) + int(call["boost_polls"]) + 1
            for call in calls
        )
        block["local_rescue"].update(
            timer_threads_created=0,
            armed_status_exact=status_exact,
            armed_observed_by_worker=observed_exact,
            poll_accounting_exact=poll_exact,
        )
    gates = dict(result["candidate_gates"])
    gates.pop("all_rescue_timers_joined", None)
    gates.update(
        {
            "no_rescue_timer_threads": no_timers,
            "all_armed_status_is_PROC": status_exact,
            "all_armed_observed_by_worker": observed_exact,
            "all_candidate_poll_accounting_exact": poll_exact,
        }
    )
    result["config"].update(
        candidate_mode=CANDIDATE_MODE,
        controller=CONTROLLER_DECISION,
        local_rescue_scope="per_source_same_polling_loop",
        rescue_timer_threads=0,
    )
    result["local_rescue"].update(
        scope="per_source_same_polling_loop",
        timer_threads_created=0,
        definitive_timer_join="not_applicable_no_timer",
    )
    result["candidate_gates"] = gates
    result["screen_outcome"] = (
        "invalid_correctness_output_or_trace"
        if not result["overall_correctness_met"]
        else "safe_local_rescue950_candidate_pass"
        if all(gates.values())
        else "safe_local_rescue950_candidate_revise"
    )
    return result


def main() -> None:
    _install_candidate_mode()
    for module in (c9, e11):
        module.CANDIDATE_MODE = CANDIDATE_MODE
        module.CONTRACT_ID = CONTRACT_ID
        module.RESULT_SCHEMA = RESULT_SCHEMA
        module.CONTROLLER_DECISION = CONTROLLER_DECISION
        module.BLOCKS = BLOCKS
    e11.LOCAL_RESCUE_TRIGGER_MS = LOCAL_RESCUE_TRIGGER_MS
    fixed._transfer_worker = _safe_local_rescue_transfer_worker
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
    v8._run_block = e11._run_block
    v8._validate_trace = c9._validate_trace
    v8._aggregate = _aggregate
    v8.main()


if __name__ == "__main__":
    main()
