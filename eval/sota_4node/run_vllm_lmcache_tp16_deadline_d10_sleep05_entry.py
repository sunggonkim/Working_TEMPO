#!/usr/bin/env python3
"""Candidate D10: C9 deadline-defer with a real 0.5 ms poll sleep."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import threading
import time
from typing import Any

from eval.sota_4node import run_vllm_lmcache_tp16_deadline_c9_entry as c9
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_d10_entry as abandoned_sleep2
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as v8
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol


CANDIDATE_MODE = "tempo_prelaunch_deadline_defer_sleep05"
CONTRACT_ID = "tp16-single-flight-deadline-defer-sleep05-d10"
RESULT_SCHEMA = "tempo-vllm-tp16-single-flight-deadline-sleep05-result-10"
CONTROLLER_DECISION = "defer/no_rescue"
LOW_PRIORITY_SLEEP_S = 0.0005
LOW_PRIORITY_SLEEP_NS = 500_000
PRIOR_OBSERVED_MIN_SLACK_MS = 263.210418
OBSERVED_SLACK_MIN_MS = 200.0
SERVICE_DELTA_MAX_MS = -5.0
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

_ORIGINAL_LOAD_CHANNEL = old._load_channel
_ABANDONED_AGGREGATE = abandoned_sleep2._aggregate


def _install_candidate_mode() -> None:
    old.TEMPO = CANDIDATE_MODE
    old.MODES = (old.FG, old.LMCACHE, CANDIDATE_MODE)
    old.BLOCKS = BLOCKS


def _deadline_sleep05_channel_class(base_channel: Any) -> Any:
    class DeadlineSleep05Channel(base_channel):
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
                        "configured_low_priority_sleep_ns": LOW_PRIORITY_SLEEP_NS,
                    }
                if status != "PROC":
                    raise RuntimeError(f"unexpected NIXL state {status}")
                if boost.is_set():
                    boost_polls += 1
                    if boost_polls % 64 == 0:
                        time.sleep(0)
                        yields += 1
                else:
                    time.sleep(LOW_PRIORITY_SLEEP_S)
                    low_priority_sleeps += 1

    DeadlineSleep05Channel.__name__ = "DeadlineSleep05Channel"
    return DeadlineSleep05Channel


def _load_channel(repo_root: Path) -> tuple[Any, Any, Any, Any]:
    channel, tensor, metadata, memory_format = _ORIGINAL_LOAD_CHANNEL(repo_root)
    return (
        _deadline_sleep05_channel_class(channel),
        tensor,
        metadata,
        memory_format,
    )


def _expected_contract() -> dict[str, Any]:
    payload = abandoned_sleep2._expected_contract()
    payload["schema_version"] = "tempo-tp16-single-flight-deadline-sleep05-contract-10"
    payload["contract_id"] = CONTRACT_ID
    payload["deadline_controller"].update(
        {
            "candidate_mode": CANDIDATE_MODE,
            "decode_progress_sleep_ms": 0.5,
        }
    )
    payload["campaign"]["modes"] = [old.FG, old.LMCACHE, CANDIDATE_MODE]
    payload["campaign"]["candidate_gates"]["all_candidate_configured_sleep_ms"] = 0.5
    payload["campaign"]["candidate_gates"]["paired_service_win_max_delta_ms"] = -5.0
    return payload


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_contract():
        raise ValueError("TP16 deadline defer sleep05 D10 contract changed")
    return payload


def _run_block(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = abandoned_sleep2._run_block(*args, **kwargs)
    source = int(kwargs["rank"]) < old.SOURCE_COUNT
    candidate = kwargs["mode"] == CANDIDATE_MODE
    result["source_call"]["configured_low_priority_sleep_ns"] = (
        LOW_PRIORITY_SLEEP_NS if source and candidate else 0
    )
    return result


def _aggregate(records: list[dict[str, Any]], trace: dict[str, Any], args: Any):
    result = _ABANDONED_AGGREGATE(records, trace, args)
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    candidate_blocks = [
        block for block in result["blocks"] if block["mode"] == CANDIDATE_MODE
    ]
    service_deltas = [
        row["tempo_minus_lmcache_service_makespan_ms"] for row in result["paired"]
    ]
    progress_rows = result["deadline_controller"]["blocks"]
    configured_exact = all(
        row["configured_low_priority_sleep_ms"] == 0.5 for row in progress_rows
    )
    gates = dict(result["candidate_gates"])
    gates.pop("all_candidate_configured_sleep_exact_2ms", None)
    gates.pop("candidate_service_makespan_median_le_minus_5ms_and_2of3", None)
    gates.update(
        {
            "all_candidate_configured_sleep_exact_0_5ms": configured_exact,
            "candidate_service_makespan_median_le_minus_5ms_and_2_meaningful_wins": (
                statistics.median(service_deltas) <= SERVICE_DELTA_MAX_MS
                and sum(delta <= SERVICE_DELTA_MAX_MS for delta in service_deltas) >= 2
            ),
        }
    )
    result["config"].update(
        {
            "candidate_mode": CANDIDATE_MODE,
            "decode_progress_sleep_ms": 0.5,
            "prior_observed_min_slack_ms": PRIOR_OBSERVED_MIN_SLACK_MS,
            "defer_threshold_ms": OBSERVED_SLACK_MIN_MS,
        }
    )
    result["deadline_controller"].update(
        {
            "decode_progress_sleep_ms": 0.5,
            "prior_observed_min_slack_ms": PRIOR_OBSERVED_MIN_SLACK_MS,
            "defer_threshold_ms": OBSERVED_SLACK_MIN_MS,
            "observed_min_slack_ms": min(
                block["observed_completion_slack_ms"] for block in candidate_blocks
            ),
            "paired_service_delta_median_ms": statistics.median(service_deltas),
            "meaningful_service_wins_le_minus_5ms": sum(
                delta <= SERVICE_DELTA_MAX_MS for delta in service_deltas
            ),
        }
    )
    result["candidate_gates"] = gates
    result["screen_outcome"] = (
        "invalid_correctness_output_or_trace"
        if not result["overall_correctness_met"]
        else "deadline_sleep05_candidate_pass"
        if all(gates.values())
        else "deadline_sleep05_candidate_revise"
    )
    return result


def main() -> None:
    _install_candidate_mode()
    for module in (c9, abandoned_sleep2):
        module.CANDIDATE_MODE = CANDIDATE_MODE
        module.CONTRACT_ID = CONTRACT_ID
        module.RESULT_SCHEMA = RESULT_SCHEMA
        module.CONTROLLER_DECISION = CONTROLLER_DECISION
        module.PRIOR_OBSERVED_MIN_SLACK_MS = PRIOR_OBSERVED_MIN_SLACK_MS
        module.OBSERVED_SLACK_MIN_MS = OBSERVED_SLACK_MIN_MS
        module.DEFER_THRESHOLD_MS = OBSERVED_SLACK_MIN_MS
        module.BLOCKS = BLOCKS
    abandoned_sleep2.LOW_PRIORITY_SLEEP_S = LOW_PRIORITY_SLEEP_S
    abandoned_sleep2.LOW_PRIORITY_SLEEP_NS = LOW_PRIORITY_SLEEP_NS
    abandoned_sleep2.SERVICE_DELTA_MAX_MS = SERVICE_DELTA_MAX_MS
    old._load_channel = _load_channel
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
