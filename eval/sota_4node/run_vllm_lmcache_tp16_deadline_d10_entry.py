#!/usr/bin/env python3
"""Candidate D10: C9 deadline-defer with a real 2 ms progress cadence.

The sole algorithmic change from C9 is the unboosted prepared-transfer poll
sleep: 2.0 ms instead of 1.0 ms.  Candidate requests remain invisible to the
token-31 hook and retain the same worker-entry prelaunch, transfer geometry,
Latin block order, and post-response exact receiver verification.
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
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as v8
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol


CANDIDATE_MODE = "tempo_prelaunch_deadline_defer_sleep2"
CONTRACT_ID = "tp16-single-flight-deadline-defer-sleep2-d10"
RESULT_SCHEMA = "tempo-vllm-tp16-single-flight-deadline-result-10"
CONTROLLER_DECISION = "defer/no_rescue"
LOW_PRIORITY_SLEEP_S = 0.002
LOW_PRIORITY_SLEEP_NS = 2_000_000
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
_C9_AGGREGATE = c9._aggregate


def _install_candidate_mode() -> None:
    old.TEMPO = CANDIDATE_MODE
    old.MODES = (old.FG, old.LMCACHE, CANDIDATE_MODE)
    old.BLOCKS = BLOCKS


def _deadline_sleep2_channel_class(base_channel: Any) -> Any:
    class DeadlineSleep2Channel(base_channel):
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

    DeadlineSleep2Channel.__name__ = "DeadlineSleep2Channel"
    return DeadlineSleep2Channel


def _load_channel(repo_root: Path) -> tuple[Any, Any, Any, Any]:
    channel, tensor, metadata, memory_format = _ORIGINAL_LOAD_CHANNEL(repo_root)
    return (
        _deadline_sleep2_channel_class(channel),
        tensor,
        metadata,
        memory_format,
    )


def _expected_contract() -> dict[str, Any]:
    return {
        "schema_version": "tempo-tp16-single-flight-deadline-contract-10",
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
            "basis_mode": "lmcache_prelaunch_no_gate",
            "prior_observed_min_slack_ms": PRIOR_OBSERVED_MIN_SLACK_MS,
            "defer_threshold_ms": OBSERVED_SLACK_MIN_MS,
            "decode_progress_sleep_ms": LOW_PRIORITY_SLEEP_S * 1000.0,
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
                "all_candidate_poll_accounting_exact": True,
                "all_candidate_configured_sleep_ms": 2.0,
                "all_candidate_post_foreground_drain_zero": True,
                "all_candidate_observed_slack_ge_ms": OBSERVED_SLACK_MIN_MS,
                "paired_service_win_min_prompts": 2,
                "paired_service_delta_median_le_ms": SERVICE_DELTA_MAX_MS,
                "candidate_e2e_p50_le_fg_ratio": E2E_RATIO_MAX,
                "candidate_tpot_p99_le_lmcache_ratio": TPOT_P99_RATIO_MAX,
            },
        },
    }


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_contract():
        raise ValueError("TP16 deadline defer D10 contract changed")
    return payload


def _run_block(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = c9._run_block(*args, **kwargs)
    source = int(kwargs["rank"]) < old.SOURCE_COUNT
    candidate = kwargs["mode"] == CANDIDATE_MODE
    result["source_call"]["configured_low_priority_sleep_ns"] = (
        LOW_PRIORITY_SLEEP_NS if source and candidate else 0
    )
    if candidate:
        result["controller_decision"] = CONTROLLER_DECISION
        result["candidate_hook_invocations"] = 0
        result["rescue_armed_sources"] = 0
    return result


def _aggregate(records: list[dict[str, Any]], trace: dict[str, Any], args: Any):
    result = _C9_AGGREGATE(records, trace, args)
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    result["config"].update(
        {
            "candidate_mode": CANDIDATE_MODE,
            "deadline_controller_decision": CONTROLLER_DECISION,
            "deadline_controller_basis": (
                "candidate_b_lmcache_prior_observed_min_completion_slack"
            ),
            "prior_observed_min_slack_ms": PRIOR_OBSERVED_MIN_SLACK_MS,
            "defer_threshold_ms": OBSERVED_SLACK_MIN_MS,
            "decode_progress_sleep_ms": LOW_PRIORITY_SLEEP_S * 1000.0,
            "measured_candidate_hook_enabled": False,
            "candidate_gate_collectives": 0,
            "candidate_rescue_armed_sources": 0,
            "permanent_boost": False,
        }
    )

    ordered = sorted(records, key=lambda item: int(item["rank"]))
    candidate_blocks = [
        block for block in result["blocks"] if block["mode"] == CANDIDATE_MODE
    ]
    accounting_rows = []
    for index, block in enumerate(result["blocks"]):
        if block["mode"] != CANDIDATE_MODE:
            continue
        calls = [item["blocks"][index]["source_call"] for item in ordered[:8]]
        exact = all(
            int(call["configured_low_priority_sleep_ns"]) == LOW_PRIORITY_SLEEP_NS
            and int(call["boost_polls"]) == 0
            and int(call["yields"]) == 0
            and int(call["polls"]) == int(call["low_priority_sleeps"]) + 1
            for call in calls
        )
        block["worker_progress"].update(
            {
                "configured_low_priority_sleep_ms": LOW_PRIORITY_SLEEP_S * 1000.0,
                "poll_accounting_exact": exact,
                "low_priority_sleep_budget_ms": (
                    block["worker_progress"]["low_priority_sleeps_sum"]
                    * LOW_PRIORITY_SLEEP_S
                    * 1000.0
                ),
            }
        )
        accounting_rows.append(
            {
                "block_index": index,
                "prompt_index": block["prompt_index"],
                "configured_low_priority_sleep_ms": LOW_PRIORITY_SLEEP_S * 1000.0,
                "poll_accounting_exact": exact,
            }
        )

    service_deltas = [
        row["tempo_minus_lmcache_service_makespan_ms"] for row in result["paired"]
    ]
    candidate_e2e = result["mode_metrics"][CANDIDATE_MODE]["e2e_p50_ms"]
    fg_e2e = result["mode_metrics"][old.FG]["e2e_p50_ms"]
    candidate_tpot = result["mode_metrics"][CANDIDATE_MODE]["tpot_p99_max_ms"]
    lmcache_tpot = result["mode_metrics"][old.LMCACHE]["tpot_p99_max_ms"]
    progress_rows = result["deadline_controller"]["blocks"]
    for progress, accounting in zip(progress_rows, accounting_rows, strict=True):
        progress.update(accounting)

    gates = {
        "correctness_output_trace": bool(result["overall_correctness_met"]),
        "no_measured_candidate_hook_events": trace.get("candidate_hook_events") == 0,
        "all_candidate_gate_bubbles_zero": all(
            block["total_gate_bubble_ms"] == 0.0 for block in candidate_blocks
        ),
        "all_candidate_boost_polls_zero": all(
            block["worker_progress"]["boost_polls_sum"] == 0
            for block in candidate_blocks
        ),
        "all_candidate_poll_accounting_exact": all(
            row["poll_accounting_exact"] for row in accounting_rows
        ),
        "all_candidate_configured_sleep_exact_2ms": all(
            row["configured_low_priority_sleep_ms"] == 2.0
            for row in accounting_rows
        ),
        "all_candidate_decisions_defer": all(
            row["all_rank_decisions_defer"] for row in progress_rows
        ),
        "all_candidate_post_foreground_drain_zero": all(
            block["post_foreground_drain_ms"] == 0.0 for block in candidate_blocks
        ),
        "all_candidate_observed_slack_ge_200ms": all(
            block["observed_completion_slack_ms"] >= OBSERVED_SLACK_MIN_MS
            for block in candidate_blocks
        ),
        "candidate_service_makespan_median_le_minus_5ms_and_2of3": (
            statistics.median(service_deltas) <= SERVICE_DELTA_MAX_MS
            and sum(delta < 0.0 for delta in service_deltas) >= 2
        ),
        "candidate_e2e_p50_le_1_05x_fg": candidate_e2e <= E2E_RATIO_MAX * fg_e2e,
        "candidate_tpot_p99_le_1_10x_lmcache": (
            candidate_tpot <= TPOT_P99_RATIO_MAX * lmcache_tpot
        ),
    }
    result["deadline_controller"].update(
        {
            "decision": CONTROLLER_DECISION,
            "basis_mode": old.LMCACHE,
            "prior_observed_min_slack_ms": PRIOR_OBSERVED_MIN_SLACK_MS,
            "defer_threshold_ms": OBSERVED_SLACK_MIN_MS,
            "decode_progress_sleep_ms": LOW_PRIORITY_SLEEP_S * 1000.0,
        }
    )
    result["candidate_gates"] = gates
    result["screen_outcome"] = (
        "invalid_correctness_output_or_trace"
        if not result["overall_correctness_met"]
        else "deadline_sleep2_candidate_pass"
        if all(gates.values())
        else "deadline_sleep2_candidate_revise"
    )
    return result


def main() -> None:
    _install_candidate_mode()
    c9.CANDIDATE_MODE = CANDIDATE_MODE
    c9.CONTRACT_ID = CONTRACT_ID
    c9.RESULT_SCHEMA = RESULT_SCHEMA
    c9.CONTROLLER_DECISION = CONTROLLER_DECISION
    c9.PRIOR_OBSERVED_MIN_SLACK_MS = PRIOR_OBSERVED_MIN_SLACK_MS
    c9.DEFER_THRESHOLD_MS = OBSERVED_SLACK_MIN_MS
    c9.BLOCKS = BLOCKS
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
