#!/usr/bin/env python3
"""E11 0.25ms branch for D10 slack-fail/service-pass outcome."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any

from eval.sota_4node import run_vllm_lmcache_tp16_deadline_c9_entry as c9
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_d10_sleep05_entry as d10
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as v8
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol


CANDIDATE_MODE = "tempo_prelaunch_deadline_defer_sleep025"
CONTRACT_ID = "tp16-single-flight-deadline-defer-sleep025-e11"
RESULT_SCHEMA = "tempo-vllm-tp16-single-flight-deadline-sleep025-result-11"
CONTROLLER_DECISION = "defer/no_rescue"
LOW_PRIORITY_SLEEP_S = 0.00025
LOW_PRIORITY_SLEEP_NS = 250_000
PRIOR_OBSERVED_MIN_SLACK_MS = 263.210418
OBSERVED_SLACK_MIN_MS = 200.0
SERVICE_DELTA_MAX_MS = -5.0
E2E_RATIO_MAX = 1.05
TPOT_P99_RATIO_MAX = 1.10
BRANCH_RULE = "d10_slack_fail_and_service_pass"
DECISION_EVIDENCE_RESULTS = (
    "results/vllm_lmcache_tp16_deadline_C9_job_56972950/result.json",
    "results/vllm_lmcache_tp16_deadline_D10_sleep05_job_56972950/result.json",
)

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
_D10_AGGREGATE = d10._aggregate


def _install_candidate_mode() -> None:
    old.TEMPO = CANDIDATE_MODE
    old.MODES = (old.FG, old.LMCACHE, CANDIDATE_MODE)
    old.BLOCKS = BLOCKS


def _deadline_sleep025_channel_class(base_channel: Any) -> Any:
    class DeadlineSleep025Channel(base_channel):
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

    DeadlineSleep025Channel.__name__ = "DeadlineSleep025Channel"
    return DeadlineSleep025Channel


def _load_channel(repo_root: Path) -> tuple[Any, Any, Any, Any]:
    channel, tensor, metadata, memory_format = _ORIGINAL_LOAD_CHANNEL(repo_root)
    return (
        _deadline_sleep025_channel_class(channel),
        tensor,
        metadata,
        memory_format,
    )


def _expected_contract() -> dict[str, Any]:
    payload = d10._expected_contract()
    payload["schema_version"] = "tempo-tp16-single-flight-deadline-sleep025-contract-11"
    payload["contract_id"] = CONTRACT_ID
    controller = payload["deadline_controller"]
    controller.pop("basis_result", None)
    controller.update(
        {
            "candidate_mode": CANDIDATE_MODE,
            "decode_progress_sleep_ms": 0.25,
            "branch_rule": BRANCH_RULE,
            "decision_evidence_results": list(DECISION_EVIDENCE_RESULTS),
        }
    )
    payload["campaign"]["modes"] = [old.FG, old.LMCACHE, CANDIDATE_MODE]
    payload["campaign"]["candidate_gates"]["all_candidate_configured_sleep_ms"] = 0.25
    return payload


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_contract():
        raise ValueError("TP16 deadline sleep025 E11 contract changed")
    return payload


def _run_block(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = d10._run_block(*args, **kwargs)
    source = int(kwargs["rank"]) < old.SOURCE_COUNT
    candidate = kwargs["mode"] == CANDIDATE_MODE
    result["source_call"]["configured_low_priority_sleep_ns"] = (
        LOW_PRIORITY_SLEEP_NS if source and candidate else 0
    )
    return result


def _aggregate(records: list[dict[str, Any]], trace: dict[str, Any], args: Any):
    result = _D10_AGGREGATE(records, trace, args)
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    gates = dict(result["candidate_gates"])
    gates.pop("all_candidate_configured_sleep_exact_0_5ms", None)
    progress_rows = result["deadline_controller"]["blocks"]
    gates["all_candidate_configured_sleep_exact_0_25ms"] = all(
        row["configured_low_priority_sleep_ms"] == 0.25 for row in progress_rows
    )
    result["config"].update(
        {
            "candidate_mode": CANDIDATE_MODE,
            "decode_progress_sleep_ms": 0.25,
            "branch_rule": BRANCH_RULE,
            "decision_evidence_results": list(DECISION_EVIDENCE_RESULTS),
        }
    )
    result["deadline_controller"].update(
        {
            "decode_progress_sleep_ms": 0.25,
            "branch_rule": BRANCH_RULE,
            "decision_evidence_results": list(DECISION_EVIDENCE_RESULTS),
        }
    )
    result["candidate_gates"] = gates
    result["screen_outcome"] = (
        "invalid_correctness_output_or_trace"
        if not result["overall_correctness_met"]
        else "deadline_sleep025_candidate_pass"
        if all(gates.values())
        else "deadline_sleep025_candidate_revise"
    )
    return result


def main() -> None:
    _install_candidate_mode()
    modules = (c9, d10, d10.abandoned_sleep2)
    for module in modules:
        module.CANDIDATE_MODE = CANDIDATE_MODE
        module.CONTRACT_ID = CONTRACT_ID
        module.RESULT_SCHEMA = RESULT_SCHEMA
        module.CONTROLLER_DECISION = CONTROLLER_DECISION
        module.PRIOR_OBSERVED_MIN_SLACK_MS = PRIOR_OBSERVED_MIN_SLACK_MS
        module.OBSERVED_SLACK_MIN_MS = OBSERVED_SLACK_MIN_MS
        module.DEFER_THRESHOLD_MS = OBSERVED_SLACK_MIN_MS
        module.BLOCKS = BLOCKS
    for module in (d10, d10.abandoned_sleep2):
        module.LOW_PRIORITY_SLEEP_S = LOW_PRIORITY_SLEEP_S
        module.LOW_PRIORITY_SLEEP_NS = LOW_PRIORITY_SLEEP_NS
        module.SERVICE_DELTA_MAX_MS = SERVICE_DELTA_MAX_MS
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
