#!/usr/bin/env python3
"""G14: F13 with only post-rescue polling sleep changed 50us -> 25us."""
from __future__ import annotations
import json
from pathlib import Path
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_c9_entry as c9
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_e11_localrescue950_entry as e11
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_e12_localrescue950_safe_entry as e12
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_f13_localrescue950_paced50us_entry as f13
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as v8
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol

CANDIDATE_MODE = "tempo_prelaunch_local_rescue_950ms_paced25us"
CONTRACT_ID = "tp16-single-flight-local-rescue-950ms-paced25us-g14"
RESULT_SCHEMA = "tempo-vllm-tp16-local-rescue-paced25us-result-14"
BLOCKS = ((0, old.FG), (0, old.LMCACHE), (0, CANDIDATE_MODE),
          (1, CANDIDATE_MODE), (1, old.FG), (1, old.LMCACHE),
          (2, old.LMCACHE), (2, CANDIDATE_MODE), (2, old.FG))
_ORIGINAL_PUBLISH = e11._publish_rescue_record

def _expected_contract():
    return {
        "schema_version": "tempo-tp16-local-rescue-paced25us-contract-14",
        "contract_id": CONTRACT_ID,
        "algorithm": {
            "mode": CANDIDATE_MODE, "trigger_ms": 950.0,
            "eligibility": "same_worker_observed_PROC",
            "pre_trigger_sleep_ms": 1.0, "post_arm_sleep_us": 25.0,
            "post_arm_yields": 0, "timer_threads": 0, "hook_events": 0,
            "global_rescue_collectives": 0,
            "single_factor_from": "F13 post_arm_sleep_us 50.0 -> 25.0",
            "basis_result": "results/vllm_lmcache_tp16_deadline_F13_localrescue950_paced50us_job_56972950/result.json",
        },
        "transfer": {
            "bytes_per_source": 16777216, "global_bytes": 134217728,
            "calls_global": 8, "physical_descriptors_global": 8,
            "prepared_handle_repost": True,
            "worker_entry_precedes_http_request": True,
        },
        "campaign": {
            "modes": [old.FG, old.LMCACHE, CANDIDATE_MODE], "blocks": 9,
            "replicates_per_mode": 3,
            "hard_gates": {
                "correctness_exact_geometry": True,
                "no_hook_gate_drain": True,
                "service_median_delta_le_ms": -5.0,
                "meaningful_wins_le_minus_5ms": 2,
                "e2e_le_fg_ratio": 1.05,
                "tpot_p99_le_lmcache_ratio": 1.1,
                "armed_implies_boost_polls": True,
                "paced_poll_sleep_us": 25.0, "yields": 0,
            },
            "reported_checks": {"observed_slack_ge_ms": 200.0},
        },
    }

def _load_contract(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_contract():
        raise ValueError("G14 contract changed")
    return payload

def _publish(name, rescue):
    rescue["paced_boost_sleep_us"] = 25.0
    _ORIGINAL_PUBLISH(name, rescue)

def _aggregate(records, trace, args):
    result = f13._aggregate(records, trace, args)
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    result["config"].update(
        candidate_mode=CANDIDATE_MODE,
        post_arm_progress="sleep_25us_every_PROC_poll",
        post_arm_sleep_us=25.0,
    )
    result["paced_rescue"]["post_arm_sleep_us"] = 25.0
    gates = dict(result["candidate_gates"])
    exact = gates.pop("all_paced_boost_polls_sleep_exact_50us")
    gates["all_paced_boost_polls_sleep_exact_25us"] = exact
    result["candidate_gates"] = gates
    result["screen_outcome"] = (
        "paced25us_candidate_pass"
        if result["overall_correctness_met"] and all(gates.values())
        else "paced25us_candidate_revise"
    )
    return result

def main():
    f13.CANDIDATE_MODE = CANDIDATE_MODE
    f13.CONTRACT_ID = CONTRACT_ID
    f13.RESULT_SCHEMA = RESULT_SCHEMA
    f13.BLOCKS = BLOCKS
    f13.PACED_SLEEP_S = 0.000025
    f13._install_mode()
    for module in (c9, e11, e12):
        module.CANDIDATE_MODE = CANDIDATE_MODE
        module.CONTRACT_ID = CONTRACT_ID
        module.RESULT_SCHEMA = RESULT_SCHEMA
        module.BLOCKS = BLOCKS
    e11.LOCAL_RESCUE_TRIGGER_MS = 950.0
    e11._publish_rescue_record = _publish
    fixed._transfer_worker = f13._paced_worker
    protocol.install_async_release_protocol()
    old.protocol.ReleaseFrame = protocol.ReleaseFrame
    old.protocol.install_generic_release_protocol = protocol.install_async_release_protocol
    old.bulk.protocol.ReleaseFrame = protocol.ReleaseFrame
    old.bulk.protocol.install_generic_release_protocol = protocol.install_async_release_protocol
    v8.CONTRACT_ID = CONTRACT_ID
    v8.RESULT_SCHEMA = RESULT_SCHEMA
    v8._load_contract = _load_contract
    v8._run_block = f13._run_block
    v8._validate_trace = c9._validate_trace
    v8._aggregate = _aggregate
    v8.main()

if __name__ == "__main__":
    main()
