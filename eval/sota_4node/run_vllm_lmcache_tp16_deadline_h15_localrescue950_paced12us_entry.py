#!/usr/bin/env python3
"""H15: G14 with only post-rescue polling sleep changed 25us -> 12.5us."""
from __future__ import annotations
import json
from pathlib import Path
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_c9_entry as c9
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_e11_localrescue950_entry as e11
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_e12_localrescue950_safe_entry as e12
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_f13_localrescue950_paced50us_entry as f13
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_g14_localrescue950_paced25us_entry as g14
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as v8
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol

CANDIDATE_MODE = "tempo_prelaunch_local_rescue_950ms_paced12_5us"
CONTRACT_ID = "tp16-single-flight-local-rescue-950ms-paced12-5us-h15"
RESULT_SCHEMA = "tempo-vllm-tp16-local-rescue-paced12-5us-result-15"
BLOCKS = ((0, old.FG), (0, old.LMCACHE), (0, CANDIDATE_MODE),
          (1, CANDIDATE_MODE), (1, old.FG), (1, old.LMCACHE),
          (2, old.LMCACHE), (2, CANDIDATE_MODE), (2, old.FG))

def _expected_contract():
    payload = g14._expected_contract()
    payload["schema_version"] = "tempo-tp16-local-rescue-paced12-5us-contract-15"
    payload["contract_id"] = CONTRACT_ID
    payload["algorithm"].update(
        mode=CANDIDATE_MODE, post_arm_sleep_us=12.5,
        single_factor_from="G14 post_arm_sleep_us 25.0 -> 12.5",
        basis_result="results/vllm_lmcache_tp16_deadline_G14v2_job_56975950/result.json",
    )
    payload["campaign"]["modes"] = [old.FG, old.LMCACHE, CANDIDATE_MODE]
    payload["campaign"]["hard_gates"]["paced_poll_sleep_us"] = 12.5
    return payload

def _load_contract(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_contract(): raise ValueError("H15 contract changed")
    return payload

def _publish(name, rescue):
    rescue["paced_boost_sleep_us"] = 12.5
    g14._ORIGINAL_PUBLISH(name, rescue)

def _aggregate(records, trace, args):
    result = g14._aggregate(records, trace, args)
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    result["config"].update(candidate_mode=CANDIDATE_MODE,
                            post_arm_progress="sleep_12_5us_every_PROC_poll",
                            post_arm_sleep_us=12.5)
    result["paced_rescue"]["post_arm_sleep_us"] = 12.5
    gates = dict(result["candidate_gates"])
    exact = gates.pop("all_paced_boost_polls_sleep_exact_25us")
    gates["all_paced_boost_polls_sleep_exact_12_5us"] = exact
    result["candidate_gates"] = gates
    result["screen_outcome"] = (
        "paced12_5us_candidate_pass"
        if result["overall_correctness_met"] and all(gates.values())
        else "paced12_5us_candidate_revise")
    return result

def main():
    g14.CANDIDATE_MODE = f13.CANDIDATE_MODE = CANDIDATE_MODE
    g14.CONTRACT_ID = f13.CONTRACT_ID = CONTRACT_ID
    g14.RESULT_SCHEMA = f13.RESULT_SCHEMA = RESULT_SCHEMA
    g14.BLOCKS = f13.BLOCKS = BLOCKS
    f13.PACED_SLEEP_S = 0.0000125
    f13._install_mode()
    for module in (c9, e11, e12):
        module.CANDIDATE_MODE = CANDIDATE_MODE; module.CONTRACT_ID = CONTRACT_ID
        module.RESULT_SCHEMA = RESULT_SCHEMA; module.BLOCKS = BLOCKS
    e11.LOCAL_RESCUE_TRIGGER_MS = 950.0; e11._publish_rescue_record = _publish
    fixed._transfer_worker = f13._paced_worker
    protocol.install_async_release_protocol()
    old.protocol.ReleaseFrame = protocol.ReleaseFrame
    old.protocol.install_generic_release_protocol = protocol.install_async_release_protocol
    old.bulk.protocol.ReleaseFrame = protocol.ReleaseFrame
    old.bulk.protocol.install_generic_release_protocol = protocol.install_async_release_protocol
    v8.CONTRACT_ID = CONTRACT_ID; v8.RESULT_SCHEMA = RESULT_SCHEMA
    v8._load_contract = _load_contract; v8._run_block = f13._run_block
    v8._validate_trace = c9._validate_trace; v8._aggregate = _aggregate
    v8.main()

if __name__ == "__main__": main()
