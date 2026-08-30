#!/usr/bin/env python3
"""K18: frozen G14 with only local rescue trigger changed 950ms -> 975ms."""
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
CANDIDATE_MODE="tempo_prelaunch_local_rescue_975ms_paced25us"; CONTRACT_ID="tp16-single-flight-local-rescue-975ms-paced25us-k18"; RESULT_SCHEMA="tempo-vllm-tp16-local-rescue-975ms-paced25us-result-18"
BLOCKS=((0,old.FG),(0,old.LMCACHE),(0,CANDIDATE_MODE),(1,CANDIDATE_MODE),(1,old.FG),(1,old.LMCACHE),(2,old.LMCACHE),(2,CANDIDATE_MODE),(2,old.FG))
def _expected_contract():
    p=g14._expected_contract(); p["schema_version"]="tempo-tp16-local-rescue-975ms-paced25us-contract-18"; p["contract_id"]=CONTRACT_ID
    p["algorithm"].update(mode=CANDIDATE_MODE,trigger_ms=975.0,single_factor_from="G14 trigger_ms 950.0 -> 975.0",basis_result="results/vllm_lmcache_tp16_deadline_G14v2_job_56975950/result.json")
    p["campaign"]["modes"]=[old.FG,old.LMCACHE,CANDIDATE_MODE]; return p
def _load_contract(path:Path):
    p=json.loads(path.read_text(encoding="utf-8"))
    if p!=_expected_contract(): raise ValueError("K18 contract changed")
    return p
def _publish(name,rescue):
    rescue["configured_trigger_ms"]=975.0; g14._ORIGINAL_PUBLISH(name,rescue)
def _aggregate(records,trace,args):
    result=g14._aggregate(records,trace,args); result["schema_version"]=RESULT_SCHEMA; result["contract_id"]=CONTRACT_ID
    result["config"].update(candidate_mode=CANDIDATE_MODE,local_rescue_trigger_ms=975.0); result["local_rescue"]["trigger_ms"]=975.0
    result["screen_outcome"]="trigger975_candidate_pass" if result["overall_correctness_met"] and all(result["candidate_gates"].values()) else "trigger975_candidate_revise"; return result
def main():
    g14.CANDIDATE_MODE=f13.CANDIDATE_MODE=CANDIDATE_MODE; g14.CONTRACT_ID=f13.CONTRACT_ID=CONTRACT_ID; g14.RESULT_SCHEMA=f13.RESULT_SCHEMA=RESULT_SCHEMA; g14.BLOCKS=f13.BLOCKS=BLOCKS; f13.PACED_SLEEP_S=0.000025; f13.TRIGGER_NS=975_000_000; f13._install_mode()
    for m in (c9,e11,e12): m.CANDIDATE_MODE=CANDIDATE_MODE; m.CONTRACT_ID=CONTRACT_ID; m.RESULT_SCHEMA=RESULT_SCHEMA; m.BLOCKS=BLOCKS
    e11.LOCAL_RESCUE_TRIGGER_MS=975.0; e11._publish_rescue_record=_publish; fixed._transfer_worker=f13._paced_worker
    protocol.install_async_release_protocol(); old.protocol.ReleaseFrame=protocol.ReleaseFrame; old.protocol.install_generic_release_protocol=protocol.install_async_release_protocol; old.bulk.protocol.ReleaseFrame=protocol.ReleaseFrame; old.bulk.protocol.install_generic_release_protocol=protocol.install_async_release_protocol
    v8.CONTRACT_ID=CONTRACT_ID; v8.RESULT_SCHEMA=RESULT_SCHEMA; v8._load_contract=_load_contract; v8._run_block=f13._run_block; v8._validate_trace=c9._validate_trace; v8._aggregate=_aggregate; v8.main()
if __name__=="__main__": main()
