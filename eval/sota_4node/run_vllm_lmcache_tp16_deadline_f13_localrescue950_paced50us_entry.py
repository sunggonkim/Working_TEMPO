#!/usr/bin/env python3
"""F13: E12 PROC-only rescue with 50us paced boosted polling."""
from __future__ import annotations
import json
from pathlib import Path
import threading
import time
from typing import Any
import numpy as np
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_c9_entry as c9
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_e11_localrescue950_entry as e11
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_e12_localrescue950_safe_entry as e12
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as v8
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol

CANDIDATE_MODE="tempo_prelaunch_local_rescue_950ms_paced50us"
CONTRACT_ID="tp16-single-flight-local-rescue-950ms-paced50us-f13"
RESULT_SCHEMA="tempo-vllm-tp16-local-rescue-paced50us-result-13"
TRIGGER_NS=950_000_000
PACED_SLEEP_S=0.00005
BLOCKS=((0,old.FG),(0,old.LMCACHE),(0,CANDIDATE_MODE),(1,CANDIDATE_MODE),(1,old.FG),(1,old.LMCACHE),(2,old.LMCACHE),(2,CANDIDATE_MODE),(2,old.FG))
_E12_AGGREGATE=e12._aggregate

def _install_mode():
    old.TEMPO=CANDIDATE_MODE; old.MODES=(old.FG,old.LMCACHE,CANDIDATE_MODE); old.BLOCKS=BLOCKS

def _paced_worker(*,channel,obj,receiver_id,mode,boost,entered,done,state):
    name=threading.current_thread().name; start=time.perf_counter_ns()
    state["started_ns"]=start; entered.set(); candidate=mode==CANDIDATE_MODE
    rescue={"configured_trigger_ms":950.0,"timer_threads_created":0,"timer_started_ns":0,"timer_callback_ns":0,"trigger_callback_observed":False,"unfinished_at_trigger":False,"status_at_arm":"","rescue_armed":False,"rescue_armed_ns":0,"rescue_armed_from_worker_start_ms":0.0,"boost_observed_by_worker":False,"completed_before_rescue":False,"timer_cancelled":False,"timer_joined":True,"paced_boost_sleep_us":50.0,"paced_boost_sleeps":0}
    try:
        spec={"receiver_id":receiver_id,"remote_indexes":np.asarray([0],dtype=np.uint64)}
        if mode==old.LMCACHE:
            state["completed"]=int(channel.batched_write(objects=[obj],transfer_spec=spec))
        elif candidate:
            handle=channel.tempo_prepare([obj],spec)
            if channel.nixl_agent.transfer(handle)=="ERR": raise RuntimeError("prepared transfer post failed")
            polls=low=boost_polls=0
            while True:
                status=channel.nixl_agent.check_xfer_state(handle); polls+=1
                if status=="ERR": raise RuntimeError("prepared transfer failed")
                if status=="DONE":
                    state.update(completed=1,polls=polls,low_priority_sleeps=low,boost_polls=boost_polls,yields=0)
                    rescue["completed_before_rescue"]=not rescue["rescue_armed"]; break
                if status!="PROC": raise RuntimeError(f"unexpected NIXL state {status}")
                now=time.perf_counter_ns()
                if not rescue["rescue_armed"] and now-start>=TRIGGER_NS:
                    boost.set(); rescue.update(trigger_callback_observed=True,unfinished_at_trigger=True,status_at_arm="PROC",rescue_armed=True,rescue_armed_ns=now,rescue_armed_from_worker_start_ms=(now-start)/1e6)
                if boost.is_set():
                    boost_polls+=1; rescue["boost_observed_by_worker"]=True
                    time.sleep(PACED_SLEEP_S); rescue["paced_boost_sleeps"]+=1
                else:
                    time.sleep(0.001); low+=1
        else: raise RuntimeError(f"invalid mode {mode}")
    except BaseException as exc: state["error"]=f"{type(exc).__name__}: {exc}"
    finally:
        if candidate: e11._publish_rescue_record(name,rescue)
        state["finished_ns"]=time.perf_counter_ns(); done.set()

def _expected_contract():
    return {"schema_version":"tempo-tp16-local-rescue-paced50us-contract-13","contract_id":CONTRACT_ID,"algorithm":{"mode":CANDIDATE_MODE,"trigger_ms":950.0,"eligibility":"same_worker_observed_PROC","pre_trigger_sleep_ms":1.0,"post_arm_sleep_us":50.0,"post_arm_yields":0,"timer_threads":0,"hook_events":0,"global_rescue_collectives":0,"basis_result":"results/vllm_lmcache_tp16_deadline_E12_localrescue950_safe_job_56972950/result.json"},"transfer":{"bytes_per_source":16777216,"global_bytes":134217728,"calls_global":8,"physical_descriptors_global":8,"prepared_handle_repost":True,"worker_entry_precedes_http_request":True},"campaign":{"modes":[old.FG,old.LMCACHE,CANDIDATE_MODE],"blocks":9,"replicates_per_mode":3,"hard_gates":{"correctness_exact_geometry":True,"no_hook_gate_drain":True,"service_median_delta_le_ms":-5.0,"meaningful_wins_le_minus_5ms":2,"e2e_le_fg_ratio":1.05,"tpot_p99_le_lmcache_ratio":1.1,"armed_implies_boost_polls":True,"paced_poll_sleep_us":50.0,"yields":0},"reported_checks":{"observed_slack_ge_ms":200.0}}}

def _load_contract(path:Path):
    p=json.loads(path.read_text(encoding="utf-8"))
    if p!=_expected_contract(): raise ValueError("F13 contract changed")
    return p

def _run_block(*args,**kwargs):
    r=e11._run_block(*args,**kwargs); rank=int(kwargs["rank"]); mode=kwargs["mode"]
    if mode==CANDIDATE_MODE:
        call=r["source_call"]
        if rank<old.SOURCE_COUNT:
            exact=(call["calls"]==1 and call["completed"]==1 and call["descriptors"]==1 and call["bytes"]==old.BYTES_PER_SOURCE and call["error"] is None and (not call["rescue_armed"] or call["status_at_arm"]=="PROC" and call["boost_observed_by_worker"] and call["boost_polls"]>0) and call["paced_boost_sleeps"]==call["boost_polls"] and call["yields"]==0)
        else:
            exact=(call["calls"]==0 and r["receiver_verified_bytes"]==old.BYTES_PER_SOURCE)
        r["correctness_met"]=bool(exact); r["correctness_recomputed_by_f13"]=True
    return r

def _aggregate(records,trace,args):
    result=_E12_AGGREGATE(records,trace,args); result["schema_version"]=RESULT_SCHEMA; result["contract_id"]=CONTRACT_ID
    ordered=sorted(records,key=lambda x:int(x["rank"])); pace=[]
    for i,b in enumerate(result["blocks"]):
        if b["mode"]!=CANDIDATE_MODE: continue
        calls=[x["blocks"][i]["source_call"] for x in ordered[:8]]
        row={"block_index":i,"paced_boost_polls":sum(int(c["boost_polls"]) for c in calls),"paced_boost_sleeps":sum(int(c["paced_boost_sleeps"]) for c in calls),"no_yields":all(int(c["yields"])==0 for c in calls),"paced_exact":all(int(c["paced_boost_sleeps"])==int(c["boost_polls"]) for c in calls),"armed_observed":all(not c["rescue_armed"] or c["boost_observed_by_worker"] and int(c["boost_polls"])>0 for c in calls)}
        b["paced_rescue"]=row; pace.append(row)
    gates=dict(result["candidate_gates"]); slack=gates.pop("all_candidate_observed_slack_ge_200ms",False)
    gates.update(all_paced_boost_polls_sleep_exact_50us=all(x["paced_exact"] for x in pace),all_candidate_yields_zero=all(x["no_yields"] for x in pace),all_armed_sources_observed_paced_poll=all(x["armed_observed"] for x in pace))
    result["config"].update(candidate_mode=CANDIDATE_MODE,post_arm_progress="sleep_50us_every_PROC_poll",post_arm_sleep_us=50.0)
    result["reported_checks"]={"all_candidate_observed_slack_ge_200ms":slack,"observed_min_slack_ms":min(b["observed_completion_slack_ms"] for b in result["blocks"] if b["mode"]==CANDIDATE_MODE)}
    result["paced_rescue"]={"post_arm_sleep_us":50.0,"blocks":pace}; result["candidate_gates"]=gates
    result["screen_outcome"]="paced50us_candidate_pass" if result["overall_correctness_met"] and all(gates.values()) else "paced50us_candidate_revise"
    return result

def main():
    _install_mode()
    for m in (c9,e11,e12): m.CANDIDATE_MODE=CANDIDATE_MODE; m.CONTRACT_ID=CONTRACT_ID; m.RESULT_SCHEMA=RESULT_SCHEMA; m.BLOCKS=BLOCKS
    e11.LOCAL_RESCUE_TRIGGER_MS=950.0; fixed._transfer_worker=_paced_worker
    protocol.install_async_release_protocol(); old.protocol.ReleaseFrame=protocol.ReleaseFrame; old.protocol.install_generic_release_protocol=protocol.install_async_release_protocol; old.bulk.protocol.ReleaseFrame=protocol.ReleaseFrame; old.bulk.protocol.install_generic_release_protocol=protocol.install_async_release_protocol
    v8.CONTRACT_ID=CONTRACT_ID; v8.RESULT_SCHEMA=RESULT_SCHEMA; v8._load_contract=_load_contract; v8._run_block=_run_block; v8._validate_trace=c9._validate_trace; v8._aggregate=_aggregate; v8.main()
if __name__=="__main__": main()
