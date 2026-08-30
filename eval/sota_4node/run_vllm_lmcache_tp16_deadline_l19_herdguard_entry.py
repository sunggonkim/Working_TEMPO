#!/usr/bin/env python3
"""L19: G14 with a one-shot source-count guard against rescue herds."""
from __future__ import annotations
import json, statistics, threading, time
from pathlib import Path
from typing import Any
import numpy as np
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_c9_entry as c9
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_e11_localrescue950_entry as e11
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_e12_localrescue950_safe_entry as e12
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_f13_localrescue950_paced50us_entry as f13
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_g14_localrescue950_paced25us_entry as g14
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as v8
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol
CANDIDATE_MODE="tempo_prelaunch_herdguard4_950ms_paced25us"; CONTRACT_ID="tp16-single-flight-herdguard4-950ms-paced25us-l19"; RESULT_SCHEMA="tempo-vllm-tp16-herdguard4-result-19"
TRIGGER_NS=950_000_000; HERD_LIMIT=4; BLOCKS=((0,old.FG),(0,old.LMCACHE),(0,CANDIDATE_MODE),(1,CANDIDATE_MODE),(1,old.FG),(1,old.LMCACHE),(2,old.LMCACHE),(2,CANDIDATE_MODE),(2,old.FG))
_SOURCE_GROUP=None

def _expected_contract():
    p=g14._expected_contract(); p["schema_version"]="tempo-tp16-herdguard4-contract-19"; p["contract_id"]=CONTRACT_ID
    p["algorithm"].update(mode=CANDIDATE_MODE,single_factor_from="G14 one-shot source unfinished count; rescue only when count<=4",basis_result="results/vllm_lmcache_tp16_deadline_G14_validation1_job_56975950/result.json",herd_limit=HERD_LIMIT,source_control_collectives_per_candidate=1)
    p["campaign"]["modes"]=[old.FG,old.LMCACHE,CANDIDATE_MODE]; return p
def _load_contract(path:Path):
    p=json.loads(path.read_text(encoding="utf-8"))
    if p!=_expected_contract(): raise ValueError("L19 contract changed")
    return p

def _worker(*,channel,obj,receiver_id,mode,boost,entered,done,state):
    import torch, torch.distributed as dist
    name=threading.current_thread().name; start=time.perf_counter_ns(); state["started_ns"]=start; entered.set(); candidate=mode==CANDIDATE_MODE
    rescue={"configured_trigger_ms":950.0,"timer_threads_created":0,"timer_started_ns":0,"timer_callback_ns":0,"trigger_callback_observed":False,"unfinished_at_trigger":False,"rescue_armed":False,"rescue_armed_ns":0,"rescue_armed_from_worker_start_ms":0.0,"completed_before_rescue":False,"timer_cancelled":False,"timer_joined":True,"status_at_arm":"","boost_observed_by_worker":False,"paced_boost_sleep_us":25.0,"paced_boost_sleeps":0,"source_control_collectives":0,"global_unfinished_sources":0,"herd_suppressed":False}
    actual_finished_ns=0
    try:
        spec={"receiver_id":receiver_id,"remote_indexes":np.asarray([0],dtype=np.uint64)}
        if mode==old.LMCACHE:
            state["completed"]=int(channel.batched_write(objects=[obj],transfer_spec=spec)); actual_finished_ns=time.perf_counter_ns()
        elif candidate:
            handle=channel.tempo_prepare([obj],spec)
            if channel.nixl_agent.transfer(handle)=="ERR": raise RuntimeError("prepared transfer post failed")
            polls=low=boost_polls=0; status="PROC"
            while time.perf_counter_ns()-start<TRIGGER_NS:
                status=channel.nixl_agent.check_xfer_state(handle); polls+=1
                if status=="ERR": raise RuntimeError("prepared transfer failed")
                if status=="DONE": actual_finished_ns=time.perf_counter_ns(); break
                if status!="PROC": raise RuntimeError(f"unexpected NIXL state {status}")
                time.sleep(0.001); low+=1
            remaining=max(0,TRIGGER_NS-(time.perf_counter_ns()-start))
            if remaining: time.sleep(remaining/1e9)
            unfinished=status!="DONE"; rescue.update(timer_callback_ns=time.perf_counter_ns(),trigger_callback_observed=True,unfinished_at_trigger=unfinished,completed_before_rescue=not unfinished,source_control_collectives=1)
            flag=torch.tensor([1 if unfinished else 0],dtype=torch.int64); dist.all_reduce(flag,op=dist.ReduceOp.SUM,group=_SOURCE_GROUP); count=int(flag.item()); rescue["global_unfinished_sources"]=count
            if unfinished:
                status=channel.nixl_agent.check_xfer_state(handle); polls+=1
                if status=="DONE": actual_finished_ns=time.perf_counter_ns()
                elif status=="ERR": raise RuntimeError("prepared transfer failed after guard")
                elif status!="PROC": raise RuntimeError(f"unexpected NIXL state {status}")
            if unfinished and status=="PROC" and count<=HERD_LIMIT:
                now=time.perf_counter_ns(); boost.set(); rescue.update(rescue_armed=True,rescue_armed_ns=now,rescue_armed_from_worker_start_ms=(now-start)/1e6,status_at_arm="PROC")
            elif unfinished and count>HERD_LIMIT: rescue["herd_suppressed"]=True
            while status!="DONE":
                if status=="ERR": raise RuntimeError("prepared transfer failed")
                if rescue["rescue_armed"]:
                    boost_polls+=1; rescue["boost_observed_by_worker"]=True; time.sleep(0.000025); rescue["paced_boost_sleeps"]+=1
                else: time.sleep(0.001); low+=1
                status=channel.nixl_agent.check_xfer_state(handle); polls+=1
            if not actual_finished_ns: actual_finished_ns=time.perf_counter_ns()
            state.update(completed=1,polls=polls,low_priority_sleeps=low,boost_polls=boost_polls,yields=0)
        else: raise RuntimeError(f"invalid mode {mode}")
    except BaseException as exc: state["error"]=f"{type(exc).__name__}: {exc}"; actual_finished_ns=time.perf_counter_ns()
    finally:
        if candidate: e11._publish_rescue_record(name,rescue)
        state["finished_ns"]=actual_finished_ns or time.perf_counter_ns(); done.set()

def _run_block(*args,**kwargs):
    global _SOURCE_GROUP
    torch,dist=args[0],args[1]
    if _SOURCE_GROUP is None: _SOURCE_GROUP=dist.new_group(ranks=list(range(old.SOURCE_COUNT)))
    rank=int(kwargs["rank"]); mode=kwargs["mode"]; block_index=int(kwargs["block_index"]); result=c9._run_block(*args,**kwargs)
    if rank<old.SOURCE_COUNT and mode==CANDIDATE_MODE:
        record=e11._take_rescue_record(f"deadline-c9-transfer-rank{rank}-block{block_index}")
        if record is None: raise RuntimeError("missing herd-guard record")
        result["source_call"].update(record); call=result["source_call"]
        exact=(call["calls"]==1 and call["completed"]==1 and call["descriptors"]==1 and call["bytes"]==old.BYTES_PER_SOURCE and call["error"] is None and call["source_control_collectives"]==1 and (not call["rescue_armed"] or call["boost_observed_by_worker"] and call["boost_polls"]>0) and call["paced_boost_sleeps"]==call["boost_polls"])
        result["correctness_met"]=bool(exact)
    elif rank>=old.SOURCE_COUNT and mode==CANDIDATE_MODE: result["correctness_met"]=bool(result["receiver_verified_bytes"]==old.BYTES_PER_SOURCE)
    return result

def _aggregate(records,trace,args):
    result=g14._aggregate(records,trace,args); result["schema_version"]=RESULT_SCHEMA; result["contract_id"]=CONTRACT_ID
    ordered=sorted(records,key=lambda x:int(x["rank"])); rows=[]
    for i,block in enumerate(result["blocks"]):
        if block["mode"]!=CANDIDATE_MODE: continue
        calls=[r["blocks"][i]["source_call"] for r in ordered[:8]]; counts={int(c["global_unfinished_sources"]) for c in calls}; count=counts.pop() if len(counts)==1 else -1; armed=sum(bool(c["rescue_armed"]) for c in calls); suppressed=sum(bool(c["herd_suppressed"]) for c in calls)
        row={"block_index":i,"global_unfinished_sources":count,"rescue_armed_sources":armed,"herd_suppressed_sources":suppressed,"collective_exact":all(int(c["source_control_collectives"])==1 for c in calls),"decision_exact":(count<=HERD_LIMIT and suppressed==0 and armed<=count) or (count>HERD_LIMIT and armed==0 and suppressed==count)}; block["herd_guard"]=row; rows.append(row)
    gates=dict(result["candidate_gates"])
    for key in ("triggered_count_equals_unfinished_at_950ms","at_least_one_rescue_observed","global_rescue_collectives_zero","rescue_only_unfinished_sources"):
        gates.pop(key,None)
    gates.update(all_candidate_source_count_collectives_exact=all(r["collective_exact"] for r in rows),all_herd_guard_decisions_exact=all(r["decision_exact"] for r in rows),all_armed_sources_are_unfinished=all(r["rescue_armed_sources"]<=r["global_unfinished_sources"] for r in rows))
    result["config"].update(candidate_mode=CANDIDATE_MODE,herd_limit=HERD_LIMIT,source_control_collectives_per_candidate=1); result["herd_guard"]={"limit":HERD_LIMIT,"blocks":rows}; result["candidate_gates"]=gates
    result["screen_outcome"]="herdguard4_candidate_pass" if result["overall_correctness_met"] and all(gates.values()) else "herdguard4_candidate_revise"; return result

def main():
    g14.CANDIDATE_MODE=f13.CANDIDATE_MODE=CANDIDATE_MODE; g14.CONTRACT_ID=f13.CONTRACT_ID=CONTRACT_ID; g14.RESULT_SCHEMA=f13.RESULT_SCHEMA=RESULT_SCHEMA; g14.BLOCKS=f13.BLOCKS=BLOCKS; f13._install_mode()
    for m in (c9,e11,e12): m.CANDIDATE_MODE=CANDIDATE_MODE; m.CONTRACT_ID=CONTRACT_ID; m.RESULT_SCHEMA=RESULT_SCHEMA; m.BLOCKS=BLOCKS
    fixed._transfer_worker=_worker; protocol.install_async_release_protocol(); old.protocol.ReleaseFrame=protocol.ReleaseFrame; old.protocol.install_generic_release_protocol=protocol.install_async_release_protocol; old.bulk.protocol.ReleaseFrame=protocol.ReleaseFrame; old.bulk.protocol.install_generic_release_protocol=protocol.install_async_release_protocol
    v8.CONTRACT_ID=CONTRACT_ID; v8.RESULT_SCHEMA=RESULT_SCHEMA; v8._load_contract=_load_contract; v8._run_block=_run_block; v8._validate_trace=c9._validate_trace; v8._aggregate=_aggregate; v8.main()
if __name__=="__main__": main()
