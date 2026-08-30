#!/usr/bin/env python3
"""T27: interleaved rolling-median admission, three cycles per prompt."""
from __future__ import annotations
import json, statistics
from pathlib import Path
from typing import Any
from eval.sota_4node import run_vllm_lmcache_tp16_calibrated_admission_q24_entry as q
from eval.sota_4node import run_vllm_lmcache_tp16_robust_admission_s26_entry as s
CANDIDATE_MODE="tempo_rolling_cache_or_decode";CONTRACT_ID="tp16-rolling-cache-admission-t27";RESULT_SCHEMA="tempo-vllm-tp16-rolling-admission-result-27";TOKENS=256
BLOCKS=tuple(item for p in range(3) for _cycle in range(3) for item in ((p,q.old.FG),(p,q.old.LMCACHE),(p,CANDIDATE_MODE)))
_SAMPLES:dict[int,dict[str,list[int]]]={}
_ORIGINAL_Q_AGGREGATE=q._aggregate;_S26_AGGREGATE=s._aggregate
def _expected_contract()->dict[str,Any]:
 return {"schema_version":"tempo-tp16-rolling-admission-contract-27","contract_id":CONTRACT_ID,"topology":{"nodes":4,"world_size":16,"source_ranks":list(range(8)),"receiver_ranks":list(range(8,16)),"pairing":[[r,r+8] for r in range(8)]},"transfer":{"bytes_per_source":16<<20,"global_bytes":128<<20,"calls_global_if_remote":8,"physical_descriptors_global_if_remote":8,"prepared_handle_repost":True},"controller":{"candidate_mode":CANDIDATE_MODE,"tokens":256,"cycles_per_prompt":3,"cycle_order":[q.old.FG,q.old.LMCACHE,CANDIDATE_MODE],"decision_statistic":"rolling_median_service_makespan","candidate_uses_only_current_and_prior_calibration":True,"remote_branch":"prepared_repost_sleep_1ms","decode_branch":"no_background_transfer","hook_events":0,"boost_enabled":False},"campaign":{"prompts":3,"blocks":27,"candidate_gates":{"exact_correctness":True,"repeat_decisions_stable":True,"paired_prompt_median_lmcache_delta_le_ms":-5.0,"meaningful_prompt_wins":2,"median_prompt_oracle_regret_le_ms":10.0,"e2e_le_fg_ratio":1.05,"tpot_p99_le_lmcache_ratio":1.10}}}
def _load_contract(path:Path):
 p=json.loads(path.read_text());
 if p!=_expected_contract():raise ValueError("T27 contract changed")
 return p
def _run_block(torch,dist,*,mode:str,rank:int,prompt_index:int,**kwargs):
 if mode!=CANDIDATE_MODE:
  row=q.c9._run_block(torch,dist,mode=mode,rank=rank,prompt_index=prompt_index,**kwargs);service=q._global_service(torch,dist,row,rank);_SAMPLES.setdefault(prompt_index,{}).setdefault(mode,[]).append(service);row["effective_mode"]=mode;row["calibration_service_ns"]=service;return row
 samples=_SAMPLES.get(prompt_index,{})
 if not samples.get(q.old.FG) or len(samples[q.old.FG])!=len(samples.get(q.old.LMCACHE,[])):raise RuntimeError("T27 incomplete rolling calibration")
 fg=int(statistics.median(samples[q.old.FG]));lm=int(statistics.median(samples[q.old.LMCACHE]));selected=q.old.FG if fg<=lm else CANDIDATE_MODE
 row=q.c9._run_block(torch,dist,mode=selected,rank=rank,prompt_index=prompt_index,**kwargs);row["mode"]=CANDIDATE_MODE;row["effective_mode"]="decode_no_transfer" if selected==q.old.FG else "remote_prepared";row["controller_decision"]=row["effective_mode"];row["calibration_fg_service_ns"]=fg;row["calibration_lmcache_service_ns"]=lm;row["decision_matches_calibration_argmin"]=True;row["rolling_calibration_count"]=len(samples[q.old.FG]);return row
def _aggregate(records,trace,args):
 installed=q._aggregate;q._aggregate=_ORIGINAL_Q_AGGREGATE
 try:result=_S26_AGGREGATE(records,trace,args)
 finally:q._aggregate=installed
 result["schema_version"]=RESULT_SCHEMA;result["contract_id"]=CONTRACT_ID;result["config"].update(candidate_mode=CANDIDATE_MODE,tokens=256,cycles_per_prompt=3,decision_statistic="rolling_median_service_makespan");gates=dict(result["candidate_gates"]);result["candidate_gates"]=gates;result["screen_outcome"]=("invalid_correctness_output_or_trace" if not result["overall_correctness_met"] else "rolling_admission_pass" if all(gates.values()) else "rolling_admission_revise");return result
def main():
 q.CANDIDATE_MODE=CANDIDATE_MODE;q.CONTRACT_ID=CONTRACT_ID;q.RESULT_SCHEMA=RESULT_SCHEMA;q.BLOCKS=BLOCKS;q.old.TOKENS=256;q._load_contract=_load_contract;q._run_block=_run_block;q._aggregate=_aggregate
 s.CANDIDATE_MODE=CANDIDATE_MODE;s.CONTRACT_ID=CONTRACT_ID;s.RESULT_SCHEMA=RESULT_SCHEMA;s.BLOCKS=BLOCKS;s._run_block=_run_block;s._aggregate=_aggregate
 q.main()
if __name__=="__main__":main()
