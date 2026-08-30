#!/usr/bin/env python3
"""V29: require two consecutive 50ms remote advantages before switching."""
from __future__ import annotations
import json,statistics
from pathlib import Path
from typing import Any
from eval.sota_4node import run_vllm_lmcache_tp16_calibrated_admission_q24_entry as q
from eval.sota_4node import run_vllm_lmcache_tp16_robust_admission_s26_entry as s
from eval.sota_4node import run_vllm_lmcache_tp16_rolling_admission_t27_entry as t
CANDIDATE_MODE="tempo_confirmed_cache_or_decode";CONTRACT_ID="tp16-confirmed-cache-admission-v29";RESULT_SCHEMA="tempo-vllm-tp16-confirmed-admission-result-29";MARGIN_MS=50.0;BLOCKS=tuple(item for p in range(3) for _ in range(3) for item in ((p,q.old.FG),(p,q.old.LMCACHE),(p,CANDIDATE_MODE)))
_SAMPLES:dict[int,dict[str,list[int]]]={};_DECISION:dict[int,str]={};_REMOTE_STREAK:dict[int,int]={}
def _expected_contract()->dict[str,Any]:
 return {"schema_version":"tempo-tp16-confirmed-admission-contract-29","contract_id":CONTRACT_ID,"topology":{"nodes":4,"world_size":16,"source_ranks":list(range(8)),"receiver_ranks":list(range(8,16)),"pairing":[[r,r+8] for r in range(8)]},"transfer":{"bytes_per_source":16<<20,"global_bytes":128<<20,"calls_global_if_remote":8,"physical_descriptors_global_if_remote":8,"prepared_handle_repost":True},"controller":{"candidate_mode":CANDIDATE_MODE,"tokens":256,"cycles_per_prompt":3,"cycle_order":[q.old.FG,q.old.LMCACHE,CANDIDATE_MODE],"decision_statistic":"rolling_median_service_makespan","initial_path":"decode_no_transfer","remote_advantage_margin_ms":50.0,"remote_confirmation_cycles":2,"fallback_to_decode_when_fg_not_slower":True,"remote_branch":"prepared_repost_sleep_1ms","decode_branch":"no_background_transfer","hook_events":0,"boost_enabled":False},"campaign":{"prompts":3,"blocks":27,"candidate_gates":{"exact_correctness":True,"path_transition_valid":True,"paired_prompt_median_lmcache_delta_le_ms":-5.0,"meaningful_prompt_wins":2,"median_prompt_oracle_regret_le_ms":10.0,"e2e_le_fg_ratio":1.05,"tpot_p99_le_lmcache_ratio":1.10}}}
def _load_contract(path:Path):
 p=json.loads(path.read_text());
 if p!=_expected_contract():raise ValueError("V29 contract changed")
 return p
def _run_block(torch,dist,*,mode:str,rank:int,prompt_index:int,**kwargs):
 if mode!=CANDIDATE_MODE:
  row=q.c9._run_block(torch,dist,mode=mode,rank=rank,prompt_index=prompt_index,**kwargs);service=q._global_service(torch,dist,row,rank);_SAMPLES.setdefault(prompt_index,{}).setdefault(mode,[]).append(service);row["effective_mode"]=mode;row["calibration_service_ns"]=service;return row
 samples=_SAMPLES[prompt_index];fg=int(statistics.median(samples[q.old.FG]));lm=int(statistics.median(samples[q.old.LMCACHE]));current=_DECISION.get(prompt_index,"decode_no_transfer");margin=int(MARGIN_MS*1e6)
 if current=="remote_prepared" and fg<=lm:current="decode_no_transfer";_REMOTE_STREAK[prompt_index]=0
 elif current=="decode_no_transfer":
  _REMOTE_STREAK[prompt_index]=_REMOTE_STREAK.get(prompt_index,0)+1 if lm+margin<fg else 0
  if _REMOTE_STREAK[prompt_index]>=2:current="remote_prepared"
 _DECISION[prompt_index]=current;selected=q.old.FG if current=="decode_no_transfer" else CANDIDATE_MODE
 row=q.c9._run_block(torch,dist,mode=selected,rank=rank,prompt_index=prompt_index,**kwargs);row["mode"]=CANDIDATE_MODE;row["effective_mode"]=current;row["controller_decision"]=current;row["calibration_fg_service_ns"]=fg;row["calibration_lmcache_service_ns"]=lm;row["decision_matches_calibration_argmin"]=(current==("decode_no_transfer" if fg<=lm else "remote_prepared"));row["rolling_calibration_count"]=len(samples[q.old.FG]);row["remote_advantage_streak"]=_REMOTE_STREAK.get(prompt_index,0);return row
def _aggregate(records,trace,args):
 result=t._aggregate(records,trace,args);by_prompt={p:[b["effective_mode"] for b in result["blocks"] if b["mode"]==CANDIDATE_MODE and b["prompt_index"]==p] for p in range(3)}
 valid=all(seq in (["decode_no_transfer"]*3,["decode_no_transfer","remote_prepared","remote_prepared"],["decode_no_transfer","remote_prepared","decode_no_transfer"]) for seq in by_prompt.values())
 result["schema_version"]=RESULT_SCHEMA;result["contract_id"]=CONTRACT_ID;result["config"].update(candidate_mode=CANDIDATE_MODE,remote_advantage_margin_ms=50.0,remote_confirmation_cycles=2,fallback_to_decode_when_fg_not_slower=True);result["path_sequences"]=by_prompt;g=dict(result["candidate_gates"]);g.pop("all_repeat_decisions_agree",None);g["all_path_transitions_valid"]=valid;result["candidate_gates"]=g;result["screen_outcome"]=("invalid_correctness_output_or_trace" if not result["overall_correctness_met"] else "confirmed_admission_pass" if all(g.values()) else "confirmed_admission_revise");return result
def main():
 for module in (q,s,t):module.CANDIDATE_MODE=CANDIDATE_MODE;module.CONTRACT_ID=CONTRACT_ID;module.RESULT_SCHEMA=RESULT_SCHEMA;module.BLOCKS=BLOCKS
 q.old.TOKENS=256;q._load_contract=_load_contract;q._run_block=_run_block;q._aggregate=_aggregate;q.main()
if __name__=="__main__":main()
