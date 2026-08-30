#!/usr/bin/env python3
"""S26: two-sample median calibration and three candidate replicates per prompt."""
from __future__ import annotations
import json, statistics
from pathlib import Path
from typing import Any
from eval.sota_4node import run_vllm_lmcache_tp16_calibrated_admission_q24_entry as q
CANDIDATE_MODE="tempo_robust_cache_or_decode";CONTRACT_ID="tp16-robust-cache-admission-s26";RESULT_SCHEMA="tempo-vllm-tp16-robust-admission-result-26";TOKENS=256
BLOCKS=tuple(item for p in range(3) for item in ((p,q.old.FG),(p,q.old.FG),(p,q.old.LMCACHE),(p,q.old.LMCACHE),(p,CANDIDATE_MODE),(p,CANDIDATE_MODE),(p,CANDIDATE_MODE)))
_SAMPLES:dict[int,dict[str,list[int]]]={}
def _expected_contract()->dict[str,Any]:
 return {"schema_version":"tempo-tp16-robust-admission-contract-26","contract_id":CONTRACT_ID,"topology":{"nodes":4,"world_size":16,"source_ranks":list(range(8)),"receiver_ranks":list(range(8,16)),"pairing":[[r,r+8] for r in range(8)]},"transfer":{"bytes_per_source":16<<20,"global_bytes":128<<20,"calls_global_if_remote":8,"physical_descriptors_global_if_remote":8,"prepared_handle_repost":True},"controller":{"candidate_mode":CANDIDATE_MODE,"tokens":256,"calibration_replicates_per_path_prompt":2,"decision_statistic":"median_service_makespan","candidate_replicates_per_prompt":3,"candidate_uses_only_prior_calibration":True,"remote_branch":"prepared_repost_sleep_1ms","decode_branch":"no_background_transfer","hook_events":0,"boost_enabled":False},"campaign":{"prompts":3,"blocks":21,"candidate_gates":{"exact_correctness":True,"all_rank_decisions_agree":True,"paired_prompt_median_lmcache_delta_le_ms":-5.0,"meaningful_prompt_wins":2,"median_prompt_oracle_regret_le_ms":10.0,"e2e_le_fg_ratio":1.05,"tpot_p99_le_lmcache_ratio":1.10}}}
def _load_contract(path:Path):
 p=json.loads(path.read_text());
 if p!=_expected_contract():raise ValueError("S26 contract changed")
 return p
def _run_block(torch,dist,*,mode:str,rank:int,prompt_index:int,**kwargs):
 if mode!=CANDIDATE_MODE:
  row=q.c9._run_block(torch,dist,mode=mode,rank=rank,prompt_index=prompt_index,**kwargs);service=q._global_service(torch,dist,row,rank);_SAMPLES.setdefault(prompt_index,{}).setdefault(mode,[]).append(service);row["effective_mode"]=mode;row["calibration_service_ns"]=service;return row
 samples=_SAMPLES.get(prompt_index,{})
 if len(samples.get(q.old.FG,[]))!=2 or len(samples.get(q.old.LMCACHE,[]))!=2:raise RuntimeError("S26 incomplete calibration")
 fg=int(statistics.median(samples[q.old.FG]));lm=int(statistics.median(samples[q.old.LMCACHE]));selected=q.old.FG if fg<=lm else CANDIDATE_MODE
 row=q.c9._run_block(torch,dist,mode=selected,rank=rank,prompt_index=prompt_index,**kwargs);row["mode"]=CANDIDATE_MODE;row["effective_mode"]="decode_no_transfer" if selected==q.old.FG else "remote_prepared";row["controller_decision"]=row["effective_mode"];row["calibration_fg_service_ns"]=fg;row["calibration_lmcache_service_ns"]=lm;row["decision_matches_calibration_argmin"]=True;return row
def _aggregate(records,trace,args):
 result=q._aggregate(records,trace,args);blocks=result["blocks"]
 by={m:[b for b in blocks if b["mode"]==m] for m in (q.old.FG,q.old.LMCACHE,CANDIDATE_MODE)}
 def metrics(mode):
  rows=by[mode];return {"replicates":len(rows),"ttft_p50_ms":statistics.median(r["ttft_ms"] for r in rows),"tpot_p50_ms":statistics.median(r["tpot_p50_ms"] for r in rows),"tpot_p99_max_ms":max(r["tpot_p99_ms"] for r in rows),"e2e_p50_ms":statistics.median(r["request_e2e_ms"] for r in rows),"service_makespan_p50_ms":statistics.median(r["service_makespan_ms"] for r in rows),"post_foreground_drain_max_ms":max(r["post_foreground_drain_ms"] for r in rows)}
 result["mode_metrics"]={m:metrics(m) for m in by};paired=[];decisions=[]
 for p in range(3):
  fg=statistics.median(b["service_makespan_ms"] for b in by[q.old.FG] if b["prompt_index"]==p);lm=statistics.median(b["service_makespan_ms"] for b in by[q.old.LMCACHE] if b["prompt_index"]==p);cand=statistics.median(b["service_makespan_ms"] for b in by[CANDIDATE_MODE] if b["prompt_index"]==p);ce=statistics.median(b["request_e2e_ms"] for b in by[CANDIDATE_MODE] if b["prompt_index"]==p);le=statistics.median(b["request_e2e_ms"] for b in by[q.old.LMCACHE] if b["prompt_index"]==p);fe=statistics.median(b["request_e2e_ms"] for b in by[q.old.FG] if b["prompt_index"]==p);ct=max(b["tpot_p99_ms"] for b in by[CANDIDATE_MODE] if b["prompt_index"]==p);lt=max(b["tpot_p99_ms"] for b in by[q.old.LMCACHE] if b["prompt_index"]==p);effective={b["effective_mode"] for b in by[CANDIDATE_MODE] if b["prompt_index"]==p};selected=effective.pop() if len(effective)==1 else "rank_or_repeat_divergence";paired.append({"prompt_index":p,"tempo_minus_lmcache_service_makespan_ms":cand-lm,"tempo_minus_lmcache_e2e_ms":ce-le,"tempo_minus_fg_e2e_ms":ce-fe,"tempo_minus_lmcache_tpot_p99_ms":ct-lt});decisions.append({"prompt_index":p,"selected":selected,"calibration_fg_median_ms":fg,"calibration_lmcache_median_ms":lm,"candidate_service_median_ms":cand,"oracle_regret_ms":cand-min(fg,lm),"repeat_decisions_agree":selected!="rank_or_repeat_divergence"})
 result["paired"]=paired;result["admission_decisions"]=decisions;deltas=[x["tempo_minus_lmcache_service_makespan_ms"] for x in paired];regrets=[x["oracle_regret_ms"] for x in decisions];metrics_out=result["mode_metrics"]
 gates={"correctness_output_trace":bool(result["overall_correctness_met"]),"all_repeat_decisions_agree":all(x["repeat_decisions_agree"] for x in decisions),"paired_prompt_median_lmcache_delta_le_minus_5ms":statistics.median(deltas)<=-5,"meaningful_prompt_wins_ge_2":sum(x<=-5 for x in deltas)>=2,"median_prompt_oracle_regret_le_10ms":statistics.median(regrets)<=10,"candidate_e2e_p50_le_1_05x_fg":metrics_out[CANDIDATE_MODE]["e2e_p50_ms"]<=1.05*metrics_out[q.old.FG]["e2e_p50_ms"],"candidate_tpot_p99_le_1_10x_lmcache":metrics_out[CANDIDATE_MODE]["tpot_p99_max_ms"]<=1.10*metrics_out[q.old.LMCACHE]["tpot_p99_max_ms"]}
 result["schema_version"]=RESULT_SCHEMA;result["contract_id"]=CONTRACT_ID;result["config"].update(candidate_mode=CANDIDATE_MODE,tokens=256,calibration_replicates_per_path_prompt=2,candidate_replicates_per_prompt=3);result["candidate_gates"]=gates;result["screen_outcome"]=("invalid_correctness_output_or_trace" if not result["overall_correctness_met"] else "robust_admission_pass" if all(gates.values()) else "robust_admission_revise");return result
def main():
 q.CANDIDATE_MODE=CANDIDATE_MODE;q.CONTRACT_ID=CONTRACT_ID;q.RESULT_SCHEMA=RESULT_SCHEMA;q.BLOCKS=BLOCKS;q.old.TOKENS=256;q._load_contract=_load_contract;q._run_block=_run_block;q._aggregate=_aggregate;q.main()
if __name__=="__main__":main()
