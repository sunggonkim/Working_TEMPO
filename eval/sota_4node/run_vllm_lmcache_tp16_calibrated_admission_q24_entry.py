#!/usr/bin/env python3
"""Q24: causal per-prompt calibration chooses remote KV or no-transfer decode."""
from __future__ import annotations
import copy, json, statistics
from pathlib import Path
from typing import Any
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_c9_entry as c9
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as v8
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol
CANDIDATE_MODE="tempo_calibrated_cache_or_decode";CONTRACT_ID="tp16-calibrated-cache-admission-q24";RESULT_SCHEMA="tempo-vllm-tp16-calibrated-admission-result-24"
BLOCKS=((0,old.FG),(0,old.LMCACHE),(0,CANDIDATE_MODE),(1,old.FG),(1,old.LMCACHE),(1,CANDIDATE_MODE),(2,old.FG),(2,old.LMCACHE),(2,CANDIDATE_MODE))
_CALIBRATION:dict[int,dict[str,int]]={}
def _expected_contract()->dict[str,Any]:
 return {"schema_version":"tempo-tp16-calibrated-admission-contract-24","contract_id":CONTRACT_ID,"topology":{"nodes":4,"world_size":16,"source_ranks":list(range(8)),"receiver_ranks":list(range(8,16)),"pairing":[[r,r+8] for r in range(8)]},"transfer":{"bytes_per_source":16<<20,"global_bytes":128<<20,"calls_global_if_remote":8,"physical_descriptors_global_if_remote":8,"prepared_handle_repost":True},"controller":{"candidate_mode":CANDIDATE_MODE,"calibration_order_per_prompt":[old.FG,old.LMCACHE,CANDIDATE_MODE],"decision":"choose_lower_observed_service_makespan","decision_causal":True,"candidate_uses_only_prior_blocks":True,"remote_branch":"prepared_repost_sleep_1ms","decode_branch":"no_background_transfer","hook_events":0,"boost_enabled":False},"campaign":{"prompts":3,"blocks":9,"candidate_gates":{"exact_correctness":True,"decision_matches_calibration_argmin":True,"paired_lmcache_delta_median_le_ms":-5.0,"meaningful_lmcache_wins":2,"median_oracle_regret_le_ms":10.0,"e2e_le_fg_ratio":1.05,"tpot_p99_le_lmcache_ratio":1.10}}}
def _load_contract(path:Path):
 payload=json.loads(path.read_text());
 if payload!=_expected_contract():raise ValueError("Q24 contract changed")
 return payload
def _global_service(torch,dist,row,rank):
 client=torch.tensor([int(float(row["client"]["request_e2e_ms"])*1e6) if rank==0 else 0],dtype=torch.int64);dist.broadcast(client,src=0)
 completion=torch.tensor([int(row["source_call"]["completion_from_origin_ns"])],dtype=torch.int64);dist.all_reduce(completion,op=dist.ReduceOp.MAX)
 return max(int(client.item()),int(completion.item()))
def _run_block(torch,dist,*,mode:str,rank:int,prompt_index:int,**kwargs):
 if mode!=CANDIDATE_MODE:
  row=c9._run_block(torch,dist,mode=mode,rank=rank,prompt_index=prompt_index,**kwargs)
  _CALIBRATION.setdefault(prompt_index,{})[mode]=_global_service(torch,dist,row,rank)
  row["effective_mode"]=mode;row["calibration_service_ns"]=_CALIBRATION[prompt_index][mode];return row
 values=_CALIBRATION.get(prompt_index,{})
 if set(values)!={old.FG,old.LMCACHE}:raise RuntimeError("Q24 candidate lacks causal calibration")
 selected=old.FG if values[old.FG]<=values[old.LMCACHE] else CANDIDATE_MODE
 row=c9._run_block(torch,dist,mode=selected,rank=rank,prompt_index=prompt_index,**kwargs)
 row["mode"]=CANDIDATE_MODE;row["effective_mode"]="decode_no_transfer" if selected==old.FG else "remote_prepared"
 row["controller_decision"]="decode_no_transfer" if selected==old.FG else "remote_prepared"
 row["calibration_fg_service_ns"]=values[old.FG];row["calibration_lmcache_service_ns"]=values[old.LMCACHE]
 row["decision_matches_calibration_argmin"]=True
 return row
def _aggregate(records,trace,args):
 result=c9._aggregate(records,trace,args);ordered=sorted(records,key=lambda r:int(r["rank"]));blocks=result["blocks"]
 decisions=[]
 for block in blocks:
  idx=int(block["block_index"]);raw=[r["blocks"][idx] for r in ordered]
  if block["mode"]!=CANDIDATE_MODE:continue
  effective={r["effective_mode"] for r in raw};match=all(bool(r["decision_matches_calibration_argmin"]) for r in raw)
  if len(effective)!=1:raise ValueError("Q24 rank-divergent decision")
  effective_mode=effective.pop();remote=effective_mode=="remote_prepared"
  completed=sum(int(r["source_call"]["bytes"]) for r in raw[:8]);verified=sum(int(r["receiver_verified_bytes"]) for r in raw[8:]);calls=sum(int(r["source_call"]["calls"]) for r in raw[:8]);descs=sum(int(r["source_call"]["descriptors"]) for r in raw[:8]);expected=128<<20 if remote else 0
  block["correctness_met"]=all(bool(r["correctness_met"]) for r in raw) and completed==verified==expected and calls==descs==(8 if remote else 0)
  block["effective_mode"]=effective_mode;block["background_completed_bytes"]=completed;block["receiver_verified_bytes"]=verified;block["source_calls"]=calls;block["physical_descriptors"]=descs
  fg=float(raw[0]["calibration_fg_service_ns"])/1e6;lm=float(raw[0]["calibration_lmcache_service_ns"])/1e6
  block["calibration_fg_service_ms"]=fg;block["calibration_lmcache_service_ms"]=lm;block["decision_matches_calibration_argmin"]=match
  decisions.append({"prompt_index":block["prompt_index"],"selected":effective_mode,"calibration_fg_ms":fg,"calibration_lmcache_ms":lm,"calibration_margin_ms":lm-fg,"actual_candidate_service_ms":block["service_makespan_ms"],"oracle_regret_ms":block["service_makespan_ms"]-min(fg,lm),"matches_argmin":match})
 output_equal=all(len({b["output_token_sha256"] for b in blocks if b["prompt_index"]==p})==1 for p in range(3));overall=trace.get("validated") is True and output_equal and all(bool(b["correctness_met"]) for b in blocks);result["overall_correctness_met"]=overall;result["output_equivalence_met"]=output_equal
 by={m:[b for b in blocks if b["mode"]==m] for m in (old.FG,old.LMCACHE,CANDIDATE_MODE)}
 metrics={}
 for mode,rows in by.items():metrics[mode]={"replicates":3,"ttft_p50_ms":statistics.median(r["ttft_ms"] for r in rows),"tpot_p50_ms":statistics.median(r["tpot_p50_ms"] for r in rows),"tpot_p99_max_ms":max(r["tpot_p99_ms"] for r in rows),"e2e_p50_ms":statistics.median(r["request_e2e_ms"] for r in rows),"service_makespan_p50_ms":statistics.median(r["service_makespan_ms"] for r in rows),"post_foreground_drain_max_ms":max(r["post_foreground_drain_ms"] for r in rows)}
 result["mode_metrics"]=metrics;paired=[]
 for p in range(3):
  fg=next(b for b in by[old.FG] if b["prompt_index"]==p);lm=next(b for b in by[old.LMCACHE] if b["prompt_index"]==p);cand=next(b for b in by[CANDIDATE_MODE] if b["prompt_index"]==p);paired.append({"prompt_index":p,"tempo_minus_lmcache_service_makespan_ms":cand["service_makespan_ms"]-lm["service_makespan_ms"],"tempo_minus_lmcache_e2e_ms":cand["request_e2e_ms"]-lm["request_e2e_ms"],"tempo_minus_fg_e2e_ms":cand["request_e2e_ms"]-fg["request_e2e_ms"],"tempo_minus_lmcache_tpot_p99_ms":cand["tpot_p99_ms"]-lm["tpot_p99_ms"]})
 result["paired"]=paired;deltas=[p["tempo_minus_lmcache_service_makespan_ms"] for p in paired];regrets=[d["oracle_regret_ms"] for d in decisions]
 gates={"correctness_output_trace":overall,"all_decisions_match_calibration_argmin":all(d["matches_argmin"] for d in decisions),"paired_lmcache_delta_median_le_minus_5ms":statistics.median(deltas)<=-5,"meaningful_lmcache_wins_ge_2":sum(d<=-5 for d in deltas)>=2,"median_oracle_regret_le_10ms":statistics.median(regrets)<=10,"candidate_e2e_p50_le_1_05x_fg":metrics[CANDIDATE_MODE]["e2e_p50_ms"]<=1.05*metrics[old.FG]["e2e_p50_ms"],"candidate_tpot_p99_le_1_10x_lmcache":metrics[CANDIDATE_MODE]["tpot_p99_max_ms"]<=1.10*metrics[old.LMCACHE]["tpot_p99_max_ms"]}
 result["schema_version"]=RESULT_SCHEMA;result["contract_id"]=CONTRACT_ID;result["config"].update(candidate_mode=CANDIDATE_MODE,controller="causal_calibration_argmin");result["admission_decisions"]=decisions;result["candidate_gates"]=gates;result["screen_outcome"]=("invalid_correctness_output_or_trace" if not overall else "calibrated_admission_pass" if all(gates.values()) else "calibrated_admission_revise");return result
def main():
 old._transfer_worker=fixed._transfer_worker
 for module in (c9,):module.CANDIDATE_MODE=CANDIDATE_MODE;module.CONTRACT_ID=CONTRACT_ID;module.RESULT_SCHEMA=RESULT_SCHEMA;module.BLOCKS=BLOCKS
 c9._install_candidate_mode();protocol.install_async_release_protocol();old.protocol.ReleaseFrame=protocol.ReleaseFrame;old.bulk.protocol.ReleaseFrame=protocol.ReleaseFrame;v8.CONTRACT_ID=CONTRACT_ID;v8.RESULT_SCHEMA=RESULT_SCHEMA;v8._load_contract=_load_contract;v8._run_block=_run_block;v8._validate_trace=c9._validate_trace;v8._aggregate=_aggregate;v8.main()
if __name__=="__main__":main()
