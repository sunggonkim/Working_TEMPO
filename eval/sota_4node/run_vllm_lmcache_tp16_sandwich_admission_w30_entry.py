#!/usr/bin/env python3
"""W30: V29 controller with before/after sandwich baselines around candidate."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from eval.sota_4node import run_vllm_lmcache_tp16_calibrated_admission_q24_entry as q
from eval.sota_4node import run_vllm_lmcache_tp16_robust_admission_s26_entry as s
from eval.sota_4node import run_vllm_lmcache_tp16_rolling_admission_t27_entry as t
from eval.sota_4node import run_vllm_lmcache_tp16_confirmed_admission_v29_entry as v
CANDIDATE_MODE="tempo_sandwich_confirmed_cache_or_decode";CONTRACT_ID="tp16-sandwich-confirmed-admission-w30";RESULT_SCHEMA="tempo-vllm-tp16-sandwich-admission-result-30"
BLOCKS=tuple(item for p in range(3) for _ in range(3) for item in ((p,q.old.FG),(p,q.old.LMCACHE),(p,CANDIDATE_MODE),(p,q.old.LMCACHE),(p,q.old.FG)))
def _expected_contract()->dict[str,Any]:
 return {"schema_version":"tempo-tp16-sandwich-admission-contract-30","contract_id":CONTRACT_ID,"topology":{"nodes":4,"world_size":16,"source_ranks":list(range(8)),"receiver_ranks":list(range(8,16)),"pairing":[[r,r+8] for r in range(8)]},"transfer":{"bytes_per_source":16<<20,"global_bytes":128<<20,"calls_global_if_remote":8,"physical_descriptors_global_if_remote":8,"prepared_handle_repost":True},"controller":{"candidate_mode":CANDIDATE_MODE,"tokens":256,"cycles_per_prompt":3,"cycle_order":[q.old.FG,q.old.LMCACHE,CANDIDATE_MODE,q.old.LMCACHE,q.old.FG],"candidate_position":"sandwiched_between_before_and_after_baselines","decision_uses_only_prior_rows":True,"decision_statistic":"rolling_median_service_makespan","initial_path":"decode_no_transfer","remote_advantage_margin_ms":50.0,"remote_confirmation_cycles":2,"fallback_to_decode_when_fg_not_slower":True,"remote_branch":"prepared_repost_sleep_1ms","decode_branch":"no_background_transfer","hook_events":0,"boost_enabled":False},"campaign":{"prompts":3,"blocks":45,"baseline_replicates_per_mode_prompt":6,"candidate_replicates_per_prompt":3,"candidate_gates":{"exact_correctness":True,"path_transition_valid":True,"paired_prompt_median_lmcache_delta_le_ms":-5.0,"meaningful_prompt_wins":2,"median_prompt_oracle_regret_le_ms":10.0,"e2e_le_fg_ratio":1.05,"tpot_p99_le_lmcache_ratio":1.10}}}
def _load_contract(path:Path):
 p=json.loads(path.read_text());
 if p!=_expected_contract():raise ValueError("W30 contract changed")
 return p
def _aggregate(records,trace,args):
 result=v._aggregate(records,trace,args);result["schema_version"]=RESULT_SCHEMA;result["contract_id"]=CONTRACT_ID;result["config"].update(candidate_mode=CANDIDATE_MODE,comparison_design="before_after_sandwich",baseline_replicates_per_mode_prompt=6,candidate_replicates_per_prompt=3);result["screen_outcome"]=("invalid_correctness_output_or_trace" if not result["overall_correctness_met"] else "sandwich_admission_pass" if all(result["candidate_gates"].values()) else "sandwich_admission_revise");return result
def main():
 for module in (q,s,t,v):module.CANDIDATE_MODE=CANDIDATE_MODE;module.CONTRACT_ID=CONTRACT_ID;module.RESULT_SCHEMA=RESULT_SCHEMA;module.BLOCKS=BLOCKS
 q.old.TOKENS=256;q._load_contract=_load_contract;q._run_block=v._run_block;q._aggregate=_aggregate;q.main()
if __name__=="__main__":main()
