#!/usr/bin/env python3
"""P23: O22 with only predecode head start changed 10ms -> 2ms."""
from pathlib import Path
from typing import Any
import json, statistics
from eval.sota_4node import run_vllm_lmcache_tp16_headstart25_n21_entry as n21
from eval.sota_4node import run_vllm_lmcache_tp16_headstart25_n21_v2_entry as n21v2
CANDIDATE_MODE="tempo_predecode_post_ordered_2ms"; CONTRACT_ID="tp16-predecode-post-ordered2-p23"; RESULT_SCHEMA="tempo-vllm-tp16-post-ordered-result-23"; HEADSTART_MS=2.0
BLOCKS=((0,n21.old.FG),(0,n21.old.LMCACHE),(0,CANDIDATE_MODE),(1,CANDIDATE_MODE),(1,n21.old.FG),(1,n21.old.LMCACHE),(2,n21.old.LMCACHE),(2,CANDIDATE_MODE),(2,n21.old.FG))
def _expected_contract()->dict[str,Any]:
 return {"schema_version":"tempo-tp16-post-ordered-contract-23","contract_id":CONTRACT_ID,"topology":{"nodes":4,"world_size":16,"source_ranks":list(range(8)),"receiver_ranks":list(range(8,16)),"pairing":[[r,r+8] for r in range(8)]},"transfer":{"bytes_per_source":16<<20,"global_bytes":128<<20,"calls_global":8,"physical_descriptors_global":8,"prepared_handle_repost":True},"algorithm":{"candidate_mode":CANDIDATE_MODE,"predecode_headstart_ms":2.0,"headstart_included_in_admission_latency":True,"completion_wait_before_decode":False,"decode_progress_sleep_ms":1.0,"hook_events_per_measured_candidate":0,"boost_enabled":False,"single_factor_from":"O22 headstart 10ms -> 2ms post-ordering guard"},"campaign":{"modes":[n21.old.FG,n21.old.LMCACHE,CANDIDATE_MODE],"blocks":9,"replicates_per_mode":3,"candidate_gates":{"exact_correctness":True,"headstart_min_ms":2.0,"post_foreground_drain_zero":True,"paired_service_delta_median_le_ms":-5.0,"meaningful_service_wins":2,"request_e2e_le_fg_ratio":1.03,"tpot_p99_le_lmcache_ratio":1.10}}}
def _load_contract(path:Path):
 payload=json.loads(path.read_text());
 if payload!=_expected_contract():raise ValueError("P23 contract changed")
 return payload
def _aggregate(records,trace,args):
 result=n21v2._aggregate(records,trace,args); candidates=[b for b in result["blocks"] if b["mode"]==CANDIDATE_MODE]; deltas=[float(p["tempo_minus_lmcache_service_makespan_ms"]) for p in result["paired"]]; ce=result["mode_metrics"][CANDIDATE_MODE]["e2e_p50_ms"]; fe=result["mode_metrics"][n21.old.FG]["e2e_p50_ms"]; ct=result["mode_metrics"][CANDIDATE_MODE]["tpot_p99_max_ms"]; lt=result["mode_metrics"][n21.old.LMCACHE]["tpot_p99_max_ms"]
 gates={"correctness_output_trace":bool(result["overall_correctness_met"]),"headstart_at_least_2ms":all(b["headstart_elapsed_ms"]>=2 for b in candidates),"all_candidate_post_foreground_drain_zero":all(b["post_foreground_drain_ms"]==0 for b in candidates),"paired_service_median_le_minus_5ms":statistics.median(deltas)<=-5,"paired_service_meaningful_wins_ge_2":sum(d<=-5 for d in deltas)>=2,"candidate_request_e2e_p50_le_1_03x_fg":ce<=1.03*fe,"candidate_tpot_p99_le_1_10x_lmcache":ct<=1.10*lt}
 result["schema_version"]=RESULT_SCHEMA;result["contract_id"]=CONTRACT_ID;result["config"].update(candidate_mode=CANDIDATE_MODE,predecode_headstart_ms=2.0);result["candidate_gates"]=gates;result["screen_outcome"]=("invalid_correctness_output_or_trace" if not result["overall_correctness_met"] else "post_ordered2_candidate_pass" if all(gates.values()) else "post_ordered2_candidate_revise");return result
def main():
 n21.CANDIDATE_MODE=CANDIDATE_MODE;n21.CONTRACT_ID=CONTRACT_ID;n21.RESULT_SCHEMA=RESULT_SCHEMA;n21.HEADSTART_MS=2.0;n21.BLOCKS=BLOCKS;n21._load_contract=_load_contract;n21._aggregate=_aggregate;n21.main()
if __name__=="__main__":main()
