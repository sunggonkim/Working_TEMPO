#!/usr/bin/env python3
"""Launch entry correcting W30's intended confirmed-transition gate."""
from eval.sota_4node import run_vllm_lmcache_tp16_sandwich_admission_w30_entry as w
_W30_AGGREGATE=w._aggregate
def _aggregate(records,trace,args):
 result=_W30_AGGREGATE(records,trace,args);valid=all(seq in (["decode_no_transfer"]*3,["decode_no_transfer","decode_no_transfer","remote_prepared"],["decode_no_transfer","remote_prepared","decode_no_transfer"]) for seq in result["path_sequences"].values());result["candidate_gates"]["all_path_transitions_valid"]=valid;result["screen_outcome"]=("invalid_correctness_output_or_trace" if not result["overall_correctness_met"] else "sandwich_admission_pass" if all(result["candidate_gates"].values()) else "sandwich_admission_revise");return result
def main():w._aggregate=_aggregate;w.main()
if __name__=="__main__":main()
