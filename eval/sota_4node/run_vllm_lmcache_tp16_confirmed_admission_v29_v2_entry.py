#!/usr/bin/env python3
"""V29 launch entry accepting the intended decode,decode,remote transition."""
from eval.sota_4node import run_vllm_lmcache_tp16_confirmed_admission_v29_entry as v
_V29_AGGREGATE=v._aggregate
def _aggregate(records,trace,args):
 result=_V29_AGGREGATE(records,trace,args);valid=all(seq in (["decode_no_transfer"]*3,["decode_no_transfer","decode_no_transfer","remote_prepared"],["decode_no_transfer","remote_prepared","decode_no_transfer"]) for seq in result["path_sequences"].values());result["candidate_gates"]["all_path_transitions_valid"]=valid;result["screen_outcome"]=("invalid_correctness_output_or_trace" if not result["overall_correctness_met"] else "confirmed_admission_pass" if all(result["candidate_gates"].values()) else "confirmed_admission_revise");return result
def main():v._aggregate=_aggregate;v.main()
if __name__=="__main__":main()
