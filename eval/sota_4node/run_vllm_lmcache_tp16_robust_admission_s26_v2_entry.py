#!/usr/bin/env python3
"""Recursion-safe launch entry for S26 aggregation."""
from eval.sota_4node import run_vllm_lmcache_tp16_calibrated_admission_q24_entry as q
from eval.sota_4node import run_vllm_lmcache_tp16_robust_admission_s26_entry as s
_ORIGINAL_Q24_AGGREGATE=q._aggregate
_S26_AGGREGATE=s._aggregate
def _aggregate(records,trace,args):
 installed=q._aggregate;q._aggregate=_ORIGINAL_Q24_AGGREGATE
 try:return _S26_AGGREGATE(records,trace,args)
 finally:q._aggregate=installed
def main():
 s._aggregate=_aggregate;s.main()
if __name__=="__main__":main()
