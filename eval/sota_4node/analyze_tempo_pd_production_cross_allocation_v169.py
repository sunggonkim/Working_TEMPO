#!/usr/bin/env python3
"""Cross-allocation production-controller reproduction analysis."""

from __future__ import annotations
import argparse,json,statistics
from pathlib import Path

LOCAL="decoder_local_recompute_or_cache";REMOTE="remote_prefill_live_kv"

def _load(path:Path)->dict:
 v=json.loads(path.read_text())
 if v.get('schema')!='tempo-pd-production-hybrid-controller-analysis-151':raise ValueError('production schema mismatch')
 if v['tempo']['routes']!={LOCAL:32,REMOTE:16}:raise ValueError('production routes changed')
 if v['tempo']['reasons']!={'same_server_tempo_measured:cache_affinity_warm_hit':48}:raise ValueError('production reason changed')
 return v
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='append',type=Path,required=True);p.add_argument('--allocation',action='append',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if len(a.run)!=2 or len(a.allocation)!=2 or len(set(a.allocation))!=2:raise ValueError('exactly two distinct allocations and reports required')
 rows=[]
 for path,allocation in zip(a.run,a.allocation,strict=True):
  v=_load(path.resolve());t,l,m=(v[x]['performance'] for x in ('tempo','fixed_local','lmcache_remote'));pair=v['paired_tempo_minus_lmcache']
  rows.append({'allocation_id':allocation,'path':str(path.resolve()),'tempo_throughput_per_s':t['request_throughput_per_s'],'local_throughput_per_s':l['request_throughput_per_s'],'lmcache_throughput_per_s':m['request_throughput_per_s'],'throughput_gain_vs_lmcache_percent':100*(t['request_throughput_per_s']/m['request_throughput_per_s']-1),'tempo_e2e_p99_ms':t['e2e_ms']['p99'],'local_e2e_p99_ms':l['e2e_ms']['p99'],'lmcache_e2e_p99_ms':m['e2e_ms']['p99'],'e2e_p99_reduction_vs_lmcache_percent':100*(1-t['e2e_ms']['p99']/m['e2e_ms']['p99']),'local_p99_regression_percent':100*(t['e2e_ms']['p99']/l['e2e_ms']['p99']-1),'tempo_tpot_p99_ms':t['tpot_ms']['p99'],'lmcache_tpot_p99_ms':m['tpot_ms']['p99'],'tpot_p99_reduction_vs_lmcache_percent':100*(1-t['tpot_ms']['p99']/m['tpot_ms']['p99']),'paired_win_count':pair['e2e_win_count'],'paired_e2e_delta_median_ms':pair['e2e_delta_median_ms']})
 gates={'both_allocations_throughput_beat_lmcache':all(x['tempo_throughput_per_s']>x['lmcache_throughput_per_s'] for x in rows),'both_allocations_throughput_beat_local':all(x['tempo_throughput_per_s']>x['local_throughput_per_s'] for x in rows),'both_allocations_e2e_p99_beat_lmcache':all(x['tempo_e2e_p99_ms']<x['lmcache_e2e_p99_ms'] for x in rows),'both_allocations_e2e_p99_within_0_1pct_local':all(x['local_p99_regression_percent']<=.1 for x in rows),'both_allocations_tpot_p99_beat_lmcache':all(x['tempo_tpot_p99_ms']<x['lmcache_tpot_p99_ms'] for x in rows),'both_allocations_at_least_25_paired_wins':all(x['paired_win_count']>=25 for x in rows),'both_allocations_paired_median_beat_lmcache':all(x['paired_e2e_delta_median_ms']<0 for x in rows)}
 out={'schema':'tempo-pd-production-cross-allocation-reproduction-169','runs':rows,'median_throughput_gain_vs_lmcache_percent':statistics.median(x['throughput_gain_vs_lmcache_percent'] for x in rows),'median_e2e_p99_reduction_vs_lmcache_percent':statistics.median(x['e2e_p99_reduction_vs_lmcache_percent'] for x in rows),'median_tpot_p99_reduction_vs_lmcache_percent':statistics.median(x['tpot_p99_reduction_vs_lmcache_percent'] for x in rows),'gates':gates,'passes':all(gates.values()),'claim_boundary':'Two distinct four-node A100 allocations; actual Qwen2.5-7B TP4+TP4 vLLM P/D; pinned official LMCache; rate48; arm-isolated warm keys; production HybridPDController.'}
 out['verdict']='cross_allocation_production_win' if out['passes'] else 'production_win_not_cross_allocation_reproduced';a.output.resolve().write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps({'verdict':out['verdict'],'failed':[k for k,v in gates.items() if not v]},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
