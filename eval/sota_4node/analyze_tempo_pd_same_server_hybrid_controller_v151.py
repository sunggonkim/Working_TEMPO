#!/usr/bin/env python3
"""Final workload-level analysis for the production hybrid controller."""

from __future__ import annotations
import json
from pathlib import Path
import sys
from eval.sota_4node import analyze_tempo_pd_same_server_warm_reuse_v132 as warm

LOCAL="decoder_local_recompute_or_cache"; REMOTE="remote_prefill_live_kv"

def main()->int:
 status=warm.main(); output=Path(sys.argv[sys.argv.index('--output')+1]).resolve(); value=json.loads(output.read_text())
 tempo,local,lm=(value[k] for k in ('tempo','fixed_local','lmcache_remote')); tp,lp,rp=(x['performance'] for x in (tempo,local,lm)); pair=value['paired_tempo_minus_lmcache']
 reasons=tempo['reasons']; gates={
  'arm_isolated_warm_reuse_contract':value['gates']['arm_isolated_warm_reuse_contract'],
  'stable_cache_catalog_identity':all(x.get('cache_catalog_identity')=='stable-item-index-v136' for x in value['contracts_by_sequence']),
  'exact_workload':value['gates']['exact_normalized_workload_schedule_outputs'],
  'fixed_baselines_exact':local['routes']=={LOCAL:48} and lm['routes']=={REMOTE:48},
  'production_hybrid_routes_32_local_16_remote':tempo['routes']=={LOCAL:32,REMOTE:16},
  'production_hybrid_reason_exact':reasons=={'same_server_tempo_measured:cache_affinity_warm_hit':48},
  'all_slo_valid':tp['slo_goodput']['success_fraction']==1.0,
  'throughput_beats_lmcache':tp['request_throughput_per_s']>rp['request_throughput_per_s'],
  'throughput_beats_local':tp['request_throughput_per_s']>lp['request_throughput_per_s'],
  'e2e_p99_beats_lmcache':tp['e2e_ms']['p99']<rp['e2e_ms']['p99'],
  'e2e_p99_beats_local':tp['e2e_ms']['p99']<lp['e2e_ms']['p99'],
  'tpot_p99_beats_lmcache':tp['tpot_ms']['p99']<rp['tpot_ms']['p99'],
  'paired_beats_lmcache':pair['e2e_win_count']>=25 and pair['e2e_delta_median_ms']<0,
 }
 value['schema']='tempo-pd-production-hybrid-controller-analysis-151'; value['gates']=gates; value['passes']=all(gates.values()); value['verdict']='production_hybrid_controller_validated' if value['passes'] else 'production_hybrid_controller_revise'
 value['claim_boundary']='One actual Qwen2.5-7B TP4+TP4 P/D lifecycle, pinned official LMCache, stable warm keys, production HybridPDController adapter.'
 output.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n'); print(json.dumps({'verdict':value['verdict'],'failed':[k for k,v in gates.items() if not v]},sort_keys=True)); return status
if __name__=='__main__':raise SystemExit(main())
