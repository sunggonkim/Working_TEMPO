#!/usr/bin/env python3
"""Analyze sparse cache routing with preserved remote prompt work."""

from __future__ import annotations
import json
from pathlib import Path
import sys
from eval.sota_4node import analyze_tempo_pd_same_server_cache_catalog_v137 as prior

def main() -> int:
    status=prior.main(); output=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    value=json.loads(output.read_text()); tempo,local,remote=(value[k] for k in ('tempo','fixed_local','lmcache_remote'))
    tp,lp,rp=(x['performance'] for x in (tempo,local,remote)); pair=value['paired_tempo_minus_lmcache']
    matched=value['route_matched_pairs']; reasons=tempo['reasons']
    lc=sum(v for k,v in reasons.items() if k.endswith('cache_catalog_hit_local'))
    rc=sum(v for k,v in reasons.items() if k.endswith('cache_catalog_hit_remote'))
    gates={
      'arm_isolated_stable_cache_catalog': value['gates']['arm_isolated_warm_reuse_contract'] and value['gates']['stable_cache_catalog_identity'],
      'exact_workload': value['gates']['exact_normalized_workload_schedule_outputs'],
      'fixed_baselines_exact': local['routes']=={prior.LOCAL_ROUTE:48} and remote['routes']=={prior.REMOTE_ROUTE:48},
      'tempo_routes_36_local_12_remote': tempo['routes']=={prior.LOCAL_ROUTE:36,prior.REMOTE_ROUTE:12},
      'exact_cache_hit_reasons': lc==36 and rc==12 and sum(reasons.values())==48,
      'all_slo_valid': tp['slo_goodput']['success_fraction']==1.0,
      'goodput_retains_95pct_local': tp['slo_goodput']['request_goodput_per_s']>=.95*lp['slo_goodput']['request_goodput_per_s'],
      'goodput_beats_lmcache': tp['slo_goodput']['request_goodput_per_s']>rp['slo_goodput']['request_goodput_per_s'],
      'throughput_beats_lmcache': tp['request_throughput_per_s']>rp['request_throughput_per_s'],
      'paired_beats_lmcache': pair['e2e_win_count']>=25 and pair['e2e_delta_median_ms']<0,
      'selected_local_median_noninferior': matched['local']['count']==36 and matched['local']['e2e_delta_median_ms']<=10,
      'remote_sacrifice_bounded': matched['remote']['count']==12 and matched['remote']['e2e_delta_median_ms']<=125,
      'e2e_p99_within_5pct_local': tp['e2e_ms']['p99']<=1.05*lp['e2e_ms']['p99'],
      'e2e_p99_beats_lmcache': tp['e2e_ms']['p99']<rp['e2e_ms']['p99'],
      'tpot_p99_beats_lmcache': tp['tpot_ms']['p99']<rp['tpot_ms']['p99'],
    }
    value['schema']='tempo-pd-sparse-byte-balanced-cache-catalog-analysis-147'; value['gates']=gates
    value['passes']=all(gates.values()); value['verdict']='promising_sparse_byte_balanced_catalog' if value['passes'] else 'revise_sparse_byte_balanced_catalog'
    output.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'verdict':value['verdict'],'failed':[k for k,v in gates.items() if not v]},sort_keys=True))
    return status

if __name__=='__main__': raise SystemExit(main())
