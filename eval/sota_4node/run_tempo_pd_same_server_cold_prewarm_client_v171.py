#!/usr/bin/env python3
"""Serially initialize remote transport, then run cold-key balanced blocks."""

from __future__ import annotations
import json,subprocess,sys
from eval.sota_4node import run_tempo_pd_same_server_balanced_client_v70 as balanced

def _prewarm(args):
 rows=balanced._load_rows(args.workload);row=dict(rows[0]);row['request_id']='ssb-remote-r0-warm-transport-prewarm';workload=args.output.parent/'cold_remote_transport_prewarm.jsonl';output=args.output.parent/'cold_remote_transport_prewarm.raw.json'
 if workload.exists() or output.exists():raise ValueError('stale cold transport prewarm')
 balanced._write_jsonl(workload,[row]);command=[sys.executable,'-m','eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_v38','--base-url',args.base_url,'--model',str(args.model),'--served-model-name',args.served_model_name,'--workload',str(workload),'--output',str(output),'--mode','lmcache_always_remote','--run-id','cold-remote-transport-prewarm','--default-max-tokens',str(args.default_max_tokens),'--max-workers','1','--timeout-s','120','--seed',str(args.seed)];subprocess.run(command,check=True,timeout=180.);value=json.loads(output.read_text());
 if len(value.get('requests',[]))!=1:raise ValueError('cold remote transport prewarm failed')
def main():
 args=balanced._parse()
 if args.run_id.endswith('-warmup'):_prewarm(args)
 return balanced.main()
if __name__=='__main__':raise SystemExit(main())
