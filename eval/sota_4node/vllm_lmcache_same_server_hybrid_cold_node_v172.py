#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import subprocess,sys
from eval.sota_4node import vllm_lmcache_same_server_balanced_node_v72 as balanced
_ORIGINAL_ROUTER=balanced._router_command
_ORIGINAL_CLIENT=balanced._client_command

def _router_command(*args,**kwargs):
 c=_ORIGINAL_ROUTER(*args,**kwargs);old='eval.sota_4node.tempo_pd_same_server_balanced_router_v70';c[c.index(old)]='eval.sota_4node.tempo_pd_same_server_hybrid_cold_router_v170';return c
def _client_command(*args,**kwargs):
 c=_ORIGINAL_CLIENT(*args,**kwargs);old='eval.sota_4node.run_tempo_pd_same_server_balanced_client_v70';c[c.index(old)]='eval.sota_4node.run_tempo_pd_same_server_cold_prewarm_client_v171';return c
def _arg(name):return Path(sys.argv[sys.argv.index(name)+1]).resolve()
def main():
 original_router=balanced._router_command;original_client=balanced._client_command;balanced._router_command=_router_command;balanced._client_command=_client_command
 try:return balanced.main()
 finally:balanced._router_command=original_router;balanced._client_command=original_client
if __name__=='__main__':raise SystemExit(main())
