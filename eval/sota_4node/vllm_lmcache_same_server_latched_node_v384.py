#!/usr/bin/env python3
"""Generic node binding for the latched controller's three workloads."""
import os
from eval.sota_4node import vllm_lmcache_same_server_online_regime_mixed_node_v293 as base


def install(client_module, workload_class, raw_parts):
    base_client, base_router = base._client_command, base._router_command
    def client(*a, **k):
        c = base_client(*a, **k); c[c.index("eval.sota_4node.run_tempo_pd_same_server_mixed_only_client_v265")] = client_module; return c
    def router(*a, **k):
        c = base_router(*a, **k); c[c.index("eval.sota_4node.tempo_pd_same_server_online_regime_router_v291")] = "eval.sota_4node.tempo_pd_same_server_latched_microburst25_v382"; return c
    def run(c, *a, **k):
        if isinstance(c, list) and "eval.sota_4node.analyze_tempo_pd_same_server_hybrid_controller_v160" in c:
            o = c[c.index("--output") + 1]; raw = os.path.join(os.path.dirname(o), "tempo_credit_admission", *raw_parts)
            return base._REAL_RUN([c[0], "-m", "eval.sota_4node.analyze_tempo_pd_latched_controller_v383", "--raw", raw, "--allocation", os.environ["SLURM_JOB_ID"], "--workload-class", workload_class, "--output", o], *a, **k)
        return base._REAL_RUN(c, *a, **k)
    old = base._client_command, base._router_command, base._bounded_run
    base._client_command, base._router_command, base._bounded_run = client, router, run
    return old


def main(client_module, workload_class, raw_parts):
    old = install(client_module, workload_class, raw_parts)
    try: return base.main()
    finally: base._client_command, base._router_command, base._bounded_run = old
