#!/usr/bin/env python3
"""Adaptive controller on collision-free steady traffic."""
import os
from eval.sota_4node import vllm_lmcache_same_server_online_regime_mixed_node_v293 as base
_BC, _BR = base._client_command, base._router_command
def _client_command(*a, **k):
    c = _BC(*a, **k); c[c.index("eval.sota_4node.run_tempo_pd_same_server_mixed_only_client_v265")] = "eval.sota_4node.run_tempo_pd_steady_prefixswap_v374"; return c
def _router_command(*a, **k):
    c = _BR(*a, **k); c[c.index("eval.sota_4node.tempo_pd_same_server_online_regime_router_v291")] = "eval.sota_4node.tempo_pd_same_server_adaptive_microburst25_v367"; return c
def _bounded_run(c, *a, **k):
    if isinstance(c, list) and "eval.sota_4node.analyze_tempo_pd_same_server_hybrid_controller_v160" in c:
        o = c[c.index("--output") + 1]; raw = os.path.join(os.path.dirname(o), "tempo_credit_admission", "mixed_request_crossover_unique_chunks_v305", "measured.raw.json")
        return base._REAL_RUN([c[0], "-m", "eval.sota_4node.analyze_tempo_pd_adaptive_stationary_v375", "--raw", raw, "--allocation", os.environ["SLURM_JOB_ID"], "--workload-class", "steady", "--output", o], *a, **k)
    return base._REAL_RUN(c, *a, **k)
def main():
    old = base._client_command, base._router_command, base._bounded_run; base._client_command, base._router_command, base._bounded_run = _client_command, _router_command, _bounded_run
    try: return base.main()
    finally: base._client_command, base._router_command, base._bounded_run = old
if __name__ == "__main__": raise SystemExit(main())
