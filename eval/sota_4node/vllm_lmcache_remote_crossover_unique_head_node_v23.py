#!/usr/bin/env python3
"""Head-nonce cold-cache wrapper around the two-arm P/D crossover."""

from eval.sota_4node import tempo_pd_unique_head_short_workload_v23 as unique
from eval.sota_4node import vllm_lmcache_remote_crossover_node_v9 as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v2 as context_safe


def main() -> int:
    context_safe._prepare_workloads = unique.prepare
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
