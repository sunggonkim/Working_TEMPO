#!/usr/bin/env python3
"""Latin-balanced wrapper around the two-arm crossover scout."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_balanced_workload_v11 as balanced
from eval.sota_4node import vllm_lmcache_remote_crossover_node_v9 as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v2 as context_safe


def main() -> int:
    context_safe._prepare_workloads = balanced.prepare
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
