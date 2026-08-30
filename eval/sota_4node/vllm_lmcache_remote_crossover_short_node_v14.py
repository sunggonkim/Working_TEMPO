#!/usr/bin/env python3
"""Equal-short-context wrapper around the two-arm crossover scout."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_short_workload_v14 as short
from eval.sota_4node import vllm_lmcache_remote_crossover_node_v9 as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v2 as context_safe


def main() -> int:
    context_safe._prepare_workloads = short.prepare
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
