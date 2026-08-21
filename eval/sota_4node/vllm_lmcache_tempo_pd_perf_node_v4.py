#!/usr/bin/env python3
"""TEMPO-PD performance node using context-safe and coalescing-aware metrics."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v2 as context_safe


_ORIGINAL_CLIENT_COMMAND = base._client_command


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT_COMMAND(*args, **kwargs)
    index = command.index("eval.sota_4node.run_tempo_pd_stream_metrics_v1")
    command[index] = "eval.sota_4node.run_tempo_pd_stream_metrics_v3"
    return command


def main() -> int:
    base._prepare_workloads = context_safe._prepare_workloads
    base._client_command = _client_command
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
