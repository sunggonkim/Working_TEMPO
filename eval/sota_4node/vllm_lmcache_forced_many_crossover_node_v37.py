#!/usr/bin/env python3
"""High-load unique-head forced-token local/LMCache crossover."""

from eval.sota_4node import tempo_pd_unique_head_many_workload_v37 as workload
from eval.sota_4node import vllm_lmcache_remote_crossover_node_v9 as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v2 as context_safe


_ORIGINAL_CLIENT_COMMAND = base.stream_v3._client_command


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT_COMMAND(*args, **kwargs)
    index = command.index("eval.sota_4node.run_tempo_pd_stream_metrics_v3")
    command[index] = "eval.sota_4node.run_tempo_pd_stream_metrics_forced_v32"
    return command


def main() -> int:
    context_safe._prepare_workloads = workload.prepare
    base.stream_v3._client_command = _client_command
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
