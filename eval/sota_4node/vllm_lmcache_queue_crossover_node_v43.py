#!/usr/bin/env python3
"""Threshold-nine queue-crossover candidate node entry."""

from eval.sota_4node import vllm_lmcache_capacity_candidate_node_v13 as base


_ORIGINAL_ROUTER_COMMAND = base._router_command
_ORIGINAL_CLIENT_COMMAND = base.stream_v3._client_command


def _router_command(*args, **kwargs):
    command = _ORIGINAL_ROUTER_COMMAND(*args, **kwargs)
    index = command.index("eval.sota_4node.tempo_pd_capacity_router_v13")
    command[index] = "eval.sota_4node.tempo_pd_queue_crossover_router_v43"
    return command


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT_COMMAND(*args, **kwargs)
    index = command.index("eval.sota_4node.run_tempo_pd_stream_metrics_v3")
    command[index] = "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_v38"
    return command


def main() -> int:
    base._router_command = _router_command
    base.stream_v3._client_command = _client_command
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
