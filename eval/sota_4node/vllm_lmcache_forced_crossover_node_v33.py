#!/usr/bin/env python3
"""Unique-head local/LMCache baseline with deterministic output tokens."""

from eval.sota_4node import vllm_lmcache_remote_crossover_unique_head_node_v23 as v23


_ORIGINAL_CLIENT_COMMAND = v23.base.stream_v3._client_command


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT_COMMAND(*args, **kwargs)
    index = command.index("eval.sota_4node.run_tempo_pd_stream_metrics_v3")
    command[index] = "eval.sota_4node.run_tempo_pd_stream_metrics_forced_v32"
    return command


def main() -> int:
    v23.base.stream_v3._client_command = _client_command
    return v23.main()


if __name__ == "__main__":
    raise SystemExit(main())
