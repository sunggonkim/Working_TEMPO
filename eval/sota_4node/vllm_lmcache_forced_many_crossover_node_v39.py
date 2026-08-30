#!/usr/bin/env python3
"""High-load crossover with deterministic tokens and EOF-drained streams."""

from eval.sota_4node import vllm_lmcache_forced_many_crossover_node_v37 as v37


_ORIGINAL_CLIENT_COMMAND = v37.base.stream_v3._client_command


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT_COMMAND(*args, **kwargs)
    index = command.index("eval.sota_4node.run_tempo_pd_stream_metrics_v3")
    command[index] = "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_v38"
    return command


def main() -> int:
    v37.base.stream_v3._client_command = _client_command
    return v37.main()


if __name__ == "__main__":
    raise SystemExit(main())
