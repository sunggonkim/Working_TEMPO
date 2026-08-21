#!/usr/bin/env python3
"""Node correction wiring the cache-isolated Elastic-PD client."""

from eval.sota_4node import vllm_lmcache_elastic_pd_node_v445 as prior


def _client_command(*args, **kwargs):
    command = prior._ORIGINAL_CLIENT(*args, **kwargs)
    old = "eval.sota_4node.run_tempo_pd_stream_metrics_v1"
    command[command.index(old)] = (
        "eval.sota_4node.run_tempo_pd_elastic_balanced_client_v446")
    return command


def main():
    old = prior._client_command
    prior._client_command = _client_command
    try:
        return prior.main()
    finally:
        prior._client_command = old


if __name__ == "__main__":
    raise SystemExit(main())
