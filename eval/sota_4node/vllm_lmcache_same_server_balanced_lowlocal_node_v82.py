#!/usr/bin/env python3
"""Corrected low-load node selecting the context-preserving client."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_same_server_balanced_lowlocal_node_v81 as prior


_ORIGINAL_CLIENT_COMMAND = prior._ORIGINAL_CLIENT_COMMAND


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT_COMMAND(*args, **kwargs)
    command[command.index("eval.sota_4node.run_tempo_pd_stream_metrics_v3")] = (
        "eval.sota_4node.run_tempo_pd_same_server_balanced_low_client_v82")
    return command


def main() -> int:
    prior._client_command = _client_command
    return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
