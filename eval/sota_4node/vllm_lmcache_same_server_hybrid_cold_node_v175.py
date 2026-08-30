#!/usr/bin/env python3
"""Hybrid cold node using nonce-agnostic serial transport prewarm."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_same_server_hybrid_cold_node_v172 as base


_ORIGINAL_CLIENT = base._client_command


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT(*args, **kwargs)
    old = "eval.sota_4node.run_tempo_pd_same_server_cold_prewarm_client_v171"
    command[command.index(old)] = (
        "eval.sota_4node.run_tempo_pd_same_server_cold_prewarm_client_v174")
    return command


def main() -> int:
    original = base._client_command
    base._client_command = _client_command
    try:
        return base.main()
    finally:
        base._client_command = original


if __name__ == "__main__":
    raise SystemExit(main())
