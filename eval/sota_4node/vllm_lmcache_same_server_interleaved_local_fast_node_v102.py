#!/usr/bin/env python3
"""Run the request-interleaved output16 direct-local fast path."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_same_server_balanced_node_v72 as balanced
from eval.sota_4node import vllm_lmcache_same_server_interleaved_local_node_v101 as v101


def _router_command(*args, **kwargs):
    command = balanced._router_command(*args, **kwargs)
    module = "eval.sota_4node.tempo_pd_same_server_balanced_router_v70"
    command[command.index(module)] = (
        "eval.sota_4node.tempo_pd_same_server_interleaved_local_fast_router_v102")
    return command


def main() -> int:
    original = v101._router_command
    v101._router_command = _router_command
    try:
        return v101.main()
    finally:
        v101._router_command = original


if __name__ == "__main__":
    raise SystemExit(main())
