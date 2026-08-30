#!/usr/bin/env python3
"""Production node entry for one-epoch cold→seed→hit validation."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_same_server_hybrid_controller_node_v166 as base


_ORIGINAL_CLIENT = base._client_command
_ORIGINAL_ROUTER = base._router_command


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT(*args, **kwargs)
    old = "eval.sota_4node.run_tempo_pd_same_server_cache_catalog_client_v163"
    command[command.index(old)] = (
        "eval.sota_4node.run_tempo_pd_same_server_hybrid_phase_client_v182")
    return command


def _router_command(*args, **kwargs):
    command = _ORIGINAL_ROUTER(*args, **kwargs)
    old = "eval.sota_4node.tempo_pd_same_server_hybrid_controller_router_v150"
    command[command.index(old)] = (
        "eval.sota_4node.tempo_pd_same_server_hybrid_phase_router_v181")
    return command


def main() -> int:
    original_client = base._client_command
    original_router = base._router_command
    base._client_command = _client_command
    base._router_command = _router_command
    try:
        return base.main()
    finally:
        base._client_command = original_client
        base._router_command = original_router


if __name__ == "__main__":
    raise SystemExit(main())
