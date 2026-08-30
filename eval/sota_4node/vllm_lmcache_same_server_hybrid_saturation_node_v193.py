#!/usr/bin/env python3
"""Node wrapper for rate-saturation TEMPO/fixed-local isolation."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_same_server_hybrid_controller_node_v166 as base


_ORIGINAL_CLIENT = base._client_command
_ORIGINAL_ROUTER = base._router_command
_ORIGINAL_SUBPROCESS = base.subprocess


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT(*args, **kwargs)
    old = "eval.sota_4node.run_tempo_pd_same_server_cache_catalog_client_v163"
    command[command.index(old)] = (
        "eval.sota_4node.run_tempo_pd_same_server_hybrid_saturation_client_v190")
    return command


def _router_command(*args, **kwargs):
    command = _ORIGINAL_ROUTER(*args, **kwargs)
    old = "eval.sota_4node.tempo_pd_same_server_hybrid_controller_router_v150"
    command[command.index(old)] = (
        "eval.sota_4node.tempo_pd_same_server_hybrid_saturation_router_v191")
    return command


class _AnalyzerProxy:
    def __getattr__(self, name):
        return getattr(_ORIGINAL_SUBPROCESS, name)

    @staticmethod
    def run(command, *args, **kwargs):
        old = "eval.sota_4node.analyze_tempo_pd_same_server_hybrid_controller_v160"
        if old in command:
            command = list(command)
            command[command.index(old)] = (
                "eval.sota_4node.analyze_tempo_pd_hybrid_saturation_v192")
        return _ORIGINAL_SUBPROCESS.run(command, *args, **kwargs)


def main() -> int:
    original_client = base._client_command
    original_router = base._router_command
    original_subprocess = base.subprocess
    base._client_command = _client_command
    base._router_command = _router_command
    base.subprocess = _AnalyzerProxy()
    try:
        return base.main()
    finally:
        base._client_command = original_client
        base._router_command = original_router
        base.subprocess = original_subprocess


if __name__ == "__main__":
    raise SystemExit(main())
