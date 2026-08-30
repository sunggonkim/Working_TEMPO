#!/usr/bin/env python3
"""One-factor tail-aware wrapper around the audited saturation harness."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_same_server_hybrid_saturation_node_v193 as base


_ORIGINAL_ROUTER = base._router_command


def _router_command(*args, **kwargs):
    command = _ORIGINAL_ROUTER(*args, **kwargs)
    old = "eval.sota_4node.tempo_pd_same_server_hybrid_saturation_router_v191"
    command[command.index(old)] = (
        "eval.sota_4node.tempo_pd_same_server_hybrid_tailaware_router_v195")
    return command


class _AnalyzerProxy:
    def __getattr__(self, name):
        return getattr(base._ORIGINAL_SUBPROCESS, name)

    @staticmethod
    def run(command, *args, **kwargs):
        old = "eval.sota_4node.analyze_tempo_pd_same_server_hybrid_controller_v160"
        if old in command:
            command = list(command)
            command[command.index(old)] = (
                "eval.sota_4node.analyze_tempo_pd_hybrid_tailaware_v197")
        return base._ORIGINAL_SUBPROCESS.run(command, *args, **kwargs)


def main() -> int:
    original_router = base._router_command
    original_proxy = base._AnalyzerProxy
    base._router_command = _router_command
    base._AnalyzerProxy = _AnalyzerProxy
    try:
        return base.main()
    finally:
        base._router_command = original_router
        base._AnalyzerProxy = original_proxy


if __name__ == "__main__":
    raise SystemExit(main())
