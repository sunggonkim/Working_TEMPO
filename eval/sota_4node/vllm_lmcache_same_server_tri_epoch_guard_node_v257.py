#!/usr/bin/env python3
"""Wire the three-arm epoch guard into the existing TP4+TP4 node lifecycle."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from eval.sota_4node import vllm_lmcache_same_server_hybrid_phase_node_v183 as phase
from eval.sota_4node.tempo_pd_same_server_tri_epoch_guard_router_v255 import MODE_ENV


_PHASE_CLIENT = phase._client_command
_PHASE_ROUTER = phase._router_command


def _client_command(*args, **kwargs):
    command = _PHASE_CLIENT(*args, **kwargs)
    old = "eval.sota_4node.run_tempo_pd_same_server_hybrid_phase_client_v182"
    command[command.index(old)] = (
        "eval.sota_4node.run_tempo_pd_same_server_tri_epoch_guard_client_v256")
    return command


def _router_command(*args, **kwargs):
    command = _PHASE_ROUTER(*args, **kwargs)
    old = "eval.sota_4node.tempo_pd_same_server_hybrid_phase_router_v181"
    command[command.index(old)] = (
        "eval.sota_4node.tempo_pd_same_server_tri_epoch_guard_router_v255")
    return command


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> int:
    result_dir = Path(_argument("--result-dir")).resolve()
    os.environ[MODE_ENV] = str(
        result_dir / "tempo_credit_admission" / "epoch_mode.json")
    original_client = phase._client_command
    original_router = phase._router_command
    phase._client_command = _client_command
    phase._router_command = _router_command
    try:
        return phase.main()
    finally:
        phase._client_command = original_client
        phase._router_command = original_router


if __name__ == "__main__":
    raise SystemExit(main())
