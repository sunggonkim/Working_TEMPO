#!/usr/bin/env python3
"""Use policy10 in the arm-separated phase node with serial LMCache warmup."""

from eval.sota_4node import vllm_lmcache_same_server_hybrid_phase_node_v183 as phase
from eval.sota_4node import vllm_lmcache_same_server_hybrid_phase_node_v233 as serial


_PHASE_ROUTER = phase._router_command


def _router_command(*args, **kwargs):
    command = _PHASE_ROUTER(*args, **kwargs)
    old = "eval.sota_4node.tempo_pd_same_server_hybrid_phase_router_v181"
    command[command.index(old)] = (
        "eval.sota_4node.tempo_pd_same_server_policy10_router_v274")
    return command


def main() -> int:
    original = phase._router_command
    phase._router_command = _router_command
    try:
        return serial.main()
    finally:
        phase._router_command = original


if __name__ == "__main__":
    raise SystemExit(main())
