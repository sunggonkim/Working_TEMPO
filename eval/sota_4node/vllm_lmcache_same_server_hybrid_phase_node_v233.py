#!/usr/bin/env python3
"""Correct phase-node wiring with serial unmeasured LMCache warmup."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_same_server_hybrid_phase_node_v183 as phase


_PHASE_CLIENT = phase._client_command


def _client_command(*args, **kwargs):
    command = _PHASE_CLIENT(*args, **kwargs)
    old = "eval.sota_4node.run_tempo_pd_same_server_hybrid_phase_client_v182"
    command[command.index(old)] = (
        "eval.sota_4node.run_tempo_pd_same_server_hybrid_phase_client_serial_lm_warm_v230"
    )
    return command


def main() -> int:
    original = phase._client_command
    phase._client_command = _client_command
    try:
        # phase.main installs both this client and the cold-aware phase router.
        return phase.main()
    finally:
        phase._client_command = original


if __name__ == "__main__":
    raise SystemExit(main())
