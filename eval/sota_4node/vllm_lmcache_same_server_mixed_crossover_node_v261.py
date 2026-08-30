#!/usr/bin/env python3
"""Wire same-window mixed crossover client into the policy8 lifecycle."""

from eval.sota_4node import vllm_lmcache_same_server_hybrid_phase_node_v183 as phase


_ORIGINAL = phase._client_command


def _client_command(*args, **kwargs):
    command = _ORIGINAL(*args, **kwargs)
    old = "eval.sota_4node.run_tempo_pd_same_server_hybrid_phase_client_v182"
    command[command.index(old)] = (
        "eval.sota_4node.run_tempo_pd_same_server_mixed_crossover_client_v260")
    return command


def main() -> int:
    original = phase._client_command
    phase._client_command = _client_command
    try:
        return phase.main()
    finally:
        phase._client_command = original


if __name__ == "__main__":
    raise SystemExit(main())
