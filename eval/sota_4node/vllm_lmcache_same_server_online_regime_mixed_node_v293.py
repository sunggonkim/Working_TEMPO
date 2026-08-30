#!/usr/bin/env python3
"""Wire the online-regime router into the mixed-only lifecycle."""

import os

from eval.sota_4node import vllm_lmcache_same_server_hybrid_controller_node_v166 as controller
from eval.sota_4node import vllm_lmcache_same_server_hybrid_phase_node_v183 as phase


_PHASE_CLIENT = phase._client_command
_PHASE_ROUTER = phase._router_command
_REAL_RUN = controller.subprocess.run


def _client_command(*args, **kwargs):
    command = _PHASE_CLIENT(*args, **kwargs)
    old = "eval.sota_4node.run_tempo_pd_same_server_hybrid_phase_client_v182"
    command[command.index(old)] = (
        "eval.sota_4node.run_tempo_pd_same_server_mixed_only_client_v265")
    return command


def _router_command(*args, **kwargs):
    command = _PHASE_ROUTER(*args, **kwargs)
    old = "eval.sota_4node.tempo_pd_same_server_hybrid_phase_router_v181"
    command[command.index(old)] = (
        "eval.sota_4node.tempo_pd_same_server_online_regime_router_v291")
    return command


def _bounded_run(command, *args, **kwargs):
    if (isinstance(command, list) and
            "eval.sota_4node.analyze_tempo_pd_same_server_hybrid_controller_v160"
            in command):
        output = command[command.index("--output") + 1]
        root = os.path.dirname(output)
        raw = os.path.join(root, "tempo_credit_admission",
                           "mixed_request_crossover_v265", "measured.raw.json")
        replacement = [command[0], "-m",
                       "eval.sota_4node.analyze_tempo_pd_online_regime_mixed_v292",
                       "--raw", raw, "--allocation", os.environ["SLURM_JOB_ID"],
                       "--output", output]
        return _REAL_RUN(replacement, *args, **kwargs)
    return _REAL_RUN(command, *args, **kwargs)


def main() -> int:
    original_client = phase._client_command
    original_router = phase._router_command
    original_run = controller.subprocess.run
    phase._client_command = _client_command
    phase._router_command = _router_command
    controller.subprocess.run = _bounded_run
    try:
        return phase.main()
    finally:
        phase._client_command = original_client
        phase._router_command = original_router
        controller.subprocess.run = original_run


if __name__ == "__main__":
    raise SystemExit(main())
