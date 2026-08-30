#!/usr/bin/env python3
"""Wire the policy11 router into the mixed-only lifecycle."""

import os

from eval.sota_4node import vllm_lmcache_same_server_policy10_mixed_node_v275 as base
from eval.sota_4node import vllm_lmcache_same_server_hybrid_controller_node_v166 as controller


_POLICY10_ROUTER = base._router_command
_REAL_RUN = controller.subprocess.run


def _router_command(*args, **kwargs):
    command = _POLICY10_ROUTER(*args, **kwargs)
    old = "eval.sota_4node.tempo_pd_same_server_policy10_router_v274"
    command[command.index(old)] = (
        "eval.sota_4node.tempo_pd_same_server_policy11_highload_router_v285")
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
                       "eval.sota_4node.analyze_tempo_pd_policy11_mixed_v286",
                       "--raw", raw, "--allocation", os.environ["SLURM_JOB_ID"],
                       "--output", output]
        return _REAL_RUN(replacement, *args, **kwargs)
    return _REAL_RUN(command, *args, **kwargs)


def main() -> int:
    original_router = base._router_command
    original_bounded = base._bounded_run
    base._router_command = _router_command
    base._bounded_run = _bounded_run
    try:
        return base.main()
    finally:
        base._router_command = original_router
        base._bounded_run = original_bounded


if __name__ == "__main__":
    raise SystemExit(main())
