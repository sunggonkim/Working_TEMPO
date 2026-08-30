#!/usr/bin/env python3
"""Frozen phase-change node with same-length unique leading chunks."""
import os
from eval.sota_4node import vllm_lmcache_same_server_online_regime_mixed_node_v293 as base

_BC, _BR = base._client_command, base._router_command


def _client_command(*args, **kwargs):
    command = _BC(*args, **kwargs)
    command[command.index(
        "eval.sota_4node.run_tempo_pd_same_server_mixed_only_client_v265"
    )] = "eval.sota_4node.run_tempo_pd_same_server_phasechange_prefixswap_v361"
    return command


def _router_command(*args, **kwargs):
    command = _BR(*args, **kwargs)
    command[command.index(
        "eval.sota_4node.tempo_pd_same_server_online_regime_router_v291"
    )] = "eval.sota_4node.tempo_pd_same_server_online_regime_microburst25_v342"
    return command


def _bounded_run(command, *args, **kwargs):
    if isinstance(command, list) and (
        "eval.sota_4node.analyze_tempo_pd_same_server_hybrid_controller_v160"
        in command
    ):
        output = command[command.index("--output") + 1]
        raw = os.path.join(os.path.dirname(output), "tempo_credit_admission",
                           "phasechange_paired_v353", "measured.raw.json")
        replacement = [
            command[0], "-m",
            "eval.sota_4node.analyze_tempo_pd_phasechange_frozen_v355",
            "--raw", raw, "--allocation", os.environ["SLURM_JOB_ID"],
            "--output", output,
        ]
        return base._REAL_RUN(replacement, *args, **kwargs)
    return base._REAL_RUN(command, *args, **kwargs)


def main():
    old = base._client_command, base._router_command, base._bounded_run
    base._client_command = _client_command
    base._router_command = _router_command
    base._bounded_run = _bounded_run
    try:
        return base.main()
    finally:
        base._client_command, base._router_command, base._bounded_run = old


if __name__ == "__main__":
    raise SystemExit(main())
