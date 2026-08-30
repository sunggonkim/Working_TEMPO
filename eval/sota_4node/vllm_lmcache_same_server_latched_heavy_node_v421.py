#!/usr/bin/env python3
"""Generic heavy-burst node binding for a fixed-cap latched controller."""

import os

from eval.sota_4node import vllm_lmcache_same_server_online_regime_mixed_node_v293 as base


def main(variant):
    module = {
        "cap5": "eval.sota_4node.tempo_pd_same_server_latched_microburst25_v382",
        "cap6": "eval.sota_4node.tempo_pd_same_server_latched_microburst25_cap6_v401",
    }[variant]
    base_client, base_router = base._client_command, base._router_command

    def client(*args, **kwargs):
        command = base_client(*args, **kwargs)
        command[command.index(
            "eval.sota_4node.run_tempo_pd_same_server_mixed_only_client_v265")] = (
            "eval.sota_4node.run_tempo_pd_heavyburst_prefixswap_v419")
        return command

    def router(*args, **kwargs):
        command = base_router(*args, **kwargs)
        command[command.index(
            "eval.sota_4node.tempo_pd_same_server_online_regime_router_v291")] = module
        return command

    def run(command, *args, **kwargs):
        if (isinstance(command, list)
                and "eval.sota_4node.analyze_tempo_pd_same_server_hybrid_controller_v160"
                in command):
            output = command[command.index("--output") + 1]
            raw = os.path.join(os.path.dirname(output), "tempo_credit_admission",
                               "bursty_paired_v322", "measured.raw.json")
            replacement = [
                command[0], "-m", "eval.sota_4node.analyze_tempo_pd_latched_heavyburst_v420",
                "--raw", raw, "--allocation", os.environ["SLURM_JOB_ID"],
                "--variant", variant, "--output", output,
            ]
            return base._REAL_RUN(replacement, *args, **kwargs)
        return base._REAL_RUN(command, *args, **kwargs)

    old = base._client_command, base._router_command, base._bounded_run
    base._client_command, base._router_command, base._bounded_run = client, router, run
    try:
        return base.main()
    finally:
        base._client_command, base._router_command, base._bounded_run = old
