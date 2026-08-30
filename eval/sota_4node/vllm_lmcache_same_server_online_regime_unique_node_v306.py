#!/usr/bin/env python3
"""Bind the online router to collision-free prompt chunks and its analyzer."""

import os

from eval.sota_4node import vllm_lmcache_same_server_online_regime_mixed_node_v293 as base


_BASE_CLIENT = base._client_command


def _client_command(*args, **kwargs):
    command = _BASE_CLIENT(*args, **kwargs)
    old = "eval.sota_4node.run_tempo_pd_same_server_mixed_only_client_v265"
    command[command.index(old)] = (
        "eval.sota_4node.run_tempo_pd_same_server_mixed_only_client_unique_chunks_v305")
    return command


def _bounded_run(command, *args, **kwargs):
    if (isinstance(command, list) and
            "eval.sota_4node.analyze_tempo_pd_same_server_hybrid_controller_v160"
            in command):
        output = command[command.index("--output") + 1]
        root = os.path.dirname(output)
        raw = os.path.join(root, "tempo_credit_admission",
                           "mixed_request_crossover_unique_chunks_v305",
                           "measured.raw.json")
        replacement = [
            command[0], "-m",
            "eval.sota_4node.analyze_tempo_pd_online_regime_mixed_v292",
            "--raw", raw, "--allocation", os.environ["SLURM_JOB_ID"],
            "--output", output,
        ]
        return base._REAL_RUN(replacement, *args, **kwargs)
    return base._REAL_RUN(command, *args, **kwargs)


def main() -> int:
    original_client = base._client_command
    original_run = base._bounded_run
    base._client_command = _client_command
    base._bounded_run = _bounded_run
    try:
        return base.main()
    finally:
        base._client_command = original_client
        base._bounded_run = original_run


if __name__ == "__main__":
    raise SystemExit(main())
