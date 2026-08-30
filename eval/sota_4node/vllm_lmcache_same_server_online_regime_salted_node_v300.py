#!/usr/bin/env python3
"""Bind the salted client artifact path into the frozen online analyzer."""

import os

from eval.sota_4node import vllm_lmcache_same_server_online_regime_mixed_node_v293 as base
from eval.sota_4node import vllm_lmcache_same_server_online_regime_salted_node_v298 as salted


def _bounded_run(command, *args, **kwargs):
    if (isinstance(command, list) and
            "eval.sota_4node.analyze_tempo_pd_same_server_hybrid_controller_v160"
            in command):
        output = command[command.index("--output") + 1]
        root = os.path.dirname(output)
        raw = os.path.join(root, "tempo_credit_admission",
                           "mixed_request_crossover_salted_v297",
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
    original = base._bounded_run
    base._bounded_run = _bounded_run
    try:
        return salted.main()
    finally:
        base._bounded_run = original


if __name__ == "__main__":
    raise SystemExit(main())
