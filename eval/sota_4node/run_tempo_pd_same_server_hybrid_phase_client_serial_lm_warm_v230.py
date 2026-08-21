#!/usr/bin/env python3
"""Hybrid phase client with only the unmeasured LMCache seed serialized.

The measured six-block crossover remains byte-for-byte unchanged at rate 48
and 32 workers.  Serializing the first LMCache warm block avoids a documented
KV-ready liveness stall while changing no measured arm.
"""

from __future__ import annotations

import subprocess

from eval.sota_4node import run_tempo_pd_same_server_hybrid_phase_client_v182 as phase


_TARGET_RUN_ID_FRAGMENT = "-warmup-00_lmcache_remote_r0"


def _serial_warm_command(command):
    if not isinstance(command, list) or _TARGET_RUN_ID_FRAGMENT not in " ".join(command):
        return command
    value = list(command)
    worker_index = value.index("--max-workers") + 1
    value[worker_index] = "1"
    return value


def main() -> int:
    original = subprocess.run

    def bounded_run(command, *args, **kwargs):
        return original(_serial_warm_command(command), *args, **kwargs)

    subprocess.run = bounded_run
    try:
        return phase.main()
    finally:
        subprocess.run = original


if __name__ == "__main__":
    raise SystemExit(main())
