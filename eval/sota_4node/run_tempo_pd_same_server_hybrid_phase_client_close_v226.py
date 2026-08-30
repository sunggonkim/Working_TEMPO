#!/usr/bin/env python3
"""Hybrid phase client using the bounded close-after-DONE metrics client."""

from __future__ import annotations

import subprocess

from eval.sota_4node import run_tempo_pd_same_server_hybrid_phase_client_v182 as phase


_OLD = "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_v38"
_NEW = "eval.sota_4node.run_tempo_pd_stream_metrics_close_after_done_v225"


def main() -> int:
    original = subprocess.run

    def bounded_run(command, *args, **kwargs):
        if isinstance(command, list) and _OLD in command:
            command = list(command)
            command[command.index(_OLD)] = _NEW
        return original(command, *args, **kwargs)

    subprocess.run = bounded_run
    try:
        return phase.main()
    finally:
        subprocess.run = original


if __name__ == "__main__":
    raise SystemExit(main())
