#!/usr/bin/env python3
"""Run the hybrid cold workload without the obsolete scout-analysis coupling."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from eval.sota_4node import vllm_lmcache_capacity_candidate_node_v13 as capacity
from eval.sota_4node import vllm_lmcache_same_server_hybrid_cold_node_v175 as base


_ORIGINAL_REQUIRE = capacity.base._require
_ORIGINAL_SUBPROCESS = capacity.subprocess


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def _require(condition: bool, message: str) -> None:
    if message == "scout artifacts missing":
        validation = Path(_argument("--scout-root")) / "workloads/validation.jsonl"
        _ORIGINAL_REQUIRE(validation.is_file(), "cold validation workload missing")
        return
    _ORIGINAL_REQUIRE(condition, message)


class _CapacityAnalysisBypass:
    """Delegate subprocesses except the obsolete pre-balanced analyzer."""

    def __getattr__(self, name):
        return getattr(_ORIGINAL_SUBPROCESS, name)

    @staticmethod
    def run(command, *args, **kwargs):
        marker = "eval.sota_4node.analyze_tempo_pd_capacity_v13"
        if marker not in command:
            return _ORIGINAL_SUBPROCESS.run(command, *args, **kwargs)
        output = Path(command[command.index("--output") + 1])
        if output.exists():
            raise ValueError(f"refusing stale legacy result: {output}")
        output.write_text(json.dumps({
            "schema": "tempo-pd-obsolete-capacity-analysis-bypass-178",
            "reason": "final same-server balanced analyzer consumes measured blocks directly",
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)


def main() -> int:
    original_require = capacity.base._require
    original_subprocess = capacity.subprocess
    capacity.base._require = _require
    capacity.subprocess = _CapacityAnalysisBypass()
    try:
        return base.main()
    finally:
        capacity.base._require = original_require
        capacity.subprocess = original_subprocess


if __name__ == "__main__":
    raise SystemExit(main())
