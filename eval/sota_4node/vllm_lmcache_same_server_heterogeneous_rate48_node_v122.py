#!/usr/bin/env python3
"""Run the production multiepoch controller at 48 request/s."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from eval.sota_4node import vllm_lmcache_same_server_balanced_node_v72 as balanced
from eval.sota_4node import vllm_lmcache_same_server_output128_diagnostic_node_v96 as v96


def _router_command(*args, **kwargs):
    return balanced._router_command(*args, **kwargs)


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> int:
    original = v96._router_command
    v96._router_command = _router_command
    try:
        status = v96.main()
    finally:
        v96._router_command = original
    if int(_argument("--node-index")) == 0:
        repo_root = Path(_argument("--repo-root")).resolve()
        result_dir = Path(_argument("--result-dir")).resolve()
        subprocess.run([
            str(repo_root / ".sota_venv/bin/python"), "-m",
            "eval.sota_4node.analyze_tempo_pd_heterogeneous_rate48_v122",
            "--input", str(result_dir / "same_server_final.json"),
            "--output", str(result_dir / "rate48_final.json"),
        ], cwd=repo_root, check=True, timeout=120.0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
