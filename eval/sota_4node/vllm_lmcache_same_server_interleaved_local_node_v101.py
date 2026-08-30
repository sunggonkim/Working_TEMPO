#!/usr/bin/env python3
"""Run the request-interleaved output16 all-local candidate."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from eval.sota_4node import vllm_lmcache_same_server_balanced_node_v72 as balanced
from eval.sota_4node import vllm_lmcache_same_server_interleaved_node_v100 as v100


def _router_command(*args, **kwargs):
    command = balanced._router_command(*args, **kwargs)
    module = "eval.sota_4node.tempo_pd_same_server_balanced_router_v70"
    command[command.index(module)] = (
        "eval.sota_4node.tempo_pd_same_server_interleaved_local_router_v101")
    return command


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> int:
    original = v100._router_command
    v100._router_command = _router_command
    try:
        status = v100.main()
    finally:
        v100._router_command = original
    if int(_argument("--node-index")) == 0:
        repo_root = Path(_argument("--repo-root")).resolve()
        result_dir = Path(_argument("--result-dir")).resolve()
        subprocess.run([
            str(repo_root / ".sota_venv/bin/python"), "-m",
            "eval.sota_4node.analyze_tempo_pd_interleaved_local_v101",
            "--input", str(result_dir / "tempo_credit_admission/raw.json"),
            "--output", str(result_dir / "interleaved_local_final.json"),
        ], cwd=repo_root, check=True, timeout=120.0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
