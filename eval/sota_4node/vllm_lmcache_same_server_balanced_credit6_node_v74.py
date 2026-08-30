#!/usr/bin/env python3
"""Thin node wrapper applying credit six to the audited balanced harness."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from eval.sota_4node import vllm_lmcache_same_server_balanced_credit7_node_v73 as prior


_ORIGINAL_ROUTER_COMMAND = prior._ORIGINAL_ROUTER_COMMAND


def _router_command(*args, **kwargs):
    command = _ORIGINAL_ROUTER_COMMAND(*args, **kwargs)
    command[command.index("eval.sota_4node.tempo_pd_capacity_router_v13")] = (
        "eval.sota_4node.tempo_pd_same_server_balanced_credit6_router_v74")
    return command


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> int:
    node_index = int(_argument("--node-index"))
    result_dir = Path(_argument("--result-dir")).resolve()
    repo_root = Path(_argument("--repo-root")).resolve()
    prior._router_command = _router_command
    status = prior.main()
    if node_index == 0:
        subprocess.run([
            str(repo_root / ".vllm_venv/bin/python"), "-m",
            "eval.sota_4node.analyze_tempo_pd_same_server_balanced_credit6_v74",
            "--input", str(result_dir / "same_server_final.json"),
            "--output", str(result_dir / "same_server_credit6_final.json"),
        ], cwd=repo_root, check=True, timeout=120.0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
