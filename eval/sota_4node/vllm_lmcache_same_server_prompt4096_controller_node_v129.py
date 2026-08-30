#!/usr/bin/env python3
"""Run output32/output64 prompt4096 through the production controller."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from eval.sota_4node import vllm_lmcache_same_server_interleaved_node_v100 as v100
from eval.sota_4node import vllm_lmcache_same_server_output16_production_node_v108 as v108


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> int:
    original_router = v100._router_command
    original_run = v100.subprocess.run

    def skip_default(command, *args, **kwargs):
        if "eval.sota_4node.analyze_tempo_pd_interleaved_v100" in command:
            return subprocess.CompletedProcess(command, 0)
        return original_run(command, *args, **kwargs)

    v100._router_command = v108._router_command
    v100.subprocess.run = skip_default
    try:
        status = v100.main()
    finally:
        v100._router_command = original_router
        v100.subprocess.run = original_run
    if int(_argument("--node-index")) == 0:
        repo_root = Path(_argument("--repo-root")).resolve()
        result_dir = Path(_argument("--result-dir")).resolve()
        subprocess.run([
            str(repo_root / ".sota_venv/bin/python"), "-m",
            "eval.sota_4node.analyze_tempo_pd_prompt4096_controller_v129",
            "--input", str(result_dir / "tempo_credit_admission/raw.json"),
            "--output", str(result_dir / "controller_final.json"),
        ], cwd=repo_root, check=True, timeout=120.0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
