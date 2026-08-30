#!/usr/bin/env python3
"""Run output128 through the production router, then attach production provenance."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from eval.sota_4node import vllm_lmcache_same_server_balanced_node_v72 as balanced
from eval.sota_4node import vllm_lmcache_same_server_output128_diagnostic_node_v96 as v96


def _argument(name: str) -> str:
    index = sys.argv.index(name)
    return sys.argv[index + 1]


def main() -> int:
    original = v96._router_command
    v96._router_command = balanced._router_command
    try:
        status = v96.main()
    finally:
        v96._router_command = original
    node_index = int(_argument("--node-index"))
    if node_index == 0:
        repo_root = Path(_argument("--repo-root")).resolve()
        result_dir = Path(_argument("--result-dir")).resolve()
        subprocess.run([
            str(repo_root / ".sota_venv/bin/python"), "-m",
            "eval.sota_4node.finalize_tempo_pd_output128_production_v97",
            "--input", str(result_dir / "output128_final.json"),
            "--output", str(result_dir / "production_output128_final.json"),
        ], cwd=repo_root, check=True, timeout=120.0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
