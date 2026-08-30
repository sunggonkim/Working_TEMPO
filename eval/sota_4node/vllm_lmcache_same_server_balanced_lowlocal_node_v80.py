#!/usr/bin/env python3
"""Low-load local-policy wrapper around the audited balanced harness."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from eval.sota_4node import vllm_lmcache_same_server_balanced_node_v72 as prior


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> int:
    node_index = int(_argument("--node-index"))
    result_dir = Path(_argument("--result-dir")).resolve()
    repo_root = Path(_argument("--repo-root")).resolve()
    status = prior.main()
    if node_index == 0:
        subprocess.run([
            str(repo_root / ".vllm_venv/bin/python"), "-m",
            "eval.sota_4node.analyze_tempo_pd_same_server_balanced_lowlocal_v80",
            "--input", str(result_dir / "same_server_final.json"),
            "--output", str(result_dir / "lowlocal_final.json"),
        ], cwd=repo_root, check=True, timeout=120.0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
