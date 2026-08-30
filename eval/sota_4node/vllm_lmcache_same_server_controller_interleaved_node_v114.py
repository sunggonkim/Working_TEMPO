#!/usr/bin/env python3
"""Run production output32/output64 with request-level interleaving."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from eval.sota_4node import vllm_lmcache_same_server_output16_production_node_v108 as v108
from eval.sota_4node import vllm_lmcache_same_server_interleaved_node_v100 as v100


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> int:
    original = v100._router_command
    v100._router_command = v108._router_command
    try:
        status = v100.main()
    finally:
        v100._router_command = original
    if int(_argument("--node-index")) == 0:
        repo_root = Path(_argument("--repo-root")).resolve()
        result_dir = Path(_argument("--result-dir")).resolve()
        output_tokens = _argument("--output-tokens")
        subprocess.run([
            str(repo_root / ".sota_venv/bin/python"), "-m",
            "eval.sota_4node.analyze_tempo_pd_controller_interleaved_v114",
            "--input", str(result_dir / "tempo_credit_admission/raw.json"),
            "--output", str(result_dir / "controller_final.json"),
            "--output-tokens", output_tokens,
        ], cwd=repo_root, check=True, timeout=120.0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
