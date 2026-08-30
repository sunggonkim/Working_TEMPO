#!/usr/bin/env python3
"""Independent-run wrapper applying the frozen v103 output16 gates."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from eval.sota_4node import vllm_lmcache_same_server_interleaved_local_fast_node_v102 as v102


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> int:
    status = v102.main()
    if int(_argument("--node-index")) == 0:
        repo_root = Path(_argument("--repo-root")).resolve()
        result_dir = Path(_argument("--result-dir")).resolve()
        subprocess.run([
            str(repo_root / ".sota_venv/bin/python"), "-m",
            "eval.sota_4node.analyze_tempo_pd_interleaved_local_fast_v103",
            "--input", str(result_dir / "tempo_credit_admission/raw.json"),
            "--output", str(result_dir / "independent_final.json"),
        ], cwd=repo_root, check=True, timeout=120.0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
