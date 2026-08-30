#!/usr/bin/env python3
"""Order-balanced same-lifecycle local/TEMPO/LMCache P/D node entry."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from eval.sota_4node import vllm_lmcache_same_server_node_v63 as prior
from eval.sota_4node import vllm_lmcache_capacity_candidate_node_v13 as base


_ORIGINAL_CLIENT_COMMAND = base.stream_v3._client_command


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT_COMMAND(*args, **kwargs)
    command[command.index("eval.sota_4node.run_tempo_pd_stream_metrics_v3")] = (
        "eval.sota_4node.run_tempo_pd_same_server_balanced_client_v70")
    return command


def _argument(name: str) -> str:
    index = sys.argv.index(name)
    return sys.argv[index + 1]


def main() -> int:
    result_dir = Path(_argument("--result-dir")).resolve()
    node_index = int(_argument("--node-index"))
    repo_root = Path(_argument("--repo-root")).resolve()
    base._router_command = prior._router_command
    base.stream_v3._client_command = _client_command
    status = base.main()
    if node_index == 0:
        subprocess.run([
            str(repo_root / ".vllm_venv/bin/python"), "-m",
            "eval.sota_4node.analyze_tempo_pd_same_server_balanced_v71",
            "--stage-root", str(result_dir / "tempo_credit_admission"),
            "--output", str(result_dir / "same_server_final.json"),
        ], cwd=repo_root, check=True, timeout=120.0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
