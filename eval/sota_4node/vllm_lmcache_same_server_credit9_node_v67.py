#!/usr/bin/env python3
"""Single-lifecycle credit-nine diagnostic node entry."""

from __future__ import annotations
from pathlib import Path
import subprocess, sys
from eval.sota_4node import vllm_lmcache_capacity_candidate_node_v13 as base


_ORIGINAL_ROUTER_COMMAND = base._router_command
_ORIGINAL_CLIENT_COMMAND = base.stream_v3._client_command


def _router_command(*args, **kwargs):
    command = _ORIGINAL_ROUTER_COMMAND(*args, **kwargs)
    command[command.index("eval.sota_4node.tempo_pd_capacity_router_v13")] = (
        "eval.sota_4node.tempo_pd_same_server_router_credit9_v66")
    return command


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT_COMMAND(*args, **kwargs)
    command[command.index("eval.sota_4node.run_tempo_pd_stream_metrics_v3")] = (
        "eval.sota_4node.run_tempo_pd_same_server_client_v62")
    return command


def _argument(name): return sys.argv[sys.argv.index(name) + 1]


def main() -> int:
    result_dir = Path(_argument("--result-dir")).resolve()
    node_index = int(_argument("--node-index"))
    repo_root = Path(_argument("--repo-root")).resolve()
    base._router_command = _router_command
    base.stream_v3._client_command = _client_command
    status = base.main()
    if node_index == 0:
        subprocess.run([
            str(repo_root / ".vllm_venv/bin/python"), "-m",
            "eval.sota_4node.analyze_tempo_pd_same_server_credit9_v67",
            "--stage-root", str(result_dir / "tempo_credit_admission"),
            "--output", str(result_dir / "same_server_final.json"),
        ], cwd=repo_root, check=True, timeout=120.0)
    return status


if __name__ == "__main__": raise SystemExit(main())
