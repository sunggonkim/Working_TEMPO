#!/usr/bin/env python3
"""One stock-UCX actual-vLLM LMCache remote lifecycle at chunk size 256."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess

from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v2 as context_safe
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v4 as stream_v3
from eval.sota_4node import vllm_lmcache_nixl_hotpath_snapshot_node_v2 as prior


_ORIGINAL_CONFIG = base._config_text
_ORIGINAL_PROXY = legacy._proxy_command


def _config_text(**kwargs):
    text = _ORIGINAL_CONFIG(**kwargs)
    base._require("chunk_size: 64" in text, "chunk config seam drifted")
    return text.replace("chunk_size: 64", "chunk_size: 256")


def _proxy_command(*args, **kwargs):
    command = _ORIGINAL_PROXY(*args, **kwargs)
    index = command.index("--chunk-size") + 1
    base._require(command[index] == "64", "proxy chunk seam drifted")
    command[index] = "256"
    return command


def main() -> int:
    args = prior._parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    args.stock_reference = args.stock_reference.resolve()
    base._require(args.repo_root in args.result_dir.parents, "result must be below repo")
    base._require(args.stock_reference.is_file(), "stock reference is missing")
    hosts = args.hosts.split(",")
    base._require(len(hosts) == 4 and len(set(hosts)) == 4, "four unique hosts required")
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    model_revision = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    base._prepare_workloads = context_safe._prepare_workloads
    base._client_command = stream_v3._client_command
    base._config_text = _config_text
    legacy._proxy_command = _proxy_command
    _, validation = context_safe._prepare_workloads(args, model, python)
    raw = base._lifecycle(
        args, lifecycle=5, stage_name="lmcache_chunk256_remote",
        router_mode="lmcache_always_remote", workload_kind="validation",
        workload=validation, manifest=args.result_dir / "unused-manifest.json",
        hosts=hosts, model=model, python=python, model_revision=model_revision,
    )
    marker = args.result_dir / f"node-{args.node_index}-complete"
    marker.write_text("complete\n", encoding="utf-8")
    result = args.result_dir / "result.json"
    if args.node_index == 0:
        for node_index in range(4):
            common._wait_file(args.result_dir / f"node-{node_index}-complete", [])
        stock_copy = args.result_dir / "stock_reference.raw.json"
        shutil.copy2(args.stock_reference, stock_copy)
        subprocess.run([
            str(python), "-m", "eval.sota_4node.analyze_lmcache_chunk256_ab_v7",
            "--stock", str(stock_copy), "--candidate", str(raw),
            "--config-root", str(args.result_dir / "lmcache_chunk256_remote"),
            "--output", str(result),
        ], cwd=args.repo_root, check=True, timeout=60.0)
    else:
        common._wait_file(result, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

