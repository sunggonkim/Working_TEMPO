#!/usr/bin/env python3
"""One actual-vLLM LMCache remote lifecycle using NIXL LIBFABRIC/CXI."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v2 as context_safe
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v4 as stream_v3
from eval.sota_4node import vllm_lmcache_nixl_hotpath_snapshot_node_v2 as prior


_ORIGINAL_ENVIRONMENT = legacy._environment
_ORIGINAL_CONFIG = base._config_text


def _config_text(**kwargs):
    text = _ORIGINAL_CONFIG(**kwargs)
    base._require("nixl_backends: [UCX]" in text, "UCX config seam drifted")
    return text.replace("nixl_backends: [UCX]", "nixl_backends: [LIBFABRIC]")


def _environment(base_env, *, config: Path, mode: str, node_index: int):
    env = _ORIGINAL_ENVIRONMENT(base_env, config=config, mode=mode, node_index=node_index)
    site = Path(base_env["VIRTUAL_ENV"]) / "lib/python3.12/site-packages"
    libraries = site / ".nixl_cu12.mesonpy.libs"
    plugin = libraries / "plugins/libplugin_LIBFABRIC.so"
    base._require(plugin.is_file(), f"missing LIBFABRIC plugin: {plugin}")
    for name in ("TEMPO_LMCACHE_NIXL_HOTPATH", "TEMPO_LMCACHE_NIXL_STATS_DIR"):
        env.pop(name, None)
    env.update({
        "NIXL_PLUGIN_DIR": str(plugin.parent),
        "LD_LIBRARY_PATH": os.pathsep.join([
            str(libraries), "/opt/cray/libfabric/1.22.0/lib64",
            env.get("LD_LIBRARY_PATH", ""),
        ]).rstrip(os.pathsep),
        "FI_PROVIDER": "cxi",
    })
    return env


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
    legacy._environment = _environment
    _, validation = context_safe._prepare_workloads(args, model, python)
    raw = base._lifecycle(
        args, lifecycle=4, stage_name="lmcache_libfabric_remote",
        router_mode="lmcache_always_remote", workload_kind="validation",
        workload=validation, manifest=args.result_dir / "unused-manifest.json",
        hosts=hosts, model=model, python=python, model_revision=model_revision,
    )
    log_path = args.result_dir / "lmcache_libfabric_remote" / f"node-{args.node_index}-vllm.log"
    log = log_path.read_text(encoding="utf-8", errors="replace")
    instantiated = "Backend LIBFABRIC was instantiated" in log
    evidence = {
        "node_index": args.node_index,
        "libfabric_instantiated": instantiated,
        "fi_provider": "cxi",
        "nixl_api": "nixl_cu12._api",
    }
    base._require(instantiated, f"node {args.node_index} lacks LIBFABRIC evidence")
    evidence_path = args.result_dir / f"node-{args.node_index}-libfabric-evidence.json"
    evidence_path.write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")
    result = args.result_dir / "result.json"
    if args.node_index == 0:
        for node_index in range(4):
            common._wait_file(args.result_dir / f"node-{node_index}-libfabric-evidence.json", [])
        stock_copy = args.result_dir / "stock_reference.raw.json"
        shutil.copy2(args.stock_reference, stock_copy)
        subprocess.run([
            str(python), "-m", "eval.sota_4node.analyze_lmcache_libfabric_ab_v5",
            "--stock", str(stock_copy), "--candidate", str(raw),
            "--evidence-root", str(args.result_dir), "--output", str(result),
        ], cwd=args.repo_root, check=True, timeout=60.0)
    else:
        common._wait_file(result, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
