#!/usr/bin/env python3
"""Optimized-only actual-vLLM run with cache-free aggressive NIXL progress."""

from __future__ import annotations

import hashlib
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
_LOCAL_STATS: dict[int, Path] = {}


def _environment(base_env, *, config: Path, mode: str, node_index: int):
    env = _ORIGINAL_ENVIRONMENT(base_env, config=config, mode=mode, node_index=node_index)
    site = Path(__file__).resolve().parent / "tempo_lmcache_nixl_site_v2"
    fingerprint = hashlib.sha256(str(config.parent.parent).encode()).hexdigest()[:16]
    local = Path(f"/tmp/tempo-nixl-poll-v3-{base_env['SLURM_JOB_ID']}-{fingerprint}-n{node_index}")
    base._require(not local.exists(), f"stale local telemetry directory: {local}")
    _LOCAL_STATS[node_index] = local
    current = [part for part in env.get("PYTHONPATH", "").split(os.pathsep)
               if part and Path(part).resolve() != site.resolve()]
    env.update({
        "TEMPO_LMCACHE_NIXL_HOTPATH": "2",
        "TEMPO_LMCACHE_NIXL_STATS_DIR": str(local),
        "TEMPO_NIXL_CACHE_CAPACITY": "0",
        "TEMPO_NIXL_YIELD_POLLS": "4096",
        "TEMPO_NIXL_SLEEP_US": "100",
        "PYTHONPATH": os.pathsep.join([str(site), *current]),
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
    legacy._environment = _environment
    _, validation = context_safe._prepare_workloads(args, model, python)
    raw = base._lifecycle(
        args, lifecycle=3, stage_name="tempo_nixl_polling_v3",
        router_mode="lmcache_always_remote", workload_kind="validation",
        workload=validation, manifest=args.result_dir / "unused-manifest.json",
        hosts=hosts, model=model, python=python, model_revision=model_revision,
    )
    local = _LOCAL_STATS[args.node_index]
    paths = sorted(local.glob("nixl-hotpath-*.json"))
    if args.node_index % 2 == 0:
        base._require(paths, f"sender node {args.node_index} has no snapshots")
    else:
        base._require(not paths, f"receiver node {args.node_index} unexpectedly wrote snapshots")
    destination = args.result_dir / "tempo_nixl_polling_v3" / "hotpath-stats" / f"node-{args.node_index}"
    destination.mkdir(parents=True, exist_ok=False)
    for path in paths:
        shutil.copy2(path, destination / path.name)
    marker = args.result_dir / f"node-{args.node_index}-telemetry-complete"
    marker.write_text(f"{len(paths)}\n", encoding="utf-8")
    result = args.result_dir / "result.json"
    if args.node_index == 0:
        for node_index in range(4):
            common._wait_file(args.result_dir / f"node-{node_index}-telemetry-complete", [])
        stock_copy = args.result_dir / "stock_reference.raw.json"
        shutil.copy2(args.stock_reference, stock_copy)
        subprocess.run([
            str(python), "-m", "eval.sota_4node.analyze_lmcache_nixl_polling_ab_v3",
            "--stock", str(stock_copy), "--optimized", str(raw),
            "--telemetry-root", str(args.result_dir / "tempo_nixl_polling_v3" / "hotpath-stats"),
            "--output", str(result),
        ], cwd=args.repo_root, check=True, timeout=60.0)
    else:
        common._wait_file(result, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
