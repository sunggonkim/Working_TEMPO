#!/usr/bin/env python3
"""Run one stock and one optimized actual-vLLM LMCache remote lifecycle."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v2 as context_safe
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v4 as stream_v3


STAGES = (
    ("stock_lmcache_remote", False),
    ("tempo_nixl_remote", True),
)
_ORIGINAL_ENVIRONMENT = legacy._environment


def _environment(base_env, *, config: Path, mode: str, node_index: int):
    env = _ORIGINAL_ENVIRONMENT(base_env, config=config, mode=mode, node_index=node_index)
    for name in (
        "TEMPO_LMCACHE_NIXL_HOTPATH",
        "TEMPO_LMCACHE_NIXL_HOTPATH_INSTALLED",
        "TEMPO_LMCACHE_NIXL_STATS_DIR",
        "TEMPO_NIXL_CACHE_CAPACITY",
        "TEMPO_NIXL_YIELD_POLLS",
        "TEMPO_NIXL_SLEEP_US",
    ):
        env.pop(name, None)
    site = str(Path(__file__).resolve().parent / "tempo_lmcache_nixl_site_v1")
    current = [part for part in env.get("PYTHONPATH", "").split(os.pathsep)
               if part and Path(part).resolve() != Path(site).resolve()]
    if mode == "tempo_nixl_remote":
        env.update({
            "TEMPO_LMCACHE_NIXL_HOTPATH": "1",
            "TEMPO_LMCACHE_NIXL_STATS_DIR": str(
                config.parent / "hotpath-stats" / f"node-{node_index}"
            ),
            "TEMPO_NIXL_CACHE_CAPACITY": "128",
            "TEMPO_NIXL_YIELD_POLLS": "16",
            "TEMPO_NIXL_SLEEP_US": "100",
            "PYTHONPATH": os.pathsep.join([site, *current]),
        })
    else:
        env["PYTHONPATH"] = os.pathsep.join(current)
    return env


def main() -> int:
    args = base._parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    base._require(args.repo_root in args.result_dir.parents,
                  "result directory must be below repository")
    hosts = args.hosts.split(",")
    base._require(len(hosts) == 4 and len(set(hosts)) == 4,
                  "four unique hosts required")
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    base._require((model / "config.json").is_file(), "Qwen model is missing")
    model_revision = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    base._prepare_workloads = context_safe._prepare_workloads
    base._client_command = stream_v3._client_command
    legacy._environment = _environment
    _, validation = context_safe._prepare_workloads(args, model, python)
    manifest = args.result_dir / "unused-policy-manifest.json"
    raw: dict[str, Path] = {}
    for lifecycle, (stage_name, optimized) in enumerate(STAGES):
        raw[stage_name] = base._lifecycle(
            args,
            lifecycle=lifecycle,
            stage_name=stage_name,
            router_mode="lmcache_always_remote",
            workload_kind="validation",
            workload=validation,
            manifest=manifest,
            hosts=hosts,
            model=model,
            python=python,
            model_revision=model_revision,
        )
        marker = args.result_dir / stage_name / f"node-{args.node_index}-complete"
        marker.write_text("complete\n", encoding="utf-8")
        if args.node_index == 0:
            for node_index in range(4):
                common._wait_file(
                    args.result_dir / stage_name / f"node-{node_index}-complete", []
                )
    result = args.result_dir / "result.json"
    if args.node_index == 0:
        subprocess.run([
            str(python), "-m", "eval.sota_4node.analyze_lmcache_nixl_hotpath_ab_v1",
            "--stock", str(raw["stock_lmcache_remote"]),
            "--optimized", str(raw["tempo_nixl_remote"]),
            "--telemetry-root", str(args.result_dir / "tempo_nixl_remote" / "hotpath-stats"),
            "--output", str(result),
        ], cwd=args.repo_root, check=True, timeout=60.0)
    else:
        common._wait_file(result, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
