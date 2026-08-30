#!/usr/bin/env python3
"""Two-stage actual-vLLM local/LMCache-remote crossover scout."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v2 as context_safe
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v4 as stream_v3
from eval.sota_4node import vllm_lmcache_chunk256_node_v7 as chunk256


def main() -> int:
    args = base._parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    base._require(args.repo_root in args.result_dir.parents, "result must be below repo")
    base._require(args.request_rate > 0 and args.max_workers >= 2,
                  "positive rate and at least two workers required")
    hosts = args.hosts.split(",")
    base._require(len(hosts) == 4 and len(set(hosts)) == 4, "four unique hosts required")
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    base._require((model / "config.json").is_file(), "Qwen model is missing")
    model_revision = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    base._prepare_workloads = context_safe._prepare_workloads
    base._client_command = stream_v3._client_command
    base._config_text = chunk256._config_text
    legacy._proxy_command = chunk256._proxy_command
    _, validation = context_safe._prepare_workloads(args, model, python)
    raw: dict[str, Path] = {}
    stages = (
        ("crossover_local", "fixed_local"),
        ("crossover_remote", "lmcache_always_remote"),
    )
    for lifecycle, (stage_name, router_mode) in enumerate(stages):
        raw[stage_name] = base._lifecycle(
            args, lifecycle=lifecycle, stage_name=stage_name,
            router_mode=router_mode, workload_kind="validation",
            workload=validation, manifest=args.result_dir / "unused-manifest.json",
            hosts=hosts, model=model, python=python, model_revision=model_revision,
        )
    marker = args.result_dir / f"node-{args.node_index}-complete"
    marker.write_text("complete\n", encoding="utf-8")
    result = args.result_dir / "result.json"
    if args.node_index == 0:
        for node_index in range(4):
            common._wait_file(args.result_dir / f"node-{node_index}-complete", [])
        subprocess.run([
            str(python), "-m", "eval.sota_4node.analyze_tempo_pd_remote_crossover_v9",
            "--local", str(raw["crossover_local"]),
            "--remote", str(raw["crossover_remote"]),
            "--output", str(result),
        ], cwd=args.repo_root, check=True, timeout=60.0)
    else:
        common._wait_file(result, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
