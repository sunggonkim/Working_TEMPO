#!/usr/bin/env python3
"""Production output64 same-server node for an explicit workload."""

from __future__ import annotations

import hashlib
import json
import subprocess

from eval.sota_4node import vllm_lmcache_capacity_candidate_node_v13 as capacity
from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base
from eval.sota_4node import vllm_lmcache_chunk256_node_v7 as chunk256
from eval.sota_4node import vllm_lmcache_same_server_balanced_node_v72 as balanced


def main() -> int:
    args = capacity._parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    args.scout_root = args.scout_root.resolve()
    validation = args.scout_root / "workloads/validation.jsonl"
    base._require(validation.is_file(), "explicit validation workload missing")
    hosts = args.hosts.split(",")
    base._require(len(hosts) == 4 and len(set(hosts)) == 4, "four unique hosts required")
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    model_revision = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    base._client_command = balanced._client_command
    base._config_text = chunk256._config_text
    legacy._proxy_command = chunk256._proxy_command
    base._router_command = balanced._router_command
    candidate = base._lifecycle(
        args, lifecycle=0, stage_name="tempo_credit_admission",
        router_mode="tempo_auto", workload_kind="validation", workload=validation,
        manifest=args.result_dir / "unused-manifest.json", hosts=hosts, model=model,
        python=python, model_revision=model_revision,
    )
    marker = args.result_dir / f"node-{args.node_index}-complete"
    marker.write_text("complete\n", encoding="utf-8")
    result = args.result_dir / "result.json"
    if args.node_index == 0:
        for node_index in range(4):
            common._wait_file(args.result_dir / f"node-{node_index}-complete", [])
        base_report = args.result_dir / "same_server_final.json"
        subprocess.run([
            str(python), "-m", "eval.sota_4node.analyze_tempo_pd_same_server_balanced_v71",
            "--stage-root", str(args.result_dir / "tempo_credit_admission"),
            "--output", str(base_report),
        ], cwd=args.repo_root, check=True, timeout=120.0)
        final = args.result_dir / "output64_final.json"
        subprocess.run([
            str(python), "-m", "eval.sota_4node.analyze_tempo_pd_same_server_balanced_output64_v77",
            "--input", str(base_report), "--output", str(final),
        ], cwd=args.repo_root, check=True, timeout=120.0)
        result.write_text(json.dumps({
            "schema": "tempo-pd-explicit-output64-result-91",
            "candidate": str(candidate.resolve()), "final": str(final.resolve()),
        }, sort_keys=True) + "\n", encoding="utf-8")
    else:
        common._wait_file(result, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
