#!/usr/bin/env python3
"""Continue a crossover scout with calibration and adaptive validation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess

from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v4 as stream_v3
from eval.sota_4node import vllm_lmcache_chunk256_node_v7 as chunk256


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--scout-root", type=Path, required=True)
    parser.add_argument("--node-index", type=int, choices=range(4), required=True)
    parser.add_argument("--hosts", required=True)
    parser.add_argument("--port-slot", type=int, required=True)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--samples-per-bucket", type=int, default=3)
    parser.add_argument("--ttft-slo-ms", type=float, default=3000)
    parser.add_argument("--tpot-slo-ms", type=float, default=250)
    parser.add_argument("--e2e-slo-ms", type=float, default=12000)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    args.scout_root = args.scout_root.resolve()
    base._require(args.repo_root in args.result_dir.parents, "result must be below repo")
    base._require(args.repo_root in args.scout_root.parents, "scout must be below repo")
    base._require((args.scout_root / "result.json").is_file(), "scout result missing")
    calibration = args.scout_root / "workloads/calibration.jsonl"
    validation = args.scout_root / "workloads/validation.jsonl"
    local_reference = args.scout_root / "crossover_local/raw.json"
    remote_reference = args.scout_root / "crossover_remote/raw.json"
    for path in (calibration, validation, local_reference, remote_reference):
        base._require(path.is_file(), f"required scout artifact missing: {path}")
    hosts = args.hosts.split(",")
    base._require(len(hosts) == 4 and len(set(hosts)) == 4, "four unique hosts required")
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    model_revision = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    base._client_command = stream_v3._client_command
    base._config_text = chunk256._config_text
    legacy._proxy_command = chunk256._proxy_command
    raw: dict[str, Path] = {}
    for lifecycle, (stage_name, mode) in enumerate((
        ("calibration_local", "fixed_local"),
        ("calibration_remote", "lmcache_always_remote"),
    )):
        raw[stage_name] = base._lifecycle(
            args, lifecycle=lifecycle, stage_name=stage_name,
            router_mode=mode, workload_kind="calibration",
            workload=calibration, manifest=args.result_dir / "unused-manifest.json",
            hosts=hosts, model=model, python=python, model_revision=model_revision,
        )
    manifest = base._build_manifest(
        args, python, raw["calibration_local"], raw["calibration_remote"]
    )
    raw["validation_tempo"] = base._lifecycle(
        args, lifecycle=2, stage_name="validation_tempo",
        router_mode="tempo_auto", workload_kind="validation",
        workload=validation, manifest=manifest,
        hosts=hosts, model=model, python=python, model_revision=model_revision,
    )
    marker = args.result_dir / f"node-{args.node_index}-complete"
    marker.write_text("complete\n", encoding="utf-8")
    result = args.result_dir / "result.json"
    if args.node_index == 0:
        for node_index in range(4):
            common._wait_file(args.result_dir / f"node-{node_index}-complete", [])
        subprocess.run([
            str(python), "-m", "eval.sota_4node.analyze_tempo_pd_performance_v1",
            "--run", f"local={local_reference}",
            "--run", f"lmcache={remote_reference}",
            "--run", f"tempo={raw['validation_tempo']}",
            "--output", str(result),
            "--ttft-slo-ms", str(args.ttft_slo_ms),
            "--tpot-slo-ms", str(args.tpot_slo_ms),
            "--e2e-slo-ms", str(args.e2e_slo_ms),
        ], cwd=args.repo_root, check=True, timeout=60.0)
    else:
        common._wait_file(result, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
