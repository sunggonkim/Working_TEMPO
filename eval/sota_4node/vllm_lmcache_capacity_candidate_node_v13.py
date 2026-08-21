#!/usr/bin/env python3
"""One-stage credit-admission candidate on an explicit crossover workload."""

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
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--samples-per-bucket", type=int, default=3)
    parser.add_argument("--ttft-slo-ms", type=float, default=3000)
    parser.add_argument("--tpot-slo-ms", type=float, default=250)
    parser.add_argument("--e2e-slo-ms", type=float, default=12000)
    return parser.parse_args()


def _router_command(python: Path, *, pair_router_port: int, decode_host: str,
                    proxy_host: str, ports: dict[str, int], model_revision: str,
                    decoder_load_bucket: str, **_ignored) -> list[str]:
    return [
        str(python), "-m", "eval.sota_4node.tempo_pd_capacity_router_v13",
        "--host", "0.0.0.0", "--port", str(pair_router_port),
        "--local-url", f"http://{decode_host}:{ports['decode_api']}",
        "--remote-url", f"http://{proxy_host}:{ports['proxy_http']}",
        "--tokenizer-url", f"http://{decode_host}:{ports['decode_api']}",
        "--served-model-name", base.SERVED_MODEL,
        "--model-id", "Qwen2.5-7B-Instruct",
        "--model-revision", model_revision,
        "--topology-id", base.TOPOLOGY_ID,
        "--remote-backend", base.REMOTE_BACKEND,
        "--classifier-version", base.CLASSIFIER_VERSION,
        "--decoder-load-bucket", decoder_load_bucket,
        "--kv-bytes-per-token", str(base.KV_BYTES_PER_TOKEN),
    ]


def main() -> int:
    args = _parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    args.scout_root = args.scout_root.resolve()
    validation = args.scout_root / "workloads/validation.jsonl"
    local_reference = args.scout_root / "crossover_local/raw.json"
    base._require(validation.is_file() and local_reference.is_file(), "scout artifacts missing")
    hosts = args.hosts.split(",")
    base._require(len(hosts) == 4 and len(set(hosts)) == 4, "four unique hosts required")
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    model_revision = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    base._client_command = stream_v3._client_command
    base._config_text = chunk256._config_text
    legacy._proxy_command = chunk256._proxy_command
    base._router_command = _router_command
    candidate = base._lifecycle(
        args, lifecycle=0, stage_name="tempo_credit_admission",
        router_mode="tempo_auto", workload_kind="validation",
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
            str(python), "-m", "eval.sota_4node.analyze_tempo_pd_capacity_v13",
            "--local", str(local_reference), "--candidate", str(candidate),
            "--failed-remote-root", str(args.scout_root), "--output", str(result),
        ], cwd=args.repo_root, check=True, timeout=60.0)
    else:
        common._wait_file(result, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

