#!/usr/bin/env python3
"""One native-Nixl remote stage against an explicit LMCache A/B reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v4 as stream_v3


_ORIGINAL_ENVIRONMENT = legacy._environment
_ORIGINAL_VLLM = base._vllm_command
_ORIGINAL_ROUTER = base._router_command
_HOSTS: list[str] = []
_SIDE_PORT_BASE = 0


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
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


def _environment(base_env, *, config: Path, mode: str, node_index: int):
    env = _ORIGINAL_ENVIRONMENT(
        base_env, config=config, mode=mode, node_index=node_index
    )
    env.update({
        "VLLM_NIXL_SIDE_CHANNEL_HOST": _HOSTS[node_index],
        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(_SIDE_PORT_BASE + node_index),
        "VLLM_NIXL_BACKENDS": "UCX",
    })
    return env


def _vllm_command(*args, is_prefill: bool, mode: str, pair: int, **kwargs):
    command = _ORIGINAL_VLLM(
        *args, is_prefill=is_prefill, mode=mode, pair=pair, **kwargs
    )
    role = "kv_producer" if is_prefill else "kv_consumer"
    connector = json.dumps({
        "kv_connector": "NixlConnector",
        "kv_role": role,
        "engine_id": f"native-nixl-pair-{pair}-{'p' if is_prefill else 'd'}",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {"backends": ["UCX"]},
    }, separators=(",", ":"))
    command[command.index("--kv-transfer-config") + 1] = connector
    return command


def _proxy_command(python: Path, _proxy_script: Path, _model: Path, *,
                   prefill_host: str, decode_host: str, ports: dict[str, int]):
    return [
        str(python), "-m", "eval.sota_4node.native_nixl_pd_proxy_v15",
        "--host", "0.0.0.0", "--port", str(ports["proxy_http"]),
        "--prefill-url", f"http://{prefill_host}:{ports['prefill_api']}",
        "--decode-url", f"http://{decode_host}:{ports['decode_api']}",
        "--served-model", base.SERVED_MODEL,
    ]


def _router_command(*args, **kwargs):
    command = _ORIGINAL_ROUTER(*args, **kwargs)
    command[command.index("eval.sota_4node.tempo_pd_router_v1")] = (
        "eval.sota_4node.tempo_pd_native_router_v15"
    )
    backend_index = command.index("--remote-backend") + 1
    command[backend_index] = "native-vllm-nixl-pull-ucx"
    return command


def main() -> int:
    global _HOSTS, _SIDE_PORT_BASE
    args = _parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    args.reference_root = args.reference_root.resolve()
    validation = args.reference_root / "workloads/validation.jsonl"
    local = args.reference_root / "crossover_local/raw.json"
    lmcache = args.reference_root / "crossover_remote/raw.json"
    for path in (validation, local, lmcache):
        base._require(path.is_file(), f"reference artifact missing: {path}")
    _HOSTS = args.hosts.split(",")
    base._require(len(_HOSTS) == 4 and len(set(_HOSTS)) == 4,
                  "four unique hosts required")
    _SIDE_PORT_BASE = 10000 + args.port_slot
    base._require(_SIDE_PORT_BASE + 3 < 12000, "native side ports exceed band")
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    model_revision = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    base._client_command = stream_v3._client_command
    base._vllm_command = _vllm_command
    base._router_command = _router_command
    legacy._proxy_command = _proxy_command
    legacy._environment = _environment
    raw = base._lifecycle(
        args, lifecycle=0, stage_name="native_nixl_remote",
        router_mode="lmcache_always_remote", workload_kind="validation",
        workload=validation, manifest=args.result_dir / "unused-manifest.json",
        hosts=_HOSTS, model=model, python=python, model_revision=model_revision,
    )
    marker = args.result_dir / f"node-{args.node_index}-complete"
    marker.write_text("complete\n", encoding="utf-8")
    result = args.result_dir / "result.json"
    if args.node_index == 0:
        for node_index in range(4):
            common._wait_file(args.result_dir / f"node-{node_index}-complete", [])
        subprocess.run([
            str(python), "-m", "eval.sota_4node.analyze_native_nixl_vs_lmcache_v15",
            "--local", str(local), "--lmcache", str(lmcache),
            "--native", str(raw), "--output", str(result),
        ], cwd=args.repo_root, check=True, timeout=60.0)
    else:
        common._wait_file(result, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
