#!/usr/bin/env python3
"""Run the four-arm Elastic-PD screen on the proven actual P/D lifecycle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from eval.sota_4node import vllm_lmcache_capacity_candidate_node_v13 as capacity
from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base
from eval.sota_4node import vllm_lmcache_chunk256_node_v7 as chunk256


PROFILE = "eval/sota_4node/real_tempo_pd_elastic_profile_v445.json"
_ORIGINAL_CLIENT = base._client_command


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT(*args, **kwargs)
    old = "eval.sota_4node.run_tempo_pd_stream_metrics_v1"
    command[command.index(old)] = (
        "eval.sota_4node.run_tempo_pd_elastic_balanced_client_v445")
    return command


def _router_command(
    python: Path, *, router_mode: str, pair_router_port: int,
    decode_host: str, proxy_host: str, ports: dict[str, int],
    model_revision: str, decoder_load_bucket: str, manifest: Path,
):
    del router_mode
    return [
        str(python), "-m", "eval.sota_4node.tempo_pd_elastic_router_v445",
        "--host", "0.0.0.0", "--port", str(pair_router_port),
        "--local-url", f"http://{decode_host}:{ports['decode_api']}",
        "--remote-url", f"http://{proxy_host}:{ports['proxy_http']}",
        "--tokenizer-url", f"http://{decode_host}:{ports['decode_api']}",
        "--served-model-name", base.SERVED_MODEL,
        "--model-id", "Qwen2.5-7B-Instruct", "--model-revision", model_revision,
        "--topology-id", base.TOPOLOGY_ID, "--remote-backend", base.REMOTE_BACKEND,
        "--classifier-version", base.CLASSIFIER_VERSION,
        "--decoder-load-bucket", decoder_load_bucket,
        "--kv-bytes-per-token", str(base.KV_BYTES_PER_TOKEN),
        "--profile", str(manifest), "--allow-screen-profile",
        "--queue-wait-ms", "250",
    ]


def _frontend_command(python: Path, *, host0: str, host2: str,
                      ports: dict[str, int]):
    return [
        str(python), "-m", "eval.sota_4node.tempo_pd_elastic_frontend_v445",
        "--host", "0.0.0.0", "--port", str(ports["frontend"]),
        "--pair-url", f"http://{host0}:{ports['pair_router']}",
        "--pair-url", f"http://{host2}:{ports['pair_router']}",
    ]


def main() -> int:
    args = capacity._parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    args.scout_root = args.scout_root.resolve()
    workload = args.scout_root
    if workload.is_dir():
        workload = workload / "workloads/validation.jsonl"
    base._require(workload.is_file(), "explicit validation workload missing")
    hosts = args.hosts.split(",")
    base._require(len(hosts) == 4 and len(set(hosts)) == 4, "four unique hosts")
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    revision = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    profile = args.repo_root / PROFILE
    base._require(profile.is_file(), "frozen Elastic-PD profile missing")
    base._client_command = _client_command
    base._config_text = chunk256._config_text
    legacy._proxy_command = chunk256._proxy_command
    base._router_command = _router_command
    base._frontend_command = _frontend_command
    candidate = base._lifecycle(
        args, lifecycle=0, stage_name="tempo_elastic_pd_v445",
        router_mode="tempo_auto", workload_kind="validation", workload=workload,
        manifest=profile, hosts=hosts, model=model, python=python,
        model_revision=revision,
    )
    marker = args.result_dir / f"node-{args.node_index}-complete"
    marker.write_text("complete\n")
    result = args.result_dir / "result.json"
    if args.node_index == 0:
        for index in range(4):
            common._wait_file(args.result_dir / f"node-{index}-complete", [])
        final = args.result_dir / "elastic_pd_final.json"
        subprocess.run([
            str(python), "-m", "eval.sota_4node.analyze_tempo_pd_elastic_balanced_v445",
            "--stage-root", str(args.result_dir / "tempo_elastic_pd_v445"),
            "--output", str(final),
        ], cwd=args.repo_root, check=True, timeout=120.0)
        result.write_text(json.dumps({
            "schema": "tempo-elastic-pd-result-445",
            "candidate": str(candidate.resolve()), "final": str(final.resolve()),
            "profile": str(profile.resolve()),
        }, sort_keys=True) + "\n")
    else:
        common._wait_file(result, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
