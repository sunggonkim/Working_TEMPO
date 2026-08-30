#!/usr/bin/env python3
"""Run one actual TP4 P/D pair with the TEMPO-GO global frontend.

This is the inference half of the ``P1PAIR+COJOB`` experiment.  The caller
places this node entry on two nodes and launches the official NCCL/LMCache
observer on the other two nodes.  The observer path is inherited by the
router processes through the allocation environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from eval.sota_4node import vllm_lmcache_elastic_pd_node as canonical
from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_chunk256_node_v7 as chunk256
from eval.sota_4node import vllm_lmcache_tempo_go_c5_node as c5
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base
from tempo.pd_global_profile import load_global_profile


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--node-index", type=int, choices=(0, 1), default=None)
    parser.add_argument("--hosts", required=True)
    parser.add_argument("--port-slot", type=int, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--global-profile", type=Path, required=True)
    parser.add_argument("--elastic-profile", type=Path, required=True)
    parser.add_argument("--endpoint-profile", type=Path, required=True)
    parser.add_argument(
        "--cojob-ready-file", type=Path, default=None,
        help="wait for the official co-job peer-init marker before requests",
    )
    parser.add_argument("--arm", choices=("tempo", "local", "remote"), default="tempo")
    parser.add_argument("--request-rate", type=float, default=2.0)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def _frontend_command(
    python: Path, *, host: str, ports: dict[str, int]
) -> list[str]:
    return [
        str(python), "-m", "eval.sota_4node.tempo_pd_elastic_frontend",
        "--host", "0.0.0.0", "--port", str(ports["frontend"]),
        "--pair-url", f"http://{host}:{ports['pair_router']}",
    ]


def _configure_environment(args: argparse.Namespace, *, decode_host: str, ports: dict[str, int]) -> None:
    repo = args.repo_root.resolve()
    os.environ["TEMPO_GO_PROFILE"] = str(args.global_profile.resolve())
    os.environ["TEMPO_GO_ELASTIC_PROFILE"] = str(args.elastic_profile.resolve())
    os.environ["TEMPO_GO_ENDPOINT_PROFILE"] = str(args.endpoint_profile.resolve())
    os.environ["TEMPO_PD_ENDPOINT_SERVICE_PROFILE"] = str(
        args.endpoint_profile.resolve())
    os.environ["TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256"] = hashlib.sha256(
        args.workload_manifest.resolve().read_bytes()).hexdigest()
    os.environ["TEMPO_GO_C5_SOURCE_WORKLOAD"] = str(args.workload.resolve())
    os.environ["TEMPO_GO_PROFILE_SHA256"] = load_global_profile(
        args.global_profile.resolve()).fingerprint_sha256
    os.environ["TEMPO_GO_TOKENIZER_URL"] = (
        f"http://{decode_host}:{ports['decode_api']}")
    os.environ["TEMPO_GO_C5_ARM"] = args.arm
    os.environ.setdefault("TEMPO_ELASTIC_PD_PROFILE_SCOPE", "screen_only")
    os.environ.setdefault("TEMPO_PD_ENDPOINT_FEEDBACK_MODE", "adaptive")
    os.environ.setdefault("TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK", "1")
    os.environ.setdefault("TEMPO_PD_ENDPOINT_ROUTING_POLICY", "semantic_epoch_v1")
    os.environ.setdefault("TEMPO_VLLM_LOAD_SNAPSHOT_MODE", "disabled")
    os.environ.setdefault("TEMPO_PD_PRESSURE_MODE", "disabled")
    os.environ.setdefault("TEMPO_VLLM_DECODER_PREFIX_CACHING", "0")
    os.environ.setdefault("TEMPO_VLLM_MAX_NUM_SEQS", "16")
    os.environ.setdefault("TEMPO_VLLM_ASYNC_SCHEDULING", "0")
    os.environ.setdefault("TEMPO_VLLM_DECODER_MAX_NUM_BATCHED_TOKENS", "32768")
    os.environ.setdefault("TEMPO_VLLM_SCHEDULING_POLICY", "fcfs")
    os.environ.setdefault("TEMPO_LMCACHE_NIXL_BACKEND", "UCX")
    os.environ.setdefault("TEMPO_LMCACHE_LOCAL_CPU_GB", "16")
    os.environ.setdefault("TEMPO_LMCACHE_PD_BUFFER_BYTES", "2147483648")
    os.environ.setdefault("TEMPO_PD_REMOTE_DECODE_PLACEMENT", "paired")
    os.environ.setdefault("TEMPO_PD_FRONTEND_PAIR_POLICY", "tempo-min-outstanding-decode-tokens-v1")
    os.environ.setdefault("TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY", "0")
    if repo not in args.result_dir.resolve().parents:
        raise ValueError("result directory must be below repository")


def _run(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve()
    result_dir = args.result_dir.resolve()
    hosts = args.hosts.split(",")
    if len(hosts) != 2 or len(set(hosts)) != 2:
        raise ValueError("P1PAIR requires two unique inference hosts")
    workload = args.workload.resolve()
    manifest = args.workload_manifest.resolve()
    for path in (workload, manifest, args.global_profile, args.elastic_profile, args.endpoint_profile):
        if not path.is_file():
            raise FileNotFoundError(path)
    model = repo / "models/Qwen2.5-7B-Instruct"
    python = repo / ".vllm_venv/bin/python"
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model / "config.json")
    model_revision = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    c5._validate_profile_bindings(
        global_path=args.global_profile.resolve(),
        elastic_path=args.elastic_profile.resolve(),
        endpoint_path=args.endpoint_profile.resolve(),
        workload_manifest=manifest,
    )
    ports = base._ports(args.port_slot, 0)
    prefill_host, decode_host = hosts
    stage_dir = result_dir / args.arm
    stage_dir.mkdir(parents=True, exist_ok=True)
    raw_path = stage_dir / "raw.json"
    config_path = stage_dir / f"node-{args.node_index}-lmcache.yaml"
    if config_path.exists():
        raise FileExistsError(config_path)
    _configure_environment(args, decode_host=decode_host, ports=ports)
    env = legacy._environment(
        os.environ,
        config=config_path,
        mode=f"tempo_go_p1pair_{args.arm}",
        node_index=args.node_index,
    )
    env["TEMPO_PD_LOCAL_DECODER_INDEX"] = "0"
    env["TEMPO_PD_REQUIRE_DECODER_INDEX"] = "0"
    config_path.write_text(
        canonical._config_text(
            is_prefill=args.node_index == 0,
            prefill_host=prefill_host,
            decode_host=decode_host,
            ports=ports,
        ),
        encoding="utf-8",
    )
    vllm = repo / ".vllm_venv/bin/vllm"
    proxy_script = repo / "third_party/lmcache/examples/disagg_prefill/disagg_proxy_server.py"
    engine = proxy = router = frontend = None
    handles: list[Any] = []
    try:
        engine, handle = common._spawn(
            canonical._vllm_command(
                vllm,
                model,
                is_prefill=args.node_index == 0,
                mode=f"tempo_go_p1pair_{args.arm}",
                pair=0,
                ports=ports,
            ),
            stage_dir / f"node-{args.node_index}-vllm.log",
            env,
        )
        handles.append(handle)
        engine_port = ports["prefill_api"] if args.node_index == 0 else ports["decode_api"]
        common._wait_url(f"http://{hosts[args.node_index]}:{engine_port}/health", [engine])
        if args.node_index == 0:
            common._wait_url(f"http://{decode_host}:{ports['decode_api']}/health", [engine])
            proxy, handle = common._spawn(
                chunk256._proxy_command(
                    python,
                    proxy_script,
                    model,
                    prefill_host=prefill_host,
                    decode_host=decode_host,
                    ports=ports,
                ),
                stage_dir / "proxy.log",
                env,
            )
            handles.append(handle)
            common._wait_url(f"http://{prefill_host}:{ports['proxy_http']}/docs", [engine, proxy])
            router, handle = common._spawn(
                canonical._router_command(
                    python,
                    router_mode="tempo_auto",
                    pair_router_port=ports["pair_router"],
                    decode_host=decode_host,
                    proxy_host=prefill_host,
                    ports=ports,
                    model_revision=model_revision,
                    decoder_load_bucket=(
                        f"openloop-rps-{args.request_rate:g}-workers-{args.max_workers}"
                    ),
                    manifest=args.elastic_profile.resolve(),
                ),
                stage_dir / "router.log",
                env,
            )
            handles.append(handle)
            common._wait_url(f"http://{prefill_host}:{ports['pair_router']}/health", [engine, proxy, router])
            frontend, handle = common._spawn(
                _frontend_command(python, host=prefill_host, ports=ports),
                stage_dir / "frontend.log",
                env,
            )
            handles.append(handle)
            frontend_url = f"http://{prefill_host}:{ports['frontend']}"
            common._wait_url(frontend_url + "/health", [engine, proxy, router, frontend])
            if args.cojob_ready_file is not None:
                common._wait_file(args.cojob_ready_file.resolve(), [
                    engine, proxy, router, frontend,
                ])
            warmup = stage_dir / "warmup.jsonl"
            warm_rows = []
            for index, line in enumerate(workload.read_text(encoding="utf-8").splitlines()):
                if line.strip():
                    value = json.loads(line)
                    value["request_id"] = f"warm-tempo-go-p1pair-{args.arm}-{index}"
                    warm_rows.append(value)
            warmup.write_text(
                "".join(json.dumps(value, separators=(",", ":")) + "\n" for value in warm_rows),
                encoding="utf-8",
            )
            if not warm_rows:
                raise RuntimeError(
                    "frozen C5 workload produced no warmup rows; "
                    "source workload/ID contract is inconsistent"
                )
            for run_id, run_workload, output in (
                (f"tempo-go-p1pair-{args.arm}-warmup", warmup, stage_dir / "warmup.raw.json"),
                (f"tempo-go-p1pair-{args.arm}", workload, raw_path),
            ):
                subprocess.run(
                    c5._client_command(
                        python,
                        base_url=frontend_url,
                        model=model,
                        workload=run_workload,
                        output=output,
                        mode="tempo_auto",
                        run_id=run_id,
                        request_rate=args.request_rate,
                        max_workers=args.max_workers,
                    ),
                    cwd=repo,
                    env=env,
                    check=True,
                    timeout=1200.0,
                )
        else:
            common._wait_file(raw_path, [engine])
    finally:
        common._stop(frontend)
        common._stop(router)
        common._stop(proxy)
        common._stop(engine)
        for handle in handles:
            handle.close()
    marker = result_dir / f"node-{args.node_index}-complete"
    marker.write_text("complete\n", encoding="utf-8")
    final = result_dir / "result.json"
    if args.node_index == 0:
        common._wait_file(result_dir / "node-1-complete", [])
        final.write_text(json.dumps({
            "schema": "tempo-go-p1pair-native-result-v1",
            "arm": args.arm,
            "raw": str(raw_path),
            "global_profile": str(args.global_profile.resolve()),
            "elastic_profile": str(args.elastic_profile.resolve()),
            "endpoint_profile": str(args.endpoint_profile.resolve()),
            "workload": str(workload),
            "workload_manifest": str(manifest),
            "observer_path": os.environ.get("TEMPO_GO_NCCL_TELEMETRY_PATH"),
            "hosts": hosts,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        common._wait_file(final, [])


def main() -> int:
    args = _parse()
    if args.node_index is None:
        raw_rank = os.environ.get("SLURM_PROCID")
        if raw_rank not in {"0", "1"}:
            raise ValueError("node index must be 0 or 1")
        args.node_index = int(raw_rank)
    if args.request_rate <= 0 or args.max_workers <= 0:
        raise ValueError("request rate and max workers must be positive")
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
