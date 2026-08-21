#!/usr/bin/env python3
"""Four-node, two-replica TEMPO-PD calibration and performance lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy


SERVED_MODEL = "tempo-qwen25-7b-pd-perf"
CLASSIFIER_VERSION = "tempo-pd-qwen25-tp4-pairs-v1"
TOPOLOGY_ID = "perlmutter-4n-2replica-tp4-prefill-tp4-decode"
REMOTE_BACKEND = "official-lmcacheconnectorv1-nixl-ucx"
KV_BYTES_PER_TOKEN = 28 * 4 * 128 * 2 * 2
STAGES = (
    ("calibration_local", "fixed_local", "calibration"),
    ("calibration_remote", "lmcache_always_remote", "calibration"),
    ("validation_local", "fixed_local", "validation"),
    ("validation_remote", "lmcache_always_remote", "validation"),
    ("validation_tempo", "tempo_auto", "validation"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _decode_hosts(
    hosts: list[str], pair: int,
) -> tuple[str, str, str]:
    """Resolve local and remote decoder hosts for one ingress pair."""
    _require(len(hosts) == 4 and len(set(hosts)) == 4,
             "four unique hosts required")
    _require(pair in (0, 1), "pair must be 0 or 1")
    placement = os.environ.get(
        "TEMPO_PD_REMOTE_DECODE_PLACEMENT", "paired")
    _require(
        placement in ("paired", "cross", "long_decode_cross"),
        "TEMPO_PD_REMOTE_DECODE_PLACEMENT must be "
        "paired, cross, or long_decode_cross",
    )
    local_decode_host = hosts[pair * 2 + 1]
    remote_decode_host = (
        local_decode_host
        if placement != "cross"
        else hosts[(1 - pair) * 2 + 1]
    )
    return placement, local_decode_host, remote_decode_host


def _multi_decoder_proxy_command(
    command: list[str], hosts: list[str], wrapper: Path,
) -> list[str]:
    """Point the official proxy data plane at both decoder replicas."""
    _require(len(hosts) == 4 and len(set(hosts)) == 4,
             "four unique hosts required")
    result = list(command)
    _require(len(result) > 2, "proxy command is incomplete")
    _require(Path(result[1]).name == "disagg_proxy_server.py",
             "unexpected official proxy script seam")
    result[1] = str(wrapper)
    for marker in ("--decoder-host", "--num-decoders"):
        _require(
            result.count(marker) == 1,
            f"proxy command must contain one {marker}",
        )
    result[result.index("--decoder-host") + 1] = (
        f"{hosts[1]},{hosts[3]}")
    result[result.index("--num-decoders") + 1] = "2"
    return result


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--node-index", type=int, choices=range(4), required=True)
    parser.add_argument("--hosts", required=True)
    parser.add_argument("--port-slot", type=int, required=True)
    parser.add_argument("--request-rate", type=float, default=2.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=2)
    parser.add_argument("--samples-per-bucket", type=int, default=3)
    parser.add_argument("--ttft-slo-ms", type=float, default=1000.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=100.0)
    parser.add_argument("--e2e-slo-ms", type=float, default=2000.0)
    return parser.parse_args()


def _ports(slot: int, lifecycle: int) -> dict[str, int]:
    offset = slot + lifecycle * 40
    values = {
        "prefill_api": 12000 + offset,
        "decode_api": 14000 + offset,
        "proxy_http": 16000 + offset,
        "proxy_notify": 18000 + offset,
        "decoder_init": 20000 + offset,
        "decoder_alloc": 22000 + offset,
        "pair_router": 25000 + offset,
        "frontend": 28000 + offset,
    }
    _require(max(values.values()) + 3 < 32768, "port range exceeds 32767")
    return values


def _config_text(
    *, is_prefill: bool, prefill_host: str, decode_host: str,
    ports: dict[str, int],
) -> str:
    text = legacy._config_text(
        is_prefill=is_prefill,
        prefill_host=prefill_host,
        decode_host=decode_host,
        ports=ports,
    )
    text = text.replace("pd_max_prefill_len: 2048", "pd_max_prefill_len: 8192")
    return text + "use_gpu_connector_v3: True\n"


def _vllm_command(
    executable: Path,
    model: Path,
    *,
    is_prefill: bool,
    mode: str,
    pair: int,
    ports: dict[str, int],
) -> list[str]:
    command = legacy._vllm_command(
        executable, model,
        is_prefill=is_prefill,
        mode=mode,
        pair=pair,
        ports=ports,
    )
    if "--disable-log-requests" in command:
        command.remove("--disable-log-requests")
    command[command.index("--served-model-name") + 1] = SERVED_MODEL
    command[command.index("--max-model-len") + 1] = "8192"
    command[command.index("--max-num-seqs") + 1] = "8"
    if "--max-num-batched-tokens" in command:
        command[command.index("--max-num-batched-tokens") + 1] = "8192"
    else:
        command.extend(("--max-num-batched-tokens", "8192"))
    return command


def _router_command(
    python: Path,
    *,
    router_mode: str,
    pair_router_port: int,
    decode_host: str,
    proxy_host: str,
    ports: dict[str, int],
    model_revision: str,
    decoder_load_bucket: str,
    manifest: Path,
) -> list[str]:
    command = [
        str(python), "-m", "eval.sota_4node.tempo_pd_router_v1",
        "--host", "0.0.0.0", "--port", str(pair_router_port),
        "--mode", router_mode,
        "--local-url", f"http://{decode_host}:{ports['decode_api']}",
        "--remote-url", f"http://{proxy_host}:{ports['proxy_http']}",
        "--tokenizer-url", f"http://{decode_host}:{ports['decode_api']}",
        "--served-model-name", SERVED_MODEL,
        "--model-id", "Qwen2.5-7B-Instruct",
        "--model-revision", model_revision,
        "--topology-id", TOPOLOGY_ID,
        "--remote-backend", REMOTE_BACKEND,
        "--classifier-version", CLASSIFIER_VERSION,
        "--decoder-load-bucket", decoder_load_bucket,
        "--kv-bytes-per-token", str(KV_BYTES_PER_TOKEN),
    ]
    if router_mode == "tempo_auto":
        command.extend(("--manifest", str(manifest), "--allow-screen-profiles"))
    return command


def _frontend_command(
    python: Path, *, host0: str, host2: str, ports: dict[str, int]
) -> list[str]:
    return [
        str(python), "-m", "eval.sota_4node.tempo_pd_frontend_v1",
        "--host", "0.0.0.0", "--port", str(ports["frontend"]),
        "--pair-url", f"http://{host0}:{ports['pair_router']}",
        "--pair-url", f"http://{host2}:{ports['pair_router']}",
    ]


def _client_command(
    python: Path,
    *,
    base_url: str,
    model: Path,
    workload: Path,
    output: Path,
    mode: str,
    run_id: str,
    request_rate: float,
    max_workers: int,
) -> list[str]:
    return [
        str(python), "-m", "eval.sota_4node.run_tempo_pd_stream_metrics_v1",
        "--base-url", base_url,
        "--model", str(model),
        "--served-model-name", SERVED_MODEL,
        "--workload", str(workload),
        "--output", str(output),
        "--mode", mode,
        "--run-id", run_id,
        "--max-workers", str(max_workers),
        "--request-rate", str(request_rate),
        "--timeout-s", "600",
    ]


def _prepare_workloads(args: argparse.Namespace, model: Path, python: Path) -> tuple[Path, Path]:
    workload_dir = args.result_dir / "workloads"
    manifest_path = workload_dir / "workload_manifest.json"
    if args.node_index == 0:
        subprocess.run([
            str(python), "-m", "eval.sota_4node.make_tempo_pd_workloads_v1",
            "--model", str(model),
            "--output-dir", str(workload_dir),
            "--samples-per-bucket", str(args.samples_per_bucket),
            "--output-tokens", str(args.output_tokens),
        ], cwd=args.repo_root, check=True, timeout=120.0)
    else:
        common._wait_file(manifest_path, [])
    return workload_dir / "calibration.jsonl", workload_dir / "validation.jsonl"


def _build_manifest(
    args: argparse.Namespace, python: Path, local_raw: Path, remote_raw: Path,
) -> Path:
    manifest = args.result_dir / "policy_manifest.json"
    report = args.result_dir / "policy_build_report.json"
    if args.node_index == 0:
        subprocess.run([
            str(python), "-m", "eval.sota_4node.build_tempo_pd_profile_manifest_v1",
            "--local", str(local_raw), "--remote", str(remote_raw),
            "--classifier-version", CLASSIFIER_VERSION,
            "--policy-epoch", "1",
            "--minimum-samples-per-route", str(args.samples_per_bucket),
            "--remote-advantage-margin-ms", "5.0",
            "--output", str(manifest), "--report", str(report),
        ], cwd=args.repo_root, check=True, timeout=60.0)
    else:
        common._wait_file(manifest, [])
    return manifest


def _lifecycle(
    args: argparse.Namespace,
    *,
    lifecycle: int,
    stage_name: str,
    router_mode: str,
    workload_kind: str,
    workload: Path,
    manifest: Path,
    hosts: list[str],
    model: Path,
    python: Path,
    model_revision: str,
) -> Path:
    stage_dir = args.result_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    raw_path = stage_dir / "raw.json"
    pair = args.node_index // 2
    is_prefill = args.node_index % 2 == 0
    prefill_host = hosts[pair * 2]
    placement, decode_host, remote_decode_host = _decode_hosts(
        hosts, pair)
    ports = _ports(args.port_slot, lifecycle)
    config_path = stage_dir / f"node-{args.node_index}-lmcache.yaml"
    _require(not config_path.exists(), f"stale config: {config_path}")
    config_path.write_text(_config_text(
        is_prefill=is_prefill, prefill_host=prefill_host,
        decode_host=decode_host, ports=ports,
    ), encoding="utf-8")
    env = legacy._environment(
        os.environ, config=config_path, mode=stage_name, node_index=args.node_index
    )
    env["TEMPO_PD_LOCAL_DECODER_INDEX"] = str(pair)
    env["TEMPO_PD_REQUIRE_DECODER_INDEX"] = (
        "1" if placement == "long_decode_cross" else "0"
    )
    vllm = args.repo_root / ".vllm_venv/bin/vllm"
    proxy_script = (
        args.repo_root / "third_party/lmcache/examples/disagg_prefill/disagg_proxy_server.py"
    )
    selector_script = (
        args.repo_root / "eval/sota_4node/tempo_pd_decoder_selecting_proxy.py"
    )
    engine = proxy = router = frontend = None
    handles: list[Any] = []
    try:
        engine, handle = common._spawn(
            _vllm_command(
                vllm, model, is_prefill=is_prefill, mode=stage_name,
                pair=pair, ports=ports,
            ),
            stage_dir / f"node-{args.node_index}-vllm.log",
            env,
        )
        handles.append(handle)
        engine_port = ports["prefill_api"] if is_prefill else ports["decode_api"]
        common._wait_url(f"http://{hosts[args.node_index]}:{engine_port}/health", [engine])
        if is_prefill:
            target_decode_hosts = (
                (hosts[1], hosts[3])
                if placement == "long_decode_cross"
                else (decode_host, remote_decode_host)
            )
            for target_decode_host in dict.fromkeys(
                target_decode_hosts
            ):
                common._wait_url(
                    f"http://{target_decode_host}:{ports['decode_api']}/health",
                    [engine])
            proxy_command = legacy._proxy_command(
                python, proxy_script, model,
                prefill_host=prefill_host,
                decode_host=remote_decode_host, ports=ports,
            )
            if placement == "long_decode_cross":
                proxy_command = _multi_decoder_proxy_command(
                    proxy_command, hosts, selector_script)
            proxy, handle = common._spawn(
                proxy_command,
                stage_dir / f"node-{args.node_index}-proxy.log",
                env,
            )
            handles.append(handle)
            common._wait_url(f"http://{prefill_host}:{ports['proxy_http']}/docs",
                             [engine, proxy])
            decoder_load_bucket = (
                f"openloop-rps-{args.request_rate:g}-workers-{args.max_workers}"
            )
            router, handle = common._spawn(
                _router_command(
                    python, router_mode=router_mode,
                    pair_router_port=ports["pair_router"],
                    decode_host=decode_host, proxy_host=prefill_host, ports=ports,
                    model_revision=model_revision,
                    decoder_load_bucket=decoder_load_bucket,
                    manifest=manifest,
                ),
                stage_dir / f"node-{args.node_index}-router.log",
                env,
            )
            handles.append(handle)
            common._wait_url(f"http://{prefill_host}:{ports['pair_router']}/health",
                             [engine, proxy, router])
        if args.node_index == 0:
            common._wait_url(f"http://{hosts[2]}:{ports['pair_router']}/health",
                             [engine, proxy, router])
            frontend, handle = common._spawn(
                _frontend_command(
                    python, host0=hosts[0], host2=hosts[2], ports=ports
                ),
                stage_dir / "frontend.log",
                env,
            )
            handles.append(handle)
            frontend_url = f"http://{hosts[0]}:{ports['frontend']}"
            common._wait_url(frontend_url + "/health", [engine, proxy, router, frontend])
            # Warm the exact stage route and workload classes with distinct IDs.
            warmup = stage_dir / "warmup.jsonl"
            warm_rows = []
            for index, line in enumerate(workload.read_text(encoding="utf-8").splitlines()):
                value = json.loads(line)
                value["request_id"] = f"warm-{stage_name}-{index}"
                warm_rows.append(value)
            warmup.write_text("".join(json.dumps(value, separators=(",", ":")) + "\n"
                                      for value in warm_rows), encoding="utf-8")
            subprocess.run(
                _client_command(
                    python, base_url=frontend_url, model=model, workload=warmup,
                    output=stage_dir / "warmup.raw.json", mode=router_mode,
                    run_id=f"{stage_name}-warmup", request_rate=args.request_rate,
                    max_workers=args.max_workers,
                ),
                cwd=args.repo_root, env=env, check=True, timeout=1200.0,
            )
            subprocess.run(
                _client_command(
                    python, base_url=frontend_url, model=model, workload=workload,
                    output=raw_path, mode=router_mode,
                    run_id=stage_name, request_rate=args.request_rate,
                    max_workers=args.max_workers,
                ),
                cwd=args.repo_root, env=env, check=True, timeout=1200.0,
            )
        else:
            common._wait_file(raw_path, [engine] + ([proxy, router] if is_prefill else []))
    finally:
        common._stop(frontend)
        common._stop(router)
        common._stop(proxy)
        common._stop(engine)
        for handle in handles:
            handle.close()
    return raw_path


def main() -> int:
    args = _parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    _require(args.repo_root in args.result_dir.parents,
             "result directory must be below repository")
    _require(args.request_rate > 0 and args.max_workers > 0,
             "request rate and workers must be positive")
    hosts = args.hosts.split(",")
    _require(len(hosts) == 4 and len(set(hosts)) == 4, "four unique hosts required")
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    _require((model / "config.json").is_file(), "Qwen model is missing")
    model_revision = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    calibration, validation = _prepare_workloads(args, model, python)
    manifest = args.result_dir / "policy_manifest.json"
    raw: dict[str, Path] = {}
    for lifecycle, (stage_name, router_mode, workload_kind) in enumerate(STAGES):
        if lifecycle == 2:
            manifest = _build_manifest(
                args, python, raw["calibration_local"], raw["calibration_remote"]
            )
        selected = calibration if workload_kind == "calibration" else validation
        raw[stage_name] = _lifecycle(
            args,
            lifecycle=lifecycle,
            stage_name=stage_name,
            router_mode=router_mode,
            workload_kind=workload_kind,
            workload=selected,
            manifest=manifest,
            hosts=hosts,
            model=model,
            python=python,
            model_revision=model_revision,
        )
    result = args.result_dir / "result.json"
    if args.node_index == 0:
        subprocess.run([
            str(python), "-m", "eval.sota_4node.analyze_tempo_pd_performance_v1",
            "--run", f"local={raw['validation_local']}",
            "--run", f"lmcache={raw['validation_remote']}",
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
