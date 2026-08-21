"""Launch two official LMCacheConnectorV1 TP4 P/D replicas per lifecycle."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--node-index", type=int, choices=range(4), required=True)
    parser.add_argument("--hosts", required=True)
    parser.add_argument("--port-slot", type=int, required=True)
    return parser.parse_args()


def _ports(slot: int, lifecycle: int) -> dict[str, int]:
    offset = slot + lifecycle * 20
    values = {
        "prefill_api": 12000 + offset,
        "decode_api": 14000 + offset,
        "proxy_http": 16000 + offset,
        "proxy_notify": 18000 + offset,
        "decoder_init": 20000 + offset,
        "decoder_alloc": 22000 + offset,
    }
    common._require(max(values.values()) + 3 < 32768, "port range exceeds 32767")
    return values


def _config_text(
    *,
    is_prefill: bool,
    prefill_host: str,
    decode_host: str,
    ports: dict[str, int],
) -> str:
    lines = [
        "chunk_size: 64",
        "local_cpu: False",
        "enable_pd: True",
        'transfer_channel: "nixl"',
        f'pd_role: "{"sender" if is_prefill else "receiver"}"',
    ]
    if is_prefill:
        lines.extend([
            f'pd_proxy_host: "{prefill_host}"',
            f"pd_proxy_port: {ports['proxy_notify']}",
        ])
    else:
        init_ports = [ports["decoder_init"] + rank for rank in range(4)]
        alloc_ports = [ports["decoder_alloc"] + rank for rank in range(4)]
        lines.extend([
            f'pd_peer_host: "{decode_host}"',
            "pd_peer_init_port: [" + ", ".join(map(str, init_ports)) + "]",
            "pd_peer_alloc_port: [" + ", ".join(map(str, alloc_ports)) + "]",
        ])
    lines.extend([
        "pd_buffer_size: 2147483648",
        'pd_buffer_device: "cuda"',
        "nixl_backends: [UCX]",
        'pd_backend_mode: "async"',
        "pd_max_prefill_len: 2048",
        "pd_allocation_timeout_sec: 30",
        "pd_shutdown_timeout_sec: 10",
        "pd_condition_poll_interval_sec: 0.001",
    ])
    return "\n".join(lines) + "\n"


def _vllm_command(
    executable: Path,
    model: Path,
    *,
    is_prefill: bool,
    mode: str,
    pair: int,
    ports: dict[str, int],
) -> list[str]:
    role = "kv_producer" if is_prefill else "kv_consumer"
    extra: dict[str, object] = {
        "discard_partial_chunks": False,
        "lmcache_rpc_port": f"{role}-{mode}-pair-{pair}",
    }
    if not is_prefill:
        extra["skip_last_n_tokens"] = 1
    import json
    connector = json.dumps({
        "kv_connector": "LMCacheConnectorV1",
        "kv_role": role,
        "engine_id": f"{mode}-pair-{pair}-{'p' if is_prefill else 'd'}",
        "kv_connector_extra_config": extra,
    }, separators=(",", ":"))
    return [
        str(executable), "serve", str(model),
        "--host", "0.0.0.0",
        "--port", str(ports["prefill_api"] if is_prefill else ports["decode_api"]),
        "--served-model-name", common.SERVED_MODEL,
        "--tensor-parallel-size", "4",
        "--distributed-executor-backend", "mp",
        "--dtype", "bfloat16",
        "--max-model-len", "2048",
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.50",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-hybrid-kv-cache-manager",
        "--no-async-scheduling",
        "--disable-log-requests",
        "--kv-transfer-config", connector,
    ]


def _proxy_command(
    python: Path,
    proxy_script: Path,
    model: Path,
    *,
    prefill_host: str,
    decode_host: str,
    ports: dict[str, int],
) -> list[str]:
    init_ports = ",".join(str(ports["decoder_init"] + rank) for rank in range(4))
    alloc_ports = ",".join(str(ports["decoder_alloc"] + rank) for rank in range(4))
    return [
        str(python), str(proxy_script),
        "--host", "0.0.0.0",
        "--port", str(ports["proxy_http"]),
        "--prefiller-host", prefill_host,
        "--prefiller-port", str(ports["prefill_api"]),
        "--num-prefillers", "1",
        "--decoder-host", decode_host,
        "--decoder-port", str(ports["decode_api"]),
        "--decoder-init-port", init_ports,
        "--decoder-alloc-port", alloc_ports,
        "--num-decoders", "1",
        "--proxy-host", prefill_host,
        "--proxy-port", str(ports["proxy_notify"]),
        "--model", str(model),
        "--pd-buffer-size", "2147483648",
        "--chunk-size", "64",
    ]


def _environment(
    base: dict[str, str],
    *,
    config: Path,
    mode: str,
    node_index: int,
) -> dict[str, str]:
    env = dict(base)
    cache = f"/tmp/tempo-live-lmcache-pd-{base['SLURM_JOB_ID']}-{mode}-n{node_index}"
    env.update({
        "LMCACHE_CONFIG_FILE": str(config),
        "LMCACHE_LOG_LEVEL": "INFO",
        "LMCACHE_DISABLE_BANNER": "1",
        "PYTHONHASHSEED": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "UCX_TLS": "cuda_ipc,cuda_copy,tcp",
        "UCX_NET_DEVICES": "all",
        "UCX_RCACHE_MAX_UNRELEASED": "1024",
        "NCCL_SOCKET_IFNAME": "hsn",
        "NCCL_IB_DISABLE": "1",
        "NCCL_CUMEM_ENABLE": "1",
        "FLASHINFER_WORKSPACE_BASE": cache + "/flashinfer",
        "TRITON_CACHE_DIR": cache + "/triton",
        "TORCHINDUCTOR_CACHE_DIR": cache + "/torchinductor",
        "VLLM_CACHE_ROOT": cache + "/vllm",
    })
    return env


def _lifecycle(
    args: argparse.Namespace,
    *,
    lifecycle: int,
    mode: str,
    hosts: list[str],
) -> None:
    repo = args.repo_root.resolve()
    model = repo / "models/TinyLlama-1.1B-Chat-v1.0"
    python = repo / ".vllm_venv/bin/python"
    vllm = repo / ".vllm_venv/bin/vllm"
    proxy_script = repo / "third_party/lmcache/examples/disagg_prefill/disagg_proxy_server.py"
    mode_dir = args.result_dir.resolve() / mode
    mode_dir.mkdir(parents=True, exist_ok=False)
    result_path = mode_dir / "result.json"
    pair = args.node_index // 2
    is_prefill = args.node_index % 2 == 0
    prefill_host = hosts[pair * 2]
    decode_host = hosts[pair * 2 + 1]
    ports = _ports(args.port_slot, lifecycle)
    config = mode_dir / f"node-{args.node_index}-lmcache.yaml"
    config.write_text(_config_text(
        is_prefill=is_prefill,
        prefill_host=prefill_host,
        decode_host=decode_host,
        ports=ports,
    ), encoding="utf-8")
    env = _environment(os.environ, config=config, mode=mode, node_index=args.node_index)
    engine = None
    proxy = None
    handles: list[object] = []
    try:
        engine, handle = common._spawn(
            _vllm_command(
                vllm, model,
                is_prefill=is_prefill,
                mode=mode,
                pair=pair,
                ports=ports,
            ),
            mode_dir / f"node-{args.node_index}-vllm.log",
            env,
        )
        handles.append(handle)
        common._wait_url(
            f"http://{hosts[args.node_index]}:{ports['prefill_api'] if is_prefill else ports['decode_api']}/health",
            [engine],
        )
        if is_prefill:
            common._wait_url(f"http://{decode_host}:{ports['decode_api']}/health", [engine])
            proxy, handle = common._spawn(
                _proxy_command(
                    python, proxy_script, model,
                    prefill_host=prefill_host,
                    decode_host=decode_host,
                    ports=ports,
                ),
                mode_dir / f"node-{args.node_index}-proxy.log",
                env,
            )
            handles.append(handle)
            common._wait_url(f"http://{prefill_host}:{ports['proxy_http']}/docs", [engine, proxy])
        if args.node_index == 0:
            proxy_urls = ",".join(
                f"http://{hosts[index]}:{ports['proxy_http']}" for index in (0, 2)
            )
            decoder_urls = ",".join(
                f"http://{hosts[index]}:{ports['decode_api']}" for index in (1, 3)
            )
            common._wait_url(proxy_urls.split(",")[1] + "/docs", [engine, proxy])
            subprocess.run([
                str(python), "-m", "eval.sota_4node.live_pd_controller_lmcache_v3", "run",
                "--mode", mode,
                "--proxy-urls", proxy_urls,
                "--decoder-urls", decoder_urls,
                "--model", str(model),
                "--output", str(result_path),
            ], cwd=repo, env=env, check=True, timeout=common.LIFECYCLE_S)
        else:
            common._wait_file(result_path, [engine] + ([proxy] if proxy else []))
    finally:
        common._stop(proxy)
        common._stop(engine)
        for handle in handles:
            handle.close()


def main() -> int:
    args = _parse()
    hosts = args.hosts.split(",")
    common._require(len(hosts) == 4 and len(set(hosts)) == 4, "four unique hosts required")
    common._require(args.repo_root.resolve() in args.result_dir.resolve().parents,
                    "result directory must be below repository")
    modes = ("lmcache_always_remote", "tempo_admission")
    for lifecycle, mode in enumerate(modes):
        _lifecycle(args, lifecycle=lifecycle, mode=mode, hosts=hosts)
    final = args.result_dir / "result.json"
    if args.node_index == 0:
        subprocess.run([
            str(args.repo_root / ".vllm_venv/bin/python"),
            "-m", "eval.sota_4node.live_pd_controller_lmcache_v3", "combine",
            "--baseline", str(args.result_dir / modes[0] / "result.json"),
            "--tempo", str(args.result_dir / modes[1] / "result.json"),
            "--output", str(final),
        ], cwd=args.repo_root, check=True, timeout=60.0)
    else:
        common._wait_file(final, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
