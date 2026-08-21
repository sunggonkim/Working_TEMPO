"""One-node entry for two fresh 2xTP8 live-P/D comparison lifecycles."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SERVED_MODEL = "tempo-tinyllama-live-pd"
READINESS_S = 600.0
LIFECYCLE_S = 900.0
TERM_S = 20.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--node-index", type=int, choices=range(4), required=True)
    parser.add_argument("--hosts", required=True)
    parser.add_argument("--port-slot", type=int, required=True)
    return parser.parse_args()


def _stop(child: subprocess.Popen[Any] | None) -> None:
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    deadline = time.monotonic() + TERM_S
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5.0)


def _wait_url(url: str, children: list[subprocess.Popen[Any]]) -> None:
    deadline = time.monotonic() + READINESS_S
    last = "no response"
    while time.monotonic() < deadline:
        for child in children:
            code = child.poll()
            if code is not None:
                raise RuntimeError(f"child exited before readiness: {code}")
        try:
            with urllib.request.urlopen(url, timeout=0.75) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as error:
            last = repr(error)
        time.sleep(0.25)
    raise TimeoutError(f"readiness timeout for {url}: {last}")


def _wait_file(path: Path, children: list[subprocess.Popen[Any]]) -> None:
    deadline = time.monotonic() + LIFECYCLE_S
    while time.monotonic() < deadline:
        if path.is_file():
            return
        for child in children:
            code = child.poll()
            if code is not None:
                raise RuntimeError(f"child exited before {path.name}: {code}")
        time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for {path}")


def _ports(slot: int, lifecycle: int) -> dict[str, int]:
    offset = slot + lifecycle * 20
    values = {
        "prefill_lmcache": 12000 + offset,
        "decode_lmcache": 14000 + offset,
        "lmcache_http": 16000 + offset,
        "prefill_master": 18000 + offset,
        "decode_master": 20000 + offset,
        "prefill_api": 22000 + offset,
        "decode_api": 24000 + offset,
        "prefill_nixl": 26000 + offset,
        "decode_nixl": 28000 + offset,
    }
    _require(max(values.values()) < 32768, "port plan exceeds low-port range")
    _require(len(set(values.values())) == len(values), "port collision")
    return values


def _lmcache_command(
    executable: Path,
    *,
    node_index: int,
    mode: str,
    ports: dict[str, int],
) -> list[str]:
    is_prefill = node_index < 2
    return [
        str(executable), "server",
        "--instance-id", f"{mode}-node-{node_index}",
        "--host", "0.0.0.0",
        "--port", str(ports["prefill_lmcache"] if is_prefill else ports["decode_lmcache"]),
        "--http-host", "0.0.0.0",
        "--http-port", str(ports["lmcache_http"]),
        "--prometheus-port", str(ports["lmcache_http"] + 1),
        "--l1-size-gb", "4",
        "--l1-init-size-gb", "1",
        "--eviction-policy", "LRU",
        "--chunk-size", "64",
        "--max-workers", "4",
        "--metrics-sample-rate", "1.0",
        "--enable-extra-logging",
        "--extra-logging-interval", "1.0",
    ]


def _kv_config(
    *,
    is_prefill: bool,
    mode: str,
    hosts: list[str],
    ports: dict[str, int],
) -> str:
    role = "kv_producer" if is_prefill else "kv_consumer"
    pair_hosts = hosts[:2] if is_prefill else hosts[2:]
    server_port = ports["prefill_lmcache"] if is_prefill else ports["decode_lmcache"]
    engine = f"{mode}-{'prefill' if is_prefill else 'decode'}"
    value = {
        "kv_connector": "MultiConnector",
        "kv_role": role,
        "engine_id": engine,
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "NixlConnector",
                    "kv_role": role,
                    "engine_id": engine,
                    "kv_load_failure_policy": "fail",
                    "kv_connector_extra_config": {"backends": ["UCX"]},
                },
                {
                    "kv_connector": "LMCacheMPConnector",
                    "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
                    "kv_role": "kv_both",
                    "engine_id": engine,
                    "kv_connector_extra_config": {
                        "lmcache.mp.server_urls": [
                            f"tcp://{host}:{server_port}" for host in pair_hosts
                        ],
                        "lmcache.mp.mq_timeout": 30.0,
                    },
                },
            ]
        },
    }
    return json.dumps(value, separators=(",", ":"))


def _vllm_command(
    executable: Path,
    model: Path,
    *,
    node_index: int,
    mode: str,
    hosts: list[str],
    ports: dict[str, int],
) -> list[str]:
    is_prefill = node_index < 2
    node_rank = node_index if is_prefill else node_index - 2
    head_index = 0 if is_prefill else 2
    api_port = ports["prefill_api"] if is_prefill else ports["decode_api"]
    master_port = ports["prefill_master"] if is_prefill else ports["decode_master"]
    command = [
        str(executable), "serve", str(model),
        "--served-model-name", SERVED_MODEL,
        "--tensor-parallel-size", "8",
        "--distributed-executor-backend", "mp",
        "--nnodes", "2",
        "--node-rank", str(node_rank),
        "--master-addr", hosts[head_index],
        "--master-port", str(master_port),
        "--dtype", "bfloat16",
        "--max-model-len", "2048",
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.50",
        "--enforce-eager",
        "--enable-prefix-caching",
        "--disable-hybrid-kv-cache-manager",
        "--no-async-scheduling",
        "--disable-log-requests",
        "--kv-transfer-config", _kv_config(
            is_prefill=is_prefill, mode=mode, hosts=hosts, ports=ports
        ),
    ]
    if node_index == head_index:
        command.extend(["--host", "0.0.0.0", "--port", str(api_port)])
    else:
        command.append("--headless")
    return command


def _child_env(
    base: dict[str, str],
    *,
    node_index: int,
    is_prefill: bool,
    host: str,
    ports: dict[str, int],
    mode: str,
) -> dict[str, str]:
    env = dict(base)
    cache_root = f"/tmp/tempo-live-pd-{base['SLURM_JOB_ID']}-{mode}-n{node_index}"
    env.update({
        "PYTHONHASHSEED": "123",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_NIXL_SIDE_CHANNEL_HOST": host,
        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(
            ports["prefill_nixl"] if is_prefill else ports["decode_nixl"]
        ),
        "VLLM_NIXL_BACKENDS": "UCX",
        "UCX_TLS": "cuda_ipc,cuda_copy,tcp",
        "UCX_NET_DEVICES": "all",
        "UCX_RCACHE_MAX_UNRELEASED": "1024",
        "NCCL_CUMEM_ENABLE": "1",
        "NCCL_SOCKET_IFNAME": "hsn",
        "NCCL_IB_DISABLE": "1",
        "FLASHINFER_WORKSPACE_BASE": cache_root + "/flashinfer",
        "TRITON_CACHE_DIR": cache_root + "/triton",
        "TORCHINDUCTOR_CACHE_DIR": cache_root + "/torchinductor",
        "VLLM_CACHE_ROOT": cache_root + "/vllm",
    })
    return env


def _spawn(command: list[str], log_path: Path, env: dict[str, str]) -> tuple[subprocess.Popen[Any], Any]:
    handle = log_path.open("xb")
    child = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    return child, handle


def _run_lifecycle(
    args: argparse.Namespace,
    *,
    lifecycle: int,
    mode: str,
    hosts: list[str],
) -> None:
    repo = args.repo_root.resolve()
    result_dir = args.result_dir.resolve()
    model = repo / "models/TinyLlama-1.1B-Chat-v1.0"
    vllm_bin = repo / ".vllm_venv/bin/vllm"
    lmcache_bin = repo / ".vllm_venv/bin/lmcache"
    python_bin = repo / ".vllm_venv/bin/python"
    _require((model / "config.json").is_file(), "local model missing")
    _require(vllm_bin.is_file() and lmcache_bin.is_file(), "runtime executable missing")
    ports = _ports(args.port_slot, lifecycle)
    local_host = hosts[args.node_index]
    is_prefill = args.node_index < 2
    mode_dir = result_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=False)
    result_path = mode_dir / "result.json"
    env = _child_env(
        os.environ,
        node_index=args.node_index,
        is_prefill=is_prefill,
        host=local_host,
        ports=ports,
        mode=mode,
    )
    lmcache: subprocess.Popen[Any] | None = None
    vllm: subprocess.Popen[Any] | None = None
    handles: list[Any] = []
    try:
        lmcache, handle = _spawn(
            _lmcache_command(
                lmcache_bin, node_index=args.node_index, mode=mode, ports=ports
            ),
            mode_dir / f"node-{args.node_index}-lmcache.log",
            env,
        )
        handles.append(handle)
        _wait_url(f"http://127.0.0.1:{ports['lmcache_http']}/status", [lmcache])
        vllm, handle = _spawn(
            _vllm_command(
                vllm_bin, model,
                node_index=args.node_index,
                mode=mode,
                hosts=hosts,
                ports=ports,
            ),
            mode_dir / f"node-{args.node_index}-vllm.log",
            env,
        )
        handles.append(handle)
        if args.node_index == 0:
            children = [lmcache, vllm]
            prefill_url = f"http://{hosts[0]}:{ports['prefill_api']}"
            decode_url = f"http://{hosts[2]}:{ports['decode_api']}"
            _wait_url(prefill_url + "/health", children)
            _wait_url(decode_url + "/health", children)
            command = [
                str(python_bin), "-m", "eval.sota_4node.live_pd_controller_v1", "run",
                "--mode", mode,
                "--prefill-url", prefill_url,
                "--decode-url", decode_url,
                "--model", str(model),
                "--output", str(result_path),
            ]
            subprocess.run(command, cwd=repo, env=env, check=True, timeout=LIFECYCLE_S)
        else:
            _wait_file(result_path, [lmcache, vllm])
    finally:
        _stop(vllm)
        _stop(lmcache)
        for handle in handles:
            handle.close()


def main() -> int:
    args = _args()
    hosts = args.hosts.split(",")
    _require(len(hosts) == 4 and len(set(hosts)) == 4, "hosts must name four unique nodes")
    _require(args.repo_root.resolve() in args.result_dir.resolve().parents,
             "result directory must be below repository")
    modes = ("lmcache_always_remote", "tempo_admission")
    for lifecycle, mode in enumerate(modes):
        _run_lifecycle(args, lifecycle=lifecycle, mode=mode, hosts=hosts)
    if args.node_index == 0:
        subprocess.run(
            [
                str(args.repo_root / ".vllm_venv/bin/python"),
                "-m", "eval.sota_4node.live_pd_controller_v1", "combine",
                "--baseline", str(args.result_dir / modes[0] / "result.json"),
                "--tempo", str(args.result_dir / modes[1] / "result.json"),
                "--output", str(args.result_dir / "result.json"),
            ],
            cwd=args.repo_root,
            check=True,
            timeout=60.0,
        )
    else:
        _wait_file(args.result_dir / "result.json", [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
