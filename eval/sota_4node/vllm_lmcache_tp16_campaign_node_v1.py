#!/usr/bin/env python3
"""Run one node of a bounded four-node vLLM TP16/LMCache campaign.

This is a compute-step entrypoint, not a scheduler launcher.  Four copies are
started node-major by the allocation launcher.  Each copy owns one native
vLLM multiprocess server process group and one four-rank torchrun process
group, and tears both groups down before it exits.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Sequence
import urllib.error
import urllib.request


NODES = 4
LOCAL_RANKS = 4
WORLD_SIZE = NODES * LOCAL_RANKS
PAIR_COUNT = 8
MODEL_RELATIVE = Path("models/TinyLlama-1.1B-Chat-v1.0")
PLAN_RELATIVE = Path("eval/sota_4node/real_tp16_pair_stagger_coalesced_v2.json")
RUNNER_MODULE = "eval.sota_4node.run_vllm_lmcache_tp16_pair_stagger_coalesced_v2"


class ShutdownRequested(RuntimeError):
    """Raised when the enclosing bounded compute step asks us to stop."""


def _positive_timeout(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return parsed


def _tcp_port(value: str) -> int:
    parsed = int(value)
    if not 1024 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("port must be in 1024..65535")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--campaign-index", type=int, choices=range(3), required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--vllm-master-port", type=_tcp_port, required=True)
    parser.add_argument("--sidecar-master-port", type=_tcp_port, required=True)
    parser.add_argument("--api-port", type=_tcp_port, required=True)
    parser.add_argument("--nixl-port-base", type=_tcp_port, required=True)
    parser.add_argument(
        "--readiness-timeout-s", type=_positive_timeout, default=600.0
    )
    parser.add_argument("--sidecar-timeout-s", type=_positive_timeout, default=1100.0)
    args = parser.parse_args()
    if args.nixl_port_base + PAIR_COUNT - 1 > 65535:
        parser.error("nixl-port-base must leave eight valid TCP ports")
    return args


def _require_path_below(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"{label} must be below the repository root")
    return resolved


def _node_environment() -> tuple[str, int]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    node_id_text = os.environ.get("SLURM_NODEID", "")
    if not job_id.isdigit():
        raise RuntimeError("SLURM_JOB_ID must identify an existing allocation")
    if os.environ.get("SLURM_JOB_NUM_NODES") != str(NODES):
        raise RuntimeError("this campaign requires an existing four-node allocation")
    if not node_id_text.isdigit() or int(node_id_text) not in range(NODES):
        raise RuntimeError("SLURM_NODEID must be 0, 1, 2, or 3")
    return job_id, int(node_id_text)


def _command_record(
    *,
    result_dir: Path,
    node_id: int,
    job_id: str,
    campaign_index: int,
    vllm_command: Sequence[str],
    sidecar_command: Sequence[str],
    ports: dict[str, int],
) -> None:
    record = {
        "schema_version": "tempo-vllm-tp16-campaign-launch-record-1",
        "allocation_id": job_id,
        "node_id": node_id,
        "campaign_index": campaign_index,
        "ports": ports,
        "vllm_command": list(vllm_command),
        "sidecar_command": list(sidecar_command),
        "bounds": {
            "vllm_health_readiness_s": 600,
            "sidecar_s": 1100,
        },
    }
    path = result_dir / f"launch-node-{node_id}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(
    process: subprocess.Popen[Any], *, label: str, grace_s: float
) -> None:
    """TERM a dedicated child process group, then KILL it after a deadline."""

    process_group = process.pid
    if not _process_group_exists(process_group):
        process.poll()
        return
    print(f"stopping {label} process group {process_group}", file=sys.stderr, flush=True)
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        process.poll()
        return

    deadline = time.monotonic() + grace_s
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        if process.poll() is None:
            try:
                process.wait(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))

    if _process_group_exists(process_group):
        print(f"killing {label} process group {process_group}", file=sys.stderr, flush=True)
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        print(f"{label} leader did not reap after SIGKILL", file=sys.stderr, flush=True)


def _wait_for_health(
    *, api_host: str, api_port: int, process: subprocess.Popen[Any], timeout_s: float
) -> None:
    url = f"http://{api_host}:{api_port}/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_s
    last_error = "no response"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"local vLLM process exited before readiness: {return_code}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            with opener.open(url, timeout=min(2.0, remaining)) as response:
                if 200 <= int(response.status) < 300:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(2.0, remaining))
    raise TimeoutError(f"vLLM health endpoint was not ready within {timeout_s}s: {last_error}")


def _wait_for_result(
    result_path: Path, *, vllm_process: subprocess.Popen[Any], timeout_s: float = 30.0
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if result_path.is_file() and result_path.stat().st_size > 0:
            return
        return_code = vllm_process.poll()
        if return_code is not None:
            raise RuntimeError(f"local vLLM exited while waiting for result: {return_code}")
        time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))
    raise RuntimeError(f"sidecar completed without a nonempty result: {result_path}")


def _signal_handler(signum: int, _frame: object) -> None:
    raise ShutdownRequested(f"received signal {signum}")


def main() -> None:
    args = _parse_args()
    job_id, node_id = _node_environment()

    repo_root = args.repo_root.resolve(strict=True)
    result_dir = _require_path_below(args.result_dir, repo_root, label="result-dir")
    result_path = result_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(f"refusing to overwrite stale result: {result_path}")

    vllm_binary = repo_root / ".vllm_venv/bin/vllm"
    torchrun_binary = repo_root / ".vllm_venv/bin/torchrun"
    model_dir = repo_root / MODEL_RELATIVE
    plan_path = repo_root / PLAN_RELATIVE
    for required in (
        vllm_binary,
        torchrun_binary,
        model_dir / "config.json",
        model_dir / "model.safetensors",
        plan_path,
    ):
        if not required.exists():
            raise RuntimeError(f"required local path is missing: {required}")
    if not os.access(vllm_binary, os.X_OK) or not os.access(torchrun_binary, os.X_OK):
        raise RuntimeError("vLLM environment entrypoints must be executable")

    node_cache = Path(f"/tmp/tempo-vllm-tp16-{job_id}-node{node_id}")
    for child in ("flashinfer", "huggingface", "torch-extensions", "triton"):
        (node_cache / child).mkdir(parents=True, exist_ok=True)
    child_environment = os.environ.copy()
    child_environment.update(
        {
            "FLASHINFER_WORKSPACE_BASE": str(node_cache / "flashinfer"),
            "HF_HOME": str(node_cache / "huggingface"),
            "TORCH_EXTENSIONS_DIR": str(node_cache / "torch-extensions"),
            "TRITON_CACHE_DIR": str(node_cache / "triton"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONPATH": str(repo_root),
        }
    )

    vllm_command = [
        str(vllm_binary),
        "serve",
        str(model_dir),
        "--tensor-parallel-size",
        str(WORLD_SIZE),
        "--distributed-executor-backend",
        "mp",
        "--nnodes",
        str(NODES),
        "--node-rank",
        str(node_id),
        "--master-addr",
        args.master_addr,
        "--master-port",
        str(args.vllm_master_port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.50",
        "--enforce-eager",
        "--no-enable-prefix-caching",
    ]
    if node_id == 0:
        vllm_command.extend(["--host", "0.0.0.0", "--port", str(args.api_port)])
    else:
        vllm_command.append("--headless")

    sidecar_command = [
        str(torchrun_binary),
        f"--nnodes={NODES}",
        f"--nproc-per-node={LOCAL_RANKS}",
        f"--node-rank={node_id}",
        f"--master-addr={args.master_addr}",
        f"--master-port={args.sidecar_master_port}",
        "--max-restarts=0",
        "-m",
        RUNNER_MODULE,
        "--output-dir",
        str(result_dir),
        "--plan",
        str(plan_path),
        "--api-host",
        args.master_addr,
        "--api-port",
        str(args.api_port),
        "--model",
        str(model_dir),
        "--nixl-port-base",
        str(args.nixl_port_base),
        "--request-timeout-s",
        "180",
        "--campaign-index",
        str(args.campaign_index),
        "--allocation-id",
        job_id,
    ]
    _command_record(
        result_dir=result_dir,
        node_id=node_id,
        job_id=job_id,
        campaign_index=args.campaign_index,
        vllm_command=vllm_command,
        sidecar_command=sidecar_command,
        ports={
            "vllm_master": args.vllm_master_port,
            "sidecar_master": args.sidecar_master_port,
            "api": args.api_port,
            "nixl_base": args.nixl_port_base,
        },
    )

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    vllm_process: subprocess.Popen[Any] | None = None
    sidecar_process: subprocess.Popen[Any] | None = None
    with (
        (result_dir / f"vllm-node-{node_id}.stdout.log").open("w", encoding="utf-8") as vllm_stdout,
        (result_dir / f"vllm-node-{node_id}.stderr.log").open("w", encoding="utf-8") as vllm_stderr,
        (result_dir / f"sidecar-node-{node_id}.stdout.log").open("w", encoding="utf-8") as sidecar_stdout,
        (result_dir / f"sidecar-node-{node_id}.stderr.log").open("w", encoding="utf-8") as sidecar_stderr,
    ):
        try:
            vllm_process = subprocess.Popen(
                vllm_command,
                cwd=repo_root,
                env=child_environment,
                stdout=vllm_stdout,
                stderr=vllm_stderr,
                start_new_session=True,
            )
            _wait_for_health(
                api_host=args.master_addr,
                api_port=args.api_port,
                process=vllm_process,
                timeout_s=args.readiness_timeout_s,
            )
            sidecar_process = subprocess.Popen(
                sidecar_command,
                cwd=repo_root,
                env=child_environment,
                stdout=sidecar_stdout,
                stderr=sidecar_stderr,
                start_new_session=True,
            )
            try:
                sidecar_return_code = sidecar_process.wait(timeout=args.sidecar_timeout_s)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"TP16 sidecar exceeded {args.sidecar_timeout_s}s"
                ) from exc
            if sidecar_return_code != 0:
                raise RuntimeError(f"TP16 sidecar exited with status {sidecar_return_code}")
            _wait_for_result(result_path, vllm_process=vllm_process)
        finally:
            if sidecar_process is not None:
                _terminate_process_group(sidecar_process, label="sidecar", grace_s=5.0)
            if vllm_process is not None:
                _terminate_process_group(vllm_process, label="vLLM", grace_s=10.0)


if __name__ == "__main__":
    main()
