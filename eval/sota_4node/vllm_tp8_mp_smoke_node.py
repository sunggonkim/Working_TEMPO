from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SERVED_MODEL = "tempo-tinyllama"
READINESS_SECONDS = 180.0
REQUEST_TIMEOUT_SECONDS = 120.0
REMOTE_FINISH_SECONDS = 360.0
CHILD_TERM_SECONDS = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-bin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--api-port", type=int, required=True)
    return parser.parse_args()


def vllm_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.vllm_bin),
        "serve",
        str(args.model),
        "--served-model-name",
        SERVED_MODEL,
        "--tensor-parallel-size",
        "8",
        "--distributed-executor-backend",
        "mp",
        "--nnodes",
        "2",
        "--node-rank",
        str(args.node_rank),
        "--master-addr",
        args.master_addr,
        "--master-port",
        str(args.master_port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.50",
        "--enforce-eager",
    ]
    if args.node_rank == 0:
        command.extend(["--host", "0.0.0.0", "--port", str(args.api_port)])
    else:
        command.append("--headless")
    return command


def stop_child(child: subprocess.Popen[Any]) -> None:
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    deadline = time.monotonic() + CHILD_TERM_SECONDS
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5.0)


def wait_until_ready(child: subprocess.Popen[Any], health_url: str) -> float:
    # This is bounded localhost service readiness inside the allocation. It
    # never queries Slurm and is not a job-monitoring loop.
    started = time.monotonic()
    deadline = started + READINESS_SECONDS
    last_error = "no response"
    while time.monotonic() < deadline:
        return_code = child.poll()
        if return_code is not None:
            raise RuntimeError(f"vLLM server exited before readiness: {return_code}")
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as response:
                if response.status == 200:
                    return (time.monotonic() - started) * 1000.0
        except (OSError, urllib.error.URLError) as error:
            last_error = repr(error)
        time.sleep(0.25)
    raise TimeoutError(f"vLLM readiness exceeded 180s; last_error={last_error}")


def streaming_smoke(endpoint: str) -> dict[str, Any]:
    prompt = "In one short sentence, explain why overlapping network traffic matters."
    request_body = json.dumps(
        {
            "model": SERVED_MODEL,
            "prompt": prompt,
            "max_tokens": 16,
            "temperature": 0.0,
            "stream": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=request_body,
        headers={"Content-Type": "application/json", "Connection": "close"},
        method="POST",
    )
    started = time.monotonic()
    first_token_at: float | None = None
    chunks = 0
    output_parts: list[str] = []
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        status = response.status
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            text = event["choices"][0].get("text", "")
            if text:
                if first_token_at is None:
                    first_token_at = time.monotonic()
                output_parts.append(text)
                chunks += 1
    finished = time.monotonic()
    if status != 200 or chunks == 0 or first_token_at is None:
        raise RuntimeError(f"streaming smoke returned status={status}, chunks={chunks}")
    return {
        "endpoint": endpoint,
        "http_status": status,
        "prompt": prompt,
        "max_tokens": 16,
        "chunks": chunks,
        "output_text": "".join(output_parts),
        "ttft_ms": (first_token_at - started) * 1000.0,
        "total_ms": (finished - started) * 1000.0,
    }


def run_head(args: argparse.Namespace, child: subprocess.Popen[Any]) -> int:
    result_path = args.result_dir / "smoke_result.json"
    health_url = f"http://127.0.0.1:{args.api_port}/health"
    endpoint = f"http://127.0.0.1:{args.api_port}/v1/completions"
    result: dict[str, Any] = {
        "schema_version": 1,
        "success": False,
        "backend": "vllm-native-mp",
        "tensor_parallel_size": 8,
        "nodes": 2,
        "gpus_per_node": 4,
        "served_model": SERVED_MODEL,
        "local_model": str(args.model),
    }
    try:
        result["readiness_ms"] = wait_until_ready(child, health_url)
        result.update(streaming_smoke(endpoint))
        result["success"] = True
        return_code = 0
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return_code = 1
    finally:
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return return_code


def run_headless(args: argparse.Namespace, child: subprocess.Popen[Any]) -> int:
    result_path = args.result_dir / "smoke_result.json"
    deadline = time.monotonic() + REMOTE_FINISH_SECONDS
    # Bounded allocation-internal coordination only; no Slurm polling.
    while time.monotonic() < deadline:
        if result_path.is_file():
            return 0
        return_code = child.poll()
        if return_code is not None:
            raise RuntimeError(f"headless vLLM exited before rank 0: {return_code}")
        time.sleep(0.25)
    raise TimeoutError("rank 0 did not publish smoke_result.json within 360s")


def main() -> int:
    args = parse_args()
    child = subprocess.Popen(vllm_command(args), start_new_session=True)

    def terminate_from_signal(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate_from_signal)
    signal.signal(signal.SIGINT, terminate_from_signal)
    try:
        if args.node_rank == 0:
            return run_head(args, child)
        return run_headless(args, child)
    finally:
        stop_child(child)


if __name__ == "__main__":
    raise SystemExit(main())
