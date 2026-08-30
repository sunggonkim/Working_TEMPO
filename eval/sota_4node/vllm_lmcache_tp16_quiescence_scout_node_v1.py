#!/usr/bin/env python3
"""Run one node of the bounded TP16 token31 quiescence scout.

This is a compute-step entrypoint, never a scheduler launcher.  Four copies
are started node-major by one bounded srun.  Only node zero's vLLM process
receives the pinned sitecustomize hook; the sidecar and follower-node vLLM
processes explicitly receive no hook environment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from typing import Any

from eval.sota_4node import vllm_lmcache_tp16_campaign_node_v1 as base


NODES = 4
LOCAL_RANKS = 4
PAIR_COUNT = 8
MODEL_RELATIVE = Path("models/TinyLlama-1.1B-Chat-v1.0")
PLAN_RELATIVE = Path("eval/sota_4node/real_tp16_quiescence_scout_v1.json")
RUNNER_MODULE = "eval.sota_4node.run_vllm_lmcache_tp16_quiescence_scout_v1"
PINNED_SITE_RELATIVE = Path(
    "eval/sota_4node/vllm_quiescence_sitecustomize_v3_pinned"
)
EXPECTED_VLLM_DISTRIBUTION = "0.26.0+cu129"
EXPECTED_PROCESS_STEP_SHA256 = (
    "41295db73bb85ebda9cee7c4f32d944e5f973b6bcc0433ff6b152a9368b175b9"
)
HOOK_ENV_NAMES = (
    "TEMPO_VLLM_QUIESCENCE_ENABLED",
    "TEMPO_VLLM_QUIESCENCE_NODE_RANK",
    "TEMPO_VLLM_QUIESCENCE_SOCKET",
    "TEMPO_VLLM_QUIESCENCE_TRACE",
    "TEMPO_VLLM_QUIESCENCE_TOKEN_INDEX",
    "TEMPO_VLLM_QUIESCENCE_TIMEOUT_S",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--campaign-index", type=int, choices=range(3), required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--vllm-master-port", type=base._tcp_port, required=True)
    parser.add_argument("--sidecar-master-port", type=base._tcp_port, required=True)
    parser.add_argument("--api-port", type=base._tcp_port, required=True)
    parser.add_argument("--nixl-port-base", type=base._tcp_port, required=True)
    parser.add_argument("--quiescence-socket", type=Path, required=True)
    parser.add_argument("--quiescence-trace", type=Path, required=True)
    parser.add_argument(
        "--readiness-timeout-s", type=base._positive_timeout, default=600.0
    )
    parser.add_argument(
        "--sidecar-timeout-s", type=base._positive_timeout, default=1100.0
    )
    args = parser.parse_args()
    if args.nixl_port_base + PAIR_COUNT - 1 > 65535:
        parser.error("nixl-port-base must leave eight valid TCP ports")
    return args


def _tmp_child(path: Path, *, prefix: str, suffix: str, label: str) -> Path:
    if not path.is_absolute() or path.parent != Path("/tmp"):
        raise ValueError(f"{label} must be an immediate child of /tmp")
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError(f"{label} has an invalid frozen name")
    if len(os.fsencode(path)) > 100:
        raise ValueError(f"{label} is too long for an AF_UNIX endpoint")
    return path


def _hook_preflight(result_dir: Path) -> dict[str, str]:
    from eval.sota_4node.vllm_decode_quiescence_gate_launch_v3_hardening import (
        validate_engine_core_compatibility,
    )
    from vllm.v1.engine.core import EngineCoreProc

    identity = validate_engine_core_compatibility(EngineCoreProc)
    if identity != {
        "vllm_version": EXPECTED_VLLM_DISTRIBUTION,
        "engine_core_process_step_sha256": EXPECTED_PROCESS_STEP_SHA256,
    }:
        raise RuntimeError("pinned vLLM preflight identity changed")
    path = result_dir / "hook-preflight-node-0.json"
    if path.exists():
        raise RuntimeError(f"refusing stale hook preflight: {path}")
    path.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")
    return identity


def _trace_provenance(trace_path: Path) -> dict[str, Any]:
    if not trace_path.is_file():
        raise RuntimeError(f"hook trace was not created before sidecar: {trace_path}")
    size = trace_path.stat().st_size
    if not 0 < size <= 65_536:
        raise RuntimeError("hook provenance trace has an invalid bounded size")
    first_line = trace_path.read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(first_line)
    expected = {
        "kind": "provenance",
        "protocol": "tempo-vllm-output-quiescence-3",
        "node_rank": 0,
        "world_size": 16,
        "tensor_parallel_size": 16,
        "async_scheduling": False,
        "speculative_decoding": False,
        "output_token_index_zero_based": 30,
        "generated_token_count_one_based": 31,
        "request_id_marker": "tempo-scout-",
        "vllm_version": EXPECTED_VLLM_DISTRIBUTION,
        "engine_core_process_step_sha256": EXPECTED_PROCESS_STEP_SHA256,
    }
    if not isinstance(payload, dict) or any(payload.get(k) != v for k, v in expected.items()):
        raise RuntimeError("hook trace provenance changed")
    return payload


def _scrub_hook_environment(environment: dict[str, str]) -> None:
    for name in HOOK_ENV_NAMES:
        environment.pop(name, None)


def _copy_trace(trace_path: Path, result_dir: Path) -> Path:
    destination = result_dir / "vllm-quiescence-trace-node-0.jsonl"
    if destination.exists():
        raise RuntimeError(f"refusing stale copied hook trace: {destination}")
    if not trace_path.is_file() or trace_path.stat().st_size <= 0:
        raise RuntimeError("node-zero hook trace is missing at teardown")
    shutil.copyfile(trace_path, destination)
    return destination


def main() -> None:
    args = _parse_args()
    job_id, node_id = base._node_environment()
    repo_root = args.repo_root.resolve(strict=True)
    result_dir = base._require_path_below(args.result_dir, repo_root, label="result-dir")
    result_path = result_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(f"refusing to overwrite stale result: {result_path}")

    gate_socket = _tmp_child(
        args.quiescence_socket,
        prefix="tempo-vllm-quiescence-",
        suffix=".sock",
        label="quiescence-socket",
    )
    gate_trace = _tmp_child(
        args.quiescence_trace,
        prefix="tempo-step-gate-",
        suffix=".jsonl",
        label="quiescence-trace",
    )
    if node_id == 0 and (gate_socket.exists() or gate_trace.exists()):
        raise RuntimeError("refusing stale node-zero quiescence socket/trace")

    vllm_binary = repo_root / ".vllm_venv/bin/vllm"
    torchrun_binary = repo_root / ".vllm_venv/bin/torchrun"
    model_dir = repo_root / MODEL_RELATIVE
    plan_path = repo_root / PLAN_RELATIVE
    pinned_site = repo_root / PINNED_SITE_RELATIVE
    for required in (
        vllm_binary,
        torchrun_binary,
        model_dir / "config.json",
        model_dir / "model.safetensors",
        plan_path,
        pinned_site / "sitecustomize.py",
    ):
        if not required.exists():
            raise RuntimeError(f"required local path is missing: {required}")
    if not os.access(vllm_binary, os.X_OK) or not os.access(torchrun_binary, os.X_OK):
        raise RuntimeError("vLLM environment entrypoints must be executable")

    if node_id == 0:
        _hook_preflight(result_dir)

    node_cache = Path(f"/tmp/tempo-vllm-quiescence-{job_id}-node{node_id}")
    for child in ("flashinfer", "huggingface", "torch-extensions", "triton"):
        (node_cache / child).mkdir(parents=True, exist_ok=True)
    common_environment = os.environ.copy()
    _scrub_hook_environment(common_environment)
    common_environment.update(
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
    sidecar_environment = common_environment.copy()
    vllm_environment = common_environment.copy()
    if node_id == 0:
        vllm_environment.update(
            {
                "PYTHONPATH": f"{pinned_site}{os.pathsep}{repo_root}",
                "TEMPO_VLLM_QUIESCENCE_ENABLED": "YES",
                "TEMPO_VLLM_QUIESCENCE_NODE_RANK": "0",
                "TEMPO_VLLM_QUIESCENCE_SOCKET": str(gate_socket),
                "TEMPO_VLLM_QUIESCENCE_TRACE": str(gate_trace),
                "TEMPO_VLLM_QUIESCENCE_TOKEN_INDEX": "30",
                "TEMPO_VLLM_QUIESCENCE_TIMEOUT_S": "10",
            }
        )

    vllm_command = [
        str(vllm_binary),
        "serve",
        str(model_dir),
        "--tensor-parallel-size",
        "16",
        "--distributed-executor-backend",
        "mp",
        "--nnodes",
        "4",
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
        "1",
        "--gpu-memory-utilization",
        "0.50",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
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
        "--quiescence-socket",
        str(gate_socket),
        "--quiescence-trace",
        str(gate_trace),
    ]
    base._command_record(
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

    signal.signal(signal.SIGTERM, base._signal_handler)
    signal.signal(signal.SIGINT, base._signal_handler)
    vllm_process: subprocess.Popen[Any] | None = None
    sidecar_process: subprocess.Popen[Any] | None = None
    copied_trace: Path | None = None
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
                env=vllm_environment,
                stdout=vllm_stdout,
                stderr=vllm_stderr,
                start_new_session=True,
            )
            base._wait_for_health(
                api_host=args.master_addr,
                api_port=args.api_port,
                process=vllm_process,
                timeout_s=args.readiness_timeout_s,
            )
            if node_id == 0:
                _trace_provenance(gate_trace)
            sidecar_process = subprocess.Popen(
                sidecar_command,
                cwd=repo_root,
                env=sidecar_environment,
                stdout=sidecar_stdout,
                stderr=sidecar_stderr,
                start_new_session=True,
            )
            try:
                return_code = sidecar_process.wait(timeout=args.sidecar_timeout_s)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"TP16 quiescence sidecar exceeded {args.sidecar_timeout_s}s"
                ) from exc
            if return_code != 0:
                raise RuntimeError(f"TP16 quiescence sidecar exited with status {return_code}")
            base._wait_for_result(result_path, vllm_process=vllm_process)
        finally:
            if sidecar_process is not None:
                base._terminate_process_group(sidecar_process, label="sidecar", grace_s=5.0)
            if vllm_process is not None:
                base._terminate_process_group(vllm_process, label="vLLM", grace_s=10.0)
            if node_id == 0 and gate_trace.is_file():
                copied_trace = _copy_trace(gate_trace, result_dir)
    if node_id == 0 and copied_trace is None:
        raise RuntimeError("node-zero hook trace was not copied after teardown")


if __name__ == "__main__":
    main()
