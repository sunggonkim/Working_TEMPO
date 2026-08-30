#!/usr/bin/env python3
"""Bounded native Perlmutter capability receipt for TEMPO-GO G0."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA = "tempo-go-perlmutter-native-capability-v1"
MANIFEST_SCHEMA = "tempo-go-perlmutter-native-capability-manifest-v1"
PACKAGE_NAMES = (
    "vllm",
    "lmcache",
    "torch",
    "transformers",
    "fastapi",
    "httpx",
    "pyzmq",
)
BLOCKED_ENV_EXACT = {"CRAY_ROOTFS", "SLURM_CONTAINER"}
BLOCKED_ENV_PREFIXES = ("SHIFTER", "UDI", "SLURM_SPANK_SHIFTER")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _integer_env(name: str) -> int:
    raw = os.environ.get(name)
    _require(raw is not None and raw.isdigit(), f"missing integer {name}")
    return int(raw)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _gpu_rows() -> tuple[str, list[dict[str, object]]]:
    executable = shutil.which("nvidia-smi")
    _require(executable is not None, "nvidia-smi is unavailable")
    query = "index,uuid,name,memory.total,driver_version"
    completed = subprocess.run(
        [
            executable,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        _require(len(fields) == 5, "unexpected nvidia-smi query shape")
        rows.append({
            "index": int(fields[0]),
            "uuid": fields[1],
            "name": fields[2],
            "memory_mib": int(fields[3]),
            "driver_version": fields[4],
        })
    _require(len(rows) == 4, "native rank does not see exactly four GPUs")
    return executable, rows


def _rank(args: argparse.Namespace) -> int:
    _require(os.getuid() != 0, "native capability probe refuses UID 0")
    blocked = sorted(
        name for name in os.environ
        if name in BLOCKED_ENV_EXACT
        or any(name.startswith(prefix) for prefix in BLOCKED_ENV_PREFIXES)
    )
    _require(not blocked, f"container/rootfs environment present: {blocked}")
    _require(_integer_env("SLURM_NNODES") == 4, "probe requires four nodes")
    _require(_integer_env("SLURM_NTASKS") == 4, "probe requires four ranks")
    rank = _integer_env("SLURM_PROCID")
    _require(0 <= rank < 4, "Slurm rank is out of range")
    local_rank = _integer_env("SLURM_LOCALID")
    _require(local_rank == 0, "probe requires one rank per node")
    hosts = tuple(filter(None, os.environ.get(
        "TEMPO_GO_EXPECTED_HOSTS", "").split(",")))
    _require(len(hosts) == 4 and len(set(hosts)) == 4,
             "expected host contract must contain four unique nodes")
    hostname = socket.gethostname().split(".")[0]
    _require(hostname in hosts, "rank hostname is outside allocation contract")
    resolutions = {}
    for host in hosts:
        addresses = sorted({
            item[4][0]
            for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        })
        _require(addresses, f"allocation hostname does not resolve: {host}")
        resolutions[host] = addresses
    nvidia_smi, gpus = _gpu_rows()
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    _require(model.is_dir(), "frozen Qwen2.5-7B model directory is absent")
    _require((model / "config.json").is_file(), "model config.json is absent")
    guard_sha = os.environ.get("TEMPO_PD_NATIVE_GUARD_SHA256")
    _require(
        isinstance(guard_sha, str) and len(guard_sha) == 64,
        "native guard SHA receipt is absent",
    )
    record = {
        "schema": SCHEMA,
        "native_only": True,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nnodes": 4,
        "slurm_ntasks": 4,
        "rank": rank,
        "local_rank": local_rank,
        "hostname": hostname,
        "expected_hosts": list(hosts),
        "hostname_resolution": resolutions,
        "guard_version": os.environ.get("TEMPO_PD_NATIVE_GUARD_VERSION"),
        "guard_sha256": guard_sha,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "loaded_modules": os.environ.get("LOADEDMODULES", "").split(":"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        "nvidia_smi": nvidia_smi,
        "gpus": gpus,
        "package_versions": _package_versions(),
        "model_path": str(model),
        "model_config_sha256": _sha256(model / "config.json"),
        "transport_contract": "LMCacheConnectorV1:UCX",
        "privileged_nic_control": False,
        "forbidden_environment": blocked,
    }
    output = args.output_dir / f"rank-{rank:03d}.json"
    _require(not output.exists(), f"rank receipt already exists: {output}")
    _atomic_json(output, record)
    print(output)
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    paths = [args.output_dir / f"rank-{rank:03d}.json" for rank in range(4)]
    _require(all(path.is_file() for path in paths),
             "native capability rank receipts are incomplete")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    _require(all(row.get("schema") == SCHEMA for row in rows),
             "native capability rank schema mismatch")
    _require([row.get("rank") for row in rows] == list(range(4)),
             "native capability ranks are not exact")
    hosts = [row.get("hostname") for row in rows]
    _require(len(set(hosts)) == 4, "native capability nodes are not unique")
    _require(all(row.get("uid") != 0 for row in rows),
             "native capability observed UID 0")
    _require(all(len(row.get("gpus", [])) == 4 for row in rows),
             "native capability did not observe 4 GPUs per node")
    for name in ("vllm", "lmcache", "torch"):
        versions = {row["package_versions"].get(name) for row in rows}
        _require(None not in versions and len(versions) == 1,
                 f"required package is absent or inconsistent: {name}")
    _require(len({row.get("guard_sha256") for row in rows}) == 1,
             "native guard receipt differs across ranks")
    _require(len({row.get("model_config_sha256") for row in rows}) == 1,
             "model contract differs across ranks")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "passed": True,
        "native_only": True,
        "slurm_job_id": rows[0]["slurm_job_id"],
        "nodes": hosts,
        "node_count": 4,
        "gpu_count": sum(len(row["gpus"]) for row in rows),
        "transport_contract": "LMCacheConnectorV1:UCX",
        "privileged_nic_control": False,
        "guard_sha256": rows[0]["guard_sha256"],
        "model_config_sha256": rows[0]["model_config_sha256"],
        "package_versions": rows[0]["package_versions"],
        "srun_command": os.environ.get("TEMPO_GO_PROBE_SRUN_COMMAND"),
        "rank_receipts": [
            {"path": path.name, "sha256": _sha256(path)} for path in paths
        ],
    }
    output = args.output_dir / "manifest.json"
    _require(not output.exists(), "native capability manifest already exists")
    _atomic_json(output, manifest)
    print(output)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("rank", "aggregate"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_dir = args.output_dir.resolve()
    expected_root = (
        args.repo_root / "eval/sota_4node/results/tempo_go_g0").resolve()
    _require(
        args.output_dir.parent == expected_root,
        "output directory must be one job directory under tempo_go_g0",
    )
    if args.mode == "rank":
        return _rank(args)
    return _aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
