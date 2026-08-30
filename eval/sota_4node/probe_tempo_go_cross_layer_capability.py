#!/usr/bin/env python3
"""Bounded native capability receipt for TEMPO-GO cross-layer evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


CORE = (
    "hni_rx_paused_0",
    "hni_tx_paused_0",
    "parbs_tarb_pi_posted_blocked_cnt",
    "parbs_tarb_pi_posted_pkts",
)
OPTIONAL = (
    "hni_pkts_sent_by_tc_0",
    "hni_pkts_recv_by_tc_0",
    "parbs_tarb_pi_non_posted_blocked_cnt",
    "parbs_tarb_pi_non_posted_pkts",
    "lpe_net_match_priority_0",
    "lpe_net_match_overflow_0",
    "pct_retry_srb_requests",
    "pct_sct_timeouts",
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _command(command: list[str]) -> tuple[int, str]:
    try:
        value = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}:{exc}"
    return value.returncode, value.stdout.strip() or value.stderr.strip()


def probe() -> dict[str, object]:
    root = Path(os.environ.get("TEMPO_CASSINI_ROOT", "/sys/class/cxi"))
    nic_rows = []
    for nic in range(4):
        telemetry = root / f"cxi{nic}" / "device" / "telemetry"
        core = {
            name: (telemetry / name).is_file()
            for name in CORE
        }
        optional = {
            name: (telemetry / name).is_file()
            for name in OPTIONAL
        }
        nic_rows.append({
            "nic_index": nic,
            "telemetry_root": str(telemetry),
            "core": core,
            "optional": optional,
            "core_supported": all(core.values()),
        })
    nvidia_rc, nvidia = _command(["nvidia-smi", "-L"])
    python_probe = (
        "import json, torch; "
        "import torch.distributed as dist; "
        "print(json.dumps({"
        "'cuda_available': bool(torch.cuda.is_available()),"
        "'cuda_device_count': int(torch.cuda.device_count()),"
        "'nccl_available': bool(dist.is_nccl_available()),"
        "'torch_cuda': torch.version.cuda,"
        "'torch_version': torch.__version__}))"
    )
    torch_rc, torch_value = _command([
        os.environ.get("TEMPO_PYTHON", ".vllm_venv/bin/python"), "-c", python_probe,
    ])
    try:
        torch_info = json.loads(torch_value) if torch_rc == 0 else {
            "error": torch_value,
        }
    except json.JSONDecodeError:
        torch_info = {"error": torch_value}
    return {
        "schema": "tempo-go-cross-layer-capability-v1",
        "native_only": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": socket.gethostname(),
        "uid": os.getuid(),
        "sampled_ns": time.perf_counter_ns(),
        "cassini": {
            "root": str(root),
            "nic_count": 4,
            "traffic_class_count": 8,
            "topology_fingerprint_sha256": _sha("perlmutter-cassini-v2:nic4:tc8"),
            "nics": nic_rows,
        },
        "nvidia_smi": {"returncode": nvidia_rc, "output": nvidia},
        "torch_nccl": {"returncode": torch_rc, **torch_info},
        "observer_source": {
            "cuda_collective_observer": (
                Path("eval/sota_4node/train.py").is_file()),
            "official_lmcache_nixl_contention_harness": (
                Path("eval/sota_4node/run_lmcache_nixl_contention_2node.py").is_file()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = probe()
    encoded = json.dumps(value, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
