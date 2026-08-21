#!/usr/bin/env python3
"""One verified serial remote transport prewarm, then remote-first crossover."""

from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
from eval.sota_4node import run_tempo_pd_same_server_balanced_client_v70 as balanced
from eval.sota_4node import run_tempo_pd_same_server_cache_catalog_client_v159 as remote_first


def _transport_prewarm() -> None:
    args = balanced._parse()
    rows = balanced._load_rows(args.workload)
    row = dict(rows[0])
    row["request_id"] = "ssb-remote-r0-warm-transport-prewarm"
    workload = args.output.parent / "remote_transport_prewarm.jsonl"
    output = args.output.parent / "remote_transport_prewarm.raw.json"
    if workload.exists() or output.exists():
        raise ValueError("refusing stale remote transport prewarm")
    balanced._write_jsonl(workload, [row])
    command = [
        sys.executable, "-m",
        "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_v38",
        "--base-url", args.base_url, "--model", str(args.model),
        "--served-model-name", args.served_model_name,
        "--workload", str(workload), "--output", str(output),
        "--mode", "lmcache_always_remote", "--run-id", "remote-transport-prewarm",
        "--default-max-tokens", str(args.default_max_tokens),
        "--max-workers", "1", "--timeout-s", "120", "--seed", str(args.seed),
    ]
    if args.api_key_env:
        command.extend(("--api-key-env", args.api_key_env))
    subprocess.run(command, check=True, timeout=180.0)
    value = json.loads(output.read_text(encoding="utf-8"))
    requests = value.get("requests")
    if not isinstance(requests, list) or len(requests) != 1:
        raise ValueError("serial remote transport prewarm did not complete exactly once")
    if requests[0].get("request_id") != row["request_id"]:
        raise ValueError("serial remote transport prewarm identity mismatch")


def main() -> int:
    args = balanced._parse()
    if args.run_id.endswith("-warmup"):
        _transport_prewarm()
    return remote_first.main()


if __name__ == "__main__": raise SystemExit(main())
