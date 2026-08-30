#!/usr/bin/env python3
"""Warm cold-PD transport once without imposing the measured nonce contract."""

from __future__ import annotations

import json
import subprocess
import sys

from eval.sota_4node import run_tempo_pd_same_server_balanced_client_v70 as balanced
from eval.sota_4node import run_tempo_pd_stream_metrics_forced_drain_v38 as direct


def _prewarm(args) -> None:
    rows = [json.loads(line) for line in args.workload.read_text(encoding="utf-8").splitlines()]
    if not rows or not isinstance(rows[0].get("prompt"), str):
        raise ValueError("cold transport prewarm workload malformed")
    row = dict(rows[0])
    row["request_id"] = "ssb-remote-r0-warm-transport-prewarm"
    workload = args.output.parent / "cold_remote_transport_prewarm.jsonl"
    output = args.output.parent / "cold_remote_transport_prewarm.raw.json"
    if workload.exists() or output.exists():
        raise ValueError("stale cold transport prewarm")
    balanced._write_jsonl(workload, [row])
    command = [
        sys.executable, "-m",
        "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_v38",
        "--base-url", args.base_url,
        "--model", str(args.model),
        "--served-model-name", args.served_model_name,
        "--workload", str(workload),
        "--output", str(output),
        "--mode", "lmcache_always_remote",
        "--run-id", "cold-remote-transport-prewarm",
        "--default-max-tokens", str(args.default_max_tokens),
        "--max-workers", "1",
        "--timeout-s", "120",
        "--seed", str(args.seed),
    ]
    if args.api_key_env:
        command.extend(("--api-key-env", args.api_key_env))
    subprocess.run(command, check=True, timeout=180.0)
    value = json.loads(output.read_text(encoding="utf-8"))
    requests = value.get("requests")
    if not isinstance(requests, list) or len(requests) != 1:
        raise ValueError("cold remote transport prewarm failed")
    if requests[0].get("request_id") != row["request_id"]:
        raise ValueError("cold remote transport prewarm identity mismatch")


def main() -> int:
    args = balanced._parse()
    if args.run_id.endswith("-warmup"):
        _prewarm(args)
        rows = [json.loads(line) for line in
                args.workload.read_text(encoding="utf-8").splitlines()]
        for index, row in enumerate(rows):
            row["request_id"] = f"ssb-local-r0-warm-control-{index}"
        routed_workload = args.output.parent / "cold_controller_warmup.jsonl"
        if routed_workload.exists():
            raise ValueError("stale cold controller warmup")
        balanced._write_jsonl(routed_workload, rows)
        argv = list(sys.argv)
        argv[argv.index("--workload") + 1] = str(routed_workload)
        argv[argv.index("--mode") + 1] = "fixed_local"
        original_argv = sys.argv
        sys.argv = argv
        try:
            return direct.main()
        finally:
            sys.argv = original_argv
    return balanced.main()


if __name__ == "__main__":
    raise SystemExit(main())
