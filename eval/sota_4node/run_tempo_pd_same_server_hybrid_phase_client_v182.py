#!/usr/bin/env python3
"""Add one cold MISS block before the existing warm-seed/hit lifecycle."""

from __future__ import annotations

import json
import subprocess
import sys

from eval.sota_4node import run_tempo_pd_same_server_balanced_client_v70 as balanced
from eval.sota_4node import run_tempo_pd_same_server_cache_catalog_client_v163 as production


_ORIGINAL_TRANSPORT_PREWARM = production._transport_prewarm


def _transport_then_cold() -> None:
    _ORIGINAL_TRANSPORT_PREWARM()
    args = balanced._parse()
    rows = balanced._load_rows(args.workload)
    derived = balanced._derive(
        rows, prefix="ssb-tempo-r0-cold-", offset=50)
    workload = args.output.parent / "hybrid_cold_transition.jsonl"
    output = args.output.parent / "hybrid_cold_transition.raw.json"
    if workload.exists() or output.exists():
        raise ValueError("stale hybrid cold transition artifact")
    balanced._write_jsonl(workload, derived)
    command = [
        sys.executable, "-m",
        "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_v38",
        "--base-url", args.base_url,
        "--model", str(args.model),
        "--served-model-name", args.served_model_name,
        "--workload", str(workload),
        "--output", str(output),
        "--mode", "tempo_auto",
        "--run-id", "hybrid-cold-transition",
        "--default-max-tokens", str(args.default_max_tokens),
        "--max-workers", str(args.max_workers),
        "--request-rate", str(args.request_rate),
        "--timeout-s", str(args.timeout_s),
        "--seed", str(args.seed),
    ]
    if args.api_key_env:
        command.extend(("--api-key-env", args.api_key_env))
    subprocess.run(command, check=True, timeout=1200.0)
    value = json.loads(output.read_text(encoding="utf-8"))
    requests = value.get("requests")
    decisions = value.get("router_decisions")
    if not isinstance(requests, list) or len(requests) != len(rows):
        raise ValueError("hybrid cold transition request count mismatch")
    if any(row.get("error") is not None or row.get("contract_violations")
           for row in requests):
        raise ValueError("hybrid cold transition request failed")
    if not isinstance(decisions, list) or len(decisions) != len(rows):
        raise ValueError("hybrid cold transition decision count mismatch")
    if any(row.get("route") != "decoder_local_recompute_or_cache"
           or "hybrid_cold" not in str(row.get("reason")) for row in decisions):
        raise ValueError("hybrid cold transition routing mismatch")


def main() -> int:
    original = production._transport_prewarm
    production._transport_prewarm = _transport_then_cold
    try:
        return production.main()
    finally:
        production._transport_prewarm = original


if __name__ == "__main__":
    raise SystemExit(main())
