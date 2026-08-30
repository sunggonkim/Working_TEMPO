#!/usr/bin/env python3
"""Request-level Latin interleave of local, TEMPO, and LMCache arms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from eval.sota_4node import run_tempo_pd_same_server_balanced_client_v70 as base


_ARMS = ("local", "tempo", "remote")


def main() -> int:
    args = base._parse()
    phase = "warm" if args.run_id.endswith("-warmup") else "measured"
    if phase == "warm":
        return base.main()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    rows = base._load_rows(args.workload)
    semantic_sha = base._canonical_sha(rows)
    derived = []
    sequence = []
    for replicate in (0, 1):
        base_order = _ARMS if replicate == 0 else tuple(reversed(_ARMS))
        for row_index, row in enumerate(rows):
            rotation = row_index % 3
            order = base_order[rotation:] + base_order[:rotation]
            for arm in order:
                value = dict(row)
                nonce = 400 + len(derived)
                value["prompt"] = base._NONCE.sub(f"nonce {nonce:03d}.", value["prompt"])
                value["request_id"] = (
                    f"ssi-{arm}-r{replicate}-measured-{row['request_id']}")
                derived.append(value)
                sequence.append({"arm": arm, "replicate": replicate,
                                 "base_request_id": row["request_id"], "nonce": nonce})
    if len(derived) != 144 or max(row["nonce"] for row in sequence) > 999:
        raise ValueError("interleaved geometry mismatch")
    workload = args.output.parent / "same_server_interleaved_measured_workload.jsonl"
    base._write_jsonl(workload, derived)
    command = [
        sys.executable, "-m", "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_v38",
        "--base-url", args.base_url, "--model", str(args.model),
        "--served-model-name", args.served_model_name,
        "--workload", str(workload), "--output", str(args.output),
        "--mode", "tempo_auto", "--run-id", args.run_id,
        "--default-max-tokens", str(args.default_max_tokens),
        "--max-workers", str(args.max_workers), "--timeout-s", str(args.timeout_s),
        "--seed", str(args.seed),
    ]
    if args.request_rate is not None:
        command.extend(("--request-rate", str(args.request_rate)))
    if args.api_key_env:
        command.extend(("--api-key-env", args.api_key_env))
    subprocess.run(command, check=True, timeout=1200.0)
    raw = json.loads(args.output.read_text(encoding="utf-8"))
    raw["same_server_interleaved_contract"] = {
        "schema": "tempo-pd-same-server-interleaved-contract-100",
        "phase": phase, "request_count": len(derived),
        "arm_counts": {arm: sum(row["arm"] == arm for row in sequence) for arm in _ARMS},
        "semantic_sha256": semantic_sha,
        "base_prompt_sha256": {
            row["request_id"]: hashlib.sha256(row["prompt"].encode()).hexdigest()
            for row in rows},
        "sequence": sequence,
        "request_level_latin_interleave": True,
        "one_live_server_epoch": True,
    }
    args.output.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
