#!/usr/bin/env python3
"""Reverse-order same-server client: LMCache, TEMPO, then fixed local."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

from eval.sota_4node import run_tempo_pd_same_server_client_v62 as shared


def main() -> int:
    args = shared._parse()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    rows = shared._load_rows(args.workload)
    semantic_sha = shared._canonical_sha(rows)
    phase = "warm" if args.run_id.endswith("-warmup") else "measured"
    order = ("fixed_local", "lmcache_remote", "tempo") if phase == "warm" else (
        "lmcache_remote", "tempo", "fixed_local")
    root = args.output.parent / f"same_server_{phase}"
    workload_root = args.output.parent / f"same_server_{phase}_workloads"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    original_lengths = [len(tokenizer.encode(row["prompt"], add_special_tokens=False)) for row in rows]
    artifacts = {}
    for sequence_index, label in enumerate(order):
        arm, base_offset, client_mode = shared._ARMS[label]
        offset = base_offset + (400 if phase == "measured" else 0)
        derived = shared._arm_rows(rows, arm=arm, phase=phase, offset=offset)
        lengths = [len(tokenizer.encode(row["prompt"], add_special_tokens=False))
                   for row in derived]
        if lengths != original_lengths:
            raise ValueError(f"{label}: nonce rewrite changed prompt token counts")
        workload_path = workload_root / f"{label}.jsonl"
        raw_path = root / f"{label}.raw.json"
        shared._write_jsonl(workload_path, derived)
        command = [
            sys.executable, "-m", "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_v38",
            "--base-url", args.base_url, "--model", str(args.model),
            "--served-model-name", args.served_model_name,
            "--workload", str(workload_path), "--output", str(raw_path),
            "--mode", client_mode, "--run-id", f"{args.run_id}-{label}",
            "--default-max-tokens", str(args.default_max_tokens),
            "--max-workers", str(args.max_workers), "--timeout-s", str(args.timeout_s),
            "--seed", str(args.seed),
        ]
        if args.request_rate is not None:
            command.extend(("--request-rate", str(args.request_rate)))
        if args.api_key_env:
            command.extend(("--api-key-env", args.api_key_env))
        subprocess.run(command, check=True, timeout=1200.0)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        raw["same_server_contract"] = {
            "schema": "tempo-pd-same-server-contract-68",
            "phase": phase, "arm": label, "sequence_index": sequence_index,
            "server_epoch_root": str(args.output.parent.resolve()),
            "base_semantic_sha256": semantic_sha,
            "base_request_ids": [row["request_id"] for row in rows],
            "base_prompt_sha256": {
                row["request_id"]: hashlib.sha256(row["prompt"].encode()).hexdigest()
                for row in rows
            },
            "prompt_token_counts": original_lengths,
            "nonce_offset": offset,
            "cache_keys_disjoint_across_arms": True,
        }
        raw_path.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        artifacts[label] = raw_path
    tempo_raw = json.loads(artifacts["tempo"].read_text(encoding="utf-8"))
    tempo_raw["same_server_orchestration"] = {
        "schema": "tempo-pd-same-server-orchestration-68", "phase": phase,
        "order": list(order), "one_live_server_epoch": True,
        "artifacts": {key: str(value.resolve()) for key, value in artifacts.items()},
    }
    args.output.write_text(json.dumps(tempo_raw, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
