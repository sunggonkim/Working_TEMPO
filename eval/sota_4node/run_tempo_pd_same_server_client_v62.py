#!/usr/bin/env python3
"""Run three cold-key routing arms sequentially against one live server epoch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


_NONCE = re.compile(r"nonce ([0-9]{3})\.")
_ARMS = {
    "fixed_local": ("local", 100, "fixed_local"),
    "tempo": ("tempo", 200, "tempo_auto"),
    "lmcache_remote": ("remote", 300, "lmcache_always_remote"),
}


def _canonical_sha(rows: list[dict]) -> str:
    value = [{"request_id": row["request_id"], "prompt": row["prompt"],
              "max_tokens": row["max_tokens"]} for row in rows]
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("workload is empty")
    for row in rows:
        if not isinstance(row.get("request_id"), str) or not isinstance(row.get("prompt"), str):
            raise ValueError("workload row is malformed")
        if type(row.get("max_tokens")) is not int:
            raise ValueError("workload max_tokens is missing")
        if len(_NONCE.findall(row["prompt"])) != 1:
            raise ValueError("prompt must contain one three-digit nonce")
    return rows


def _arm_rows(rows: list[dict], *, arm: str, phase: str, offset: int) -> list[dict]:
    derived = []
    for row in rows:
        value = dict(row)
        match = _NONCE.search(value["prompt"])
        nonce = int(match.group(1))
        if nonce + offset > 999:
            raise ValueError("nonce offset exceeds three digits")
        value["prompt"] = _NONCE.sub(f"nonce {nonce + offset:03d}.", value["prompt"])
        value["request_id"] = f"ss-{arm}-{phase}-{row['request_id']}"
        derived.append(value)
    return derived


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                    encoding="utf-8")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--default-max-tokens", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--request-rate", type=float)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--api-key-env")
    return parser.parse_args()


def main() -> int:
    args = _parse()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    rows = _load_rows(args.workload)
    semantic_sha = _canonical_sha(rows)
    phase = "warm" if args.run_id.endswith("-warmup") else "measured"
    order = ("fixed_local", "lmcache_remote", "tempo") if phase == "warm" else (
        "fixed_local", "tempo", "lmcache_remote")
    root = args.output.parent / f"same_server_{phase}"
    workload_root = args.output.parent / f"same_server_{phase}_workloads"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    original_lengths = [len(tokenizer.encode(row["prompt"], add_special_tokens=False)) for row in rows]
    artifacts: dict[str, Path] = {}
    for sequence_index, label in enumerate(order):
        arm, base_offset, client_mode = _ARMS[label]
        offset = base_offset + (400 if phase == "measured" else 0)
        derived = _arm_rows(rows, arm=arm, phase=phase, offset=offset)
        derived_lengths = [len(tokenizer.encode(row["prompt"], add_special_tokens=False))
                           for row in derived]
        if derived_lengths != original_lengths:
            raise ValueError(f"{label}: nonce rewrite changed prompt token counts")
        workload_path = workload_root / f"{label}.jsonl"
        raw_path = root / f"{label}.raw.json"
        _write_jsonl(workload_path, derived)
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
            "schema": "tempo-pd-same-server-contract-62",
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
        "schema": "tempo-pd-same-server-orchestration-62",
        "phase": phase, "order": list(order),
        "artifacts": {key: str(value.resolve()) for key, value in artifacts.items()},
        "one_live_server_epoch": True,
    }
    args.output.write_text(json.dumps(tempo_raw, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
