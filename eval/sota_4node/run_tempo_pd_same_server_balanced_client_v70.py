#!/usr/bin/env python3
"""Run an order-balanced cold-key crossover against one live server epoch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


_NONCE = re.compile(r"nonce ([0-9]{3})\.")
_MODE = {
    "fixed_local": ("local", "fixed_local"),
    "tempo": ("tempo", "tempo_auto"),
    "lmcache_remote": ("remote", "lmcache_always_remote"),
}
_WARM_ORDER = ("fixed_local", "lmcache_remote", "tempo")
_MEASURED_ORDER = (
    "fixed_local", "tempo", "lmcache_remote",
    "lmcache_remote", "tempo", "fixed_local",
)


def _load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("workload is empty")
    for row in rows:
        if not isinstance(row.get("request_id"), str) or not isinstance(row.get("prompt"), str):
            raise ValueError("workload row is malformed")
        if type(row.get("max_tokens")) is not int or len(_NONCE.findall(row["prompt"])) != 1:
            raise ValueError("workload token/nonce contract mismatch")
    return rows


def _canonical_sha(rows: list[dict]) -> str:
    value = [{"request_id": row["request_id"], "prompt": row["prompt"],
              "max_tokens": row["max_tokens"]} for row in rows]
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _derive(rows: list[dict], *, prefix: str, offset: int) -> list[dict]:
    derived = []
    for row in rows:
        value = dict(row)
        match = _NONCE.search(value["prompt"])
        nonce = int(match.group(1))
        if nonce + offset > 999:
            raise ValueError("nonce offset exceeds three digits")
        value["prompt"] = _NONCE.sub(f"nonce {nonce + offset:03d}.", value["prompt"])
        value["request_id"] = prefix + value["request_id"]
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
    order = _WARM_ORDER if phase == "warm" else _MEASURED_ORDER
    offsets = (100, 200, 300) if phase == "warm" else (400, 500, 600, 700, 800, 900)
    root = args.output.parent / f"same_server_balanced_{phase}"
    workload_root = args.output.parent / f"same_server_balanced_{phase}_workloads"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    lengths = [len(tokenizer.encode(row["prompt"], add_special_tokens=False)) for row in rows]
    artifacts: dict[str, Path] = {}
    arm_counts = {key: 0 for key in _MODE}
    for sequence_index, (label, offset) in enumerate(zip(order, offsets, strict=True)):
        arm, client_mode = _MODE[label]
        replicate = arm_counts[label]
        arm_counts[label] += 1
        key = f"{sequence_index:02d}_{label}_r{replicate}"
        prefix = f"ssb-{arm}-r{replicate}-{phase}-"
        derived = _derive(rows, prefix=prefix, offset=offset)
        if [len(tokenizer.encode(row["prompt"], add_special_tokens=False)) for row in derived] != lengths:
            raise ValueError(f"{key}: nonce rewrite changed prompt token counts")
        workload_path = workload_root / f"{key}.jsonl"
        raw_path = root / f"{key}.raw.json"
        _write_jsonl(workload_path, derived)
        command = [
            sys.executable, "-m", "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_v38",
            "--base-url", args.base_url, "--model", str(args.model),
            "--served-model-name", args.served_model_name,
            "--workload", str(workload_path), "--output", str(raw_path),
            "--mode", client_mode, "--run-id", f"{args.run_id}-{key}",
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
        raw["same_server_balanced_contract"] = {
            "schema": "tempo-pd-same-server-balanced-contract-70",
            "phase": phase, "arm": label, "replicate": replicate,
            "sequence_index": sequence_index, "sequence_key": key,
            "request_prefix": prefix,
            "server_epoch_root": str(args.output.parent.resolve()),
            "base_semantic_sha256": semantic_sha,
            "base_request_ids": [row["request_id"] for row in rows],
            "base_prompt_sha256": {
                row["request_id"]: hashlib.sha256(row["prompt"].encode()).hexdigest()
                for row in rows
            },
            "prompt_token_counts": lengths, "nonce_offset": offset,
            "cache_keys_disjoint_across_all_blocks": True,
        }
        raw_path.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        artifacts[key] = raw_path

    tempo_key = next(key for key in artifacts if "_tempo_" in key)
    public = json.loads(artifacts[tempo_key].read_text(encoding="utf-8"))
    public["same_server_balanced_orchestration"] = {
        "schema": "tempo-pd-same-server-balanced-orchestration-70",
        "phase": phase, "order": list(order),
        "artifacts": {key: str(value.resolve()) for key, value in artifacts.items()},
        "one_live_server_epoch": True,
    }
    args.output.write_text(json.dumps(public, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
