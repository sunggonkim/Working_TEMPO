#!/usr/bin/env python3
"""Order-balanced four-arm Elastic-PD experiment in one live server epoch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


_NONCE = re.compile(r"nonce ([0-9]{3})\.")
_ARMS = ("local", "remote", "predictor", "tempo")
_WARM_ORDER = _ARMS
_MEASURED_ORDER = (
    "local", "remote", "predictor", "tempo",
    "tempo", "predictor", "remote", "local",
)


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


def _load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows:
        raise ValueError("workload is empty")
    for row in rows:
        if len(_NONCE.findall(row.get("prompt", ""))) != 1:
            raise ValueError("each prompt needs one three-digit nonce")
    return rows


def _semantic_sha(rows: list[dict]) -> str:
    payload = [{key: row[key] for key in ("request_id", "prompt", "max_tokens")}
               for row in rows]
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _derive(rows: list[dict], *, arm: str, replicate: int, phase: str,
            offset: int) -> list[dict]:
    derived = []
    for item, row in enumerate(rows):
        value = dict(row)
        match = _NONCE.search(value["prompt"])
        nonce = int(match.group(1)) + offset
        if nonce > 999:
            raise ValueError("nonce offset overflow")
        value["prompt"] = _NONCE.sub(f"nonce {nonce:03d}.", value["prompt"])
        value["request_id"] = f"epd-{arm}-r{replicate}-{phase}-item-{item:02d}"
        derived.append(value)
    return derived


def main() -> int:
    args = _parse()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    rows = _load(args.workload)
    semantic_sha = _semantic_sha(rows)
    phase = "warm" if args.run_id.endswith("-warmup") else "measured"
    order = _WARM_ORDER if phase == "warm" else _MEASURED_ORDER
    # Workload constructors keep the base nonce below 100.  These disjoint
    # offsets preserve token geometry while preventing cache-key reuse.
    offsets = (400, 500, 600, 700) if phase == "warm" else (
        100, 200, 300, 400, 500, 600, 700, 800)
    root = args.output.parent / f"elastic_balanced_{phase}"
    workload_root = args.output.parent / f"elastic_balanced_{phase}_workloads"
    root.mkdir()
    workload_root.mkdir()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    lengths = [len(tokenizer.encode(row["prompt"], add_special_tokens=False)) for row in rows]
    arm_counts = {arm: 0 for arm in _ARMS}
    artifacts: dict[str, Path] = {}
    for sequence, (arm, offset) in enumerate(zip(order, offsets, strict=True)):
        replicate = arm_counts[arm]
        arm_counts[arm] += 1
        key = f"{sequence:02d}_{arm}_r{replicate}"
        derived = _derive(rows, arm=arm, replicate=replicate, phase=phase, offset=offset)
        observed = [len(tokenizer.encode(row["prompt"], add_special_tokens=False))
                    for row in derived]
        if observed != lengths:
            raise ValueError(f"{key}: nonce rewrite changed prompt geometry")
        workload_path = workload_root / f"{key}.jsonl"
        raw_path = root / f"{key}.raw.json"
        workload_path.write_text("".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in derived))
        command = [
            sys.executable, "-m", "eval.sota_4node.run_tempo_pd_elastic_stream_metrics_v445",
            "--base-url", args.base_url, "--model", str(args.model),
            "--served-model-name", args.served_model_name,
            "--workload", str(workload_path), "--output", str(raw_path),
            "--mode", "tempo_auto", "--run-id", f"{args.run_id}-{key}",
            "--default-max-tokens", str(args.default_max_tokens),
            "--max-workers", str(args.max_workers), "--timeout-s", str(args.timeout_s),
            "--seed", str(args.seed),
        ]
        if args.request_rate is not None:
            command.extend(("--request-rate", str(args.request_rate)))
        if args.api_key_env:
            command.extend(("--api-key-env", args.api_key_env))
        subprocess.run(command, check=True, timeout=1200.0)
        artifact = json.loads(raw_path.read_text())
        artifact["elastic_balanced_contract"] = {
            "schema": "tempo-elastic-pd-balanced-contract-445",
            "phase": phase, "arm": arm, "replicate": replicate,
            "sequence_index": sequence, "sequence_key": key,
            "order": list(order), "nonce_offset": offset,
            "base_semantic_sha256": semantic_sha,
            "prompt_token_counts": lengths,
            "one_live_server_epoch": True,
            "cache_keys_disjoint_across_blocks": True,
        }
        raw_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        artifacts[key] = raw_path

    public_key = next(key for key in artifacts if "_tempo_" in key)
    public = json.loads(artifacts[public_key].read_text())
    public["elastic_balanced_orchestration"] = {
        "schema": "tempo-elastic-pd-balanced-orchestration-445",
        "phase": phase, "order": list(order),
        "artifacts": {key: str(path.resolve()) for key, path in artifacts.items()},
        "one_live_server_epoch": True,
    }
    args.output.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
