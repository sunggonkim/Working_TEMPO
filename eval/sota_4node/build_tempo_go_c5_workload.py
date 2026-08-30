#!/usr/bin/env python3
"""Build a tenant-labelled C5 workload from one exact-token source JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TENANTS = ("latency", "interactive", "batch", "background")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=4094)
    parser.add_argument("--output-tokens", type=int, default=16)
    args = parser.parse_args()
    if not args.source.is_file() or not args.model.joinpath("config.json").is_file():
        raise ValueError("C5 source/model artifact is missing")
    if args.output.exists() or args.count < 4 or args.output_tokens < 2:
        raise ValueError("C5 output exists or workload bounds are invalid")
    raw_rows = [
        json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(raw_rows) < args.count:
        raise ValueError("C5 source has fewer rows than requested")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False)
    rows = []
    prompts = set()
    for index, raw in enumerate(raw_rows[:args.count]):
        prompt = raw.get("prompt") if isinstance(raw, dict) else None
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("C5 source prompt is invalid")
        observed = len(tokenizer.encode(prompt, add_special_tokens=False))
        if observed != args.prompt_tokens:
            raise ValueError(
                f"C5 prompt geometry differs at {index}: {observed} "
                f"!= {args.prompt_tokens}")
        # Keep every other row distinct so cache namespaces are explicit, but
        # preserve a repeated prompt for the warmup/main affinity experiment.
        prompts.add(hashlib.sha256(prompt.encode()).hexdigest())
        tenant = TENANTS[index % len(TENANTS)]
        rows.append({
            "request_id": f"epd-tempo-{tenant}-measured-c5-{index:04d}",
            "prompt": prompt,
            "max_tokens": args.output_tokens,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8")
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({
        "schema": "tempo-go-c5-workload-v1",
        "source": str(args.source.resolve()),
        "source_sha256": _sha256(args.source),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "model": str(args.model.resolve()),
        "count": len(rows),
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "tenant_order": list(TENANTS),
        "distinct_prompt_namespaces": len(prompts),
        "arrival_pattern": "client_request_rate_8_rps_open_loop",
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
