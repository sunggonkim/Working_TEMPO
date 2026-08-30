#!/usr/bin/env python3
"""Build a token-exact 2048-prompt workload from an explicit validation JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from transformers import AutoTokenizer


_NONCE = re.compile(r"nonce ([0-9]{3})\.")
_PADDING = (
    " Preserve exact cache ownership, transfer accounting, admission timing, "
    "and streamed decoder output correctness under the same service contract."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 24:
        raise ValueError("exactly 24 source rows required")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    padding = tokenizer.encode(_PADDING, add_special_tokens=False)
    if not padding:
        raise ValueError("padding must tokenize")
    output = []
    for row in rows:
        if len(_NONCE.findall(row["prompt"])) != 1:
            raise ValueError("source prompt nonce contract mismatch")
        tokens = tokenizer.encode(row["prompt"], add_special_tokens=False)
        if len(tokens) >= 2048:
            raise ValueError("source prompt is not shorter than target")
        needed = 2048 - len(tokens)
        extended = tokens + (padding * ((needed + len(padding) - 1) // len(padding)))[:needed]
        prompt = tokenizer.decode(extended, skip_special_tokens=True)
        if tokenizer.encode(prompt, add_special_tokens=False) != extended:
            raise ValueError("decode/re-encode changed exact token IDs")
        if len(_NONCE.findall(prompt)) != 1:
            raise ValueError("derived prompt nonce contract mismatch")
        value = dict(row)
        value["prompt"] = prompt
        value["max_tokens"] = 32
        output.append(value)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in output),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(output), "prompt_tokens": 2048}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
