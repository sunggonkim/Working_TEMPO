#!/usr/bin/env python3
"""Balanced client adapter for the legacy low-load workload without nonces."""

from __future__ import annotations

import json
from pathlib import Path

from eval.sota_4node import run_tempo_pd_same_server_balanced_client_v70 as prior


_WORD = {
    100: "Verified", 200: "Exact", 300: "Strict",
    400: "Safe", 500: "Valid", 600: "Frozen",
    700: "Final", 800: "Direct", 900: "Remote",
}


def _load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("workload is empty")
    for row in rows:
        if not isinstance(row.get("request_id"), str) or not isinstance(row.get("prompt"), str):
            raise ValueError("workload row is malformed")
        if type(row.get("max_tokens")) is not int:
            raise ValueError("workload max_tokens is missing")
        if "Measured admission" not in row["prompt"]:
            raise ValueError("low workload lacks the frozen cold-key phrase")
    return rows


def _derive(rows: list[dict], *, prefix: str, offset: int) -> list[dict]:
    word = _WORD[offset]
    derived = []
    for row in rows:
        value = dict(row)
        value["prompt"] = value["prompt"].replace(
            "Measured admission", f"{word} admission", 1)
        value["request_id"] = prefix + value["request_id"]
        derived.append(value)
    return derived


def main() -> int:
    prior._load_rows = _load_rows
    prior._derive = _derive
    return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
