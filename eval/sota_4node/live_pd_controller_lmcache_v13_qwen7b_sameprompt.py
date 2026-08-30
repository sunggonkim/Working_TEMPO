"""Qwen7B fair P/D entry with identical calibration prompts per route."""

from __future__ import annotations

from eval.sota_4node import live_pd_controller_lmcache_v12_qwen7b_unloaded as qwen
from eval.sota_4node import live_pd_controller_v1 as base


_ORIGINAL_PROMPT = base._prompt


def _prompt(kind: str, bucket: int, repetitions: int) -> str:
    if kind in {"calibration-remote", "calibration-direct"}:
        kind = "calibration"
    return _ORIGINAL_PROMPT(kind, bucket, repetitions)


def main() -> int:
    old = base._prompt
    base._prompt = _prompt
    try:
        return qwen.main()
    finally:
        base._prompt = old


if __name__ == "__main__":
    raise SystemExit(main())
