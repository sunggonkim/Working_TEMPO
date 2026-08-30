"""Loaded Qwen7B long-context controller with identical calibration prompts."""

from __future__ import annotations

from eval.sota_4node import live_pd_controller_lmcache_v11_qwen7b as loaded_qwen
from eval.sota_4node import live_pd_controller_lmcache_v7 as fair
from eval.sota_4node import live_pd_controller_v1 as base


LONG_BUCKET_REPETITIONS = (64, 256, 512)
_ORIGINAL_PROMPT = base._prompt


def _prompt(kind: str, bucket: int, repetitions: int) -> str:
    if kind in {"calibration-remote", "calibration-direct"}:
        kind = "calibration"
    return _ORIGINAL_PROMPT(kind, bucket, repetitions)


def main() -> int:
    old_prompt = base._prompt
    old_buckets = fair.previous.BUCKET_REPETITIONS
    base._prompt = _prompt
    fair.previous.BUCKET_REPETITIONS = LONG_BUCKET_REPETITIONS
    try:
        return loaded_qwen.main()
    finally:
        base._prompt = old_prompt
        fair.previous.BUCKET_REPETITIONS = old_buckets


if __name__ == "__main__":
    raise SystemExit(main())
