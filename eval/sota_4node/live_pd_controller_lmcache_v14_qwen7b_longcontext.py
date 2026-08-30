"""Qwen7B live-P/D controller spanning short through 6K-token prompts."""

from __future__ import annotations

from eval.sota_4node import live_pd_controller_lmcache_v13_qwen7b_sameprompt as qwen
from eval.sota_4node import live_pd_controller_lmcache_v7 as fair


LONG_BUCKET_REPETITIONS = (64, 256, 512)


def main() -> int:
    old = fair.previous.BUCKET_REPETITIONS
    fair.previous.BUCKET_REPETITIONS = LONG_BUCKET_REPETITIONS
    try:
        return qwen.main()
    finally:
        fair.previous.BUCKET_REPETITIONS = old


if __name__ == "__main__":
    raise SystemExit(main())
