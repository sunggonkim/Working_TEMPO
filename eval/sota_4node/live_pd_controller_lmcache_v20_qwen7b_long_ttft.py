"""Long-context, saturated Qwen7B P/D crossover with two output tokens."""

from __future__ import annotations

from eval.sota_4node import live_pd_controller_lmcache_v17_qwen7b_loaded_short as short
from eval.sota_4node import live_pd_controller_lmcache_v19_qwen7b_loaded_saturated as saturated
from eval.sota_4node import live_pd_controller_v1 as base


LONG_BUCKET_REPETITIONS = (64, 256, 512)
FOREGROUND_OUTPUT_TOKENS = 2


def main() -> int:
    old_buckets = short.SHORT_BUCKET_REPETITIONS
    old_tokens = base.OUTPUT_TOKENS
    short.SHORT_BUCKET_REPETITIONS = LONG_BUCKET_REPETITIONS
    base.OUTPUT_TOKENS = FOREGROUND_OUTPUT_TOKENS
    try:
        return saturated.main()
    finally:
        short.SHORT_BUCKET_REPETITIONS = old_buckets
        base.OUTPUT_TOKENS = old_tokens


if __name__ == "__main__":
    raise SystemExit(main())
