"""Loaded Qwen7B controller with three distinct, short prompt buckets."""

from __future__ import annotations

from eval.sota_4node import live_pd_controller_lmcache_v15_qwen7b_long_loaded as loaded


SHORT_BUCKET_REPETITIONS = (64, 64, 64)


def main() -> int:
    old = loaded.LONG_BUCKET_REPETITIONS
    loaded.LONG_BUCKET_REPETITIONS = SHORT_BUCKET_REPETITIONS
    try:
        return loaded.main()
    finally:
        loaded.LONG_BUCKET_REPETITIONS = old


if __name__ == "__main__":
    raise SystemExit(main())
