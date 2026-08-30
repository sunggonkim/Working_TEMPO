"""Loaded Qwen7B controller capped below the observed 6K output divergence."""

from __future__ import annotations

from eval.sota_4node import live_pd_controller_lmcache_v15_qwen7b_long_loaded as long_loaded


LOADED_BUCKET_REPETITIONS = (64, 192, 256)


def main() -> int:
    old = long_loaded.LONG_BUCKET_REPETITIONS
    long_loaded.LONG_BUCKET_REPETITIONS = LOADED_BUCKET_REPETITIONS
    try:
        return long_loaded.main()
    finally:
        long_loaded.LONG_BUCKET_REPETITIONS = old


if __name__ == "__main__":
    raise SystemExit(main())
