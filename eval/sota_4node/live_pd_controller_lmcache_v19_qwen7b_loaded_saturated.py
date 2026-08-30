"""Qwen7B crossover with seven synchronized decoder background streams."""

from __future__ import annotations

from eval.sota_4node import live_pd_controller_lmcache_v18_qwen7b_loaded_heavy as heavy


SATURATED_BACKGROUND_STREAMS = 7


def main() -> int:
    old = heavy.BACKGROUND_STREAMS
    heavy.BACKGROUND_STREAMS = SATURATED_BACKGROUND_STREAMS
    try:
        return heavy.main()
    finally:
        heavy.BACKGROUND_STREAMS = old


if __name__ == "__main__":
    raise SystemExit(main())
