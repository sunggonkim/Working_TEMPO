"""Live-P/D client with token-accurate SSE and model-valid prompt buckets."""

from __future__ import annotations

from eval.sota_4node import live_pd_controller_lmcache_v2 as wire
from eval.sota_4node import live_pd_controller_lmcache_v5 as token_accurate


BUCKET_REPETITIONS = (16, 64, 96)


def main() -> int:
    old = wire.BUCKET_REPETITIONS
    wire.BUCKET_REPETITIONS = BUCKET_REPETITIONS
    try:
        return token_accurate.main()
    finally:
        wire.BUCKET_REPETITIONS = old


if __name__ == "__main__":
    raise SystemExit(main())
