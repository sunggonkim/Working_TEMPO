"""Qwen2.5-7B fair live-P/D entry without concurrent decoder load."""

from __future__ import annotations

from eval.sota_4node import live_pd_controller_lmcache_v7 as fair
from eval.sota_4node import live_pd_controller_v1 as base


def _potential_kv_bytes_tp4(prompt_tokens: int) -> dict[str, int]:
    base._require(prompt_tokens > 0, "prompt token count must be positive")
    logical = prompt_tokens * 28 * 4 * 128 * 2 * 2
    return {"logical_bytes": logical, "tp4_physical_bytes": logical}


def main() -> int:
    old = fair._potential_kv_bytes_tp4
    fair._potential_kv_bytes_tp4 = _potential_kv_bytes_tp4
    try:
        return fair.main()
    finally:
        fair._potential_kv_bytes_tp4 = old


if __name__ == "__main__":
    raise SystemExit(main())
