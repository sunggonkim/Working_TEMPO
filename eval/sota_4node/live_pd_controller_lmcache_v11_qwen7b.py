"""Qwen2.5-7B live-P/D crossover entry with exact TP4 KV geometry."""

from __future__ import annotations

from eval.sota_4node import live_pd_controller_lmcache_v10_streamsync as streamsync
from eval.sota_4node import live_pd_controller_lmcache_v7 as fair
from eval.sota_4node import live_pd_controller_v1 as base


def _potential_kv_bytes_tp4(prompt_tokens: int) -> dict[str, int]:
    base._require(prompt_tokens > 0, "prompt token count must be positive")
    # Qwen2.5-7B: 28 layers, 4 KV heads, head_dim 128, BF16 K+V.
    # TP4 shards the four KV heads exactly.
    logical = prompt_tokens * 28 * 4 * 128 * 2 * 2
    return {"logical_bytes": logical, "tp4_physical_bytes": logical}


def main() -> int:
    old = fair._potential_kv_bytes_tp4
    fair._potential_kv_bytes_tp4 = _potential_kv_bytes_tp4
    try:
        return streamsync.main()
    finally:
        fair._potential_kv_bytes_tp4 = old


if __name__ == "__main__":
    raise SystemExit(main())
