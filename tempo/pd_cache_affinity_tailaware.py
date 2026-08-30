"""Tail-aware saturation variant of the frozen warm cache-affinity catalog."""

from __future__ import annotations

import threading

from tempo.pd_admission import PDRoute
from tempo.pd_cache_affinity import CachePlacement, VALID_OUTPUTS, VALID_PROMPTS


POLICY_ID = "qwen25-7b-tp4x2-warm-affinity-tailaware-1"
REMOTE_BUCKETS = frozenset({(512, 32), (512, 64), (512, 128)})


def calibrated_route(prompt_tokens: int, output_tokens: int) -> PDRoute:
    if type(prompt_tokens) is not int or prompt_tokens not in VALID_PROMPTS:
        raise ValueError("tail-aware prompt geometry is unvalidated")
    if type(output_tokens) is not int or output_tokens not in VALID_OUTPUTS:
        raise ValueError("tail-aware output geometry is unvalidated")
    return (PDRoute.REMOTE_PREFILL
            if (prompt_tokens, output_tokens) in REMOTE_BUCKETS
            else PDRoute.DECODER_LOCAL)


class TailAwareCacheAffinityCatalog:
    def __init__(self) -> None:
        self._items: dict[str, CachePlacement] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(cache_item: str) -> str:
        if not isinstance(cache_item, str) or not cache_item.startswith("cache-item-"):
            raise ValueError("stable cache item identity required")
        suffix = cache_item.removeprefix("cache-item-")
        if len(suffix) != 2 or not suffix.isdigit():
            raise ValueError("stable cache item identity malformed")
        return cache_item

    def seed(self, cache_item: str, prompt_tokens: int,
             output_tokens: int) -> CachePlacement:
        key = self._key(cache_item)
        placement = CachePlacement(
            key, prompt_tokens, output_tokens,
            calibrated_route(prompt_tokens, output_tokens))
        with self._lock:
            prior = self._items.setdefault(key, placement)
            if prior != placement:
                raise ValueError("tail-aware cache placement changed after seed")
            return prior

    def hit(self, cache_item: str, prompt_tokens: int,
            output_tokens: int) -> CachePlacement:
        key = self._key(cache_item)
        with self._lock:
            placement = self._items.get(key)
        if placement is None:
            raise ValueError("tail-aware cache item was not seeded")
        expected = CachePlacement(
            key, prompt_tokens, output_tokens,
            calibrated_route(prompt_tokens, output_tokens))
        if placement != expected:
            raise ValueError("tail-aware hit geometry or placement changed")
        return placement
