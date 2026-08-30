"""Stable one-local/one-remote split for the saturated 2048/64 bucket."""

from __future__ import annotations

import threading

from tempo.pd_admission import PDRoute
from tempo.pd_cache_affinity import CachePlacement, VALID_OUTPUTS, VALID_PROMPTS


POLICY_ID = "qwen25-7b-tp4x2-warm-affinity-split-1"
REMOTE_BUCKETS = frozenset({(512, 32), (512, 64), (512, 128)})


def calibrated_route(cache_item: str, prompt_tokens: int,
                     output_tokens: int) -> PDRoute:
    if type(prompt_tokens) is not int or prompt_tokens not in VALID_PROMPTS:
        raise ValueError("split prompt geometry is unvalidated")
    if type(output_tokens) is not int or output_tokens not in VALID_OUTPUTS:
        raise ValueError("split output geometry is unvalidated")
    suffix = cache_item.removeprefix("cache-item-")
    if len(suffix) != 2 or not suffix.isdigit():
        raise ValueError("split cache item identity malformed")
    remote = ((prompt_tokens, output_tokens) in REMOTE_BUCKETS
              or ((prompt_tokens, output_tokens) == (2048, 64)
                  and int(suffix) % 2 == 1))
    return PDRoute.REMOTE_PREFILL if remote else PDRoute.DECODER_LOCAL


class SplitCacheAffinityCatalog:
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
            calibrated_route(key, prompt_tokens, output_tokens))
        with self._lock:
            prior = self._items.setdefault(key, placement)
            if prior != placement:
                raise ValueError("split cache placement changed after seed")
            return prior

    def hit(self, cache_item: str, prompt_tokens: int,
            output_tokens: int) -> CachePlacement:
        key = self._key(cache_item)
        with self._lock:
            placement = self._items.get(key)
        if placement is None:
            raise ValueError("split cache item was not seeded")
        expected = CachePlacement(
            key, prompt_tokens, output_tokens,
            calibrated_route(key, prompt_tokens, output_tokens))
        if placement != expected:
            raise ValueError("split hit geometry or placement changed")
        return placement
