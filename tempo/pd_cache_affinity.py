"""Fail-closed cache-affinity placement for validated P/D workloads.

The policy binds a stable cache item to one route. Its initial placement uses
calibrated geometry plus a bounded recent seed-composition guard; later hits
must use the same route.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading

from tempo.pd_admission import PDRoute


POLICY_ID = "qwen25-7b-tp4x2-warm-affinity-8"
VALID_PROMPTS = (512, 1230, 2048, 4094, 4096)
VALID_OUTPUTS = (16, 32, 64, 128, 256)
REMOTE_BUCKETS = frozenset({
    (512, 32), (512, 64), (512, 128), (2048, 64), (2048, 256)})
COMPOSITION_WINDOW = 8


def calibrated_route(prompt_tokens: int, output_tokens: int) -> PDRoute:
    """Return the frozen geometry placement, rejecting unvalidated inputs."""
    if type(prompt_tokens) is not int or prompt_tokens not in VALID_PROMPTS:
        raise ValueError("cache-affinity prompt geometry is unvalidated")
    if type(output_tokens) is not int or output_tokens not in VALID_OUTPUTS:
        raise ValueError("cache-affinity output geometry is unvalidated")
    if prompt_tokens in (4094, 4096) and output_tokens not in (16, 128):
        raise ValueError("prompt4096 is validated only for output16/output128")
    return (PDRoute.REMOTE_PREFILL
            if (prompt_tokens, output_tokens) in REMOTE_BUCKETS
            else PDRoute.DECODER_LOCAL)


def calibrated_partition() -> dict[str, int | float]:
    rows = []
    for prompt in VALID_PROMPTS:
        for output in VALID_OUTPUTS:
            try:
                calibrated_route(prompt, output)
            except ValueError:
                continue
            rows.extend(((prompt, output), (prompt, output)))
    remote = sum(prompt for prompt, output in rows
                 if calibrated_route(prompt, output) is PDRoute.REMOTE_PREFILL)
    total = sum(prompt for prompt, _ in rows)
    return {
        "request_count": len(rows),
        "remote_request_count": sum(
            calibrated_route(prompt, output) is PDRoute.REMOTE_PREFILL
            for prompt, output in rows),
        "prompt_token_work": total,
        "remote_prompt_token_work": remote,
        "remote_prompt_token_work_fraction": remote / total,
    }


@dataclass(frozen=True)
class CachePlacement:
    cache_item: str
    prompt_tokens: int
    output_tokens: int
    route: PDRoute


class CacheAffinityCatalog:
    """Thread-safe seed/hit catalog with immutable, composition-aware placement."""

    def __init__(self) -> None:
        self._items: dict[str, CachePlacement] = {}
        self._recent_seed_outputs: list[int] = []
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
        base_route = calibrated_route(prompt_tokens, output_tokens)
        with self._lock:
            prior = self._items.get(key)
            if prior is not None:
                expected = CachePlacement(
                    key, prompt_tokens, output_tokens, prior.route)
                if prior != expected:
                    raise ValueError("cache placement changed after warm seed")
                return prior
            route = base_route
            if ((prompt_tokens, output_tokens) == (2048, 256)
                    and any(output != 256 for output in self._recent_seed_outputs)):
                route = PDRoute.DECODER_LOCAL
            placement = CachePlacement(key, prompt_tokens, output_tokens, route)
            self._items[key] = placement
            self._recent_seed_outputs.append(output_tokens)
            del self._recent_seed_outputs[:-COMPOSITION_WINDOW]
            return placement

    def hit(self, cache_item: str, prompt_tokens: int,
            output_tokens: int) -> CachePlacement:
        key = self._key(cache_item)
        calibrated_route(prompt_tokens, output_tokens)
        with self._lock:
            placement = self._items.get(key)
        if placement is None:
            raise ValueError("cache item was not seeded on this engine pair")
        expected = CachePlacement(key, prompt_tokens, output_tokens, placement.route)
        if placement != expected:
            raise ValueError("cache hit geometry or placement changed")
        return placement
