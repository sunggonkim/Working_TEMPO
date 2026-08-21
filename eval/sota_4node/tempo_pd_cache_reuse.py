"""Fail-closed cache-state assignment for canonical Elastic-PD workloads."""

from __future__ import annotations

import hashlib
import re


_ITEM = re.compile(r"-item-(\d+)$")
_REPLICATE = re.compile(r"-r(\d+)-(?:warm|measured)-")


def parse_reuse_items(raw: str) -> frozenset[int] | None:
    """Return None for all items, otherwise an explicit immutable item set."""
    if raw == "all":
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError("decoder reuse items must be 'all' or comma-separated")
    parts = raw.split(",")
    if any(not part.isdigit() for part in parts):
        raise ValueError("decoder reuse items must be canonical integers")
    values = [int(part) for part in parts]
    if any(not 0 <= value <= 999 for value in values):
        raise ValueError("decoder reuse item is outside [0, 999]")
    if len(values) != len(set(values)):
        raise ValueError("decoder reuse items contain duplicates")
    if raw != ",".join(str(value) for value in values):
        raise ValueError("decoder reuse items are not canonical")
    return frozenset(values)


def item_index(request_id: str) -> int:
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be nonempty")
    match = _ITEM.search(request_id)
    if match is None:
        raise ValueError("request_id lacks an item suffix")
    return int(match.group(1))


def replicate_index(request_id: str) -> int:
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be nonempty")
    match = _REPLICATE.search(request_id)
    if match is None:
        raise ValueError("request_id lacks a replicate marker")
    return int(match.group(1))


def reuses_decoder_cache(
    request_id: str, reuse_items: frozenset[int] | None,
) -> bool:
    return reuse_items is None or item_index(request_id) in reuse_items


def cache_salt(
    request_id: str, reuse_items: frozenset[int] | None,
) -> str:
    if "-warm-" in request_id:
        domain = "warm"
    elif "-measured-" in request_id:
        domain = (
            "measured-reuse"
            if reuses_decoder_cache(request_id, reuse_items)
            else f"measured-r{replicate_index(request_id)}"
        )
    else:
        raise ValueError("prefix-cached request lacks a phase marker")
    return f"tempo-elastic-pd-{domain}-cache-domain-v1"


def namespace_cache_salt(*, arm: str, prompt_key: str) -> str:
    """Return one arm-isolated salt stable across seed/probe/measurement.

    ``prompt_key`` is the router's SHA-256 of exact token IDs.  Hashing the
    arm and prompt key again keeps the public salt compact while ensuring that
    fixed baselines cannot populate one another's physical vLLM or LMCache
    namespaces.
    """

    if not isinstance(arm, str) or not arm.strip():
        raise ValueError("cache-salt arm must be nonempty")
    if (
        not isinstance(prompt_key, str)
        or len(prompt_key) != 64
        or any(value not in "0123456789abcdef" for value in prompt_key)
    ):
        raise ValueError("prompt_key must be a lowercase SHA-256 digest")
    digest = hashlib.sha256(
        f"tempo-elastic-pd-cache-v2:{arm}:{prompt_key}".encode("ascii")
    ).hexdigest()[:32]
    return f"tempo-epd-v2-{digest}"


__all__ = [
    "cache_salt",
    "item_index",
    "namespace_cache_salt",
    "parse_reuse_items",
    "replicate_index",
    "reuses_decoder_cache",
]
