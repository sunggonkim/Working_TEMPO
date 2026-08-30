"""Executable cache-state preparation plan for actual vLLM/LMCache P/D.

The plan deliberately separates source-side and decoder-side preparation:

1. remote seed/full-hit probes establish P_ONLY for every P_ONLY/BOTH prompt;
2. one quiescent decoder APC reset removes probe-created decoder residency
   while preserving producer LMCache state;
3. local miss-seed/full-hit probes establish D_ONLY/BOTH on one decoder pair;
4. only then may measured requests start.

Request-ID labels never establish residency.  They select the fixed probe path;
the router still requires LMCache response headers or final vLLM SSE usage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable

from tempo.pd_contention_workload import CacheState


SCHEMA = "tempo-pd-cache-state-preparation-plan-v1"
PREP_OUTPUT_TOKENS = 2
_ARM = re.compile(r"^epd-(local|remote|predictor|queue_gpu|tempo)-")
_ITEM = re.compile(r"-item-([0-9]+)$")
_STATE_MARKERS = {
    CacheState.MISS: "-cache-miss-measured-",
    CacheState.P_ONLY: "-cache-p-only-measured-",
    CacheState.D_ONLY: "-cache-d-only-measured-",
    CacheState.BOTH: "-cache-both-measured-",
}


def _canonical_sha256(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


@dataclass(frozen=True)
class CacheProtocolItem:
    """One measured request bound to an exact physical cache namespace."""

    request_id: str
    prompt: str
    prompt_token_sha256: str
    prompt_tokens: int
    output_tokens: int
    cache_state: CacheState
    terminal_item: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be nonempty")
        arm = _ARM.match(self.request_id)
        if arm is None:
            raise ValueError("request_id lacks a canonical Elastic-PD arm")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be nonempty")
        _canonical_sha256(
            self.prompt_token_sha256, name="prompt_token_sha256")
        for name in ("prompt_tokens", "output_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if not isinstance(self.cache_state, CacheState):
            raise TypeError("cache_state must be CacheState")
        if type(self.terminal_item) is not int or self.terminal_item < 0:
            raise ValueError("terminal_item must be a non-negative integer")
        match = _ITEM.search(self.request_id)
        if match is None or int(match.group(1)) != self.terminal_item:
            raise ValueError(
                "request_id terminal item differs from cache owner item")
        markers = [
            state for state, marker in _STATE_MARKERS.items()
            if marker in self.request_id
        ]
        if markers != [self.cache_state]:
            raise ValueError(
                "request_id cache marker differs from declared cache state")

    @property
    def arm(self) -> str:
        match = _ARM.match(self.request_id)
        assert match is not None
        return match.group(1)

    @property
    def namespace_key(self) -> tuple[str, str]:
        return self.arm, self.prompt_token_sha256

    def protocol_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "arm": self.arm,
            "prompt_token_sha256": self.prompt_token_sha256,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "cache_state": self.cache_state.value,
            "terminal_item": self.terminal_item,
        }


@dataclass(frozen=True)
class CachePreparationPlan:
    items: tuple[CacheProtocolItem, ...]
    source_probe_rows: tuple[dict[str, object], ...]
    decoder_prepare_rows: tuple[dict[str, object], ...]
    fingerprint_sha256: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("cache preparation plan schema differs")
        _canonical_sha256(self.fingerprint_sha256, name="fingerprint_sha256")

    def manifest_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fingerprint_sha256": self.fingerprint_sha256,
            "preparation_order": [
                "source_probe_rows",
                "quiescent_decoder_apc_reset_preserving_external_lmcache",
                "decoder_prepare_rows",
                "measured_rows",
            ],
            "counts": {
                "measured": len(self.items),
                "source_probe": len(self.source_probe_rows),
                "decoder_prepare": len(self.decoder_prepare_rows),
            },
            "items": [item.protocol_dict() for item in self.items],
            "source_probe_request_ids": [
                str(row["request_id"]) for row in self.source_probe_rows
            ],
            "decoder_prepare_request_ids": [
                str(row["request_id"]) for row in self.decoder_prepare_rows
            ],
            "measurement_includes_preparation_requests": False,
            "request_id_labels_establish_residency": False,
        }


def _row(*, request_id: str, item: CacheProtocolItem) -> dict[str, object]:
    return {
        "request_id": request_id,
        "prompt": item.prompt,
        "max_tokens": PREP_OUTPUT_TOKENS,
        "arrival_offset_ms": 0.0,
    }


def _prep_stem(item: CacheProtocolItem) -> str:
    digest = item.prompt_token_sha256[:16]
    return f"epd-{item.arm}-c4prep-{digest}"


def _source_probe_row(item: CacheProtocolItem) -> dict[str, object]:
    request_id = (
        f"{_prep_stem(item)}-warm-cache-p-probe-"
        f"item-{item.terminal_item:06d}"
    )
    return _row(request_id=request_id, item=item)


def _decoder_rows(item: CacheProtocolItem) -> tuple[dict[str, object], ...]:
    stem = _prep_stem(item)
    suffix = f"item-{item.terminal_item:06d}"
    seed = _row(
        request_id=(
            f"{stem}-warm-seed-o{item.output_tokens}-cache-d-seed-{suffix}"
        ),
        item=item,
    )
    probe = _row(
        request_id=f"{stem}-warm-cache-d-probe-{suffix}",
        item=item,
    )
    return seed, probe


def _fingerprint(
    items: tuple[CacheProtocolItem, ...],
    source_rows: tuple[dict[str, object], ...],
    decoder_rows: tuple[dict[str, object], ...],
) -> str:
    payload = {
        "schema": SCHEMA,
        "items": [item.protocol_dict() for item in items],
        "source_probe_rows": list(source_rows),
        "decoder_prepare_rows": list(decoder_rows),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_cache_preparation_plan(
    items: Iterable[CacheProtocolItem],
) -> CachePreparationPlan:
    measured = tuple(items)
    if not measured:
        raise ValueError("cache preparation requires measured items")
    request_ids = [item.request_id for item in measured]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("measured cache protocol request IDs are duplicated")

    namespace_states: dict[tuple[str, str], CacheState] = {}
    namespace_owner_items: dict[tuple[str, str], int] = {}
    namespace_counts: dict[tuple[str, str], int] = {}
    representatives: dict[tuple[str, str], CacheProtocolItem] = {}
    ordered_namespaces: list[tuple[str, str]] = []
    for item in measured:
        key = item.namespace_key
        prior = namespace_states.get(key)
        if prior is not None and prior is not item.cache_state:
            raise ValueError(
                "one physical cache namespace has conflicting cache states")
        owner = namespace_owner_items.get(key)
        if owner is not None and owner != item.terminal_item:
            raise ValueError(
                "one physical cache namespace maps to multiple decoder pairs")
        if key not in representatives:
            representatives[key] = item
            ordered_namespaces.append(key)
        namespace_states[key] = item.cache_state
        namespace_owner_items[key] = item.terminal_item
        namespace_counts[key] = namespace_counts.get(key, 0) + 1

    for key, count in namespace_counts.items():
        state = namespace_states[key]
        if state in {CacheState.MISS, CacheState.D_ONLY} and count != 1:
            raise ValueError(
                f"{state.value} namespace cannot be measured more than once")

    source_rows = tuple(
        _source_probe_row(representatives[key])
        for key in ordered_namespaces
        if namespace_states[key] in {CacheState.P_ONLY, CacheState.BOTH}
    )
    decoder_rows = tuple(
        row
        for key in ordered_namespaces
        if namespace_states[key] in {CacheState.D_ONLY, CacheState.BOTH}
        for row in _decoder_rows(representatives[key])
    )
    prep_ids = [
        str(row["request_id"]) for row in (*source_rows, *decoder_rows)
    ]
    if len(prep_ids) != len(set(prep_ids)):
        raise ValueError("cache preparation request IDs are duplicated")
    return CachePreparationPlan(
        items=measured,
        source_probe_rows=source_rows,
        decoder_prepare_rows=decoder_rows,
        fingerprint_sha256=_fingerprint(
            measured, source_rows, decoder_rows),
    )


__all__ = [
    "CachePreparationPlan",
    "CacheProtocolItem",
    "PREP_OUTPUT_TOKENS",
    "SCHEMA",
    "build_cache_preparation_plan",
]
