#!/usr/bin/env python3
"""Canonical two-replica frontend for the actual-vLLM Elastic-PD path.

The fixed and predictor baselines retain the historical item-modulo pair
placement. The full TEMPO arm additionally balances outstanding decode-token
reservations across the two decoder pairs. A reservation lives until HTTP EOF,
so prefill completion cannot make a still-decoding pair look idle.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from eval.sota_4node.tempo_pd_frontend_v1 import pair_index
from eval.sota_4node import tempo_pd_cache_reuse as cache_reuse


ROUTER_SCHEMA = "tempo-elastic-pd-router-canonical"
FRONTEND_SCHEMA = "tempo-elastic-pd-frontend-canonical-semantic-pressure-4"
PAIR_POLICY = "tempo-min-outstanding-decode-tokens-v1"
PAIR_AFFINITY_POLICY = "warm-prompt-sha256-owner-set-v2"
BUCKET_ROTATION_PAIR_POLICY = (
    "tempo-cache-stable-log2-decode-bucket-rotation-v1")
PAIR_POLICY_ENV = "TEMPO_PD_FRONTEND_PAIR_POLICY"
REPLICATE_WARM_AFFINITY_ENV = "TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY"
COLD_MEASURED_ENV = "TEMPO_PD_BENCHMARK_COLD_MEASURED"
MAX_NUM_SEQS_ENV = "TEMPO_VLLM_MAX_NUM_SEQS"
P_ONLY_MEASURED_MARKER = "-cache-p-only-measured-"
D_ONLY_MEASURED_MARKER = "-cache-d-only-measured-"
BOTH_MEASURED_MARKER = "-cache-both-measured-"
MISS_MEASURED_MARKER = "-cache-miss-measured-"
D_CACHE_SEED_MARKER = "-cache-d-seed-"
D_CACHE_PROBE_MARKER = "-cache-d-probe-"
AFFINITY_SHADOW_MARKER = "-affinity-shadow-p"
C4_PHYSICAL_WARM_PREFIX = "-c4-cache-p-only-warm"
C4_PHYSICAL_MARKER = "-physical-"
_PAIR_POLICIES = {PAIR_POLICY, BUCKET_ROTATION_PAIR_POLICY}
# Request IDs use benchmark arm labels; response headers use policy labels.
_ARM = re.compile(r"^epd-(local|remote|predictor|tempo)-")
_WARM_SEED_OUTPUT = re.compile(r"-warm-seed-o([1-9][0-9]*)-")
_FORWARDED = (
    "x-tempo-pd-schema", "x-tempo-pd-request-id", "x-tempo-pd-arm",
    "x-tempo-pd-route", "x-tempo-pd-reason", "x-tempo-pd-profile",
    "x-tempo-pd-profile-sha256",
)


def request_arm(request_id: str) -> str:
    match = _ARM.match(request_id)
    if match is None:
        raise ValueError("request ID does not encode an Elastic-PD arm")
    return match.group(1)


def c4_physical_pair_pin(request_id: str, arm: str) -> bool:
    """Pin only unmeasured C4 physical seeds to their item owner pair."""
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be nonempty")
    if arm not in {"local", "remote", "predictor", "tempo"}:
        raise ValueError("Elastic-PD arm is invalid")
    return (
        arm == "tempo"
        and C4_PHYSICAL_WARM_PREFIX in request_id
        and C4_PHYSICAL_MARKER in request_id
    )


def requires_warm_pair_affinity(
    request_id: str, arm: str, *, cold_measured: bool,
) -> bool:
    """Require warm ownership only for cache-conditioned measurements."""
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be nonempty")
    if arm not in {"local", "remote", "predictor", "tempo"}:
        raise ValueError("Elastic-PD arm is invalid")
    if type(cold_measured) is not bool:
        raise TypeError("cold_measured must be bool")
    return (
        arm == "tempo"
        and (
            P_ONLY_MEASURED_MARKER in request_id
            or D_ONLY_MEASURED_MARKER in request_id
            or BOTH_MEASURED_MARKER in request_id
            or D_CACHE_PROBE_MARKER in request_id
            or "-measured-" in request_id and not cold_measured
        )
    )


def placement_decode_tokens(request_id: str, decode_tokens: int) -> int:
    """Recover the measured decode shape for a short unmeasured seed."""
    match = _WARM_SEED_OUTPUT.search(request_id)
    if match is None:
        return decode_tokens
    return int(match.group(1))


def uses_decoder_affinity(
    request_id: str, *, decoder_prefix_caching: bool,
    affinity_required: bool,
    decoder_reuse_items: frozenset[int] | None,
) -> bool:
    """Return whether this request may select/register a decoder owner."""

    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be nonempty")
    if type(decoder_prefix_caching) is not bool:
        raise TypeError("decoder_prefix_caching must be bool")
    if type(affinity_required) is not bool:
        raise TypeError("affinity_required must be bool")
    if not decoder_prefix_caching:
        return False
    if any(marker in request_id for marker in (
        D_CACHE_SEED_MARKER,
        D_CACHE_PROBE_MARKER,
        D_ONLY_MEASURED_MARKER,
        BOTH_MEASURED_MARKER,
    )):
        return True
    if any(marker in request_id for marker in (
        P_ONLY_MEASURED_MARKER,
        MISS_MEASURED_MARKER,
    )):
        return False
    return affinity_required and cache_reuse.reuses_decoder_cache(
        request_id, decoder_reuse_items)


def bucket_rotated_pair(
    preferred: int, decode_tokens: int, pair_count: int,
) -> int:
    """Rotate logical stripes across pairs without breaking cache affinity."""
    if type(decode_tokens) is not int or decode_tokens <= 0:
        raise ValueError("decode_tokens must be positive")
    if type(pair_count) is not int or pair_count <= 0:
        raise ValueError("pair_count must be positive")
    if type(preferred) is not int or not 0 <= preferred < pair_count:
        raise ValueError("preferred pair is out of range")
    decode_bucket = decode_tokens.bit_length() - 1
    return (preferred + decode_bucket) % pair_count


def affinity_shadow_request_id(request_id: str, pair_index_value: int) -> str:
    """Derive an auditable warm-only request ID for the other producer pair."""
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be nonempty")
    if "-warm-" not in request_id:
        raise ValueError("affinity shadow requests must be warm traffic")
    if AFFINITY_SHADOW_MARKER in request_id:
        raise ValueError("request ID is already an affinity shadow")
    if type(pair_index_value) is not int or pair_index_value < 0:
        raise ValueError("shadow pair index must be nonnegative")
    item_marker = "-item-"
    if item_marker not in request_id:
        raise ValueError("warm request ID lacks item marker")
    return request_id.replace(
        item_marker,
        f"{AFFINITY_SHADOW_MARKER}{pair_index_value}{item_marker}", 1)


class PairLoadLedger:
    """Race-free, auditable decoder-load reservations."""

    def __init__(self, pair_count: int, tempo_policy: str = PAIR_POLICY):
        if type(pair_count) is not int or pair_count <= 0:
            raise ValueError("pair_count must be positive")
        if tempo_policy not in _PAIR_POLICIES:
            raise ValueError("unsupported TEMPO frontend pair policy")
        self._loads = [0] * pair_count
        self._tempo_policy = tempo_policy
        self._active: dict[str, int] = {}
        self._rows: dict[str, dict[str, Any]] = {}
        self._affinity: dict[str, set[int]] = {}
        self._affinity_evidence: dict[str, dict[int, str]] = {}
        self._decode_affinity: dict[str, int] = {}
        self._decode_affinity_evidence: dict[str, str] = {}
        self._lock = asyncio.Lock()

    def _active_counts_locked(self) -> list[int]:
        counts = [0] * len(self._loads)
        for selected in self._active.values():
            counts[selected] += 1
        return counts

    async def reserve(
        self, request_id: str, decode_tokens: int, preferred: int, *,
        dynamic: bool, placement_tokens: int | None = None,
        affinity_key: str | None = None, affinity_seed: bool = False,
        affinity_required: bool = False,
        affinity_owner_count_required: int | None = 1,
        prefer_decode_affinity: bool = False,
        decode_affinity_required: bool = False,
        pair_pin_preferred: bool = False,
    ) -> dict[str, Any]:
        if not request_id:
            raise ValueError("request_id must be nonempty")
        if type(decode_tokens) is not int or decode_tokens <= 0:
            raise ValueError("decode_tokens must be positive")
        if type(preferred) is not int or not 0 <= preferred < len(self._loads):
            raise ValueError("preferred pair is out of range")
        if placement_tokens is None:
            placement_tokens = decode_tokens
        if type(placement_tokens) is not int or placement_tokens <= 0:
            raise ValueError("placement_tokens must be positive")
        if affinity_key is not None and (
            not isinstance(affinity_key, str) or len(affinity_key) != 64
        ):
            raise ValueError("affinity_key must be a SHA-256 hex digest")
        if affinity_seed and affinity_required:
            raise ValueError("pair affinity cannot be seeded and required")
        if affinity_owner_count_required is not None and (
            type(affinity_owner_count_required) is not int
            or not 1 <= affinity_owner_count_required <= len(self._loads)
        ):
            raise ValueError("required affinity owner count is out of range")
        if (affinity_seed or affinity_required) and (
            not dynamic or affinity_key is None
        ):
            raise ValueError("pair affinity requires a dynamic prompt key")
        if (prefer_decode_affinity or decode_affinity_required) and (
            not dynamic or affinity_key is None
        ):
            raise ValueError("decode affinity requires a dynamic prompt key")
        if decode_affinity_required and not prefer_decode_affinity:
            raise ValueError(
                "required decode affinity must also be preferred")
        if type(pair_pin_preferred) is not bool:
            raise TypeError("preferred pair pin must be bool")
        if pair_pin_preferred and not dynamic:
            raise ValueError("preferred pair pin requires dynamic routing")
        async with self._lock:
            if request_id in self._rows:
                raise ValueError(f"duplicate pair reservation: {request_id}")
            before = tuple(self._loads)
            active_before = tuple(self._active_counts_locked())
            selected = preferred
            affinity_hit = False
            affinity_created = False
            decode_affinity_hit = False
            if dynamic:
                source_owners = (
                    self._affinity.get(affinity_key)
                    if affinity_key is not None else None
                )
                if affinity_required and not source_owners:
                    raise ValueError(
                        "measured TEMPO prompt lacks warm pair affinity")
                if (
                    affinity_required
                    and affinity_owner_count_required is not None
                    and source_owners is not None
                    and len(source_owners) != affinity_owner_count_required
                ):
                    raise ValueError(
                        "measured TEMPO prompt lacks replicated pair affinity")
                decode_owner = (
                    self._decode_affinity.get(affinity_key)
                    if prefer_decode_affinity and affinity_key is not None
                    else None
                )
                if decode_affinity_required and decode_owner is None:
                    raise ValueError(
                        "TEMPO prompt lacks proven decoder pair affinity")
                owners = (
                    {decode_owner} if decode_owner is not None else source_owners)
                decode_affinity_hit = decode_owner is not None
                if owners:
                    minimum = min(before[index] for index in owners)
                    candidates = {
                        index for index in owners if before[index] == minimum
                    }
                    selected = (
                        preferred if preferred in candidates else min(candidates))
                    affinity_hit = True
                else:
                    if pair_pin_preferred:
                        selected = preferred
                    elif self._tempo_policy == PAIR_POLICY:
                        minimum = min(before)
                        candidates = {
                            index for index, load in enumerate(before)
                            if load == minimum
                        }
                        selected = (
                            preferred if preferred in candidates
                            else min(candidates))
                    else:
                        selected = bucket_rotated_pair(
                            preferred, placement_tokens, len(self._loads))
                    if affinity_seed:
                        assert affinity_key is not None
                        self._affinity[affinity_key] = {selected}
                        affinity_created = True
            self._loads[selected] += decode_tokens
            active_after = list(active_before)
            active_after[selected] += 1
            owner_indices = sorted(
                self._affinity.get(affinity_key, set()))
            evidence_by_owner = self._affinity_evidence.get(
                affinity_key, {})
            evidence_request_ids = [
                evidence_by_owner[index] for index in owner_indices
                if index in evidence_by_owner
            ]
            row = {
                "request_id": request_id,
                "frontend_pair_index": selected,
                "frontend_pair_preferred_index": preferred,
                "frontend_pair_policy": (
                    self._tempo_policy if dynamic else "item_modulo_v1"),
                "frontend_pair_physical_seed_pin": pair_pin_preferred,
                "frontend_decode_tokens_reserved": decode_tokens,
                "frontend_pair_placement_decode_tokens": placement_tokens,
                "frontend_pair_affinity_policy": (
                    PAIR_AFFINITY_POLICY if dynamic and affinity_key is not None
                    else "not_applicable"),
                "frontend_pair_affinity_key_sha256": affinity_key,
                "frontend_pair_affinity_hit": affinity_hit,
                "frontend_pair_decode_affinity_preferred": (
                    prefer_decode_affinity),
                "frontend_pair_decode_affinity_required": (
                    decode_affinity_required),
                "frontend_pair_decode_affinity_hit": decode_affinity_hit,
                "frontend_pair_decode_affinity_owner": (
                    self._decode_affinity.get(affinity_key)),
                "frontend_pair_affinity_created": affinity_created,
                "frontend_pair_affinity_required": affinity_required,
                "frontend_pair_affinity_owner_count_required": (
                    affinity_owner_count_required),
                "frontend_pair_affinity_owner_index": (
                    selected if dynamic and affinity_key is not None else None),
                "frontend_pair_affinity_owner_indices": owner_indices,
                "frontend_pair_affinity_replica_count": len(owner_indices),
                "frontend_pair_affinity_evidence_request_ids": (
                    evidence_request_ids),
                "frontend_pair_affinity_registration_source": (
                    "completed_warm_probe_eof"
                    if owner_indices
                    and len(evidence_request_ids) == len(owner_indices)
                    else "reservation_or_unproven"),
                "frontend_pair_load_before": list(before),
                "frontend_pair_load_after_reserve": list(self._loads),
                "frontend_pair_active_requests_before": list(active_before),
                "frontend_pair_active_requests_after_reserve": active_after,
                "frontend_pair_released": False,
            }
            self._active[request_id] = selected
            self._rows[request_id] = row
            return dict(row)

    async def register_affinity_replicas(
        self, affinity_key: str, owner_indices: set[int], *,
        evidence_request_ids: dict[int, str] | None = None,
    ) -> list[int]:
        """Register only producer owners proven by completed warm probes."""
        if not isinstance(affinity_key, str) or len(affinity_key) != 64:
            raise ValueError("affinity_key must be a SHA-256 hex digest")
        if (
            not isinstance(owner_indices, set)
            or not owner_indices
            or any(
                type(index) is not int
                or not 0 <= index < len(self._loads)
                for index in owner_indices
            )
        ):
            raise ValueError("affinity replica owners are invalid")
        if evidence_request_ids is not None and (
            not isinstance(evidence_request_ids, dict)
            or set(evidence_request_ids) != owner_indices
            or any(
                type(index) is not int
                or not isinstance(value, str)
                or "-warm-" not in value
                for index, value in evidence_request_ids.items()
            )
        ):
            raise ValueError("affinity replica evidence is invalid")
        async with self._lock:
            evidence = self._affinity_evidence.get(affinity_key, {})
            if evidence_request_ids is not None:
                for index, value in evidence_request_ids.items():
                    prior = evidence.get(index)
                    if prior is not None and prior != value:
                        raise ValueError("affinity replica evidence changed")
            owners = self._affinity.setdefault(affinity_key, set())
            owners.update(owner_indices)
            if evidence_request_ids is not None:
                evidence = self._affinity_evidence.setdefault(
                    affinity_key, {})
                evidence.update(evidence_request_ids)
            return sorted(owners)

    async def register_decode_affinity(
        self, affinity_key: str, owner_index: int, *,
        evidence_request_id: str,
    ) -> int:
        """Pin a repeated prompt to the decoder proven by completed EOF."""
        if not isinstance(affinity_key, str) or len(affinity_key) != 64:
            raise ValueError("decode affinity key must be a SHA-256 digest")
        if (
            type(owner_index) is not int
            or not 0 <= owner_index < len(self._loads)
        ):
            raise ValueError("decode affinity owner is invalid")
        if not isinstance(evidence_request_id, str) or not (
            "-measured-" in evidence_request_id
            or D_CACHE_SEED_MARKER in evidence_request_id
            or D_CACHE_PROBE_MARKER in evidence_request_id
        ):
            raise ValueError(
                "decode affinity requires completed seed/probe/measured EOF")
        async with self._lock:
            prior = self._decode_affinity.get(affinity_key)
            if prior is not None and prior != owner_index:
                raise ValueError("decode affinity owner changed")
            if prior is None:
                self._decode_affinity[affinity_key] = owner_index
                self._decode_affinity_evidence[
                    affinity_key] = evidence_request_id
            return owner_index
    async def clear_decode_affinity_for_cache_reset(self) -> int:
        async with self._lock:
            if self._active or any(self._loads):
                raise RuntimeError(
                    "decoder cache reset requires a quiescent pair ledger")
            cleared = len(self._decode_affinity)
            self._decode_affinity.clear()
            self._decode_affinity_evidence.clear()
            return cleared



    async def release(self, request_id: str) -> bool:
        async with self._lock:
            selected = self._active.pop(request_id, None)
            if selected is None:
                return False
            row = self._rows[request_id]
            tokens = int(row["frontend_decode_tokens_reserved"])
            if self._loads[selected] < tokens:
                raise RuntimeError("pair load ledger underflow")
            self._loads[selected] -= tokens
            row["frontend_pair_released"] = True
            row["frontend_pair_load_after_release"] = list(self._loads)
            row["frontend_pair_active_requests_after_release"] = (
                self._active_counts_locked())
            return True

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "loads": list(self._loads),
                "active": len(self._active),
                "active_by_pair": self._active_counts_locked(),
                "pair_affinity_policy": PAIR_AFFINITY_POLICY,
                "pair_affinity_entries": len(self._affinity),
                "pair_affinity_replicas": sum(
                    len(owners) for owners in self._affinity.values()),
                "decode_affinity_entries": len(self._decode_affinity),
                "decode_affinity_evidence": len(self._decode_affinity_evidence),
                "tempo_pair_policy": self._tempo_policy,
                "rows": {
                    key: dict(value) for key, value in self._rows.items()
                },
            }


def _completion_shape(payload: bytes) -> tuple[int, str]:
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("completion body must be JSON") from exc
    tokens = value.get("max_tokens") if isinstance(value, dict) else None
    if type(tokens) is not int or tokens <= 0:
        raise ValueError(
            "completion max_tokens must be a positive integer")
    prompt = value.get("prompt") if isinstance(value, dict) else None
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("completion prompt must be a nonempty string")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return tokens, prompt_sha256


def _decode_tokens(payload: bytes) -> int:
    return _completion_shape(payload)[0]


async def _drain_affinity_shadow(
    client: httpx.AsyncClient, payload: bytes, headers: dict[str, str],
    shadow_request_id: str,
) -> None:
    """Execute and fully verify one unmeasured warm copy."""
    shadow_headers = dict(headers)
    shadow_headers["X-Tempo-Request-Id"] = shadow_request_id
    shadow_request = client.build_request(
        "POST", "/v1/completions", content=payload, headers=shadow_headers)
    response = None
    try:
        response = await client.send(shadow_request, stream=True)
        response.raise_for_status()
        if (
            response.headers.get("x-tempo-pd-schema") != ROUTER_SCHEMA
            or response.headers.get("x-tempo-pd-request-id")
            != shadow_request_id
            or response.headers.get("x-tempo-pd-route")
            != "official_lmcache_remote_prefill"
        ):
            raise ValueError(
                "replicated-affinity shadow provenance mismatch")
        async for _chunk in response.aiter_raw():
            pass
    finally:
        if response is not None:
            await response.aclose()


async def _get_pair_decisions(
    client: httpx.AsyncClient,
) -> tuple[httpx.Response, int]:
    """Retry one idempotent telemetry GET after a stale keep-alive close."""
    try:
        return await client.get("/tempo/decisions"), 0
    except httpx.RemoteProtocolError:
        # Uvicorn may close an idle keep-alive connection immediately before
        # HTTPX reuses it. The failed GET has no side effects; one bounded
        # retry opens a usable connection while a repeated failure remains
        # fail-closed and visible to the benchmark client.
        return await client.get("/tempo/decisions"), 1


def build_app(pair_urls: list[str]) -> FastAPI:
    if len(pair_urls) != 2 or len(set(pair_urls)) != 2:
        raise ValueError("exactly two unique pair router URLs are required")
    if any(not value.startswith(("http://", "https://"))
           for value in pair_urls):
        raise ValueError("pair router URLs must be HTTP(S)")
    tempo_pair_policy = os.environ.get(PAIR_POLICY_ENV, PAIR_POLICY)
    if tempo_pair_policy not in _PAIR_POLICIES:
        raise ValueError(f"unsupported {PAIR_POLICY_ENV}")
    raw_replicate_warm_affinity = os.environ.get(
        REPLICATE_WARM_AFFINITY_ENV, "0")
    if raw_replicate_warm_affinity not in ("0", "1"):
        raise ValueError(
            f"{REPLICATE_WARM_AFFINITY_ENV} must be 0 or 1")
    replicate_warm_affinity = raw_replicate_warm_affinity == "1"
    raw_cold_measured = os.environ.get(COLD_MEASURED_ENV, "0")
    if raw_cold_measured not in ("0", "1"):
        raise ValueError(
            f"{COLD_MEASURED_ENV} must be 0 or 1")
    cold_measured = raw_cold_measured == "1"
    raw_decoder_prefix_caching = os.environ.get(
        "TEMPO_VLLM_DECODER_PREFIX_CACHING", "0")
    if raw_decoder_prefix_caching not in ("0", "1"):
        raise ValueError(
            "TEMPO_VLLM_DECODER_PREFIX_CACHING must be 0 or 1")
    decoder_prefix_caching = raw_decoder_prefix_caching == "1"
    decoder_reuse_items = cache_reuse.parse_reuse_items(
        os.environ.get("TEMPO_PD_DECODER_REUSE_ITEMS", "all"))
    raw_max_num_seqs = os.environ.get(MAX_NUM_SEQS_ENV, "8")
    try:
        max_num_seqs = int(raw_max_num_seqs)
    except ValueError as exc:
        raise ValueError(f"{MAX_NUM_SEQS_ENV} must be 8 or 16") from exc
    if max_num_seqs not in (8, 16):
        raise ValueError(f"{MAX_NUM_SEQS_ENV} must be 8 or 16")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.clients = [
            httpx.AsyncClient(base_url=value, timeout=None)
            for value in pair_urls
        ]
        app.state.pair_load = PairLoadLedger(
            len(pair_urls), tempo_policy=tempo_pair_policy)
        app.state.decision_fetch_remote_protocol_retries = 0
        try:
            yield
        finally:
            for client in app.state.clients:
                await client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        state = await app.state.pair_load.snapshot()
        return {
            "schema": FRONTEND_SCHEMA,
            "ok": True,
            "pairs": len(pair_urls),
            "pair_policy": PAIR_POLICY,
            "tempo_pair_policy": state["tempo_pair_policy"],
            "pair_loads": state["loads"],
            "active_pair_reservations": state["active"],
            "active_pair_reservations_by_pair": state["active_by_pair"],
            "vllm_max_num_seqs": max_num_seqs,
            "pair_affinity_policy": state["pair_affinity_policy"],
            "pair_affinity_entries": state["pair_affinity_entries"],
            "pair_affinity_replicas": state["pair_affinity_replicas"],
            "replicate_warm_affinity": replicate_warm_affinity,
            "benchmark_cold_measured": cold_measured,
            "decision_fetch_remote_protocol_retries": (
                app.state.decision_fetch_remote_protocol_retries),
        }

    @app.post("/tempo/reset_decoder_prefix_cache")
    async def reset_decoder_prefix_cache() -> dict[str, Any]:
        state = await app.state.pair_load.snapshot()
        if state["active"] != 0 or any(state["loads"]):
            raise HTTPException(
                status_code=409,
                detail="decoder APC reset requires quiescent frontend",
            )
        responses = await asyncio.gather(*[
            client.post("/tempo/reset_decoder_prefix_cache")
            for client in app.state.clients
        ])
        for response in responses:
            response.raise_for_status()
            value = response.json()
            if not (
                isinstance(value, dict)
                and value.get("schema") == ROUTER_SCHEMA
                and value.get("success") is True
                and value.get("external_cache_reset") is False
            ):
                raise HTTPException(
                    status_code=502,
                    detail="pair decoder APC reset evidence mismatch",
                )
        cleared = await (
            app.state.pair_load.clear_decode_affinity_for_cache_reset())
        return {
            "schema": FRONTEND_SCHEMA,
            "success": True,
            "pair_decoder_resets": len(responses),
            "external_cache_reset": False,
            "decode_affinity_entries_cleared": cleared,
        }

    @app.get("/tempo/decisions")
    async def decisions() -> dict[str, Any]:
        fetched = await asyncio.gather(*[
            _get_pair_decisions(client)
            for client in app.state.clients
        ])
        responses = [response for response, _retries in fetched]
        app.state.decision_fetch_remote_protocol_retries += sum(
            retries for _response, retries in fetched)
        all_rows = []
        for response in responses:
            response.raise_for_status()
            value = response.json()
            if value.get("schema") != ROUTER_SCHEMA:
                raise HTTPException(
                    status_code=502, detail="pair decision schema mismatch")
            all_rows.extend(value.get("decisions", []))
        identifiers = [row.get("request_id") for row in all_rows]
        if len(identifiers) != len(set(identifiers)):
            raise HTTPException(
                status_code=502, detail="duplicate pair decision IDs")
        ledger = await app.state.pair_load.snapshot()
        assignments = ledger["rows"]
        rows = []
        shadow_rows = []
        for row in all_rows:
            row_request_id = row.get("request_id")
            if (
                isinstance(row_request_id, str)
                and AFFINITY_SHADOW_MARKER in row_request_id
            ):
                expected_cached = (
                    row.get("lmcache_source_cached_tokens") == 0
                    if "-warm-seed-" in row_request_id
                    else row.get("lmcache_source_full_hit_observed") is True
                )
                if not (
                    row.get("phase") == "complete"
                    and row.get("finished_ns") is not None
                    and row.get("error") is None
                    and row.get("route") == "official_lmcache_remote_prefill"
                    and expected_cached
                ):
                    raise HTTPException(
                        status_code=502,
                        detail="invalid replicated-affinity shadow evidence")
                shadow_rows.append(row)
                continue
            assignment = assignments.get(row_request_id)
            if assignment is None:
                raise HTTPException(
                    status_code=502,
                    detail="missing frontend pair assignment")
            row.update(assignment)
            rows.append(row)
        rows.sort(key=lambda row: str(row.get("request_id")))
        return {
            "schema": ROUTER_SCHEMA,
            "count": len(rows),
            "decisions": rows,
            "frontend_schema": FRONTEND_SCHEMA,
            "frontend_pair_policy": ledger["tempo_pair_policy"],
            "frontend_pair_loads": ledger["loads"],
            "frontend_active_pair_reservations": ledger["active"],
            "frontend_pair_affinity_policy": ledger["pair_affinity_policy"],
            "frontend_pair_affinity_entries": ledger["pair_affinity_entries"],
            "frontend_pair_affinity_replicas": ledger[
                "pair_affinity_replicas"],
            "frontend_replicate_warm_affinity": replicate_warm_affinity,
            "benchmark_cold_measured": cold_measured,
            "frontend_affinity_shadow_count": len(shadow_rows),
            "frontend_decision_fetch_remote_protocol_retries": (
                app.state.decision_fetch_remote_protocol_retries),
        }

    @app.post("/v1/completions")
    async def completions(request: Request):
        request_id = request.headers.get("x-tempo-request-id")
        if not request_id:
            raise HTTPException(
                status_code=400, detail="missing x-tempo-request-id")
        payload = await request.body()
        try:
            arm = request_arm(request_id)
            tokens, prompt_sha256 = _completion_shape(payload)
            placement_tokens = placement_decode_tokens(request_id, tokens)
            preferred = pair_index(request_id, len(pair_urls))
            tempo_warm = arm == "tempo" and "-warm-" in request_id
            d_cache_prepare = (
                D_CACHE_SEED_MARKER in request_id
                or D_CACHE_PROBE_MARKER in request_id)
            replicated_warm = (
                replicate_warm_affinity
                and tempo_warm
                and not d_cache_prepare)
            replicated_warm_probe = (
                replicated_warm and "-warm-seed-" not in request_id)
            affinity_required = requires_warm_pair_affinity(
                request_id, arm, cold_measured=cold_measured)
            affinity_seed = (
                tempo_warm
                and not affinity_required
                and (
                    not replicate_warm_affinity
                    or D_CACHE_SEED_MARKER in request_id))
            decoder_reuse = (
                arm == "tempo"
                and uses_decoder_affinity(
                    request_id,
                    decoder_prefix_caching=decoder_prefix_caching,
                    affinity_required=affinity_required,
                    decoder_reuse_items=decoder_reuse_items,
                ))
            decode_affinity_required = (
                arm == "tempo"
                and decoder_prefix_caching
                and (
                    D_CACHE_PROBE_MARKER in request_id
                    or D_ONLY_MEASURED_MARKER in request_id
                    or BOTH_MEASURED_MARKER in request_id))
            physical_pair_pin = c4_physical_pair_pin(request_id, arm)
            affinity_owner_count_required = (
                None
                if D_CACHE_PROBE_MARKER in request_id
                else 1
                if D_ONLY_MEASURED_MARKER in request_id
                else len(pair_urls)
                if replicate_warm_affinity and affinity_required
                else 1)
            assignment = await app.state.pair_load.reserve(
                request_id, tokens, preferred, dynamic=arm == "tempo",
                placement_tokens=placement_tokens,
                affinity_key=prompt_sha256 if arm == "tempo" else None,
                affinity_seed=affinity_seed,
                affinity_required=affinity_required,
                affinity_owner_count_required=affinity_owner_count_required,
                prefer_decode_affinity=decoder_reuse,
                decode_affinity_required=decode_affinity_required,
                pair_pin_preferred=physical_pair_pin)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        selected = int(assignment["frontend_pair_index"])
        client = app.state.clients[selected]
        headers = {
            "Content-Type": request.headers.get(
                "content-type", "application/json"),
            "X-Tempo-Request-Id": request_id,
        }
        for name in ("authorization", "x-tempo-remaining-deadline-ms"):
            value = request.headers.get(name)
            if value:
                headers[name] = value
        upstream = None
        shadow_selected = None
        try:
            if replicated_warm:
                shadow_selected = 1 - selected
                shadow_request_id = affinity_shadow_request_id(
                    request_id, shadow_selected)
                await _drain_affinity_shadow(
                    app.state.clients[shadow_selected], payload, headers,
                    shadow_request_id)
            headers.update({
                "X-Tempo-PD-Frontend-Pair-Index": str(selected),
                "X-Tempo-PD-Frontend-Decode-Tokens-Before": str(
                    assignment["frontend_pair_load_before"][selected]),
                "X-Tempo-PD-Frontend-Active-Requests-Before": str(
                    assignment[
                        "frontend_pair_active_requests_before"][selected]),
                "X-Tempo-PD-Frontend-Max-Num-Seqs": str(max_num_seqs),
            })
            upstream_request = client.build_request(
                "POST", "/v1/completions", content=payload, headers=headers)
            upstream = await client.send(upstream_request, stream=True)
            upstream.raise_for_status()
        except Exception as exc:
            if upstream is not None:
                await upstream.aclose()
            await app.state.pair_load.release(request_id)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        response_headers = {
            name: upstream.headers[name] for name in _FORWARDED
            if name in upstream.headers
        }
        response_headers.update({
            "X-Tempo-PD-Pair-Index": str(selected),
            "X-Tempo-PD-Pair-Policy": str(
                assignment["frontend_pair_policy"]),
            "X-Tempo-PD-Decode-Tokens-Reserved": str(tokens),
            "X-Tempo-PD-Placement-Decode-Tokens": str(placement_tokens),
            "X-Tempo-PD-Pair-Affinity-Policy": str(
                assignment["frontend_pair_affinity_policy"]),
            "X-Tempo-PD-Pair-Affinity-Hit": str(
                assignment["frontend_pair_affinity_hit"]).lower(),
            "X-Tempo-PD-Pair-Affinity-Owners": ",".join(
                str(index) for index in assignment[
                    "frontend_pair_affinity_owner_indices"]),
            "X-Tempo-PD-Replicate-Warm-Affinity": str(
                replicate_warm_affinity).lower(),
        })

        async def generate():
            stream_completed = False
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
                stream_completed = True
            finally:
                try:
                    await upstream.aclose()
                finally:
                    try:
                        if stream_completed and replicated_warm_probe:
                            assert shadow_selected is not None
                            assert isinstance(shadow_request_id, str)
                            await app.state.pair_load.register_affinity_replicas(
                                prompt_sha256, {selected, shadow_selected},
                                evidence_request_ids={
                                    selected: request_id,
                                    shadow_selected: shadow_request_id,
                                })
                        if stream_completed and decoder_reuse:
                            await app.state.pair_load.register_decode_affinity(
                                prompt_sha256, selected,
                                evidence_request_id=request_id)
                    finally:
                        released = await app.state.pair_load.release(request_id)
                        if not released:
                            raise RuntimeError(
                                "pair load reservation released twice")

        return StreamingResponse(
            generate(), media_type="text/event-stream",
            headers=response_headers)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pair-url", action="append", required=True)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(
        build_app(args.pair_url), host=args.host, port=args.port,
        log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
