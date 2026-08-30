#!/usr/bin/env python3
"""Canonical one- or two-replica frontend for the actual-vLLM Elastic-PD path.

The fixed and predictor baselines retain the historical item-modulo pair
placement. The full TEMPO arm additionally balances outstanding decode-token
reservations across the two decoder pairs. A reservation lives until HTTP EOF,
so prefill completion cannot make a still-decoding pair look idle.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager, suppress
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from eval.sota_4node.tempo_pd_frontend_v1 import pair_index
from eval.sota_4node import tempo_pd_cache_reuse as cache_reuse
from tempo.pd_elastic_controller import CacheResidency
from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_endpoint_profile import load_endpoint_service_profile
from tempo.pd_global_agent import RequestTriggeredTelemetryAgent
from tempo.pd_global_candidates import GlobalCandidateBuilder, PairCacheState
from tempo.pd_global_coordinator import GlobalAdmissionCoordinator
from tempo.pd_global_hierarchy import HierarchicalCandidateReducer
from tempo.pd_global_orchestrator import (
    GlobalDecisionKind,
    GlobalOrchestrator,
    GlobalRoute,
    PRIORITY_SERVICE_LANE_BINDING,
    global_failure_dict,
    global_failure_fingerprint,
    global_decision_dict,
    global_decision_fingerprint,
    global_service_lane_queue_promotion_dict,
    global_service_lane_queue_promotion_fingerprint,
    global_service_lane_reservation_failure_dict,
    global_service_lane_reservation_failure_fingerprint,
)
from tempo.pd_global_profile import load_global_profile
from tempo.pd_global_telemetry import FRONTEND_LEDGER_SCHEMA


ROUTER_SCHEMA = "tempo-elastic-pd-router-canonical"
FRONTEND_SCHEMA = "tempo-elastic-pd-frontend-canonical-semantic-pressure-4"
PAIR_POLICY = "tempo-min-outstanding-decode-tokens-v1"
PAIR_AFFINITY_POLICY = "warm-prompt-sha256-owner-set-v2"
CACHE_CHUNK_GROUP_SIZE = 256
BUCKET_ROTATION_PAIR_POLICY = (
    "tempo-cache-stable-log2-decode-bucket-rotation-v1")
PAIR_POLICY_ENV = "TEMPO_PD_FRONTEND_PAIR_POLICY"
REPLICATE_WARM_AFFINITY_ENV = "TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY"
COLD_MEASURED_ENV = "TEMPO_PD_BENCHMARK_COLD_MEASURED"
MAX_NUM_SEQS_ENV = "TEMPO_VLLM_MAX_NUM_SEQS"
GLOBAL_PROFILE_ENV = "TEMPO_GO_PROFILE"
GLOBAL_ELASTIC_PROFILE_ENV = "TEMPO_GO_ELASTIC_PROFILE"
GLOBAL_ENDPOINT_PROFILE_ENV = "TEMPO_GO_ENDPOINT_PROFILE"
GLOBAL_TOKENIZER_URL_ENV = "TEMPO_GO_TOKENIZER_URL"
GLOBAL_ABLATION_ENV = "TEMPO_GO_ABLATION"
GLOBAL_TENANT_HEADER = "x-tempo-tenant-id"
DECODER_BUSINESS_ADMISSION_SCHEMA = (
    "tempo-go-decoder-business-admission-v1")
GLOBAL_MESH_COMMIT_SCHEMA = "tempo-go-mesh-commit-v1"
GLOBAL_MESH_JOINT_COMMIT_SCHEMA = "tempo-go-mesh-joint-commit-v1"
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
_ARM = re.compile(
    r"^epd-(local|remote|predictor|queue_gpu|network_request_only|"
    r"app_global_only|tempo)-"
)
_WARM_SEED_OUTPUT = re.compile(r"-warm-seed-o([1-9][0-9]*)-")
_FORWARDED = (
    "x-tempo-pd-schema", "x-tempo-pd-request-id", "x-tempo-pd-arm",
    "x-tempo-pd-route", "x-tempo-pd-reason", "x-tempo-pd-profile",
    "x-tempo-pd-profile-sha256",
)


def tempo_route_failure_kind(error: BaseException) -> str:
    """Return a bounded, non-secret failure label for Candidate C.

    LMCache cache-key ownership failures are pair-scoped.  They are often
    surfaced by the backend as HTTP 500 after an EngineCore assertion, but
    the assertion invalidates the decoder/cache state shared by both semantic
    routes on that pair.  Keeping this classification here lets the global
    failure plane quarantine the correct domain without retrying or silently
    falling back under the same request ID.
    """

    if _is_lmcache_cache_key_failure(error):
        return "lmcache_cache_key_ownership_failure"

    if isinstance(error, httpx.HTTPStatusError):
        return f"upstream_http_status_{error.response.status_code}"
    if isinstance(error, httpx.HTTPError):
        return "upstream_transport_error"
    if isinstance(error, RuntimeError):
        return "route_commit_or_stream_runtime_error"
    return f"upstream_{type(error).__name__.lower()}"


def tempo_route_failure_scope(error: BaseException) -> str:
    """Quarantine one semantic route or the whole pair endpoint."""

    if _is_lmcache_cache_key_failure(error):
        return "pair"

    if isinstance(error, httpx.HTTPStatusError):
        return "route"
    if isinstance(error, (httpx.TransportError, RuntimeError)):
        return "pair"
    return "route"


def tempo_global_failure_scope(
    error: BaseException,
    *,
    decision: object,
    mesh_enabled: bool,
) -> str:
    """Narrow ordinary remote transport failures to one committed P->D edge."""

    scope = tempo_route_failure_scope(error)
    if (
        mesh_enabled
        and scope == "route"
        and getattr(decision, "route", None) is GlobalRoute.REMOTE
        and getattr(decision, "prefill_index", None) is not None
        and getattr(decision, "decoder_index", None) is not None
    ):
        return "edge"
    return scope


def _is_lmcache_cache_key_failure(error: BaseException) -> bool:
    """Recognize the bounded LMCache ownership failure receipt.

    ``_raise_upstream_status_with_body`` includes at most 4 KiB of the
    backend body in the exception string.  Matching the stable ownership
    markers keeps this classifier independent of a backend traceback while
    avoiding broad quarantine for unrelated HTTP 500 responses.
    """

    text = str(error).lower()
    return (
        ("cacheenginekey" in text or "cache engine key" in text)
        and "not found" in text
        and ("local data" in text or "lmcache" in text)
    )


async def _raise_upstream_status_with_body(response: httpx.Response) -> None:
    """Raise an HTTP error while retaining a bounded upstream contract body.

    ``httpx.HTTPStatusError`` normally omits the response body from its string
    form. That made a router-side global-commit rejection look like an opaque
    frontend 502 in native receipts. The bounded body is required evidence for
    distinguishing a route-contract rejection from a backend failure.
    """

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = (await response.aread()).decode(
            response.encoding or "utf-8", errors="replace").strip()
        if len(body) > 4096:
            body = body[:4096] + "..."
        detail = str(exc)
        if body:
            detail = f"{detail}; body={body}"
        raise httpx.HTTPStatusError(
            detail, request=exc.request, response=exc.response,
        ) from exc


def request_arm(request_id: str) -> str:
    match = _ARM.match(request_id)
    if match is None:
        raise ValueError("request ID does not encode an Elastic-PD arm")
    return match.group(1)


def c4_physical_pair_pin(request_id: str, arm: str) -> bool:
    """Pin only unmeasured C4 physical seeds to their item owner pair."""
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be nonempty")
    if arm not in {
        "local", "remote", "predictor", "queue_gpu",
        "network_request_only", "app_global_only", "tempo",
    }:
        raise ValueError("Elastic-PD arm is invalid")
    return (
        arm in {"tempo", "app_global_only", "queue_gpu"}
        and C4_PHYSICAL_WARM_PREFIX in request_id
        and C4_PHYSICAL_MARKER in request_id
    )


def requires_warm_pair_affinity(
    request_id: str, arm: str, *, cold_measured: bool,
) -> bool:
    """Require warm ownership only for cache-conditioned measurements."""
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be nonempty")
    if arm not in {
        "local", "remote", "predictor", "queue_gpu",
        "network_request_only", "app_global_only", "tempo",
    }:
        raise ValueError("Elastic-PD arm is invalid")
    if type(cold_measured) is not bool:
        raise TypeError("cold_measured must be bool")
    return (
        arm in {"tempo", "app_global_only"}
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
        committed_pair: int | None = None,
        queue_gpu_pair: int | None = None,
        queue_gpu_observation: dict[str, object] | None = None,
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
        if committed_pair is not None and (
            type(committed_pair) is not int
            or not 0 <= committed_pair < len(self._loads)
            or not dynamic
            or pair_pin_preferred
        ):
            raise ValueError(
                "committed pair requires unpinned dynamic routing")
        if queue_gpu_pair is not None and (
            type(queue_gpu_pair) is not int
            or not 0 <= queue_gpu_pair < len(self._loads)
            or not dynamic
            or committed_pair is not None
            or pair_pin_preferred
        ):
            raise ValueError(
                "queue-GPU pair requires unpinned dynamic routing")
        if queue_gpu_observation is not None and not isinstance(
            queue_gpu_observation, dict
        ):
            raise TypeError("queue-GPU observation must be an object")
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
                if committed_pair is not None:
                    selected = committed_pair
                    affinity_hit = bool(owners and selected in owners)
                    decode_affinity_hit = decode_owner == selected
                elif queue_gpu_pair is not None:
                    selected = queue_gpu_pair
                elif owners:
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
                    "physical-p-only-seed-owner-pin-v1"
                    if pair_pin_preferred
                    else "tempo-go-global-committed-pair-v1"
                    if committed_pair is not None
                    else "queue-gpu-vllm-scheduler-observe-only-v1"
                    if queue_gpu_pair is not None
                    else self._tempo_policy if dynamic else "item_modulo_v1"),
                "frontend_pair_global_commit": committed_pair is not None,
                "frontend_pair_queue_gpu_selection": queue_gpu_pair is not None,
                "frontend_queue_gpu_observation": (
                    dict(queue_gpu_observation)
                    if queue_gpu_observation is not None else None),
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
                "frontend_pair_affinity_invalidated": False,
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
                "schema": FRONTEND_LEDGER_SCHEMA,
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

    async def record_decoder_business_admission(
        self, request_id: str, receipt: dict[str, object],
    ) -> None:
        """Attach the profile-authorized decoder gate receipt to one row."""

        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != DECODER_BUSINESS_ADMISSION_SCHEMA
            or receipt.get("request_id") != request_id
            or receipt.get("status") not in {"held", "released"}
        ):
            raise ValueError("decoder business admission receipt is invalid")
        async with self._lock:
            row = self._rows.get(request_id)
            if row is None or request_id not in self._active:
                raise ValueError(
                    "decoder business admission lacks an active reservation")
            prior = row.get("frontend_decoder_business_admission")
            if (
                isinstance(prior, dict)
                and prior.get("status") == "released"
            ):
                raise ValueError("decoder business admission released twice")
            row["frontend_decoder_business_admission"] = dict(receipt)

    async def cache_states(
        self, affinity_key: str, *, explicit_cache_reset_miss: bool,
    ) -> tuple[PairCacheState, ...]:
        """Return only completed pair-local cache placement evidence."""

        if not isinstance(affinity_key, str) or len(affinity_key) != 64:
            raise ValueError("affinity_key must be a SHA-256 hex digest")
        if type(explicit_cache_reset_miss) is not bool:
            raise TypeError("explicit_cache_reset_miss must be bool")
        async with self._lock:
            registered_prefill = self._affinity.get(affinity_key, set())
            prefill_evidence = self._affinity_evidence.get(affinity_key, {})
            prefill_owners = {
                index for index in registered_prefill
                if index in prefill_evidence
            }
            decode_owner = self._decode_affinity.get(affinity_key)
            decode_proven = (
                decode_owner is not None
                and affinity_key in self._decode_affinity_evidence
            )
            any_completed_evidence = bool(prefill_owners) or decode_proven
            values = []
            for index in range(len(self._loads)):
                prefill = index in prefill_owners
                decode = decode_proven and decode_owner == index
                if prefill and decode:
                    residency = CacheResidency.BOTH
                elif prefill:
                    residency = CacheResidency.P_ONLY
                elif decode:
                    residency = CacheResidency.D_ONLY
                elif explicit_cache_reset_miss or any_completed_evidence:
                    residency = CacheResidency.MISS
                else:
                    residency = CacheResidency.UNKNOWN
                source = (
                    "explicit_cache_reset_miss"
                    if residency is CacheResidency.MISS
                    and explicit_cache_reset_miss
                    and not any_completed_evidence
                    else "unknown_fail_closed"
                    if residency is CacheResidency.UNKNOWN
                    else "completed_frontend_affinity_evidence"
                )
                values.append(PairCacheState(
                    pair_index=index,
                    residency=residency,
                    source=source,
                ))
            return tuple(values)


    async def record_global_decision(
        self, request_id: str, *, decision: dict[str, object],
        decision_sha256: str, tokenizer_ms: float,
        admission_arrival_ns: int | None = None,
        hierarchy_receipt: dict[str, object] | None = None,
        telemetry_preparation_receipt: dict[str, object] | None = None,
    ) -> None:
        if not isinstance(decision, dict):
            raise TypeError("global decision must be a dict")
        if (
            not isinstance(decision_sha256, str)
            or len(decision_sha256) != 64
        ):
            raise ValueError("global decision SHA must be a digest")
        if not isinstance(tokenizer_ms, (int, float)) or not math.isfinite(
            float(tokenizer_ms)
        ) or float(tokenizer_ms) < 0:
            raise ValueError("global tokenizer latency must be non-negative")
        if admission_arrival_ns is not None and (
            type(admission_arrival_ns) is not int or admission_arrival_ns < 0
        ):
            raise ValueError("global admission arrival timestamp is invalid")
        if telemetry_preparation_receipt is not None:
            if (
                not isinstance(telemetry_preparation_receipt, dict)
                or telemetry_preparation_receipt.get("schema")
                != "tempo-go-admission-preparation-v1"
            ):
                raise ValueError("telemetry preparation receipt schema mismatch")
        if hierarchy_receipt is not None:
            if not isinstance(hierarchy_receipt, dict):
                raise TypeError("hierarchy receipt must be an object")
            receipt = hierarchy_receipt.get("receipt")
            fingerprint = hierarchy_receipt.get("fingerprint_sha256")
            if not isinstance(receipt, dict) or receipt.get("schema") != (
                "tempo-go-reduction-receipt-v1"):
                raise ValueError("hierarchy receipt schema mismatch")
            if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                raise ValueError("hierarchy receipt fingerprint is invalid")
        async with self._lock:
            row = self._rows.get(request_id)
            if row is None or request_id not in self._active:
                raise ValueError("global decision lacks an active pair reservation")
            if row["frontend_pair_index"] != decision.get("pair_index"):
                raise ValueError("global decision and pair reservation differ")
            if "frontend_tempo_go_decision" in row:
                raise ValueError("global decision recorded twice")
            row["frontend_tempo_go_decision"] = dict(decision)
            row["frontend_tempo_go_decision_sha256"] = decision_sha256
            row["frontend_tempo_go_tokenizer_ms"] = float(tokenizer_ms)
            row["frontend_tempo_go_admission_arrival_ns"] = (
                admission_arrival_ns)
            row["frontend_tempo_go_hierarchy_reduction"] = (
                dict(hierarchy_receipt) if hierarchy_receipt is not None else None)
            row["frontend_tempo_go_telemetry_preparation"] = (
                dict(telemetry_preparation_receipt)
                if telemetry_preparation_receipt is not None else None)
            decided_ns = decision.get("decided_ns")
            if type(decided_ns) is int and admission_arrival_ns is not None:
                row["frontend_tempo_go_admission_wait_ns"] = max(
                    0, decided_ns - admission_arrival_ns)
            else:
                row["frontend_tempo_go_admission_wait_ns"] = None

    async def record_service_lane_queue_promotion(
        self,
        request_id: str,
        *,
        decision: dict[str, object],
        decision_sha256: str,
        promotion: dict[str, object],
        promotion_sha256: str,
    ) -> None:
        """Replace only the queue-lease bit of an already recorded decision."""

        if not isinstance(decision, dict) or not isinstance(promotion, dict):
            raise TypeError("queue promotion evidence must be objects")
        for name, value in (
            ("decision_sha256", decision_sha256),
            ("promotion_sha256", promotion_sha256),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a digest")
        if promotion.get("status") != "promoted":
            raise ValueError("queue promotion receipt is not promoted")
        if promotion.get("request_id") != request_id:
            raise ValueError("queue promotion request identity differs")
        async with self._lock:
            row = self._rows.get(request_id)
            if row is None or request_id not in self._active:
                raise ValueError("queue promotion lacks an active reservation")
            old = row.get("frontend_tempo_go_decision")
            old_sha = row.get("frontend_tempo_go_decision_sha256")
            if not isinstance(old, dict) or not isinstance(old_sha, str):
                raise ValueError("queue promotion lacks an initial decision")
            if "frontend_tempo_go_service_lane_queue_promotion" in row:
                raise ValueError("queue promotion was recorded twice")
            immutable = (
                "request_id", "tenant_id", "kind", "pair_index",
                "prefill_index", "decoder_index", "edge_id", "route",
                "selected_work", "cache_group_key", "joint_actuation",
            )
            if any(old.get(name) != decision.get(name) for name in immutable):
                raise ValueError("queue promotion changed route ownership")
            if (
                old.get("queue_lease") is not False
                or decision.get("queue_lease") is not True
            ):
                raise ValueError("queue promotion lease transition is invalid")
            if old_sha == decision_sha256:
                raise ValueError("queue promotion did not change decision SHA")
            row["frontend_tempo_go_initial_decision"] = dict(old)
            row["frontend_tempo_go_initial_decision_sha256"] = old_sha
            row["frontend_tempo_go_decision"] = dict(decision)
            row["frontend_tempo_go_decision_sha256"] = decision_sha256
            row["frontend_tempo_go_service_lane_queue_promotion"] = dict(
                promotion)
            row[
                "frontend_tempo_go_service_lane_queue_promotion_sha256"
            ] = promotion_sha256

    async def record_service_lane_queue_promotion_rejection(
        self,
        request_id: str,
        *,
        promotion: dict[str, object],
        promotion_sha256: str,
    ) -> None:
        """Preserve a fail-closed promotion decision before lease release."""

        if not isinstance(promotion, dict):
            raise TypeError("queue promotion rejection must be an object")
        if (
            not isinstance(promotion_sha256, str)
            or len(promotion_sha256) != 64
        ):
            raise ValueError("queue promotion rejection SHA is invalid")
        if (
            promotion.get("status") != "rejected"
            or promotion.get("request_id") != request_id
        ):
            raise ValueError("queue promotion rejection identity differs")
        async with self._lock:
            row = self._rows.get(request_id)
            if row is None or request_id not in self._active:
                raise ValueError(
                    "queue promotion rejection lacks an active reservation")
            if "frontend_tempo_go_service_lane_queue_promotion" in row:
                raise ValueError("queue promotion was recorded twice")
            decision = row.get("frontend_tempo_go_decision")
            if (
                not isinstance(decision, dict)
                or decision.get("queue_lease") is not False
            ):
                raise ValueError(
                    "queue promotion rejection lacks a direct global decision")
            row["frontend_tempo_go_service_lane_queue_promotion"] = dict(
                promotion)
            row[
                "frontend_tempo_go_service_lane_queue_promotion_sha256"
            ] = promotion_sha256

    def _invalidate_affinity_for_pair_locked(
        self, affinity_key: str, pair_index: int,
    ) -> bool:
        """Drop stale cache/decoder affinity after a pair-scoped failure.

        A completed warm probe proves that a cache was usable at that point;
        it does not prove that a later push-based LMCache receiver still owns
        every chunk.  A pair-scoped EngineCore/cache failure therefore
        invalidates only the failed pair's evidence.  Remaining replicas stay
        valid, while an empty owner set becomes UNKNOWN and is fail-closed by
        ``cache_states`` until a new completed probe registers evidence.
        """

        owners = self._affinity.get(affinity_key)
        if not owners or pair_index not in owners:
            return False
        owners.remove(pair_index)
        evidence = self._affinity_evidence.get(affinity_key)
        if evidence is not None:
            evidence.pop(pair_index, None)
        if not owners:
            self._affinity.pop(affinity_key, None)
            self._affinity_evidence.pop(affinity_key, None)
        if self._decode_affinity.get(affinity_key) == pair_index:
            self._decode_affinity.pop(affinity_key, None)
            self._decode_affinity_evidence.pop(affinity_key, None)
        return True

    async def record_global_failure(
        self, request_id: str, *, failure: dict[str, object],
        failure_sha256: str,
    ) -> None:
        """Persist a route-failure receipt before releasing pair load."""

        if not isinstance(failure, dict):
            raise TypeError("global failure must be a dict")
        if (
            not isinstance(failure_sha256, str)
            or len(failure_sha256) != 64
        ):
            raise ValueError("global failure SHA must be a digest")
        if failure.get("request_id") != request_id:
            raise ValueError("global failure request ID differs")
        if failure.get("terminal_phase") != "failed":
            raise ValueError("global failure is not terminally failed")
        async with self._lock:
            row = self._rows.get(request_id)
            if row is None or request_id not in self._active:
                raise ValueError("global failure lacks an active pair reservation")
            if row["frontend_pair_index"] != failure.get("pair_index"):
                raise ValueError("global failure and pair reservation differ")
            if "frontend_tempo_go_failure" in row:
                raise ValueError("global failure recorded twice")
            row["frontend_tempo_go_failure"] = dict(failure)
            row["frontend_tempo_go_failure_sha256"] = failure_sha256
            row["frontend_tempo_go_failure_kind"] = failure.get(
                "failure_kind")
            row["frontend_tempo_go_failure_scope"] = failure.get(
                "quarantine_scope")
            row["global_decision_reason"] = failure.get("reason")
            row["phase"] = "failed"
            row["error"] = failure.get("failure_kind")
            if failure.get("quarantine_scope") == "pair":
                affinity_key = row.get(
                    "frontend_pair_affinity_key_sha256")
                invalidated = self._invalidate_affinity_for_pair_locked(
                    affinity_key,
                    int(failure["pair_index"]),
                ) if isinstance(affinity_key, str) else False
                row["frontend_pair_affinity_invalidated"] = invalidated
                row["frontend_pair_affinity_invalidation_reason"] = (
                    "pair_scoped_global_failure"
                    if invalidated else None
                )

    async def record_global_reservation_failure(
        self, request_id: str, *, failure: dict[str, object],
        failure_sha256: str,
    ) -> None:
        """Persist an endpoint service-lane handshake failure.

        Unlike route failure, this receipt must not quarantine the route: the
        pair router never started the upstream request.  It is still recorded
        before the frontend decoder-load reservation is released.
        """

        if not isinstance(failure, dict):
            raise TypeError("service-lane failure must be an object")
        if failure.get("schema") != (
            "tempo-go-service-lane-reservation-v1"):
            raise ValueError("service-lane failure schema mismatch")
        if failure.get("request_id") != request_id:
            raise ValueError("service-lane failure request ID differs")
        if failure.get("terminal_phase") != "failed":
            raise ValueError("service-lane failure is not terminal")
        if (
            not isinstance(failure_sha256, str)
            or len(failure_sha256) != 64
        ):
            raise ValueError("service-lane failure SHA must be a digest")
        async with self._lock:
            row = self._rows.get(request_id)
            if row is None or request_id not in self._active:
                raise ValueError(
                    "service-lane failure lacks active pair reservation")
            if row["frontend_pair_index"] != failure.get("pair_index"):
                raise ValueError(
                    "service-lane failure and pair reservation differ")
            if "frontend_tempo_go_reservation_failure" in row:
                raise ValueError("service-lane failure recorded twice")
            row["frontend_tempo_go_reservation_failure"] = dict(failure)
            row["frontend_tempo_go_reservation_failure_sha256"] = (
                failure_sha256)
            row["frontend_tempo_go_failure_kind"] = failure.get(
                "failure_kind")
            row["frontend_tempo_go_failure_scope"] = "service_lane"
            row["global_decision_reason"] = failure.get("reason")
            row["phase"] = "failed"
            row["error"] = failure.get("failure_kind")


class DecoderBusinessAdmissionGate:
    """Per-decoder protected/background admission with bounded starvation."""

    def __init__(
        self, *, background_limits: tuple[int, ...],
        background_max_wait_ns: int, protected_tenants: frozenset[str],
    ) -> None:
        if (
            not background_limits
            or any(type(value) is not int or value <= 0
                   for value in background_limits)
        ):
            raise ValueError("decoder background limits must be positive")
        if type(background_max_wait_ns) is not int or background_max_wait_ns <= 0:
            raise ValueError("decoder background max wait must be positive")
        if not protected_tenants or any(
            not isinstance(value, str) or not value
            for value in protected_tenants
        ):
            raise ValueError("protected decoder tenant set is invalid")
        self.background_limits = background_limits
        self.background_max_wait_ns = background_max_wait_ns
        self.protected_tenants = protected_tenants
        self._condition = asyncio.Condition()
        self._foreground_active = [0] * len(background_limits)
        self._background_active = [0] * len(background_limits)
        self._background_waiting = [0] * len(background_limits)
        self._leases: dict[str, dict[str, object]] = {}
        self._foreground_admitted = 0
        self._background_admitted = 0
        self._background_forced = 0

    async def acquire(
        self, *, request_id: str, pair_index: int, tenant_id: str | None,
        globally_committed: bool,
    ) -> dict[str, object] | None:
        if (
            not isinstance(request_id, str) or not request_id
            or type(pair_index) is not int
            or not 0 <= pair_index < len(self.background_limits)
        ):
            raise ValueError("decoder business admission identity is invalid")
        foreground = bool(
            globally_committed and tenant_id in self.protected_tenants)
        background = bool(
            isinstance(tenant_id, str)
            and tenant_id not in self.protected_tenants)
        if not (foreground or background):
            return None
        admission_class = "protected" if foreground else "background"
        arrived_ns = time.perf_counter_ns()
        forced = False
        async with self._condition:
            if request_id in self._leases:
                raise ValueError("duplicate decoder business admission")
            foreground_before = self._foreground_active[pair_index]
            background_before = self._background_active[pair_index]
            if foreground:
                self._foreground_active[pair_index] += 1
                self._foreground_admitted += 1
            else:
                self._background_waiting[pair_index] += 1
                try:
                    while True:
                        waited_ns = time.perf_counter_ns() - arrived_ns
                        priority_clear = (
                            self._foreground_active[pair_index] == 0)
                        capacity_clear = (
                            self._background_active[pair_index]
                            < self.background_limits[pair_index])
                        if capacity_clear and (
                            priority_clear
                            or waited_ns >= self.background_max_wait_ns
                        ):
                            forced = not priority_clear
                            break
                        remaining_ns = max(
                            1,
                            self.background_max_wait_ns - waited_ns,
                        )
                        try:
                            await asyncio.wait_for(
                                self._condition.wait(),
                                remaining_ns / 1_000_000_000,
                            )
                        except asyncio.TimeoutError:
                            pass
                finally:
                    self._background_waiting[pair_index] -= 1
                self._background_active[pair_index] += 1
                self._background_admitted += 1
                if forced:
                    self._background_forced += 1
            admitted_ns = time.perf_counter_ns()
            receipt = {
                "schema": DECODER_BUSINESS_ADMISSION_SCHEMA,
                "request_id": request_id,
                "tenant_id": tenant_id,
                "pair_index": pair_index,
                "admission_class": admission_class,
                "status": "held",
                "arrived_ns": arrived_ns,
                "admitted_ns": admitted_ns,
                "wait_ns": admitted_ns - arrived_ns,
                "background_limit": self.background_limits[pair_index],
                "background_max_wait_ns": self.background_max_wait_ns,
                "starvation_escape": forced,
                "foreground_active_before": foreground_before,
                "background_active_before": background_before,
                "foreground_active_after": self._foreground_active[pair_index],
                "background_active_after": self._background_active[pair_index],
                "released_ns": None,
            }
            self._leases[request_id] = dict(receipt)
            return receipt

    async def release(self, request_id: str) -> dict[str, object]:
        async with self._condition:
            receipt = self._leases.pop(request_id, None)
            if receipt is None:
                raise ValueError("decoder business admission lease is absent")
            pair_index = int(receipt["pair_index"])
            if receipt["admission_class"] == "protected":
                if self._foreground_active[pair_index] <= 0:
                    raise RuntimeError("decoder foreground admission underflow")
                self._foreground_active[pair_index] -= 1
            else:
                if self._background_active[pair_index] <= 0:
                    raise RuntimeError("decoder background admission underflow")
                self._background_active[pair_index] -= 1
            released = {
                **receipt,
                "status": "released",
                "released_ns": time.perf_counter_ns(),
                "foreground_active_after_release": (
                    self._foreground_active[pair_index]),
                "background_active_after_release": (
                    self._background_active[pair_index]),
            }
            self._condition.notify_all()
            return released

    async def snapshot(self) -> dict[str, object]:
        async with self._condition:
            return {
                "schema": DECODER_BUSINESS_ADMISSION_SCHEMA,
                "mode": "priority_drain_v1",
                "background_limits": list(self.background_limits),
                "background_max_wait_ns": self.background_max_wait_ns,
                "protected_tenants": sorted(self.protected_tenants),
                "foreground_active": list(self._foreground_active),
                "background_active": list(self._background_active),
                "background_waiting": list(self._background_waiting),
                "leases": len(self._leases),
                "foreground_admitted": self._foreground_admitted,
                "background_admitted": self._background_admitted,
                "background_starvation_escapes": self._background_forced,
            }


async def _record_tempo_go_rejection(
    app: FastAPI,
    request_id: str,
    *,
    decision: dict[str, object],
    decision_sha256: str,
    tokenizer_ms: float,
    admission_arrival_ns: int,
    hierarchy_receipt: dict[str, object] | None = None,
    telemetry_preparation_receipt: dict[str, object] | None = None,
) -> None:
    """Persist a terminal global reject even though no pair was reserved."""

    if not isinstance(request_id, str) or not request_id:
        raise ValueError("global rejection request ID must be nonempty")
    if decision.get("request_id") != request_id:
        raise ValueError("global rejection request ID differs from decision")
    if decision.get("kind") != GlobalDecisionKind.REJECT.value:
        raise ValueError("global rejection decision kind is not reject")
    if (
        not isinstance(decision_sha256, str)
        or len(decision_sha256) != 64
    ):
        raise ValueError("global rejection decision SHA must be a digest")
    if not isinstance(tokenizer_ms, (int, float)) or not math.isfinite(
        float(tokenizer_ms)
    ) or float(tokenizer_ms) < 0:
        raise ValueError("global rejection tokenizer latency is invalid")
    if type(admission_arrival_ns) is not int or admission_arrival_ns < 0:
        raise ValueError("global rejection arrival timestamp is invalid")
    if telemetry_preparation_receipt is not None:
        if (
            not isinstance(telemetry_preparation_receipt, dict)
            or telemetry_preparation_receipt.get("schema")
            != "tempo-go-admission-preparation-v1"
        ):
            raise ValueError("telemetry preparation receipt schema mismatch")
    if hierarchy_receipt is not None:
        if not isinstance(hierarchy_receipt, dict):
            raise TypeError("hierarchy receipt must be an object")
        receipt = hierarchy_receipt.get("receipt")
        fingerprint = hierarchy_receipt.get("fingerprint_sha256")
        if not isinstance(receipt, dict) or receipt.get("schema") != (
            "tempo-go-reduction-receipt-v1"):
            raise ValueError("hierarchy receipt schema mismatch")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("hierarchy receipt fingerprint is invalid")
    decided_ns = decision.get("decided_ns")
    if type(decided_ns) is not int:
        raise ValueError("global rejection decision timestamp is invalid")
    row = {
        "request_id": request_id,
        "phase": "rejected",
        "error": None,
        "route": None,
        "global_decision_kind": GlobalDecisionKind.REJECT.value,
        "global_decision_reason": decision.get("reason"),
        "tempo_go_rejected": True,
        "frontend_tempo_go_decision": dict(decision),
        "frontend_tempo_go_decision_sha256": decision_sha256,
        "frontend_tempo_go_tokenizer_ms": float(tokenizer_ms),
        "frontend_tempo_go_admission_arrival_ns": admission_arrival_ns,
        "frontend_tempo_go_admission_wait_ns": max(
            0, decided_ns - admission_arrival_ns),
        "frontend_tempo_go_hierarchy_reduction": (
            dict(hierarchy_receipt) if hierarchy_receipt is not None else None),
        "frontend_tempo_go_telemetry_preparation": (
            dict(telemetry_preparation_receipt)
            if telemetry_preparation_receipt is not None else None),
        "frontend_pair_global_commit": False,
        # This is a pre-dispatch terminal decision, so no pair-router commit
        # can exist.  Emit the canonical boolean explicitly; downstream
        # performance receipts must distinguish this case from a post-commit
        # service failure without treating an omitted key as evidence.
        "tempo_go_global_commit_applied": False,
    }
    async with app.state.tempo_go_rejection_lock:
        if request_id in app.state.tempo_go_rejections:
            raise RuntimeError("global rejection recorded twice")
        app.state.tempo_go_rejections[request_id] = row


async def _tempo_go_rejection_rows(app: FastAPI) -> list[dict[str, object]]:
    async with app.state.tempo_go_rejection_lock:
        return [
            dict(app.state.tempo_go_rejections[request_id])
            for request_id in sorted(app.state.tempo_go_rejections)
        ]


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


async def _tokenize_prompt(
    client: httpx.AsyncClient, payload: bytes,
) -> tuple[int, float, str | None, str]:
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("completion body must be JSON") from exc
    prompt = value.get("prompt") if isinstance(value, dict) else None
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("global completion prompt must be a nonempty string")
    started_ns = time.perf_counter_ns()
    response = await client.post("/tokenize", json={"prompt": prompt})
    response.raise_for_status()
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list) or not tokens or any(
        type(item) is not int for item in tokens
    ):
        raise ValueError("global tokenizer returned invalid token IDs")
    finished_ns = time.perf_counter_ns()
    prompt_namespace_key = hashlib.sha256(json.dumps(
        tokens, separators=(",", ":")
    ).encode()).hexdigest()
    complete_tokens = (
        len(tokens) // CACHE_CHUNK_GROUP_SIZE * CACHE_CHUNK_GROUP_SIZE)
    cache_group_key = None
    if complete_tokens:
        # LMCache uses rolling prefix hashes for complete 256-token chunks.
        # The frontend does not reproduce a backend-specific hash function;
        # it creates a stable ownership identity from the same token prefix,
        # chunk size, and aligned endpoint.  Including the full complete
        # prefix avoids serializing unrelated prompts that merely share the
        # final raw chunk while still grouping duplicate shared-prefix
        # transfers before they reach a pair-local receiver.
        group_payload = json.dumps(
            {
                "schema": "tempo-cache-chunk-group-v1",
                "chunk_size": CACHE_CHUNK_GROUP_SIZE,
                "complete_tokens": tokens[:complete_tokens],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        cache_group_key = hashlib.sha256(group_payload).hexdigest()
    return (
        len(tokens),
        (finished_ns - started_ns) / 1_000_000,
        cache_group_key,
        prompt_namespace_key,
    )


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


async def _queue_gpu_pair_selection(
    clients: list[httpx.AsyncClient],
) -> tuple[int, dict[str, object]]:
    """Select the least-loaded decoder using request-start vLLM gauges only."""

    responses = await asyncio.gather(*[
        client.get("/tempo/runtime_telemetry") for client in clients
    ])
    observations = []
    for index, response in enumerate(responses):
        response.raise_for_status()
        value = response.json()
        scheduler = value.get("vllm_scheduler") if isinstance(value, dict) else None
        if not isinstance(scheduler, dict):
            raise ValueError("queue-GPU telemetry lacks vLLM scheduler snapshot")
        if scheduler.get("decision_mode") != "observe_only":
            raise ValueError("queue-GPU scheduler snapshot is not observe-only")
        if scheduler.get("source") != "router_local_vllm_prometheus_observe_only":
            raise ValueError("queue-GPU scheduler snapshot source differs")
        running = scheduler.get("num_requests_running")
        waiting = scheduler.get("num_requests_waiting")
        kv_usage = scheduler.get("kv_cache_usage_fraction")
        if (
            type(running) is not int or running < 0
            or type(waiting) is not int or waiting < 0
            or not isinstance(kv_usage, (int, float))
            or not math.isfinite(float(kv_usage))
            or not 0.0 <= float(kv_usage) <= 1.0
        ):
            raise ValueError("queue-GPU scheduler snapshot fields are invalid")
        observations.append({
            "pair_index": index,
            "running_requests": running,
            "waiting_requests": waiting,
            "kv_cache_usage_fraction": float(kv_usage),
        })
    selected = min(
        observations,
        key=lambda item: (
            int(item["waiting_requests"]),
            int(item["running_requests"]),
            float(item["kv_cache_usage_fraction"]),
            int(item["pair_index"]),
        ),
    )
    return int(selected["pair_index"]), {
        "schema": "tempo-go-vllm-scheduler-pair-selection-v1",
        "source": "router_local_vllm_prometheus_observe_only",
        "decision_mode": "observe_only",
        "selected_pair": int(selected["pair_index"]),
        "pairs": observations,
    }


def build_app(pair_urls: list[str]) -> FastAPI:
    if len(pair_urls) not in (1, 2) or len(set(pair_urls)) != len(pair_urls):
        raise ValueError("one or two unique pair router URLs are required")
    if any(not value.startswith(("http://", "https://"))
           for value in pair_urls):
        raise ValueError("pair router URLs must be HTTP(S)")
    global_ablation = os.environ.get(GLOBAL_ABLATION_ENV, "disabled")
    if global_ablation not in {"disabled", "app_global_only"}:
        raise ValueError(
            f"{GLOBAL_ABLATION_ENV} must be disabled or app_global_only")
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

    global_paths = {
        "global": os.environ.get(GLOBAL_PROFILE_ENV),
        "elastic": os.environ.get(GLOBAL_ELASTIC_PROFILE_ENV),
        "endpoint": os.environ.get(GLOBAL_ENDPOINT_PROFILE_ENV),
        "tokenizer": os.environ.get(GLOBAL_TOKENIZER_URL_ENV),
    }
    if any(global_paths.values()) and not all(global_paths.values()):
        raise ValueError("TEMPO-GO frontend profile environment is incomplete")
    global_profile = None
    global_candidate_builder = None
    global_hierarchical_reducer = None
    global_tokenizer_url = None
    decoder_business_admission_spec = None
    if all(global_paths.values()):
        global_profile = load_global_profile(
            Path(str(global_paths["global"])).resolve())
        elastic_profile = load_elastic_profile(
            Path(str(global_paths["elastic"])).resolve())
        endpoint_profile = load_endpoint_service_profile(
            Path(str(global_paths["endpoint"])).resolve())
        identity = global_profile.identity
        if identity.router_schema != ROUTER_SCHEMA:
            raise ValueError("TEMPO-GO router schema identity differs")
        if elastic_profile.fingerprint_sha256 != (
            identity.elastic_profile_fingerprint_sha256
        ):
            raise ValueError("TEMPO-GO elastic profile identity differs")
        if endpoint_profile.fingerprint_sha256 != (
            identity.endpoint_profile_fingerprint_sha256
        ):
            raise ValueError("TEMPO-GO endpoint profile identity differs")
        if (
            endpoint_profile.profile_id != identity.endpoint_profile_id
            or endpoint_profile.schema != identity.endpoint_profile_schema
            or endpoint_profile.deployment_scope
            != identity.endpoint_profile_deployment_scope
            or endpoint_profile.workload_manifest_sha256
            != identity.workload_manifest_sha256
        ):
            raise ValueError("TEMPO-GO endpoint profile provenance differs")
        if any(
            item.resources.active_sequences != max_num_seqs
            for item in global_profile.capacities
        ):
            raise ValueError("TEMPO-GO active-sequence capacity differs from vLLM")
        global_tokenizer_url = str(global_paths["tokenizer"])
        if not global_tokenizer_url.startswith(("http://", "https://")):
            raise ValueError("TEMPO-GO tokenizer URL must be HTTP(S)")
        global_candidate_builder = GlobalCandidateBuilder(
            elastic_profile,
            endpoint_profile,
            pair_count=len(pair_urls),
            allow_service_proxy=(
                global_profile.deployment_scope == "discovery"
                and global_profile.service_proxy_policy() is None),
            service_proxy_policy=global_profile.service_proxy_policy(),
            mesh_enabled=(
                global_profile.orchestrator_config().mesh_control_mode
                == "receiver_credit_pxd_v1"
            ),
        )
        global_config = global_profile.orchestrator_config()
        if global_config.decoder_business_admission_mode == "priority_drain_v1":
            capacities = tuple(sorted(
                global_profile.capacities,
                key=lambda item: item.pair_index,
            ))
            decoder_business_admission_spec = {
                "background_limits": tuple(
                    item.resources.active_sequences
                    - global_config.priority_service_lane_capacity
                    for item in capacities
                ),
                "background_max_wait_ns": (
                    global_config.decoder_business_background_max_wait_ns),
                "protected_tenants": frozenset(
                    policy.tenant_id for policy in global_profile.tenants
                    if policy.admission_priority
                    >= global_config.priority_service_lane_min_admission_priority
                ),
            }
        # Native Perlmutter currently exposes one or two P/D pairs.  Keep one
        # shard per pair and forward every pair frontier, so the hierarchy is
        # exercised in the actual vLLM path without changing the exact
        # two-pair candidate population.  The same reducer accepts an explicit
        # bounded many-pair map for the larger control-plane deployment.
        global_hierarchical_reducer = HierarchicalCandidateReducer(
            shard_count=len(pair_urls),
            max_pairs_per_shard=1,
            max_routes_per_pair=(
                1 + len(pair_urls)
                if global_profile.orchestrator_config().mesh_control_mode
                == "receiver_credit_pxd_v1"
                else 2
            ),
            pair_to_shard={index: index for index in range(len(pair_urls))},
            pair_capacities={
                item.pair_index: item.resources
                for item in global_profile.capacities
            },
            telemetry_fresh_ns=global_profile.telemetry.freshness_ns,
            telemetry_stale_grace_ns=(
                global_profile.orchestrator_config().telemetry_stale_grace_ns),
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.clients = [
            httpx.AsyncClient(base_url=value, timeout=None)
            for value in pair_urls
        ]
        app.state.pair_load = PairLoadLedger(
            len(pair_urls), tempo_policy=tempo_pair_policy)
        app.state.decision_fetch_remote_protocol_retries = 0
        app.state.tempo_go_profile = global_profile
        app.state.tempo_go_candidate_builder = global_candidate_builder
        app.state.tempo_go_hierarchical_reducer = global_hierarchical_reducer
        app.state.tempo_go_coordinator = None
        app.state.tempo_go_tokenizer = None
        app.state.tempo_go_rejections = {}
        app.state.tempo_go_rejection_lock = asyncio.Lock()
        app.state.decoder_business_admission = (
            DecoderBusinessAdmissionGate(**decoder_business_admission_spec)
            if decoder_business_admission_spec is not None else None
        )
        if global_profile is not None:
            job_id = os.environ.get("SLURM_JOB_ID")
            if not isinstance(job_id, str) or not job_id.isdigit():
                raise ValueError("TEMPO-GO requires a numeric Slurm job ID")
            agent_epoch = (
                f"slurm-{job_id}-frontend-{time.perf_counter_ns()}")

            async def fetch_frontend_state():
                value = await app.state.pair_load.snapshot()
                gate = app.state.decoder_business_admission
                if gate is not None:
                    value = dict(value)
                    value["decoder_business_admission"] = (
                        await gate.snapshot())
                return value

            def endpoint_fetcher(index: int):
                async def fetch():
                    response = await app.state.clients[index].get(
                        "/tempo/runtime_telemetry")
                    response.raise_for_status()
                    value = response.json()
                    if not isinstance(value, dict):
                        raise ValueError(
                            "pair endpoint telemetry is not an object")
                    if global_ablation == "app_global_only":
                        value = dict(value)
                        value.pop("cross_layer", None)
                    return value

                return fetch

            telemetry_agent = RequestTriggeredTelemetryAgent(
                global_profile.telemetry_adapter(agent_epoch=agent_epoch),
                frontend_fetcher=fetch_frontend_state,
                endpoint_fetchers=tuple(
                    endpoint_fetcher(index) for index in range(len(pair_urls))
                ),
                freshness_ns=global_profile.telemetry.freshness_ns,
                refresh_timeout_ns=(
                    global_profile.telemetry.refresh_timeout_ns),
            )
            app.state.tempo_go_coordinator = GlobalAdmissionCoordinator(
                GlobalOrchestrator(global_profile.orchestrator_config()),
                telemetry_agent,
                admission_wait_ns=(
                    global_profile.orchestrator_config().maximum_queue_wait_ns),
                hierarchical_reducer=global_hierarchical_reducer,
            )
            app.state.tempo_go_tokenizer = httpx.AsyncClient(
                base_url=global_tokenizer_url,
                timeout=global_profile.telemetry.tokenizer_timeout_ns
                / 1_000_000_000,
            )
        try:
            yield
        finally:
            if app.state.tempo_go_tokenizer is not None:
                await app.state.tempo_go_tokenizer.aclose()
            for client in app.state.clients:
                await client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        state = await app.state.pair_load.snapshot()
        decoder_admission = (
            await app.state.decoder_business_admission.snapshot()
            if app.state.decoder_business_admission is not None else None
        )
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
            "tempo_go_enabled": global_profile is not None,
            "tempo_go_profile_id": (
                global_profile.profile_id if global_profile else None),
            "tempo_go_profile_sha256": (
                global_profile.fingerprint_sha256 if global_profile else None),
            "tempo_go_coordinator": (
                app.state.tempo_go_coordinator.status()
                if app.state.tempo_go_coordinator is not None else None),
            "decoder_business_admission": decoder_admission,
        }

    @app.get("/tempo/global_state")
    async def global_state() -> dict[str, Any]:
        coordinator = app.state.tempo_go_coordinator
        if coordinator is None or global_profile is None:
            raise HTTPException(
                status_code=404, detail="TEMPO-GO is not enabled")
        now_ns = time.perf_counter_ns()
        return {
            "schema": FRONTEND_SCHEMA,
            "tempo_go_profile_id": global_profile.profile_id,
            "tempo_go_profile_sha256": global_profile.fingerprint_sha256,
            "coordinator": coordinator.status(),
            "orchestrator": coordinator.orchestrator.snapshot(now_ns=now_ns),
            "frontend_ledger": await app.state.pair_load.snapshot(),
            "decoder_business_admission": (
                await app.state.decoder_business_admission.snapshot()
                if app.state.decoder_business_admission is not None else None),
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
                    row.get("completion_cache_residency") == "prefill_only"
                    and isinstance(
                        row.get("lmcache_source_cached_tokens"), int)
                    and 0 <= row["lmcache_source_cached_tokens"] <= int(
                        row.get("prompt_tokens", 0)) + 1
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
        rows.extend(await _tempo_go_rejection_rows(app))
        row_ids = [row.get("request_id") for row in rows]
        if len(row_ids) != len(set(row_ids)):
            raise HTTPException(
                status_code=502, detail="duplicate frontend decision IDs")
        rows.sort(key=lambda row: str(row.get("request_id")))
        decoder_admission = (
            await app.state.decoder_business_admission.snapshot()
            if app.state.decoder_business_admission is not None else None
        )
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
            "tempo_go_enabled": global_profile is not None,
            "tempo_go_profile_sha256": (
                global_profile.fingerprint_sha256 if global_profile else None),
            "tempo_go_coordinator": (
                app.state.tempo_go_coordinator.status()
                if app.state.tempo_go_coordinator is not None else None),
            "decoder_business_admission": decoder_admission,
        }

    @app.post("/v1/completions")
    async def completions(request: Request):
        request_id = request.headers.get("x-tempo-request-id")
        if not request_id:
            raise HTTPException(
                status_code=400, detail="missing x-tempo-request-id")
        business_tenant_id = request.headers.get(GLOBAL_TENANT_HEADER)
        payload = await request.body()
        tempo_go_decision = None
        tempo_go_decision_payload = None
        tempo_go_decision_sha256 = None
        tempo_go_tokenizer_ms = None
        service_lane_preflighted = False
        preflight_failure_reason = None

        try:
            arm = request_arm(request_id)
            tokens, prompt_sha256 = _completion_shape(payload)
            placement_tokens = placement_decode_tokens(request_id, tokens)
            preferred = pair_index(request_id, len(pair_urls))
            tempo_warm = (
                arm in {"tempo", "app_global_only"}
                and "-warm-" in request_id
            )
            d_cache_prepare = (
                D_CACHE_SEED_MARKER in request_id
                or D_CACHE_PROBE_MARKER in request_id)
            replicated_warm = (
                replicate_warm_affinity
                and "-warm-" in request_id
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
                arm in {"tempo", "app_global_only"}
                and uses_decoder_affinity(
                    request_id,
                    decoder_prefix_caching=decoder_prefix_caching,
                    affinity_required=affinity_required,
                    decoder_reuse_items=decoder_reuse_items,
                ))
            decode_affinity_required = (
                arm in {"tempo", "app_global_only"}
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
            tempo_go_request = (
                global_profile is not None
                and arm in {"tempo", "app_global_only"}
                and not tempo_warm
                and not d_cache_prepare
            )
            queue_gpu_pair = None
            queue_gpu_observation = None
            if arm == "queue_gpu" and not physical_pair_pin:
                queue_gpu_pair, queue_gpu_observation = (
                    await _queue_gpu_pair_selection(app.state.clients))
            committed_pair = None
            hierarchy_receipt = None
            if tempo_go_request:
                coordinator = app.state.tempo_go_coordinator
                candidate_builder = app.state.tempo_go_candidate_builder
                tokenizer = app.state.tempo_go_tokenizer
                if (
                    coordinator is None
                    or candidate_builder is None
                    or tokenizer is None
                ):
                    raise RuntimeError("TEMPO-GO frontend is not initialized")
                tenant_id = business_tenant_id
                if not tenant_id:
                    raise ValueError(
                        f"missing {GLOBAL_TENANT_HEADER} for TEMPO-GO")
                arrival_ns = time.perf_counter_ns()
                raw_deadline_ms = request.headers.get(
                    "x-tempo-remaining-deadline-ms")
                deadline_ms = (
                    float(raw_deadline_ms)
                    if raw_deadline_ms is not None
                    else float(
                        candidate_builder.endpoint_profile
                        .default_e2e_deadline_ms)
                )
                if not math.isfinite(deadline_ms) or deadline_ms <= 0:
                    raise ValueError(
                        "TEMPO-GO remaining deadline must be positive")
                # Tokenization and the request-triggered all-pair scrape are
                # independent prerequisites.  Overlap their I/O so normal
                # service does not pay their sum.  This task is scoped to this
                # request, is always joined/cancelled below, and does not turn
                # the telemetry agent into a background poller.
                telemetry_preparation_task = asyncio.create_task(
                    coordinator.prepare_admission())
                try:
                    (
                        prompt_tokens,
                        tempo_go_tokenizer_ms,
                        cache_group_key,
                        tempo_go_prompt_namespace_key,
                    ) = await _tokenize_prompt(tokenizer, payload)
                    cache_states = await app.state.pair_load.cache_states(
                        prompt_sha256,
                        explicit_cache_reset_miss=(
                            cold_measured
                            and MISS_MEASURED_MARKER in request_id),
                    )
                    telemetry_preparation = (
                        await telemetry_preparation_task)
                except BaseException:
                    telemetry_preparation_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await telemetry_preparation_task
                    raise
                global_request = candidate_builder.build(
                    request_id=request_id,
                    tenant_id=tenant_id,
                    arrival_ns=arrival_ns,
                    deadline_ns=arrival_ns + int(deadline_ms * 1_000_000),
                    prompt_tokens=prompt_tokens,
                    output_tokens=tokens,
                    cache_states=cache_states,
                    cache_group_key=cache_group_key,
                )
                admitted_decision = await coordinator.admit(
                    global_request,
                    preparation=telemetry_preparation,
                )
                hierarchy_receipt = coordinator.take_hierarchy_receipt(request_id)
                if admitted_decision.kind is GlobalDecisionKind.REJECT:
                    rejection_payload = global_decision_dict(
                        admitted_decision)
                    rejection_sha256 = global_decision_fingerprint(
                        admitted_decision)
                    assert tempo_go_tokenizer_ms is not None
                    await _record_tempo_go_rejection(
                        app,
                        request_id,
                        decision=rejection_payload,
                        decision_sha256=rejection_sha256,
                        tokenizer_ms=tempo_go_tokenizer_ms,
                        admission_arrival_ns=global_request.arrival_ns,
                        hierarchy_receipt=hierarchy_receipt,
                        telemetry_preparation_receipt=(
                            telemetry_preparation.as_dict()),
                    )
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "code": "tempo_go_global_reject",
                            "reason": admitted_decision.reason,
                            "request_id": request_id,
                        },
                    )
                tempo_go_decision = admitted_decision
                if (
                    tempo_go_decision.kind is not GlobalDecisionKind.ADMIT
                    or tempo_go_decision.pair_index is None
                    or tempo_go_decision.prefill_index is None
                    or tempo_go_decision.decoder_index is None
                    or tempo_go_decision.edge_id is None
                    or tempo_go_decision.route is None
                ):
                    raise RuntimeError(
                        "TEMPO-GO coordinator returned no route commitment")
                committed_pair = tempo_go_decision.decoder_index
                tempo_go_decision_payload = global_decision_dict(
                    tempo_go_decision)
                tempo_go_decision_sha256 = global_decision_fingerprint(
                    tempo_go_decision)
            assignment = await app.state.pair_load.reserve(
                request_id, tokens, preferred,
                dynamic=arm in {"tempo", "app_global_only", "queue_gpu"},
                placement_tokens=placement_tokens,
                affinity_key=(
                    prompt_sha256
                    if arm in {"tempo", "app_global_only"} else None
                ),
                affinity_seed=affinity_seed,
                affinity_required=affinity_required,
                affinity_owner_count_required=affinity_owner_count_required,
                prefer_decode_affinity=decoder_reuse,
                decode_affinity_required=decode_affinity_required,
                pair_pin_preferred=physical_pair_pin,
                committed_pair=committed_pair,
                queue_gpu_pair=queue_gpu_pair,
                queue_gpu_observation=queue_gpu_observation)
            if tempo_go_decision is not None:
                assert tempo_go_decision_payload is not None
                assert tempo_go_decision_sha256 is not None
                assert tempo_go_tokenizer_ms is not None
                await app.state.pair_load.record_global_decision(
                    request_id,
                    decision=tempo_go_decision_payload,
                    decision_sha256=tempo_go_decision_sha256,
                    tokenizer_ms=tempo_go_tokenizer_ms,
                    admission_arrival_ns=global_request.arrival_ns,
                    hierarchy_receipt=hierarchy_receipt,
                    telemetry_preparation_receipt=(
                        telemetry_preparation.as_dict()),
                )
        except RuntimeError as exc:
            if tempo_go_decision is not None:
                await app.state.tempo_go_coordinator.fail(request_id)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            if tempo_go_decision is not None:
                await app.state.tempo_go_coordinator.fail(request_id)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            if tempo_go_decision is not None:
                await app.state.tempo_go_coordinator.fail(request_id)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        selected = int(assignment["frontend_pair_index"])
        router_selected = selected
        if (
            tempo_go_decision is not None
            and tempo_go_decision.route is GlobalRoute.REMOTE
            and tempo_go_decision.prefill_index is not None
        ):
            router_selected = tempo_go_decision.prefill_index
        client = app.state.clients[router_selected]
        headers = {
            "Content-Type": request.headers.get(
                "content-type", "application/json"),
            "X-Tempo-Request-Id": request_id,
        }
        if business_tenant_id:
            headers["X-Tempo-Tenant-Id"] = business_tenant_id
        for name in ("authorization",):
            value = request.headers.get(name)
            if value:
                headers[name] = value
        if tempo_go_decision is not None:
            # The frontend may consume part of the request's business/SLO
            # deadline while tokenizing, refreshing the global telemetry
            # batch, reducing pair frontiers, or waiting for global admission.
            # Carry the *remaining* budget to vLLM so a global endpoint queue
            # lease is bounded by the same end-to-end contract.  Forwarding
            # no header here silently selected v448's unrelated 1-second
            # default queue wait under native contention.
            remaining_deadline_ms = max(
                0.001,
                (global_request.deadline_ns - time.perf_counter_ns())
                / 1_000_000,
            )
            headers["x-tempo-remaining-deadline-ms"] = (
                f"{remaining_deadline_ms:.6f}"
            )
        else:
            value = request.headers.get("x-tempo-remaining-deadline-ms")
            if value:
                headers["x-tempo-remaining-deadline-ms"] = value
        upstream = None
        shadow_selected = None
        decoder_business_admission_held = False

        async def release_decoder_business_admission() -> None:
            nonlocal decoder_business_admission_held
            if not decoder_business_admission_held:
                return
            gate = app.state.decoder_business_admission
            if gate is None:
                raise RuntimeError("decoder business admission gate disappeared")
            released_receipt = await gate.release(request_id)
            await app.state.pair_load.record_decoder_business_admission(
                request_id, released_receipt)
            decoder_business_admission_held = False

        try:
            gate = app.state.decoder_business_admission
            if gate is not None:
                admission_receipt = await gate.acquire(
                    request_id=request_id,
                    pair_index=selected,
                    tenant_id=business_tenant_id,
                    globally_committed=tempo_go_decision is not None,
                )
                if admission_receipt is not None:
                    decoder_business_admission_held = True
                    await app.state.pair_load.record_decoder_business_admission(
                        request_id, admission_receipt)
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
            if tempo_go_decision is not None:
                assert global_profile is not None
                assert tempo_go_decision_sha256 is not None
                assert tempo_go_decision.route is not None
                sequence_values = set(
                    tempo_go_decision.telemetry_sequences.values())
                if len(sequence_values) != 1:
                    raise RuntimeError(
                        "TEMPO-GO decision has mixed telemetry sequences")
                joint_plan = tempo_go_decision.joint_actuation
                mesh_commit = (
                    tempo_go_decision.prefill_index
                    != tempo_go_decision.decoder_index
                    or global_profile.orchestrator_config().mesh_control_mode
                    == "receiver_credit_pxd_v1"
                )
                headers.update({
                    "X-Tempo-GO-Schema": (
                        GLOBAL_MESH_JOINT_COMMIT_SCHEMA
                        if mesh_commit and joint_plan is not None
                        else GLOBAL_MESH_COMMIT_SCHEMA
                        if mesh_commit
                        else "tempo-go-joint-commit-v1"
                        if joint_plan is not None
                        else "tempo-go-route-commit-v1"
                    ),
                    "X-Tempo-GO-Pair-Index": str(selected),
                    "X-Tempo-GO-Prefill-Index": str(
                        tempo_go_decision.prefill_index),
                    "X-Tempo-GO-Decoder-Index": str(
                        tempo_go_decision.decoder_index),
                    "X-Tempo-GO-Edge-Id": tempo_go_decision.edge_id,
                    "X-Tempo-GO-Route": tempo_go_decision.route.value,
                    "X-Tempo-GO-Profile-SHA256": (
                        global_profile.fingerprint_sha256),
                    "X-Tempo-GO-Decision-SHA256": (
                        tempo_go_decision_sha256),
                    "X-Tempo-GO-Telemetry-Sequence": str(
                        next(iter(sequence_values))),
                    "X-Tempo-GO-Queue-Lease": (
                        "1" if tempo_go_decision.queue_lease else "0"),
                    "X-Tempo-GO-Priority-Service-Lane": (
                        "1"
                        if PRIORITY_SERVICE_LANE_BINDING
                        in tempo_go_decision.binding_resources else "0"
                    ),
                    "X-Tempo-GO-Service-Queue-Delay-MS": (
                        str(tempo_go_decision.service_queue_delay_ms)
                        if tempo_go_decision.service_queue_delay_ms is not None
                        else None
                    ),
                    "X-Tempo-GO-Service-Forecast-MS": (
                        str(tempo_go_decision.service_forecast_ms)
                        if tempo_go_decision.service_forecast_ms is not None
                        else None
                    ),
                })
                headers = {
                    key: value for key, value in headers.items()
                    if value is not None
                }
                if joint_plan is not None:
                    headers["X-Tempo-GO-Actuation-Plan"] = json.dumps(
                        joint_plan.as_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                if mesh_commit and tempo_go_decision.route is GlobalRoute.REMOTE:
                    headers["X-Tempo-PD-Decoder-Index"] = str(selected)
                commit_payload = {
                    "schema": headers["X-Tempo-GO-Schema"],
                    # Preserve the exact canonical header representation.
                    # The router's immutable commit parser deliberately does
                    # not accept JSON-number aliases for these identities.
                    "pair_index": headers["X-Tempo-GO-Pair-Index"],
                    "prefill_index": headers["X-Tempo-GO-Prefill-Index"],
                    "decoder_index": headers["X-Tempo-GO-Decoder-Index"],
                    "edge_id": headers["X-Tempo-GO-Edge-Id"],
                    "route": headers["X-Tempo-GO-Route"],
                    "profile_sha256": (
                        headers["X-Tempo-GO-Profile-SHA256"]),
                    "decision_sha256": (
                        headers["X-Tempo-GO-Decision-SHA256"]),
                    "telemetry_sequence": headers[
                        "X-Tempo-GO-Telemetry-Sequence"],
                    "actuation_plan": (
                        json.loads(headers["X-Tempo-GO-Actuation-Plan"])
                        if "X-Tempo-GO-Actuation-Plan" in headers else None
                    ),
                    "queue_lease": headers["X-Tempo-GO-Queue-Lease"],
                    "priority_service_lane": headers[
                        "X-Tempo-GO-Priority-Service-Lane"],
                }
                for field, header in (
                    ("service_queue_delay_ms",
                     "X-Tempo-GO-Service-Queue-Delay-MS"),
                    ("service_forecast_ms",
                     "X-Tempo-GO-Service-Forecast-MS"),
                ):
                    if header in headers:
                        commit_payload[field] = headers[header]
                preflight_response = await client.post(
                    "/tempo/service_lane_preflight",
                    json={
                        "request_id": request_id,
                        "prompt_key": tempo_go_prompt_namespace_key,
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": tokens,
                        "remaining_deadline_ms": remaining_deadline_ms,
                        "commit": commit_payload,
                    },
                )
                try:
                    preflight_payload = preflight_response.json()
                except (TypeError, ValueError, json.JSONDecodeError):
                    preflight_payload = {}
                if (
                    preflight_response.status_code == 200
                    and isinstance(preflight_payload, dict)
                    and preflight_payload.get("status") == "queue_required"
                ):
                    promotion_report = await (
                        app.state.tempo_go_coordinator
                        .promote_service_lane_queue_lease(request_id)
                    )
                    promotion_receipt = promotion_report.receipt
                    promotion_payload = (
                        global_service_lane_queue_promotion_dict(
                            promotion_receipt)
                    )
                    promotion_sha256 = (
                        global_service_lane_queue_promotion_fingerprint(
                            promotion_receipt)
                    )
                    if promotion_report.decision is None:
                        await app.state.pair_load.record_service_lane_queue_promotion_rejection(
                            request_id,
                            promotion=promotion_payload,
                            promotion_sha256=promotion_sha256,
                        )
                        preflight_failure_reason = promotion_receipt.reason
                    else:
                        promoted_decision = promotion_report.decision
                        promoted_payload = global_decision_dict(
                            promoted_decision)
                        promoted_sha256 = global_decision_fingerprint(
                            promoted_decision)
                        await app.state.pair_load.record_service_lane_queue_promotion(
                            request_id,
                            decision=promoted_payload,
                            decision_sha256=promoted_sha256,
                            promotion=promotion_payload,
                            promotion_sha256=promotion_sha256,
                        )
                        tempo_go_decision = promoted_decision
                        tempo_go_decision_payload = promoted_payload
                        tempo_go_decision_sha256 = promoted_sha256
                        headers["X-Tempo-GO-Decision-SHA256"] = (
                            promoted_sha256)
                        headers["X-Tempo-GO-Queue-Lease"] = "1"
                        headers["X-Tempo-GO-Priority-Service-Lane"] = (
                            "1"
                            if PRIORITY_SERVICE_LANE_BINDING
                            in promoted_decision.binding_resources else "0"
                        )
                        commit_payload["decision_sha256"] = promoted_sha256
                        commit_payload["queue_lease"] = "1"
                        commit_payload["priority_service_lane"] = headers[
                            "X-Tempo-GO-Priority-Service-Lane"]
                        preflight_response = await client.post(
                            "/tempo/service_lane_preflight",
                            json={
                                "request_id": request_id,
                                "prompt_key": tempo_go_prompt_namespace_key,
                                "prompt_tokens": prompt_tokens,
                                "output_tokens": tokens,
                                "remaining_deadline_ms": (
                                    remaining_deadline_ms),
                                "commit": commit_payload,
                            },
                        )
                        try:
                            preflight_payload = preflight_response.json()
                        except (
                            TypeError, ValueError, json.JSONDecodeError
                        ):
                            preflight_payload = {}
                if (
                    preflight_failure_reason is not None
                    or
                    preflight_response.status_code != 200
                    or not isinstance(preflight_payload, dict)
                    or preflight_payload.get("status") != "accepted"
                ):
                    detail = (
                        preflight_payload.get("detail")
                        if isinstance(preflight_payload, dict) else None
                    )
                    detail_reason = (
                        detail.get("reason") or detail.get("code")
                        if isinstance(detail, dict) else
                        detail if isinstance(detail, str) else None
                    )
                    reason = preflight_failure_reason or (
                        preflight_payload.get("reason")
                        if isinstance(preflight_payload, dict) else None
                    ) or detail_reason or (
                        "endpoint_service_lane_preflight_failed")
                    # A queue_required response leaves a request-scoped
                    # endpoint offer.  Close it before releasing global
                    # ownership; direct commit-validation failures are also
                    # safe here because the router abort is idempotent at
                    # this boundary and 409 is ignored.
                    try:
                        await client.post(
                            "/tempo/service_lane_abort",
                            json={
                                "request_id": request_id,
                                "error": (
                                    "frontend_service_lane_preflight_abort"),
                            },
                        )
                    except (httpx.HTTPError, RuntimeError, ValueError):
                        pass
                    report = await (
                        app.state.tempo_go_coordinator
                        .fail_service_lane_reservation(
                            request_id,
                            failure_kind=(
                                "endpoint_service_lane_preflight_unavailable"),
                            reason=str(reason),
                        )
                    )
                    failure_payload = (
                        global_service_lane_reservation_failure_dict(
                            report.receipt)
                    )
                    await app.state.pair_load.record_global_reservation_failure(
                        request_id,
                        failure=failure_payload,
                        failure_sha256=(
                            global_service_lane_reservation_failure_fingerprint(
                                report.receipt)
                        ),
                    )
                    await release_decoder_business_admission()
                    await app.state.pair_load.release(request_id)
                    preflight_failure_reason = str(reason)
                    raise RuntimeError(
                        "tempo_go_service_lane_preflight_failed")
                service_lane_preflighted = True
            upstream_request = client.build_request(
                "POST", "/v1/completions", content=payload, headers=headers)
            if tempo_go_decision is not None:
                joint_plan = tempo_go_decision.joint_actuation
                stagger_us = max(
                    joint_plan.dispatch_stagger_us
                    if joint_plan is not None else 0,
                    tempo_go_decision.receiver_stagger_us,
                )
                if stagger_us:
                    await asyncio.sleep(
                        stagger_us / 1_000_000.0)
            upstream = await client.send(upstream_request, stream=True)
            await _raise_upstream_status_with_body(upstream)
            if tempo_go_decision is not None and not (
                upstream.headers.get("x-tempo-pd-schema") == ROUTER_SCHEMA
                and upstream.headers.get("x-tempo-pd-request-id") == request_id
                and upstream.headers.get("x-tempo-pd-route")
                == tempo_go_decision.route.value
                and upstream.headers.get(
                    "x-tempo-service-lane-reservation") == "accepted"
            ):
                raise RuntimeError(
                    "pair router violated TEMPO-GO service-lane commitment")
        except Exception as exc:
            service_lane_status = (
                upstream.headers.get("x-tempo-service-lane-reservation")
                if upstream is not None else None
            )
            service_lane_reason = (
                upstream.headers.get("x-tempo-service-lane-reason")
                if upstream is not None else None
            )
            if upstream is not None:
                await upstream.aclose()
            await release_decoder_business_admission()
            if service_lane_preflighted:
                # The router may already have observed the upstream failure;
                # its abort endpoint is idempotent at this campaign boundary
                # and releases a still-held preflight reservation otherwise.
                try:
                    await client.post(
                        "/tempo/service_lane_abort",
                        json={
                            "request_id": request_id,
                            "error": "frontend_upstream_abort",
                        },
                    )
                except (httpx.HTTPError, RuntimeError, ValueError):
                    pass
                service_lane_preflighted = False
            if preflight_failure_reason is not None:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "tempo_go_service_lane_preflight_failed",
                        "request_id": request_id,
                        "reason": preflight_failure_reason,
                    },
                ) from exc
            if tempo_go_decision is not None:
                reservation_unavailable = (
                    service_lane_status == "unavailable"
                )
                reservation_timeout = service_lane_status == "timeout"
                if reservation_unavailable or reservation_timeout:
                    failure_kind = (
                        "endpoint_bounded_queue_lease_timeout"
                        if reservation_timeout else
                        "endpoint_service_lane_reservation_unavailable"
                    )
                    fallback_reason = (
                        "endpoint_bounded_queue_lease_timeout"
                        if reservation_timeout else
                        "endpoint_service_lane_capacity_unavailable"
                    )
                    report = await (
                        app.state.tempo_go_coordinator
                        .fail_service_lane_reservation(
                            request_id,
                            failure_kind=failure_kind,
                            reason=service_lane_reason or fallback_reason,
                        )
                    )
                    failure_payload = (
                        global_service_lane_reservation_failure_dict(
                            report.receipt)
                    )
                    await app.state.pair_load.record_global_reservation_failure(
                        request_id,
                        failure=failure_payload,
                        failure_sha256=(
                            global_service_lane_reservation_failure_fingerprint(
                                report.receipt)
                        ),
                    )
                    await app.state.pair_load.release(request_id)
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "code": (
                                "tempo_go_service_lane_reservation_timeout"
                                if reservation_timeout else
                                "tempo_go_service_lane_reservation_unavailable"
                            ),
                            "request_id": request_id,
                            "reason": report.receipt.reason,
                        },
                    ) from exc
                failure_report = await app.state.tempo_go_coordinator.fail_route(
                    request_id,
                    failure_kind=tempo_route_failure_kind(exc),
                    scope=tempo_global_failure_scope(
                        exc,
                        decision=tempo_go_decision,
                        mesh_enabled=(
                            global_profile.orchestrator_config().mesh_control_mode
                            == "receiver_credit_pxd_v1"
                        ),
                    ),
                )
                if failure_report is not None:
                    failure_payload = global_failure_dict(
                        failure_report.receipt)
                    await app.state.pair_load.record_global_failure(
                        request_id,
                        failure=failure_payload,
                        failure_sha256=global_failure_fingerprint(
                            failure_report.receipt),
                    )
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
        if queue_gpu_observation is not None:
            response_headers.update({
                "X-Tempo-PD-Queue-GPU-Selection": "observe_only",
                "X-Tempo-PD-Queue-GPU-Pair": str(
                    queue_gpu_observation["selected_pair"]),
            })
        if tempo_go_decision is not None:
            assert global_profile is not None
            assert tempo_go_decision_sha256 is not None
            assert tempo_go_decision.route is not None
            response_headers.update({
                "X-Tempo-GO-Profile-SHA256": (
                    global_profile.fingerprint_sha256),
                "X-Tempo-GO-Decision-SHA256": tempo_go_decision_sha256,
                "X-Tempo-GO-Route": tempo_go_decision.route.value,
                "X-Tempo-GO-Pair-Index": str(selected),
                "X-Tempo-GO-Prefill-Index": str(
                    tempo_go_decision.prefill_index),
                "X-Tempo-GO-Decoder-Index": str(
                    tempo_go_decision.decoder_index),
                "X-Tempo-GO-Edge-Id": str(tempo_go_decision.edge_id),
            })

        async def generate():
            stream_completed = False
            global_first_response = False
            global_terminal = False
            stream_failure_kind = None
            stream_failure_scope = None
            try:
                async for chunk in upstream.aiter_raw():
                    if (
                        tempo_go_decision is not None
                        and not global_first_response
                    ):
                        await app.state.tempo_go_coordinator.mark_first_response(
                            request_id)
                        global_first_response = True
                    yield chunk
                if tempo_go_decision is not None:
                    if not global_first_response:
                        raise RuntimeError(
                            "TEMPO-GO upstream completed without a response chunk")
                    await app.state.tempo_go_coordinator.complete(request_id)
                    global_terminal = True
                stream_completed = True
            except Exception as exc:
                stream_failure_kind = tempo_route_failure_kind(exc)
                stream_failure_scope = tempo_global_failure_scope(
                    exc,
                    decision=tempo_go_decision,
                    mesh_enabled=(
                        global_profile is not None
                        and global_profile.orchestrator_config().mesh_control_mode
                        == "receiver_credit_pxd_v1"
                    ),
                )
                raise
            finally:
                try:
                    await upstream.aclose()
                finally:
                    try:
                        if (
                            tempo_go_decision is not None
                            and not global_terminal
                        ):
                            failure_report = (
                                await app.state.tempo_go_coordinator.fail_route(
                                    request_id,
                                    failure_kind=(
                                        stream_failure_kind
                                        or "upstream_stream_terminated"),
                                    scope=stream_failure_scope or "route",
                                )
                            )
                            if failure_report is not None:
                                failure_payload = global_failure_dict(
                                    failure_report.receipt)
                                await app.state.pair_load.record_global_failure(
                                    request_id,
                                    failure=failure_payload,
                                    failure_sha256=global_failure_fingerprint(
                                        failure_report.receipt),
                                )
                            global_terminal = True
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
                        await release_decoder_business_admission()
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
