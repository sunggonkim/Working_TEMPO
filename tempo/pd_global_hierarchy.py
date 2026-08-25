"""Bounded node -> pair -> shard -> global fan-in for TEMPO-GO.

The global controller remains the authority for business policy, fairness,
capacity, route health, cross-layer action bounds, and request lifecycle.  This
module only changes how a large candidate population reaches that authority:
node/pair agents publish an allocation-scoped identity, a shard keeps a small
route frontier for its best pairs, and the global controller evaluates the
bounded result.

The reducer is intentionally conservative about identity.  A mixed allocation
epoch, profile, sequence, or partial cross-layer view is rejected before a
reduced request can be submitted.  Topology is retained as pair/node-local
provenance and aggregated into the reduction receipt; Perlmutter pairs may
legitimately expose different local topology fingerprints within one
allocation.  This prevents a fast fan-in path from becoming an oracle or
silently combining observations from different Perlmutter states without
mistaking node-local topology for a mixed allocation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from typing import Iterable, Mapping

from tempo.pd_global_orchestrator import (
    GlobalRequest,
    GlobalDecision,
    GlobalRoute,
    PairTelemetry,
    PathHealth,
    ResourceVector,
    RouteCandidate,
)


HIERARCHY_SCHEMA = "tempo-go-hierarchical-fan-in-v1"
NODE_ENVELOPE_SCHEMA = "tempo-go-node-envelope-v1"
PAIR_ENVELOPE_SCHEMA = "tempo-go-pair-envelope-v1"
SHARD_ENVELOPE_SCHEMA = "tempo-go-shard-envelope-v1"
REDUCTION_RECEIPT_SCHEMA = "tempo-go-reduction-receipt-v1"
PAIR_FRONTIER_SCHEMA = "tempo-go-pair-frontier-v1"


class HierarchyIdentityError(ValueError):
    """The hierarchy cannot prove that its input is one current state."""


class HierarchyTelemetryStaleError(HierarchyIdentityError):
    """The hierarchy identity is known, but its observation is too old."""


class HierarchyCandidateUnavailableError(ValueError):
    """All policy-eligible hierarchy frontiers were unavailable."""


class HierarchyCandidateMode(str, Enum):
    ROUTE_FRONTIER = "route_frontier_per_pair_then_top_pairs_per_shard"


def _positive_int(name: str, value: int, *, zero: bool = False) -> None:
    if type(value) is not int or value < (0 if zero else 1):
        qualifier = "non-negative" if zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} int")


def _sha256(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _candidate_dict(candidate: RouteCandidate) -> dict[str, object]:
    return {
        "pair_index": candidate.pair_index,
        "prefill_index": candidate.prefill_index,
        "decoder_index": candidate.decoder_index,
        "edge_id": candidate.edge_id,
        "route": candidate.route.value,
        "work": candidate.work.as_dict(),
        "predicted_e2e_ms": candidate.predicted_e2e_ms,
        "predicted_ttft_ms": candidate.predicted_ttft_ms,
        "uncertainty_ms": candidate.uncertainty_ms,
        "cache_affinity": candidate.cache_affinity,
    }


@dataclass(frozen=True)
class HierarchicalRequestHeader:
    """Request identity carried separately from the full candidate population.

    A node/pair agent must be able to reduce its local candidates before the
    global coordinator sees them.  ``GlobalRequest`` intentionally still
    requires at least one candidate, so this small header is the transport
    boundary for a distributed frontier submission.
    """

    request_id: str
    tenant_id: str
    arrival_ns: int
    deadline_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("hierarchical request_id must be nonempty")
        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("hierarchical tenant_id must be nonempty")
        _positive_int("hierarchical arrival_ns", self.arrival_ns, zero=True)
        if type(self.deadline_ns) is not int or self.deadline_ns <= self.arrival_ns:
            raise ValueError("hierarchical deadline_ns must exceed arrival_ns")

    @classmethod
    def from_request(cls, request: GlobalRequest) -> "HierarchicalRequestHeader":
        if not isinstance(request, GlobalRequest):
            raise TypeError("request must be GlobalRequest")
        return cls(
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            arrival_ns=request.arrival_ns,
            deadline_ns=request.deadline_ns,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "arrival_ns": self.arrival_ns,
            "deadline_ns": self.deadline_ns,
        }


@dataclass(frozen=True)
class PairCandidateFrontier:
    """Bounded candidate result emitted by one pair/node agent.

    ``raw_candidate_count`` and ``candidate_population_fingerprint`` describe
    the full local population before truncation.  ``candidates`` is the only
    payload sent to the shard/global layer and is therefore bounded by the
    pair agent's route frontier limit.  The global coordinator can verify
    identity and omission accounting without receiving the raw population.
    """

    pair_index: int
    node_id: str
    source_epoch: str
    topology_fingerprint_sha256: str
    profile_fingerprint_sha256: str
    sequence: int
    cross_layer_supported: bool
    raw_candidate_count: int
    candidate_population_fingerprint: str
    candidates: tuple[RouteCandidate, ...]
    schema: str = PAIR_FRONTIER_SCHEMA

    def __post_init__(self) -> None:
        _positive_int("frontier pair_index", self.pair_index, zero=True)
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("frontier node_id must be nonempty")
        if not isinstance(self.source_epoch, str) or not self.source_epoch.strip():
            raise ValueError("frontier source_epoch must be nonempty")
        _sha256("frontier topology fingerprint", self.topology_fingerprint_sha256)
        _sha256("frontier profile fingerprint", self.profile_fingerprint_sha256)
        _positive_int("frontier sequence", self.sequence)
        if type(self.cross_layer_supported) is not bool:
            raise TypeError("frontier cross_layer_supported must be bool")
        _positive_int("frontier raw_candidate_count", self.raw_candidate_count)
        _sha256(
            "frontier candidate population fingerprint",
            self.candidate_population_fingerprint,
        )
        if not self.candidates:
            raise ValueError("frontier requires at least one candidate")
        if len(self.candidates) > self.raw_candidate_count:
            raise ValueError("frontier candidates exceed raw population")
        if any(
            not isinstance(item, RouteCandidate)
            or item.pair_index != self.pair_index
            for item in self.candidates
        ):
            raise ValueError("frontier candidate pair differs")
        keys = [item.identity_key for item in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("frontier candidates must be unique by P/D edge")
        if self.schema != PAIR_FRONTIER_SCHEMA:
            raise ValueError("pair frontier schema mismatch")

    @classmethod
    def from_candidates(
        cls,
        *,
        pair_index: int,
        node_id: str,
        source_epoch: str,
        topology_fingerprint_sha256: str,
        profile_fingerprint_sha256: str,
        sequence: int,
        cross_layer_supported: bool,
        candidates: Iterable[RouteCandidate],
        selected_candidates: Iterable[RouteCandidate],
    ) -> "PairCandidateFrontier":
        raw = tuple(candidates)
        selected = tuple(selected_candidates)
        if not raw:
            raise ValueError("pair frontier raw population is empty")
        return cls(
            pair_index=pair_index,
            node_id=node_id,
            source_epoch=source_epoch,
            topology_fingerprint_sha256=topology_fingerprint_sha256,
            profile_fingerprint_sha256=profile_fingerprint_sha256,
            sequence=sequence,
            cross_layer_supported=cross_layer_supported,
            raw_candidate_count=len(raw),
            candidate_population_fingerprint=_stable_hash(
                [_candidate_dict(item) for item in raw]
            ),
            candidates=selected,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "pair_index": self.pair_index,
            "node_id": self.node_id,
            "source_epoch": self.source_epoch,
            "topology_fingerprint_sha256": self.topology_fingerprint_sha256,
            "profile_fingerprint_sha256": self.profile_fingerprint_sha256,
            "sequence": self.sequence,
            "cross_layer_supported": self.cross_layer_supported,
            "raw_candidate_count": self.raw_candidate_count,
            "candidate_population_fingerprint": self.candidate_population_fingerprint,
            "candidates": [_candidate_dict(item) for item in self.candidates],
        }


@dataclass(frozen=True)
class NodeTelemetryEnvelope:
    """Identity and bounded ownership summary emitted by a node agent."""

    node_id: str
    source_epoch: str
    topology_fingerprint_sha256: str
    profile_fingerprint_sha256: str
    sequence: int
    pair_indices: tuple[int, ...]
    cross_layer_supported: bool
    schema: str = NODE_ENVELOPE_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("node_id must be nonempty")
        if not isinstance(self.source_epoch, str) or not self.source_epoch.strip():
            raise ValueError("source_epoch must be nonempty")
        _sha256("topology_fingerprint_sha256", self.topology_fingerprint_sha256)
        _sha256("profile_fingerprint_sha256", self.profile_fingerprint_sha256)
        _positive_int("sequence", self.sequence)
        if not self.pair_indices or tuple(sorted(set(self.pair_indices))) != self.pair_indices:
            raise ValueError("node pair indices must be sorted and unique")
        for pair in self.pair_indices:
            _positive_int("pair_index", pair, zero=True)
        if type(self.cross_layer_supported) is not bool:
            raise TypeError("cross_layer_supported must be bool")
        if self.schema != NODE_ENVELOPE_SCHEMA:
            raise ValueError("node envelope schema mismatch")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "node_id": self.node_id,
            "source_epoch": self.source_epoch,
            "topology_fingerprint_sha256": self.topology_fingerprint_sha256,
            "profile_fingerprint_sha256": self.profile_fingerprint_sha256,
            "sequence": self.sequence,
            "pair_indices": list(self.pair_indices),
            "cross_layer_supported": self.cross_layer_supported,
        }


@dataclass(frozen=True)
class PairTelemetryEnvelope:
    """The pair agent's identity plus candidate population cardinality."""

    pair_index: int
    node_id: str
    source_epoch: str
    topology_fingerprint_sha256: str
    profile_fingerprint_sha256: str
    sequence: int
    raw_candidate_count: int
    forwarded_candidate_count: int
    schema: str = PAIR_ENVELOPE_SCHEMA

    def __post_init__(self) -> None:
        _positive_int("pair_index", self.pair_index, zero=True)
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("pair envelope node_id must be nonempty")
        if not isinstance(self.source_epoch, str) or not self.source_epoch.strip():
            raise ValueError("pair envelope source_epoch must be nonempty")
        _sha256("pair topology fingerprint", self.topology_fingerprint_sha256)
        _sha256("pair profile fingerprint", self.profile_fingerprint_sha256)
        _positive_int("pair sequence", self.sequence)
        _positive_int("raw_candidate_count", self.raw_candidate_count)
        _positive_int(
            "forwarded_candidate_count", self.forwarded_candidate_count, zero=True)
        if self.forwarded_candidate_count > self.raw_candidate_count:
            raise ValueError("pair forwarded candidates exceed raw candidates")
        if self.schema != PAIR_ENVELOPE_SCHEMA:
            raise ValueError("pair envelope schema mismatch")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "pair_index": self.pair_index,
            "node_id": self.node_id,
            "source_epoch": self.source_epoch,
            "topology_fingerprint_sha256": self.topology_fingerprint_sha256,
            "profile_fingerprint_sha256": self.profile_fingerprint_sha256,
            "sequence": self.sequence,
            "raw_candidate_count": self.raw_candidate_count,
            "forwarded_candidate_count": self.forwarded_candidate_count,
        }


@dataclass(frozen=True)
class ShardCandidateEnvelope:
    """A shard's bounded candidate frontier and fan-in accounting."""

    shard_id: int
    source_epoch: str
    topology_fingerprint_sha256: str
    profile_fingerprint_sha256: str
    sequence: int
    pair_indices: tuple[int, ...]
    forwarded_pair_indices: tuple[int, ...]
    raw_candidate_count: int
    forwarded_candidate_count: int
    omitted_pair_count: int
    candidates: tuple[RouteCandidate, ...]
    mode: HierarchyCandidateMode = HierarchyCandidateMode.ROUTE_FRONTIER
    schema: str = SHARD_ENVELOPE_SCHEMA

    def __post_init__(self) -> None:
        _positive_int("shard_id", self.shard_id, zero=True)
        if not isinstance(self.source_epoch, str) or not self.source_epoch.strip():
            raise ValueError("shard source_epoch must be nonempty")
        _sha256("shard topology fingerprint", self.topology_fingerprint_sha256)
        _sha256("shard profile fingerprint", self.profile_fingerprint_sha256)
        _positive_int("shard sequence", self.sequence)
        if tuple(sorted(set(self.pair_indices))) != self.pair_indices:
            raise ValueError("shard pair indices must be sorted and unique")
        if tuple(sorted(set(self.forwarded_pair_indices))) != self.forwarded_pair_indices:
            raise ValueError("forwarded pair indices must be sorted and unique")
        if not set(self.forwarded_pair_indices).issubset(self.pair_indices):
            raise ValueError("forwarded pair is not owned by shard")
        _positive_int(
            "shard raw_candidate_count", self.raw_candidate_count, zero=True)
        _positive_int(
            "shard forwarded_candidate_count",
            self.forwarded_candidate_count,
            zero=True,
        )
        _positive_int("shard omitted_pair_count", self.omitted_pair_count, zero=True)
        if self.forwarded_candidate_count != len(self.candidates):
            raise ValueError("shard candidate count does not match candidates")
        if self.omitted_pair_count != len(self.pair_indices) - len(self.forwarded_pair_indices):
            raise ValueError("shard omitted pair count is inconsistent")
        if self.schema != SHARD_ENVELOPE_SCHEMA:
            raise ValueError("shard envelope schema mismatch")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "shard_id": self.shard_id,
            "source_epoch": self.source_epoch,
            "topology_fingerprint_sha256": self.topology_fingerprint_sha256,
            "profile_fingerprint_sha256": self.profile_fingerprint_sha256,
            "sequence": self.sequence,
            "pair_indices": list(self.pair_indices),
            "forwarded_pair_indices": list(self.forwarded_pair_indices),
            "raw_candidate_count": self.raw_candidate_count,
            "forwarded_candidate_count": self.forwarded_candidate_count,
            "omitted_pair_count": self.omitted_pair_count,
            "candidates": [_candidate_dict(item) for item in self.candidates],
            "mode": self.mode.value,
        }


@dataclass(frozen=True)
class HierarchicalReductionReceipt:
    """Immutable proof of what the shard layer forwarded to global."""

    request_id: str
    tenant_id: str
    source_epoch: str
    topology_fingerprint_sha256: str
    profile_fingerprint_sha256: str
    sequence: int
    raw_pair_count: int
    raw_candidate_count: int
    forwarded_candidate_count: int
    omitted_pair_count: int
    shard_count: int
    max_pairs_per_shard: int
    max_routes_per_pair: int
    shard_fan_in: tuple[tuple[int, int, int, int], ...]
    identity_mode: str
    candidate_population_fingerprint: str
    forwarded_candidate_fingerprint: str
    schema: str = REDUCTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in ("request_id", "tenant_id", "source_epoch", "identity_mode"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        _sha256("reduction topology fingerprint", self.topology_fingerprint_sha256)
        _sha256("reduction profile fingerprint", self.profile_fingerprint_sha256)
        _positive_int("reduction sequence", self.sequence)
        for name in (
            "raw_pair_count", "raw_candidate_count", "forwarded_candidate_count",
            "shard_count", "max_pairs_per_shard", "max_routes_per_pair",
        ):
            _positive_int(name, getattr(self, name))
        _positive_int("omitted_pair_count", self.omitted_pair_count, zero=True)
        if self.forwarded_candidate_count > self.raw_candidate_count:
            raise ValueError("reduction forwarded candidates exceed raw candidates")
        if len(self.shard_fan_in) != self.shard_count:
            raise ValueError("reduction shard fan-in cardinality mismatch")
        for shard_id, raw_pairs, forwarded_pairs, forwarded_candidates in self.shard_fan_in:
            _positive_int("fan-in shard_id", shard_id, zero=True)
            _positive_int("fan-in raw_pairs", raw_pairs, zero=True)
            _positive_int("fan-in forwarded_pairs", forwarded_pairs, zero=True)
            _positive_int(
                "fan-in forwarded_candidates", forwarded_candidates, zero=True)
            if forwarded_pairs > raw_pairs:
                raise ValueError("fan-in forwarded pairs exceed raw pairs")
        _sha256("candidate population fingerprint", self.candidate_population_fingerprint)
        _sha256("forwarded candidate fingerprint", self.forwarded_candidate_fingerprint)
        if self.schema != REDUCTION_RECEIPT_SCHEMA:
            raise ValueError("reduction receipt schema mismatch")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "source_epoch": self.source_epoch,
            "topology_fingerprint_sha256": self.topology_fingerprint_sha256,
            "profile_fingerprint_sha256": self.profile_fingerprint_sha256,
            "sequence": self.sequence,
            "raw_pair_count": self.raw_pair_count,
            "raw_candidate_count": self.raw_candidate_count,
            "forwarded_candidate_count": self.forwarded_candidate_count,
            "omitted_pair_count": self.omitted_pair_count,
            "shard_count": self.shard_count,
            "max_pairs_per_shard": self.max_pairs_per_shard,
            "max_routes_per_pair": self.max_routes_per_pair,
            "shard_fan_in": [
                {
                    "shard_id": shard_id,
                    "raw_pairs": raw_pairs,
                    "forwarded_pairs": forwarded_pairs,
                    "forwarded_candidates": forwarded_candidates,
                }
                for shard_id, raw_pairs, forwarded_pairs, forwarded_candidates
                in self.shard_fan_in
            ],
            "identity_mode": self.identity_mode,
            "candidate_population_fingerprint": self.candidate_population_fingerprint,
            "forwarded_candidate_fingerprint": self.forwarded_candidate_fingerprint,
        }


@dataclass(frozen=True)
class HierarchicalReduction:
    request: GlobalRequest
    receipt: HierarchicalReductionReceipt
    nodes: tuple[NodeTelemetryEnvelope, ...]
    pairs: tuple[PairTelemetryEnvelope, ...]
    shards: tuple[ShardCandidateEnvelope, ...]

    @property
    def raw_candidate_count(self) -> int:
        return self.receipt.raw_candidate_count

    @property
    def forwarded_candidate_count(self) -> int:
        return self.receipt.forwarded_candidate_count

    @property
    def fingerprint(self) -> str:
        return _stable_hash({
            "receipt": self.receipt.as_dict(),
            "shards": [item.as_dict() for item in self.shards],
        })


class HierarchicalCandidateReducer:
    """Reduce a large request population with explicit bounded fan-in.

    ``max_routes_per_pair`` preserves a small local-vs-remote route frontier.
    ``max_pairs_per_shard`` bounds how many pair frontiers reach the global
    controller.  The static frontier order is only a local admission hint;
    the global controller still applies live capacity, fairness, queue, SLO,
    and cross-layer resource prices to every forwarded candidate.
    """

    def __init__(
        self,
        *,
        shard_count: int,
        max_pairs_per_shard: int,
        max_routes_per_pair: int = 2,
        pair_to_shard: Mapping[int, int] | None = None,
        pair_capacities: Mapping[int, ResourceVector] | None = None,
        telemetry_fresh_ns: int | None = None,
        telemetry_stale_grace_ns: int = 0,
    ) -> None:
        _positive_int("shard_count", shard_count)
        _positive_int("max_pairs_per_shard", max_pairs_per_shard)
        _positive_int("max_routes_per_pair", max_routes_per_pair)
        if telemetry_fresh_ns is not None:
            _positive_int("telemetry_fresh_ns", telemetry_fresh_ns)
        _positive_int(
            "telemetry_stale_grace_ns",
            telemetry_stale_grace_ns,
            zero=True,
        )
        if pair_to_shard is not None:
            for pair, shard in pair_to_shard.items():
                _positive_int("pair_to_shard pair", pair, zero=True)
                _positive_int("pair_to_shard shard", shard, zero=True)
                if shard >= shard_count:
                    raise ValueError("pair_to_shard shard exceeds shard_count")
        if pair_capacities is not None:
            for pair, capacity in pair_capacities.items():
                _positive_int("pair_capacities pair", pair, zero=True)
                if not isinstance(capacity, ResourceVector):
                    raise TypeError("pair_capacities values must be ResourceVector")
        self.shard_count = shard_count
        self.max_pairs_per_shard = max_pairs_per_shard
        self.max_routes_per_pair = max_routes_per_pair
        self.pair_to_shard = dict(pair_to_shard or {})
        self.pair_capacities = dict(pair_capacities or {})
        self.telemetry_fresh_ns = telemetry_fresh_ns
        self.telemetry_stale_grace_ns = telemetry_stale_grace_ns

    def _shard_for(self, pair_index: int) -> int:
        return self.pair_to_shard.get(pair_index, pair_index % self.shard_count)

    @staticmethod
    def _identity(telemetry: PairTelemetry) -> tuple[str, str, str, int, str, bool]:
        cross_layer = telemetry.cross_layer
        if cross_layer is None:
            # This fallback is explicitly endpoint-profile-only.  It does not
            # claim fabric telemetry and therefore cannot be mixed with a
            # cross-layer observation in one reduction.
            return (
                telemetry.agent_epoch,
                telemetry.profile_fingerprint_sha256,
                telemetry.profile_fingerprint_sha256,
                telemetry.sequence,
                f"pair-agent-{telemetry.pair_index}",
                False,
            )
        return (
            cross_layer.source_epoch,
            telemetry.profile_fingerprint_sha256,
            cross_layer.topology_fingerprint_sha256,
            # ``PairTelemetry.sequence`` is assigned by the frontend's
            # atomic GlobalTelemetryBatch.  The cross-layer envelope's
            # sequence is producer-local provenance (the NCCL/Cassini
            # observer on one pair) and is allowed to differ between pair
            # agents in the same global poll.  Using it here would reject a
            # valid Perlmutter-wide state before global orchestration.
            telemetry.sequence,
            cross_layer.node_id,
            True,
        )

    @classmethod
    def _aggregate_topology_fingerprint(
        cls,
        telemetry: Mapping[int, PairTelemetry],
        pairs: Iterable[int],
    ) -> str:
        """Return a stable identity for the pair/node topology set.

        ``CrossLayerTelemetry.topology_fingerprint_sha256`` is endpoint-local
        evidence.  On Perlmutter it can differ by node, GPU placement, or
        communicator, so it must remain attached to each pair envelope rather
        than be compared as one scalar allocation identity.  The aggregate
        digest lets shard and reduction receipts still bind the exact mapping
        that was used for the fan-in.
        """

        members = []
        for pair in sorted(set(pairs)):
            item = telemetry[pair]
            identity = cls._identity(item)
            members.append({
                "pair_index": pair,
                "node_id": identity[4],
                "topology_fingerprint_sha256": identity[2],
            })
        return _stable_hash({
            "schema": "tempo-go-topology-set-v1",
            "members": members,
        })

    def _rank(
        self,
        candidate: RouteCandidate,
        *,
        telemetry: PairTelemetry | None = None,
    ) -> tuple[float, int, bool, int, int, str]:
        score = candidate.predicted_e2e_ms + candidate.uncertainty_ms
        unhealthy = 0
        if telemetry is not None:
            score += candidate.predicted_ttft_ms * (
                telemetry.multiplier(candidate.route) - 1.0)
            if telemetry.health(candidate.route) in {
                PathHealth.SKIP,
                PathHealth.DENIED,
            }:
                unhealthy = 1
                score += 1_000_000.0
            capacity = self.pair_capacities.get(candidate.pair_index)
            if capacity is not None:
                projected = telemetry.observed_total + candidate.work
                score += 100.0 * projected.dominant_ratio(capacity)
            if telemetry.cross_layer is not None:
                externality, _contributions, _confidence = (
                    telemetry.cross_layer.route_externality(candidate.route)
                )
                score += externality
        return (
            score,
            unhealthy,
            not candidate.cache_affinity,
            int(candidate.prefill_index),
            candidate.pair_index,
            candidate.route.value,
        )

    @staticmethod
    def _fully_quarantined(telemetry: PairTelemetry) -> bool:
        """Return whether a failed endpoint supplied no policy-eligible path."""

        return (
            isinstance(telemetry.quarantine_reason, str)
            and bool(telemetry.quarantine_reason.strip())
            and telemetry.local_health is PathHealth.DENIED
            and telemetry.remote_health is PathHealth.DENIED
        )

    def _validate_identity(
        self,
        request: GlobalRequest,
        telemetry: Mapping[int, PairTelemetry],
        *,
        now_ns: int,
    ) -> tuple[str, str, str, int, str, bool]:
        pairs = sorted({
            endpoint
            for candidate in request.candidates
            for endpoint in (
                candidate.prefill_index,
                candidate.decoder_index,
            )
        })
        missing = [pair for pair in pairs if pair not in telemetry]
        if missing:
            raise HierarchyIdentityError(
                f"hierarchy telemetry missing for pair(s): {missing}")
        identities = {
            pair: self._identity(telemetry[pair]) for pair in pairs
        }
        policy_pairs = [
            pair for pair in pairs
            if not self._fully_quarantined(telemetry[pair])
        ]
        base_pair = next(
            (pair for pair in policy_pairs if identities[pair][5]),
            policy_pairs[0] if policy_pairs else pairs[0],
        )
        base = identities[base_pair]
        for pair in pairs:
            identity = identities[pair]
            # Epoch, endpoint profile, atomic batch sequence, and whether the
            # cross-layer producer is present are allocation-wide identity.
            # Node IDs and topology fingerprints are intentionally pair-local
            # provenance and are checked on their own envelopes below.
            if (
                identity[1] != base[1]
                or identity[3] != base[3]
            ):
                raise HierarchyIdentityError(
                    f"mixed hierarchy identity at pair {pair}: {identity} != {base}")
            if self._fully_quarantined(telemetry[pair]):
                # A bounded endpoint fetch failure deliberately strips the
                # untrusted cross-layer envelope and denies both paths.  It is
                # safe to retain that pair as omitted receipt evidence; it is
                # not safe to mistake the missing envelope for a mixed live
                # policy view and reject the healthy allocation frontier.
                continue
            if identity[0] != base[0] or identity[5] != base[5]:
                raise HierarchyIdentityError(
                    f"mixed hierarchy identity at pair {pair}: {identity} != {base}")
        if self.telemetry_fresh_ns is not None:
            stale = [
                pair for pair in pairs
                if telemetry[pair].sampled_ns > now_ns
                or now_ns - telemetry[pair].sampled_ns > (
                    self.telemetry_fresh_ns
                    + self.telemetry_stale_grace_ns
                )
            ]
            if stale:
                raise HierarchyTelemetryStaleError(
                    f"hierarchy telemetry stale for pair(s): {stale}")
        return (
            base[0],
            base[1],
            self._aggregate_topology_fingerprint(telemetry, pairs),
            base[3],
            base[4],
            base[5],
        )

    @staticmethod
    def _telemetry_values(
        telemetry: Iterable[PairTelemetry] | Mapping[int, PairTelemetry],
    ) -> dict[int, PairTelemetry]:
        if isinstance(telemetry, Mapping):
            values = dict(telemetry)
        else:
            telemetry_values = tuple(telemetry)
            if any(not isinstance(item, PairTelemetry) for item in telemetry_values):
                raise TypeError("hierarchy telemetry must contain PairTelemetry")
            pair_indices = [item.pair_index for item in telemetry_values]
            if len(pair_indices) != len(set(pair_indices)):
                raise ValueError("hierarchy telemetry contains duplicate pair")
            values = {item.pair_index: item for item in telemetry_values}
        if any(not isinstance(item, PairTelemetry) for item in values.values()):
            raise TypeError("hierarchy telemetry must contain PairTelemetry")
        return values

    def build_pair_frontier(
        self,
        *,
        pair_index: int,
        candidates: Iterable[RouteCandidate],
        telemetry: PairTelemetry,
    ) -> PairCandidateFrontier:
        """Build the bounded result emitted by one pair/node agent."""

        _positive_int("frontier pair_index", pair_index, zero=True)
        if not isinstance(telemetry, PairTelemetry):
            raise TypeError("frontier telemetry must be PairTelemetry")
        if telemetry.pair_index != pair_index:
            raise ValueError("frontier telemetry pair differs")
        raw = tuple(candidates)
        if not raw:
            raise ValueError("frontier candidate population is empty")
        if any(
            not isinstance(item, RouteCandidate) or item.pair_index != pair_index
            for item in raw
        ):
            raise ValueError("frontier candidate pair differs")
        keys = [item.identity_key for item in raw]
        if len(keys) != len(set(keys)):
            raise ValueError("frontier raw candidates must be unique by P/D edge")
        selected = tuple(sorted(
            raw,
            key=lambda item: self._rank(item, telemetry=telemetry),
        )[: self.max_routes_per_pair])
        source_epoch, profile_fp, topology_fp, sequence, node_id, cross_layer = (
            self._identity(telemetry)
        )
        return PairCandidateFrontier.from_candidates(
            pair_index=pair_index,
            node_id=node_id,
            source_epoch=source_epoch,
            topology_fingerprint_sha256=topology_fp,
            profile_fingerprint_sha256=profile_fp,
            sequence=sequence,
            cross_layer_supported=cross_layer,
            candidates=raw,
            selected_candidates=selected,
        )

    def reduce_frontiers(
        self,
        header: HierarchicalRequestHeader,
        *,
        frontiers: Iterable[PairCandidateFrontier],
        telemetry: Iterable[PairTelemetry] | Mapping[int, PairTelemetry],
        now_ns: int,
    ) -> HierarchicalReduction:
        """Reduce precomputed pair frontiers without receiving raw candidates."""

        if not isinstance(header, HierarchicalRequestHeader):
            raise TypeError("frontier header must be HierarchicalRequestHeader")
        values = self._telemetry_values(telemetry)
        frontier_values = tuple(frontiers)
        if not frontier_values:
            raise ValueError("frontier reduction requires at least one pair")
        if any(not isinstance(item, PairCandidateFrontier) for item in frontier_values):
            raise TypeError("frontiers must contain PairCandidateFrontier")
        pair_indices = [item.pair_index for item in frontier_values]
        if len(pair_indices) != len(set(pair_indices)):
            raise ValueError("frontiers contain duplicate pair")
        if any(len(item.candidates) > self.max_routes_per_pair for item in frontier_values):
            raise ValueError("pair frontier exceeds configured route bound")
        missing = [pair for pair in pair_indices if pair not in values]
        if missing:
            raise HierarchyIdentityError(
                f"frontier telemetry missing for pair(s): {missing}")

        bounded_candidates = tuple(
            candidate
            for item in sorted(frontier_values, key=lambda value: value.pair_index)
            for candidate in item.candidates
        )
        bounded_request = GlobalRequest(
            request_id=header.request_id,
            tenant_id=header.tenant_id,
            arrival_ns=header.arrival_ns,
            deadline_ns=header.deadline_ns,
            candidates=bounded_candidates,
        )
        self._validate_identity(bounded_request, values, now_ns=now_ns)
        for frontier in frontier_values:
            expected = self._identity(values[frontier.pair_index])
            actual = (
                frontier.source_epoch,
                frontier.profile_fingerprint_sha256,
                frontier.topology_fingerprint_sha256,
                frontier.sequence,
                frontier.node_id,
                frontier.cross_layer_supported,
            )
            if actual != expected:
                raise HierarchyIdentityError(
                    f"frontier identity differs at pair {frontier.pair_index}: "
                    f"{actual} != {expected}"
                )

        reduction = self.reduce(
            bounded_request,
            telemetry=values,
            now_ns=now_ns,
        )
        raw_counts = {
            item.pair_index: item.raw_candidate_count for item in frontier_values
        }
        raw_candidate_count = sum(raw_counts.values())
        frontier_population = [
            {
                "pair_index": item.pair_index,
                "raw_candidate_count": item.raw_candidate_count,
                "candidate_population_fingerprint": item.candidate_population_fingerprint,
            }
            for item in sorted(frontier_values, key=lambda value: value.pair_index)
        ]
        corrected_pairs = tuple(
            replace(item, raw_candidate_count=raw_counts[item.pair_index])
            for item in reduction.pairs
        )
        corrected_shards = tuple(
            replace(
                item,
                raw_candidate_count=sum(
                    raw_counts[pair] for pair in item.pair_indices
                ),
            )
            for item in reduction.shards
        )
        identity_mode = (
            "cross_layer_frontier"
            if any(item.cross_layer_supported for item in frontier_values)
            else "endpoint_profile_frontier"
        )
        corrected_receipt = replace(
            reduction.receipt,
            raw_candidate_count=raw_candidate_count,
            identity_mode=identity_mode,
            candidate_population_fingerprint=_stable_hash(frontier_population),
        )
        return replace(
            reduction,
            receipt=corrected_receipt,
            pairs=corrected_pairs,
            shards=corrected_shards,
        )

    def reduce_shard_frontiers(
        self,
        header: HierarchicalRequestHeader,
        *,
        shards: Iterable[ShardCandidateEnvelope],
        pairs: Iterable[PairTelemetryEnvelope],
        telemetry: Iterable[PairTelemetry] | Mapping[int, PairTelemetry],
        now_ns: int,
    ) -> HierarchicalReduction:
        """Consume shard outputs at the global layer.

        Shard agents have already chosen their bounded pair frontiers.  The
        global coordinator therefore receives at most
        ``shard_count * max_pairs_per_shard * max_routes_per_pair`` candidates,
        while ``pairs`` retains compact raw/forwarded cardinality receipts for
        omitted pairs.  No raw request candidate population is reconstructed.
        """

        if not isinstance(header, HierarchicalRequestHeader):
            raise TypeError("frontier header must be HierarchicalRequestHeader")
        shard_values = tuple(shards)
        pair_values = tuple(pairs)
        if not shard_values:
            raise ValueError("shard reduction requires shard envelopes")
        if any(not isinstance(item, ShardCandidateEnvelope) for item in shard_values):
            raise TypeError("shards must contain ShardCandidateEnvelope")
        if any(not isinstance(item, PairTelemetryEnvelope) for item in pair_values):
            raise TypeError("pairs must contain PairTelemetryEnvelope")
        if len({item.shard_id for item in shard_values}) != len(shard_values):
            raise ValueError("shards contain duplicate shard")
        if len({item.pair_index for item in pair_values}) != len(pair_values):
            raise ValueError("pair envelopes contain duplicate pair")
        values = self._telemetry_values(telemetry)
        pair_by_index = {item.pair_index: item for item in pair_values}
        shard_pair_indices = {
            pair for shard in shard_values for pair in shard.pair_indices
        }
        if any(pair not in shard_pair_indices for pair in pair_by_index):
            raise HierarchyIdentityError("pair envelope is outside shard ownership")
        forwarded_by_pair: dict[int, list[RouteCandidate]] = defaultdict(list)
        for shard in shard_values:
            if shard.shard_id >= self.shard_count:
                raise HierarchyIdentityError("shard envelope exceeds reducer shard count")
            if shard.raw_candidate_count < shard.forwarded_candidate_count:
                raise HierarchyIdentityError("shard raw/forwarded count differs")
            if shard.forwarded_candidate_count != len(shard.candidates):
                raise HierarchyIdentityError("shard candidate count differs")
            for candidate in shard.candidates:
                forwarded_by_pair[candidate.pair_index].append(candidate)
            for pair in shard.forwarded_pair_indices:
                if pair not in shard_pair_indices:
                    raise HierarchyIdentityError(
                        f"shard forwarded pair is outside shard ownership: {pair}")
            expected_raw = sum(
                pair_by_index[pair].raw_candidate_count
                for pair in shard.pair_indices
                if pair in pair_by_index
            )
            expected_forwarded = sum(
                pair_by_index[pair].forwarded_candidate_count
                for pair in shard.forwarded_pair_indices
                if pair in pair_by_index
            )
            missing_pair_envelopes = [
                pair for pair in shard.pair_indices if pair not in pair_by_index
            ]
            if not missing_pair_envelopes and expected_raw != shard.raw_candidate_count:
                raise HierarchyIdentityError(
                    f"shard raw count differs at shard {shard.shard_id}")
            if expected_raw > shard.raw_candidate_count:
                raise HierarchyIdentityError(
                    f"shard raw count under-reports pair envelopes at shard {shard.shard_id}")
            if expected_forwarded > shard.forwarded_candidate_count:
                raise HierarchyIdentityError(
                    f"shard forwarded count differs at shard {shard.shard_id}")

        if not forwarded_by_pair:
            raise ValueError("shard reduction has no forwarded candidate")
        for pair, candidates in forwarded_by_pair.items():
            if pair in pair_by_index and len(candidates) != pair_by_index[pair].forwarded_candidate_count:
                raise HierarchyIdentityError(
                    f"forwarded pair count differs at pair {pair}")
            if len(candidates) > self.max_routes_per_pair:
                raise HierarchyIdentityError(
                    f"forwarded pair exceeds route bound at pair {pair}")

        frontier_values = []
        effective_pairs: list[PairTelemetryEnvelope] = []
        for pair in sorted(forwarded_by_pair):
            pair_envelope = pair_by_index.get(pair)
            expected = self._identity(values[pair])
            if pair_envelope is None:
                pair_envelope = PairTelemetryEnvelope(
                    pair_index=pair,
                    node_id=expected[4],
                    source_epoch=expected[0],
                    topology_fingerprint_sha256=expected[2],
                    profile_fingerprint_sha256=expected[1],
                    sequence=expected[3],
                    raw_candidate_count=len(forwarded_by_pair[pair]),
                    forwarded_candidate_count=len(forwarded_by_pair[pair]),
                )
            else:
                actual = (
                    pair_envelope.source_epoch,
                    pair_envelope.profile_fingerprint_sha256,
                    pair_envelope.topology_fingerprint_sha256,
                    pair_envelope.sequence,
                    pair_envelope.node_id,
                    expected[5],
                )
                if actual != expected:
                    raise HierarchyIdentityError(
                        f"pair envelope identity differs at pair {pair}: "
                        f"{actual} != {expected}"
                    )
            effective_pairs.append(pair_envelope)
            frontier_values.append(PairCandidateFrontier(
                pair_index=pair,
                node_id=pair_envelope.node_id,
                source_epoch=pair_envelope.source_epoch,
                topology_fingerprint_sha256=pair_envelope.topology_fingerprint_sha256,
                profile_fingerprint_sha256=pair_envelope.profile_fingerprint_sha256,
                sequence=pair_envelope.sequence,
                cross_layer_supported=expected[5],
                raw_candidate_count=pair_envelope.raw_candidate_count,
                candidate_population_fingerprint=_stable_hash(
                    pair_envelope.as_dict()
                ),
                candidates=tuple(forwarded_by_pair[pair]),
            ))

        bounded = self.reduce_frontiers(
            header,
            frontiers=frontier_values,
            telemetry=values,
            now_ns=now_ns,
        )
        raw_pair_count = sum(len(item.pair_indices) for item in shard_values)
        raw_candidate_count = sum(item.raw_candidate_count for item in shard_values)
        forwarded_candidate_count = sum(
            item.forwarded_candidate_count for item in shard_values
        )
        omitted_pair_count = sum(item.omitted_pair_count for item in shard_values)
        shard_fan_in = tuple(
            (
                item.shard_id,
                len(item.pair_indices),
                len(item.forwarded_pair_indices),
                item.forwarded_candidate_count,
            )
            for item in sorted(shard_values, key=lambda value: value.shard_id)
        )
        corrected_receipt = replace(
            bounded.receipt,
            raw_pair_count=raw_pair_count,
            raw_candidate_count=raw_candidate_count,
            forwarded_candidate_count=forwarded_candidate_count,
            omitted_pair_count=omitted_pair_count,
            shard_fan_in=shard_fan_in,
            identity_mode="cross_layer_shard_frontier",
            candidate_population_fingerprint=_stable_hash({
                "pairs": [item.as_dict() for item in effective_pairs],
                "shards": [item.as_dict() for item in shard_values],
            }),
        )
        nodes_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
        for item in effective_pairs:
            nodes_by_key[(item.node_id, item.topology_fingerprint_sha256)].append(
                item.pair_index
            )
        if not effective_pairs:
            raise ValueError("shard reduction has no effective pair envelope")
        nodes = tuple(
            NodeTelemetryEnvelope(
                node_id=node_id,
                source_epoch=effective_pairs[0].source_epoch,
                topology_fingerprint_sha256=topology,
                profile_fingerprint_sha256=effective_pairs[0].profile_fingerprint_sha256,
                sequence=effective_pairs[0].sequence,
                pair_indices=tuple(sorted(indices)),
                cross_layer_supported=any(
                    self._identity(values[pair])[5] for pair in indices
                ),
            )
            for (node_id, topology), indices in sorted(nodes_by_key.items())
        )
        return replace(
            bounded,
            receipt=corrected_receipt,
            nodes=nodes,
            pairs=tuple(sorted(effective_pairs, key=lambda value: value.pair_index)),
            shards=tuple(sorted(shard_values, key=lambda value: value.shard_id)),
        )

    def reduce(
        self,
        request: GlobalRequest,
        *,
        telemetry: Iterable[PairTelemetry] | Mapping[int, PairTelemetry],
        now_ns: int,
    ) -> HierarchicalReduction:
        if not isinstance(request, GlobalRequest):
            raise TypeError("request must be GlobalRequest")
        _positive_int("now_ns", now_ns, zero=True)
        values = self._telemetry_values(telemetry)
        identity = self._validate_identity(request, values, now_ns=now_ns)
        source_epoch, profile_fp, topology_fp, sequence, _node, cross_layer = identity
        candidates_by_pair: dict[int, list[RouteCandidate]] = defaultdict(list)
        for candidate in request.candidates:
            candidates_by_pair[candidate.pair_index].append(candidate)
        # One request has unique P/D-edge candidates.  Cache the live
        # cross-layer rank once and reuse it for pair frontiers, shard pair
        # selection, and the final global ordering.  Recomputing this score
        # at every hierarchy layer made the global control path grow faster
        # than the actual bounded candidate population.
        rank_cache: dict[
            tuple[int, int, GlobalRoute],
            tuple[float, int, bool, int, int, str],
        ] = {}

        def cached_rank(
            candidate: RouteCandidate,
        ) -> tuple[float, int, bool, int, int, str]:
            key = candidate.identity_key
            value = rank_cache.get(key)
            if value is None:
                value = self._rank(candidate, telemetry=values[candidate.pair_index])
                rank_cache[key] = value
            return value

        pair_frontiers: dict[int, tuple[RouteCandidate, ...]] = {}
        for pair in sorted(candidates_by_pair):
            candidates = candidates_by_pair[pair]
            pair_frontiers[pair] = tuple(
                sorted(candidates, key=cached_rank)[: self.max_routes_per_pair]
            )
        quarantined_pairs = {
            pair for pair in pair_frontiers
            if self._fully_quarantined(values[pair])
        }

        nodes_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
        pairs: list[PairTelemetryEnvelope] = []
        pairs_by_shard: list[list[int]] = [[] for _ in range(self.shard_count)]
        for pair in sorted(pair_frontiers):
            item = values[pair]
            item_epoch, item_profile, item_topology, item_sequence, node_id, item_cross = self._identity(item)
            nodes_by_key[(node_id, item_topology)].append(pair)
            pairs_by_shard[self._shard_for(pair)].append(pair)
            pairs.append(PairTelemetryEnvelope(
                pair_index=pair,
                node_id=node_id,
                source_epoch=item_epoch,
                topology_fingerprint_sha256=item_topology,
                profile_fingerprint_sha256=item_profile,
                sequence=item_sequence,
                raw_candidate_count=len(candidates_by_pair[pair]),
                forwarded_candidate_count=0,
            ))

        shards: list[ShardCandidateEnvelope] = []
        forwarded: list[RouteCandidate] = []
        forwarded_counts: dict[int, int] = {}
        fan_in: list[tuple[int, int, int, int]] = []
        for shard_id in range(self.shard_count):
            shard_pairs = pairs_by_shard[shard_id]
            selected_pairs = sorted(
                [
                    pair for pair in shard_pairs
                    if pair not in quarantined_pairs
                ],
                key=lambda pair: cached_rank(pair_frontiers[pair][0]),
            )[: self.max_pairs_per_shard]
            selected_candidates = tuple(
                candidate
                for pair in selected_pairs
                for candidate in pair_frontiers[pair]
            )
            forwarded.extend(selected_candidates)
            for pair in selected_pairs:
                forwarded_counts[pair] = len(pair_frontiers[pair])
            raw_count = sum(len(candidates_by_pair[pair]) for pair in shard_pairs)
            shards.append(ShardCandidateEnvelope(
                shard_id=shard_id,
                source_epoch=source_epoch,
                topology_fingerprint_sha256=(
                    self._aggregate_topology_fingerprint(values, shard_pairs)
                ),
                profile_fingerprint_sha256=profile_fp,
                sequence=sequence,
                pair_indices=tuple(shard_pairs),
                forwarded_pair_indices=tuple(sorted(selected_pairs)),
                raw_candidate_count=raw_count,
                forwarded_candidate_count=len(selected_candidates),
                omitted_pair_count=len(shard_pairs) - len(selected_pairs),
                candidates=selected_candidates,
            ))
            fan_in.append((
                shard_id,
                len(shard_pairs),
                len(selected_pairs),
                len(selected_candidates),
            ))

        forwarded_tuple = tuple(sorted(
            forwarded,
            key=cached_rank,
        ))
        if not forwarded_tuple:
            # A request can legitimately have no policy-eligible route when
            # every endpoint snapshot is quarantined.  That is an admission
            # rejection, not a malformed client request.  Keep the transport
            # boundary explicit so the coordinator can emit the normal
            # business/overload reject receipt instead of constructing an
            # invalid zero-candidate GlobalRequest or leaking a 400.
            raise HierarchyCandidateUnavailableError(
                "hierarchy has no policy-eligible candidate")
        for index, pair in enumerate(pairs):
            pairs[index] = PairTelemetryEnvelope(
                pair_index=pair.pair_index,
                node_id=pair.node_id,
                source_epoch=pair.source_epoch,
                topology_fingerprint_sha256=pair.topology_fingerprint_sha256,
                profile_fingerprint_sha256=pair.profile_fingerprint_sha256,
                sequence=pair.sequence,
                raw_candidate_count=pair.raw_candidate_count,
                forwarded_candidate_count=forwarded_counts.get(pair.pair_index, 0),
            )
        nodes = tuple(
            NodeTelemetryEnvelope(
                node_id=node_id,
                source_epoch=self._identity(values[pair_indices[0]])[0],
                topology_fingerprint_sha256=topology,
                profile_fingerprint_sha256=(
                    self._identity(values[pair_indices[0]])[1]),
                sequence=self._identity(values[pair_indices[0]])[3],
                pair_indices=tuple(sorted(pair_indices)),
                cross_layer_supported=any(
                    self._identity(values[pair])[5] for pair in pair_indices
                ),
            )
            for (node_id, topology), pair_indices in sorted(nodes_by_key.items())
        )
        raw_population = [_candidate_dict(item) for item in request.candidates]
        forwarded_population = [_candidate_dict(item) for item in forwarded_tuple]
        receipt = HierarchicalReductionReceipt(
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            source_epoch=source_epoch,
            topology_fingerprint_sha256=topology_fp,
            profile_fingerprint_sha256=profile_fp,
            sequence=sequence,
            raw_pair_count=len(pair_frontiers),
            raw_candidate_count=len(request.candidates),
            forwarded_candidate_count=len(forwarded_tuple),
            omitted_pair_count=sum(item.omitted_pair_count for item in shards),
            shard_count=self.shard_count,
            max_pairs_per_shard=self.max_pairs_per_shard,
            max_routes_per_pair=self.max_routes_per_pair,
            shard_fan_in=tuple(fan_in),
            identity_mode=(
                "cross_layer_with_quarantined_pairs"
                if cross_layer and quarantined_pairs
                else "cross_layer" if cross_layer
                else "endpoint_profile_only"
            ),
            candidate_population_fingerprint=_stable_hash(raw_population),
            forwarded_candidate_fingerprint=_stable_hash(forwarded_population),
        )
        return HierarchicalReduction(
            request=GlobalRequest(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                arrival_ns=request.arrival_ns,
                deadline_ns=request.deadline_ns,
                candidates=forwarded_tuple,
            ),
            receipt=receipt,
            nodes=nodes,
            pairs=tuple(pairs),
            shards=tuple(shards),
        )


def submit_hierarchical(
    orchestrator: object,
    reducer: HierarchicalCandidateReducer,
    request: GlobalRequest,
    *,
    now_ns: int,
) -> tuple[GlobalDecision, HierarchicalReduction]:
    """Reduce against one telemetry snapshot and submit to global authority.

    The duck-typed boundary keeps this module independent of the concrete
    orchestrator while making the intended production call explicit.  The
    orchestrator must expose ``telemetry_snapshot()`` and ``submit()``.
    """

    if not isinstance(reducer, HierarchicalCandidateReducer):
        raise TypeError("reducer must be HierarchicalCandidateReducer")
    snapshot = getattr(orchestrator, "telemetry_snapshot", None)
    submit = getattr(orchestrator, "submit", None)
    if not callable(snapshot) or not callable(submit):
        raise TypeError("orchestrator lacks hierarchy submission interface")
    reduction = reducer.reduce(
        request,
        telemetry=snapshot(),
        now_ns=now_ns,
    )
    return submit(reduction.request, now_ns=now_ns), reduction


def submit_hierarchical_frontiers(
    orchestrator: object,
    reducer: HierarchicalCandidateReducer,
    header: HierarchicalRequestHeader,
    frontiers: Iterable[PairCandidateFrontier],
    *,
    now_ns: int,
) -> tuple[GlobalDecision, HierarchicalReduction]:
    """Submit a precomputed pair-agent frontier to the global authority."""

    if not isinstance(reducer, HierarchicalCandidateReducer):
        raise TypeError("reducer must be HierarchicalCandidateReducer")
    snapshot = getattr(orchestrator, "telemetry_snapshot", None)
    submit = getattr(orchestrator, "submit", None)
    if not callable(snapshot) or not callable(submit):
        raise TypeError("orchestrator lacks hierarchy submission interface")
    reduction = reducer.reduce_frontiers(
        header,
        frontiers=frontiers,
        telemetry=snapshot(),
        now_ns=now_ns,
    )
    return submit(reduction.request, now_ns=now_ns), reduction


__all__ = [
    "HIERARCHY_SCHEMA",
    "NODE_ENVELOPE_SCHEMA",
    "PAIR_ENVELOPE_SCHEMA",
    "PAIR_FRONTIER_SCHEMA",
    "SHARD_ENVELOPE_SCHEMA",
    "REDUCTION_RECEIPT_SCHEMA",
    "HierarchyIdentityError",
    "HierarchyTelemetryStaleError",
    "HierarchyCandidateUnavailableError",
    "HierarchyCandidateMode",
    "HierarchicalRequestHeader",
    "PairCandidateFrontier",
    "NodeTelemetryEnvelope",
    "PairTelemetryEnvelope",
    "ShardCandidateEnvelope",
    "HierarchicalReductionReceipt",
    "HierarchicalReduction",
    "HierarchicalCandidateReducer",
    "submit_hierarchical",
    "submit_hierarchical_frontiers",
]
