"""Fail-closed admission policy for live prefill/decode KV routing.

This module deliberately contains no HTTP, vLLM, LMCache, Mooncake, CUDA, or
Slurm code.  A request router classifies a request, calls the immutable policy,
and commits the returned route *before* it starts remote prefill.  Transport
adapters execute the decision but cannot change it.

The validated 2026-08-15 mechanism used one-sample screen profiles.  Those
profiles are useful research evidence but are not production-promotable.  The
default policy therefore requires replicated evidence; callers must opt in
explicitly to reproduce a screen-only decision.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
import threading
from typing import Iterable


POLICY_SCHEMA = "tempo-live-pd-admission-policy-1"


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _finite_nonnegative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


class PDRoute(str, Enum):
    DECODER_LOCAL = "decoder_local_recompute_or_cache"
    REMOTE_PREFILL = "remote_prefill_live_kv"


class PDEvidenceLevel(str, Enum):
    SCREEN = "single_allocation_mechanism_screen"
    REPLICATED = "independent_replication"


class PDDecisionReason(str, Enum):
    REMOTE_BENEFIT_PROVEN = "remote_benefit_lower_bound_meets_margin"
    LOCAL_NO_PROFILE = "no_exact_workload_profile"
    LOCAL_REMOTE_UNAVAILABLE = "remote_backend_unavailable"
    LOCAL_SCREEN_ONLY = "profile_is_screen_only"
    LOCAL_INSUFFICIENT_SAMPLES = "insufficient_calibration_samples"
    LOCAL_CORRECTNESS_UNPROVEN = "remote_local_output_equivalence_unproven"
    LOCAL_REMOTE_FAILURE = "remote_profile_contains_transfer_failures"
    LOCAL_PROFILE_EXPIRED = "profile_epoch_out_of_range"
    LOCAL_MARGIN_NOT_MET = "remote_benefit_lower_bound_below_margin"
    LOCAL_DEADLINE_INFEASIBLE = "remote_upper_bound_exceeds_request_budget"
    LOCAL_REMOTE_PRESTART_FAILURE = "remote_failed_before_prefill_started"


@dataclass(frozen=True, order=True)
class PDWorkloadClass:
    """Exact classifier output used as a frozen profile lookup key.

    Bucket construction belongs to a versioned router classifier.  Keeping the
    labels explicit prevents the policy from silently inventing boundaries or
    conflating a new model, topology, transport, or load regime with old data.
    """

    model_id: str
    model_revision: str
    topology_id: str
    remote_backend: str
    prompt_bucket: str
    output_bucket: str
    decoder_load_bucket: str
    kv_bytes_bucket: str

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_revision",
            "topology_id",
            "remote_backend",
            "prompt_bucket",
            "output_bucket",
            "decoder_load_bucket",
            "kv_bytes_bucket",
        ):
            _nonempty(name, getattr(self, name))

    def canonical_dict(self) -> dict[str, str]:
        return {
            "decoder_load_bucket": self.decoder_load_bucket,
            "kv_bytes_bucket": self.kv_bytes_bucket,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "output_bucket": self.output_bucket,
            "prompt_bucket": self.prompt_bucket,
            "remote_backend": self.remote_backend,
            "topology_id": self.topology_id,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PDCalibrationProfile:
    workload: PDWorkloadClass
    evidence_level: PDEvidenceLevel
    local_samples: int
    remote_samples: int
    local_latency_p50_ms: float
    remote_latency_p50_ms: float
    local_latency_lower_bound_ms: float
    remote_latency_upper_bound_ms: float
    outputs_equivalent: bool
    remote_transfer_failures: int
    valid_from_epoch: int
    valid_through_epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.workload, PDWorkloadClass):
            raise TypeError("workload must be a PDWorkloadClass")
        if not isinstance(self.evidence_level, PDEvidenceLevel):
            raise TypeError("evidence_level must be a PDEvidenceLevel")
        for name in (
            "local_samples",
            "remote_samples",
            "remote_transfer_failures",
            "valid_from_epoch",
            "valid_through_epoch",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if self.local_samples == 0 or self.remote_samples == 0:
            raise ValueError("calibration profiles require both routes")
        if self.valid_through_epoch < self.valid_from_epoch:
            raise ValueError("valid_through_epoch precedes valid_from_epoch")
        for name in (
            "local_latency_p50_ms",
            "remote_latency_p50_ms",
            "local_latency_lower_bound_ms",
            "remote_latency_upper_bound_ms",
        ):
            _finite_nonnegative(name, getattr(self, name))
        if type(self.outputs_equivalent) is not bool:
            raise TypeError("outputs_equivalent must be bool")

    @property
    def remote_advantage_lower_bound_ms(self) -> float:
        return self.local_latency_lower_bound_ms - self.remote_latency_upper_bound_ms

    def canonical_dict(self) -> dict[str, object]:
        return {
            "evidence_level": self.evidence_level.value,
            "local_latency_lower_bound_ms": self.local_latency_lower_bound_ms,
            "local_latency_p50_ms": self.local_latency_p50_ms,
            "local_samples": self.local_samples,
            "outputs_equivalent": self.outputs_equivalent,
            "remote_latency_p50_ms": self.remote_latency_p50_ms,
            "remote_latency_upper_bound_ms": self.remote_latency_upper_bound_ms,
            "remote_samples": self.remote_samples,
            "remote_transfer_failures": self.remote_transfer_failures,
            "valid_from_epoch": self.valid_from_epoch,
            "valid_through_epoch": self.valid_through_epoch,
            "workload": self.workload.canonical_dict(),
        }

    @property
    def profile_id(self) -> str:
        payload = json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return f"pd-profile-{hashlib.sha256(payload).hexdigest()[:20]}"


@dataclass(frozen=True)
class PDPolicyConfig:
    remote_advantage_margin_ms: float = 5.0
    minimum_samples_per_route: int = 3
    require_replicated_evidence: bool = True
    remote_deadline_reserve_ms: float = 0.0

    def __post_init__(self) -> None:
        _finite_nonnegative(
            "remote_advantage_margin_ms", self.remote_advantage_margin_ms
        )
        _finite_nonnegative(
            "remote_deadline_reserve_ms", self.remote_deadline_reserve_ms
        )
        if type(self.minimum_samples_per_route) is not int or self.minimum_samples_per_route <= 0:
            raise ValueError("minimum_samples_per_route must be a positive int")
        if type(self.require_replicated_evidence) is not bool:
            raise TypeError("require_replicated_evidence must be bool")


@dataclass(frozen=True)
class PDRequestContext:
    request_id: str
    workload: PDWorkloadClass
    policy_epoch: int
    remote_backend_available: bool = True
    remaining_deadline_ms: float | None = None

    def __post_init__(self) -> None:
        _nonempty("request_id", self.request_id)
        if not isinstance(self.workload, PDWorkloadClass):
            raise TypeError("workload must be a PDWorkloadClass")
        if type(self.policy_epoch) is not int or self.policy_epoch < 0:
            raise ValueError("policy_epoch must be a non-negative int")
        if type(self.remote_backend_available) is not bool:
            raise TypeError("remote_backend_available must be bool")
        if self.remaining_deadline_ms is not None:
            _finite_nonnegative("remaining_deadline_ms", self.remaining_deadline_ms)


@dataclass(frozen=True)
class PDAdmissionDecision:
    request_id: str
    route: PDRoute
    reason: PDDecisionReason
    workload_fingerprint: str
    profile_id: str | None
    remote_advantage_lower_bound_ms: float | None
    local_latency_p50_ms: float | None
    remote_latency_p50_ms: float | None
    fallback_allowed_before_remote_start: bool


class FrozenPDAdmissionPolicy:
    """Immutable exact-key policy used by both experiment and production adapters."""

    def __init__(
        self,
        profiles: Iterable[PDCalibrationProfile],
        config: PDPolicyConfig = PDPolicyConfig(),
    ) -> None:
        if not isinstance(config, PDPolicyConfig):
            raise TypeError("config must be a PDPolicyConfig")
        indexed: dict[str, PDCalibrationProfile] = {}
        for profile in profiles:
            if not isinstance(profile, PDCalibrationProfile):
                raise TypeError("profiles must contain PDCalibrationProfile values")
            key = profile.workload.fingerprint
            if key in indexed:
                raise ValueError("duplicate workload profile")
            indexed[key] = profile
        self.config = config
        self._profiles = indexed

    @property
    def profile_count(self) -> int:
        return len(self._profiles)

    def decide(self, request: PDRequestContext) -> PDAdmissionDecision:
        if not isinstance(request, PDRequestContext):
            raise TypeError("request must be a PDRequestContext")
        fingerprint = request.workload.fingerprint
        profile = self._profiles.get(fingerprint)
        if profile is None:
            return self._local(request, None, PDDecisionReason.LOCAL_NO_PROFILE)
        if not request.remote_backend_available:
            return self._local(
                request, profile, PDDecisionReason.LOCAL_REMOTE_UNAVAILABLE
            )
        if (
            self.config.require_replicated_evidence
            and profile.evidence_level is not PDEvidenceLevel.REPLICATED
        ):
            return self._local(request, profile, PDDecisionReason.LOCAL_SCREEN_ONLY)
        if min(profile.local_samples, profile.remote_samples) < self.config.minimum_samples_per_route:
            return self._local(
                request, profile, PDDecisionReason.LOCAL_INSUFFICIENT_SAMPLES
            )
        if not profile.outputs_equivalent:
            return self._local(
                request, profile, PDDecisionReason.LOCAL_CORRECTNESS_UNPROVEN
            )
        if profile.remote_transfer_failures:
            return self._local(
                request, profile, PDDecisionReason.LOCAL_REMOTE_FAILURE
            )
        if not profile.valid_from_epoch <= request.policy_epoch <= profile.valid_through_epoch:
            return self._local(
                request, profile, PDDecisionReason.LOCAL_PROFILE_EXPIRED
            )
        if (
            profile.remote_advantage_lower_bound_ms
            < self.config.remote_advantage_margin_ms
        ):
            return self._local(
                request, profile, PDDecisionReason.LOCAL_MARGIN_NOT_MET
            )
        if request.remaining_deadline_ms is not None and (
            profile.remote_latency_upper_bound_ms
            + self.config.remote_deadline_reserve_ms
            > request.remaining_deadline_ms
        ):
            return self._local(
                request, profile, PDDecisionReason.LOCAL_DEADLINE_INFEASIBLE
            )
        return PDAdmissionDecision(
            request_id=request.request_id,
            route=PDRoute.REMOTE_PREFILL,
            reason=PDDecisionReason.REMOTE_BENEFIT_PROVEN,
            workload_fingerprint=fingerprint,
            profile_id=profile.profile_id,
            remote_advantage_lower_bound_ms=profile.remote_advantage_lower_bound_ms,
            local_latency_p50_ms=profile.local_latency_p50_ms,
            remote_latency_p50_ms=profile.remote_latency_p50_ms,
            fallback_allowed_before_remote_start=True,
        )

    @staticmethod
    def _local(
        request: PDRequestContext,
        profile: PDCalibrationProfile | None,
        reason: PDDecisionReason,
    ) -> PDAdmissionDecision:
        return PDAdmissionDecision(
            request_id=request.request_id,
            route=PDRoute.DECODER_LOCAL,
            reason=reason,
            workload_fingerprint=request.workload.fingerprint,
            profile_id=profile.profile_id if profile else None,
            remote_advantage_lower_bound_ms=(
                profile.remote_advantage_lower_bound_ms if profile else None
            ),
            local_latency_p50_ms=(profile.local_latency_p50_ms if profile else None),
            remote_latency_p50_ms=(profile.remote_latency_p50_ms if profile else None),
            fallback_allowed_before_remote_start=False,
        )


class PDRequestPhase(str, Enum):
    LOCAL_SELECTED = "local_selected"
    REMOTE_SELECTED = "remote_selected"
    REMOTE_STARTED = "remote_started"
    DECODE_STARTED = "decode_started"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class PDRequestRecord:
    context: PDRequestContext
    decision: PDAdmissionDecision
    phase: PDRequestPhase
    failure: str | None = None


class PDAdmissionLedger:
    """Thread-safe ownership ledger enforcing the pre-remote decision boundary."""

    def __init__(self, policy: FrozenPDAdmissionPolicy) -> None:
        if not isinstance(policy, FrozenPDAdmissionPolicy):
            raise TypeError("policy must be a FrozenPDAdmissionPolicy")
        self.policy = policy
        self._records: dict[str, PDRequestRecord] = {}
        self._lock = threading.Lock()

    def admit(self, context: PDRequestContext) -> PDAdmissionDecision:
        decision = self.policy.decide(context)
        phase = (
            PDRequestPhase.REMOTE_SELECTED
            if decision.route is PDRoute.REMOTE_PREFILL
            else PDRequestPhase.LOCAL_SELECTED
        )
        with self._lock:
            if context.request_id in self._records:
                raise ValueError("duplicate request_id")
            self._records[context.request_id] = PDRequestRecord(
                context, decision, phase
            )
        return decision

    def fallback_before_remote_start(
        self, request_id: str, failure: str
    ) -> PDAdmissionDecision:
        _nonempty("failure", failure)
        with self._lock:
            record = self._get(request_id)
            if record.phase is not PDRequestPhase.REMOTE_SELECTED:
                raise ValueError("fallback is allowed only before remote prefill starts")
            decision = replace(
                record.decision,
                route=PDRoute.DECODER_LOCAL,
                reason=PDDecisionReason.LOCAL_REMOTE_PRESTART_FAILURE,
                fallback_allowed_before_remote_start=False,
            )
            self._records[request_id] = replace(
                record,
                decision=decision,
                phase=PDRequestPhase.LOCAL_SELECTED,
                failure=failure,
            )
            return decision

    def mark_remote_started(self, request_id: str) -> None:
        self._transition(
            request_id, PDRequestPhase.REMOTE_SELECTED, PDRequestPhase.REMOTE_STARTED
        )

    def mark_decode_started(self, request_id: str) -> None:
        with self._lock:
            record = self._get(request_id)
            if record.phase not in {
                PDRequestPhase.LOCAL_SELECTED,
                PDRequestPhase.REMOTE_STARTED,
            }:
                raise ValueError("decode cannot start from the current phase")
            self._records[request_id] = replace(
                record, phase=PDRequestPhase.DECODE_STARTED
            )

    def complete(self, request_id: str) -> None:
        self._transition(
            request_id, PDRequestPhase.DECODE_STARTED, PDRequestPhase.COMPLETE
        )

    def fail(self, request_id: str, failure: str) -> None:
        _nonempty("failure", failure)
        with self._lock:
            record = self._get(request_id)
            if record.phase in {PDRequestPhase.COMPLETE, PDRequestPhase.FAILED}:
                raise ValueError("terminal request cannot fail again")
            self._records[request_id] = replace(
                record, phase=PDRequestPhase.FAILED, failure=failure
            )

    def record(self, request_id: str) -> PDRequestRecord:
        with self._lock:
            return self._get(request_id)

    def _transition(
        self,
        request_id: str,
        expected: PDRequestPhase,
        target: PDRequestPhase,
    ) -> None:
        with self._lock:
            record = self._get(request_id)
            if record.phase is not expected:
                raise ValueError(f"expected {expected.value}, got {record.phase.value}")
            self._records[request_id] = replace(record, phase=target)

    def _get(self, request_id: str) -> PDRequestRecord:
        record = self._records.get(request_id)
        if record is None:
            raise ValueError("unknown request_id")
        return record
