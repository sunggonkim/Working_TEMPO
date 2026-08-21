"""Canonical TEMPO Elastic-PD controller and cache-evidence catalog."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import threading
import time

from tempo.pd_elastic_controller_v443 import (
    CacheResidency, ElasticConfig, ElasticDecision, ElasticEstimate,
    ElasticPDController as _VersionedController, ElasticPhase, ElasticRegime,
    ElasticRequest, ElasticRoute,
)

POLICY_ID = "tempo-elastic-pd-weighted-dual-remote-credit-canonical"
CRITICAL_OUTPUT_TOKENS = 256
CRITICAL_PROMPT_TOKENS = 2048
HEADROOM_OUTPUT_TOKENS = 128
HEADROOM_PROMPT_TOKENS = 2048
HEADROOM_SHORT_PROMPT_TOKENS = 512
SHORT_INTRINSIC_MAX_OUTPUT_TOKENS = 16
SHORT_INTRINSIC_MAX_PROMPT_TOKENS = 512
SHORT_INTRINSIC_MIN_ADVANTAGE_MS = 50.0
SHORT_INTRINSIC_MAX_ADVANTAGE_MS = 250.0
COLD_MEASURED_ENV = "TEMPO_PD_BENCHMARK_COLD_MEASURED"
COLD_HEADROOM_MIN_OUTPUT_TOKENS = 256
REMOTE_HEADROOM_KV_BUDGET_ENV = "TEMPO_PD_REMOTE_HEADROOM_KV_BUDGET_BYTES"


@dataclass(frozen=True)
class CacheResidencyEvent:
    namespace: str
    completed_ns: int
    prefill_resident: bool
    decode_resident: bool
    actual_kv_bytes: int | None = None

    @property
    def residency(self) -> CacheResidency:
        if self.prefill_resident and self.decode_resident:
            return CacheResidency.BOTH
        if self.prefill_resident:
            return CacheResidency.P_ONLY
        if self.decode_resident:
            return CacheResidency.D_ONLY
        return CacheResidency.MISS


class CacheResidencyCatalog:
    """Fail-closed catalog populated only by completed backend evidence."""

    def __init__(self) -> None:
        self._events: dict[str, CacheResidencyEvent] = {}
        self._lock = threading.Lock()

    @staticmethod
    def namespace(*, arm: str, prompt_tokens: int, output_tokens: int,
                  item: str) -> str:
        if not isinstance(arm, str) or not arm.strip():
            raise ValueError("arm must be nonempty")
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if type(output_tokens) is not int or output_tokens < 2:
            raise ValueError("output_tokens must be at least two")
        if not isinstance(item, str) or not item.strip():
            raise ValueError("item must be nonempty")
        # KV residency is a property of the exact prompt namespace, not of
        # the requested decode length.  Seed requests intentionally decode
        # only two tokens while their hit probes and measured requests use the
        # final output geometry, so including output_tokens would split one
        # physical vLLM/LMCache cache into contradictory logical catalogs.
        return f"arm={arm}:prompt={prompt_tokens}:item={item}"

    def classify(self, namespace: str) -> CacheResidency:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("cache namespace must be nonempty")
        with self._lock:
            event = self._events.get(namespace)
        return event.residency if event is not None else CacheResidency.UNKNOWN

    def record_completion(self, namespace: str, *, prefill_resident: bool,
                          decode_resident: bool, actual_kv_bytes: int | None = None,
                          completed_ns: int | None = None) -> CacheResidencyEvent:
        if type(prefill_resident) is not bool or type(decode_resident) is not bool:
            raise TypeError("residency flags must be bool")
        if actual_kv_bytes is not None and (
            type(actual_kv_bytes) is not int or actual_kv_bytes < 0
        ):
            raise ValueError("actual_kv_bytes must be nonnegative")
        event = CacheResidencyEvent(
            namespace=namespace,
            completed_ns=time.perf_counter_ns() if completed_ns is None else completed_ns,
            prefill_resident=prefill_resident,
            decode_resident=decode_resident,
            actual_kv_bytes=actual_kv_bytes,
        )
        if type(event.completed_ns) is not int or event.completed_ns < 0:
            raise ValueError("completed_ns must be nonnegative")
        with self._lock:
            prior = self._events.get(namespace)
            if prior is not None:
                if event.completed_ns < prior.completed_ns:
                    raise ValueError("cache residency evidence is stale")
                if (
                    prior.prefill_resident and not event.prefill_resident
                    or prior.decode_resident and not event.decode_resident
                ):
                    raise ValueError("cache residency evidence regressed")
                if (
                    prior.prefill_resident == event.prefill_resident
                    and prior.decode_resident == event.decode_resident
                    and prior.actual_kv_bytes == event.actual_kv_bytes
                ):
                    return prior
            self._events[namespace] = event
        return event

    def event(self, namespace: str) -> CacheResidencyEvent | None:
        with self._lock:
            return self._events.get(namespace)


class ElasticPDController(_VersionedController):
    """Request-start route commit with geometry-aware contention control."""

    def __init__(self, config: ElasticConfig) -> None:
        self.profile_remote_kv_budget_bytes = config.remote_kv_budget_bytes
        raw_remote_kv_budget = os.environ.get(
            "TEMPO_PD_REMOTE_KV_BUDGET_BYTES",
            str(self.profile_remote_kv_budget_bytes),
        )
        try:
            self.effective_remote_kv_budget_bytes = int(raw_remote_kv_budget)
        except ValueError as exc:
            raise ValueError(
                "TEMPO_PD_REMOTE_KV_BUDGET_BYTES must be an integer"
            ) from exc
        if not (
            self.profile_remote_kv_budget_bytes
            <= self.effective_remote_kv_budget_bytes
            <= 2 * self.profile_remote_kv_budget_bytes
        ):
            raise ValueError(
                "TEMPO_PD_REMOTE_KV_BUDGET_BYTES must be between the profile "
                "remote KV budget and twice that budget"
            )
        super().__init__(replace(
            config,
            remote_kv_budget_bytes=self.effective_remote_kv_budget_bytes,
        ))
        raw_headroom_kv_budget = os.environ.get(
            REMOTE_HEADROOM_KV_BUDGET_ENV,
            str(self.effective_remote_kv_budget_bytes),
        )
        try:
            self.remote_headroom_kv_budget_bytes = int(
                raw_headroom_kv_budget)
        except ValueError as exc:
            raise ValueError(
                f"{REMOTE_HEADROOM_KV_BUDGET_ENV} must be an integer"
            ) from exc
        if not (
            self.effective_remote_kv_budget_bytes
            <= self.remote_headroom_kv_budget_bytes
            <= 2 * self.profile_remote_kv_budget_bytes
        ):
            raise ValueError(
                f"{REMOTE_HEADROOM_KV_BUDGET_ENV} must be between the "
                "effective remote KV budget and twice the profile budget")
        self._request_geometry: dict[str, tuple[int, int]] = {}
        raw_externality_budget = os.environ.get(
            "TEMPO_PD_EXTERNALITY_SPILL_BUDGET_MS",
            str(config.spill_regression_budget_ms),
        )
        try:
            self.externality_spill_budget_ms = float(raw_externality_budget)
        except ValueError as exc:
            raise ValueError(
                "TEMPO_PD_EXTERNALITY_SPILL_BUDGET_MS must be a finite number"
            ) from exc
        if not (
            config.spill_regression_budget_ms
            <= self.externality_spill_budget_ms
            <= 250.0
        ):
            raise ValueError(
                "TEMPO_PD_EXTERNALITY_SPILL_BUDGET_MS must be between "
                "the profile spill budget and 250ms"
            )
        raw_short_advantage = os.environ.get(
            "TEMPO_PD_SHORT_REMOTE_MIN_ADVANTAGE_MS",
            str(SHORT_INTRINSIC_MAX_ADVANTAGE_MS),
        )
        try:
            self.short_remote_min_advantage_ms = float(raw_short_advantage)
        except ValueError as exc:
            raise ValueError(
                "TEMPO_PD_SHORT_REMOTE_MIN_ADVANTAGE_MS must be a finite number"
            ) from exc
        if not (
            SHORT_INTRINSIC_MIN_ADVANTAGE_MS
            <= self.short_remote_min_advantage_ms
            <= SHORT_INTRINSIC_MAX_ADVANTAGE_MS
        ):
            raise ValueError(
                "TEMPO_PD_SHORT_REMOTE_MIN_ADVANTAGE_MS must be between "
                "50ms and 250ms"
            )
        raw_medium_output = os.environ.get(
            "TEMPO_PD_HEADROOM_MEDIUM_MIN_OUTPUT_TOKENS", "128")
        try:
            self.headroom_medium_min_output_tokens = int(raw_medium_output)
        except ValueError as exc:
            raise ValueError(
                "TEMPO_PD_HEADROOM_MEDIUM_MIN_OUTPUT_TOKENS must be 128 or 256"
            ) from exc
        if self.headroom_medium_min_output_tokens not in (128, 256):
            raise ValueError(
                "TEMPO_PD_HEADROOM_MEDIUM_MIN_OUTPUT_TOKENS must be 128 or 256"
            )

        raw_remote_requests = os.environ.get(
            "TEMPO_PD_REMOTE_REQUEST_BUDGET", "8")
        raw_headroom_requests = os.environ.get(
            "TEMPO_PD_REMOTE_HEADROOM_REQUEST_BUDGET",
            raw_remote_requests,
        )
        try:
            self.remote_request_budget = int(raw_remote_requests)
            self.remote_headroom_request_budget = int(raw_headroom_requests)
        except ValueError as exc:
            raise ValueError(
                "TEMPO remote request budgets must be integers"
            ) from exc
        if not 1 <= self.remote_request_budget <= 64:
            raise ValueError(
                "TEMPO_PD_REMOTE_REQUEST_BUDGET must be between 1 and 64")
        if not (
            1 <= self.remote_headroom_request_budget
            <= self.remote_request_budget
        ):
            raise ValueError(
                "TEMPO_PD_REMOTE_HEADROOM_REQUEST_BUDGET must be between "
                "1 and TEMPO_PD_REMOTE_REQUEST_BUDGET"
            )
        # Byte ownership remains in the versioned controller. These ledgers
        # add an independent request-count dimension and distinguish
        # system-level headroom deflections from intrinsic/bounded remote work.
        self._remote_headroom_class: set[str] = set()
        self._request_credit_evidence: dict[str, dict[str, int | bool]] = {}
        raw_cold_measured = os.environ.get(COLD_MEASURED_ENV, "0")
        if raw_cold_measured not in ("0", "1"):
            raise ValueError(
                f"{COLD_MEASURED_ENV} must be 0 or 1")
        self.cold_measured = raw_cold_measured == "1"

    def request_credit_evidence(self, request_id: str) -> dict[str, int | bool]:
        with self._lock:
            evidence = self._request_credit_evidence.get(request_id)
            return dict(evidence) if evidence is not None else {}


    def register_request_geometry(
        self, request_id: str, prompt_tokens: int, output_tokens: int,
    ) -> None:
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be nonempty")
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if type(output_tokens) is not int or output_tokens < 2:
            raise ValueError("output_tokens must be at least two")
        geometry = (prompt_tokens, output_tokens)
        with self._lock:
            prior = self._request_geometry.get(request_id)
            if prior is not None and prior != geometry:
                raise ValueError("request geometry changed")
            self._request_geometry[request_id] = geometry

    def _evaluate(self, request: ElasticRequest, estimate: ElasticEstimate, *, attempt: int) -> ElasticDecision:
        original_residency = request.cache_residency
        cold_unknown = (
            self.cold_measured
            and original_residency is CacheResidency.UNKNOWN
        )
        original_remote_score = estimate.remote_upper_bound_ms + estimate.uncertainty_ms
        remote_requests_used = len(self._remote_owned)
        remote_headroom_requests_used = sum(
            request_id in self._remote_owned
            for request_id in self._remote_headroom_class
        )
        remote_request_credit = (
            remote_requests_used < self.remote_request_budget)
        remote_headroom_request_credit = (
            remote_headroom_requests_used
            < self.remote_headroom_request_budget)
        prompt_tokens, output_tokens = self._request_geometry.get(
            request.request_id, (None, None))
        critical_output = (
            prompt_tokens is not None
            and output_tokens is not None
            and prompt_tokens >= CRITICAL_PROMPT_TOKENS
            and output_tokens >= CRITICAL_OUTPUT_TOKENS
        )
        base_local_budget = self.config.local_compute_budget_us
        effective_local_budget = (
            max(base_local_budget, 2 * request.local_compute_cost_us)
            if critical_output else base_local_budget
        )
        remote_spill_safe = (
            original_residency is CacheResidency.P_ONLY
            and estimate.remote_backend_available and estimate.remote_evidence_valid
            and estimate.remote_upper_bound_ms <= estimate.local_upper_bound_ms
            + self.config.spill_regression_budget_ms
        )
        cold_headroom_candidate = (
            cold_unknown
            and output_tokens is not None
            and output_tokens >= COLD_HEADROOM_MIN_OUTPUT_TOKENS
        )
        externality_spill_safe = (
            (
                original_residency is CacheResidency.P_ONLY
                or cold_headroom_candidate
            )
            and estimate.remote_backend_available and estimate.remote_evidence_valid
            and estimate.remote_upper_bound_ms <= estimate.local_upper_bound_ms
            + self.externality_spill_budget_ms
        )
        headroom_eligible = (
            prompt_tokens is not None
            and output_tokens is not None
            and self._regime in {
                ElasticRegime.DEFLECT_ACTIVE,
                ElasticRegime.REMOTE_STABLE,
            }
            and prompt_tokens <= HEADROOM_PROMPT_TOKENS
            and output_tokens >= HEADROOM_OUTPUT_TOKENS
            and (
                not cold_unknown
                or output_tokens >= COLD_HEADROOM_MIN_OUTPUT_TOKENS
            )
            and (
                prompt_tokens <= HEADROOM_SHORT_PROMPT_TOKENS
                or output_tokens >= self.headroom_medium_min_output_tokens
            )
            and externality_spill_safe
        )
        headroom_deflection = (
            headroom_eligible
            and remote_request_credit
            and remote_headroom_request_credit
        )
        effective_request = request
        if original_residency is CacheResidency.UNKNOWN and not cold_unknown:
            effective_request = replace(request, cache_residency=CacheResidency.MISS)
        benefit = estimate.local_upper_bound_ms - estimate.remote_upper_bound_ms
        short_intrinsic_remote = (
            original_residency is CacheResidency.P_ONLY
            and estimate.remote_backend_available and estimate.remote_evidence_valid
            and prompt_tokens is not None
            and output_tokens is not None
            and prompt_tokens <= SHORT_INTRINSIC_MAX_PROMPT_TOKENS
            and output_tokens <= SHORT_INTRINSIC_MAX_OUTPUT_TOKENS
            and benefit >= self.short_remote_min_advantage_ms
            and benefit >= 0.10 * estimate.local_upper_bound_ms
        )
        # uncertainty_ms describes uncertainty in the measured route gap. It
        # cannot be added to both arms and canceled: an intrinsic remote route
        # needs a positive lower-confidence advantage.
        remote_allowed = (
            estimate.remote_backend_available
            and estimate.remote_evidence_valid
            and benefit >= self.config.route_margin_ms + estimate.uncertainty_ms
        )
        if short_intrinsic_remote:
            remote_allowed = True
        if headroom_deflection:
            # This is a system-level contention decision, not a claim that the
            # standalone remote latency predictor beats local. The bounded
            # regression check above keeps the alternate path safe.
            remote_allowed = True
        if original_residency is CacheResidency.UNKNOWN and not cold_unknown:
            remote_allowed = False
        if original_residency in {CacheResidency.D_ONLY, CacheResidency.BOTH}:
            remote_allowed = False
        if headroom_deflection and cold_unknown:
            effective_request = replace(
                request, cache_residency=CacheResidency.P_ONLY)
        if not remote_request_credit:
            estimate = replace(
                estimate,
                remote_backend_available=False,
                remote_evidence_valid=False,
            )
            remote_allowed = False
        if not remote_allowed:
            if original_residency is CacheResidency.P_ONLY:
                # Prefer local for a near-tie P-only request, but retain the
                # proven remote path as a bounded spill target when local
                # prefill credit is exhausted. D_ONLY expresses local-first
                # preference to the base ledger without falsifying evidence.
                effective_request = replace(
                    request, cache_residency=CacheResidency.D_ONLY
                )
            else:
                # UNKNOWN/MISS without an intrinsic remote win, plus D_ONLY
                # and BOTH, must not spill to a cache-incompatible path.
                estimate = replace(
                    estimate,
                    remote_upper_bound_ms=(
                        estimate.local_upper_bound_ms
                        + self.config.spill_regression_budget_ms + 1.0
                    ),
                    remote_evidence_valid=False,
                )
        singleton_borrow = (
            effective_request.local_compute_cost_us
            > base_local_budget
            and not self._local_owned
            and not remote_spill_safe
        )
        if singleton_borrow:
            effective_request = replace(
                effective_request,
                local_compute_cost_us=base_local_budget,
            )
        base_credit_available = (
            sum(self._local_owned.values())
            + effective_request.local_compute_cost_us
            <= base_local_budget
        )
        original_config = self.config
        original_regime = self._regime
        if effective_local_budget != base_local_budget:
            self.config = replace(
                self.config,
                local_compute_budget_us=effective_local_budget,
            )
        if headroom_deflection:
            self.config = replace(
                self.config,
                remote_kv_budget_bytes=self.remote_headroom_kv_budget_bytes,
                spill_regression_budget_ms=self.externality_spill_budget_ms,
            )
        if headroom_deflection:
            # Consume bounded system-level remote headroom for this request.
            # The lock is held; config and regime are restored before return.
            self._regime = ElasticRegime.REMOTE_STABLE
        try:
            decision = super()._evaluate(
                effective_request, estimate, attempt=attempt)
        finally:
            self._regime = original_regime
            self.config = original_config
        headroom_credit_used = (
            headroom_deflection
            and decision.route is ElasticRoute.REMOTE
        )
        short_remote_used = (
            short_intrinsic_remote
            and decision.route is ElasticRoute.REMOTE
        )
        if headroom_credit_used:
            self._remote_headroom_class.add(request.request_id)
        self._request_credit_evidence[request.request_id] = {
            "remote_requests_used_before": remote_requests_used,
            "remote_request_budget": self.remote_request_budget,
            "remote_request_credit_available": remote_request_credit,
            "remote_headroom_requests_used_before": (
                remote_headroom_requests_used),
            "remote_headroom_request_budget": (
                self.remote_headroom_request_budget),
            "remote_headroom_request_credit_available": (
                remote_headroom_request_credit),
            "remote_headroom_eligible": headroom_eligible,
            "remote_headroom_credit_consumed": headroom_credit_used,
            "cold_unknown_remote_candidate": cold_unknown,
            "cold_unknown_remote_admitted": (
                cold_unknown and decision.route is ElasticRoute.REMOTE),
            "cold_high_load_headroom_candidate": (
                cold_unknown and headroom_eligible),
            "cold_high_load_headroom_consumed": (
                cold_unknown and headroom_credit_used),
        }
        critical_credit_used = (
            critical_output
            and not base_credit_available
            and decision.route is ElasticRoute.LOCAL
        )
        return replace(
            decision,
            reason=(
                "cold_high_load_remote_headroom_deflection"
                if cold_unknown and headroom_credit_used
                else "cold_unknown_remote_evidence"
                if cold_unknown and decision.route is ElasticRoute.REMOTE
                else "short_intrinsic_remote_evidence"
                if short_remote_used
                else "high_load_remote_headroom_deflection"
                if headroom_credit_used
                else "critical_output_expanded_local_credit"
                if critical_credit_used
                else "oversized_singleton_local_borrow"
                if singleton_borrow and decision.route is ElasticRoute.LOCAL
                else "remote_request_credit_exhausted_to_local"
                if not remote_request_credit
                and decision.route is ElasticRoute.LOCAL
                else "remote_headroom_credit_exhausted_to_local"
                if headroom_eligible and not remote_headroom_request_credit
                and decision.route is ElasticRoute.LOCAL
                else decision.reason
            ),
            cache_residency=original_residency,
            regime=original_regime,
            remote_score_ms=original_remote_score,
            policy_id=POLICY_ID,
        )


__all__ = [
    "CacheResidency", "CacheResidencyCatalog", "CacheResidencyEvent",
    "COLD_MEASURED_ENV",
    "COLD_HEADROOM_MIN_OUTPUT_TOKENS",
    "REMOTE_HEADROOM_KV_BUDGET_ENV",
    "CRITICAL_OUTPUT_TOKENS", "CRITICAL_PROMPT_TOKENS",
    "ElasticConfig", "ElasticDecision", "ElasticEstimate", "ElasticPDController",
    "ElasticPhase", "ElasticRegime", "ElasticRequest", "ElasticRoute",
    "HEADROOM_OUTPUT_TOKENS", "HEADROOM_PROMPT_TOKENS", "POLICY_ID",
    "SHORT_INTRINSIC_MAX_OUTPUT_TOKENS",
    "SHORT_INTRINSIC_MAX_PROMPT_TOKENS",
    "SHORT_INTRINSIC_MIN_ADVANTAGE_MS",
]
