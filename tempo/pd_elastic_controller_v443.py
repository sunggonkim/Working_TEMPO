"""Corrected add-only revision of the Elastic-PD ingress controller.

This revision makes recovery probes explicit: an arbitrary recovery-window
request cannot consume the single probe unless it has P-only cache affinity or
the conservative remote estimate wins by the configured margin.  It also
guarantees that a remote backend failure before execution tries the safe local
path directly instead of reapplying stale P-only affinity.
"""

from dataclasses import replace

from tempo.pd_elastic_controller_v442 import (
    CacheResidency,
    ElasticConfig,
    ElasticDecision,
    ElasticEstimate,
    ElasticPDController as _BaseElasticPDController,
    ElasticPhase,
    ElasticRegime,
    ElasticRequest,
    ElasticRoute,
)


POLICY_ID = "tempo-elastic-pd-ingress-dual-credit-443"


class ElasticPDController(_BaseElasticPDController):
    def _evaluate(
        self, request: ElasticRequest, estimate: ElasticEstimate, *, attempt: int
    ) -> ElasticDecision:
        effective = estimate
        if self._regime is ElasticRegime.RECOVERY_PROBE:
            remote_advantage = (
                estimate.local_upper_bound_ms - estimate.remote_upper_bound_ms
            )
            eligible = (
                request.cache_residency is CacheResidency.P_ONLY
                or remote_advantage >= self.config.route_margin_ms
            )
            if not eligible:
                effective = replace(estimate, remote_evidence_valid=False)
        decision = super()._evaluate(request, effective, attempt=attempt)
        return replace(decision, policy_id=POLICY_ID)

    def fallback_remote_before_start(
        self, request_id: str, estimate: ElasticEstimate
    ) -> ElasticDecision:
        if not isinstance(estimate, ElasticEstimate):
            raise TypeError("estimate must be ElasticEstimate")
        with self._lock:
            entry = self._get(request_id)
            if entry.phase is not ElasticPhase.REMOTE_RESERVED:
                raise ValueError("remote fallback is allowed only before start")
            self._release_remote(request_id)
            if entry.decision.remote_probe:
                self._remote_probe_request = None
                self._regime = ElasticRegime.DEFLECT_ACTIVE
            local_only = replace(
                estimate,
                remote_backend_available=False,
                remote_evidence_valid=False,
            )
            local_request = replace(
                entry.request, cache_residency=CacheResidency.D_ONLY
            )
            decision = self._evaluate(
                local_request,
                local_only,
                attempt=entry.decision.attempt + 1,
            )
            decision = replace(
                decision,
                cache_residency=entry.request.cache_residency,
                reason=("remote_prestart_failure_to_local"
                        if decision.route is ElasticRoute.LOCAL
                        else "remote_prestart_failure_queued"),
            )
            self._entries[request_id] = replace(
                entry,
                estimate=local_only,
                decision=decision,
                phase=decision.phase,
            )
            return decision


__all__ = [
    "CacheResidency",
    "ElasticConfig",
    "ElasticDecision",
    "ElasticEstimate",
    "ElasticPDController",
    "ElasticPhase",
    "ElasticRegime",
    "ElasticRequest",
    "ElasticRoute",
    "POLICY_ID",
]
