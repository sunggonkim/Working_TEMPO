"""Causal pair-by-route candidate construction for TEMPO-GO."""

from __future__ import annotations

from dataclasses import dataclass

from tempo.pd_elastic_controller import CacheResidency
from tempo.pd_elastic_profile import ElasticPDProfile
from tempo.pd_endpoint_profile import EndpointServiceProfile
from tempo.pd_endpoint_controller import EndpointRoute
from tempo.pd_global_orchestrator import (
    GlobalRequest,
    GlobalRoute,
    ResourceVector,
    RouteCandidate,
)
from tempo.pd_global_profile import FrozenServiceProxyPolicy


@dataclass(frozen=True)
class PairCacheState:
    pair_index: int
    residency: CacheResidency
    source: str

    def __post_init__(self) -> None:
        if type(self.pair_index) is not int or self.pair_index < 0:
            raise ValueError("pair_index must be non-negative")
        if not isinstance(self.residency, CacheResidency):
            raise TypeError("residency must be CacheResidency")
        if self.source not in {
            "completed_frontend_affinity_evidence",
            "explicit_cache_reset_miss",
            "unknown_fail_closed",
        }:
            raise ValueError("cache-state source is not policy-eligible")
        if (
            self.source == "unknown_fail_closed"
            and self.residency is not CacheResidency.UNKNOWN
        ):
            raise ValueError("unknown source must carry UNKNOWN residency")
        if (
            self.source == "explicit_cache_reset_miss"
            and self.residency is not CacheResidency.MISS
        ):
            raise ValueError("cache reset source must carry MISS residency")


class GlobalCandidateBuilder:
    """Join frozen route priors with proven pair-local cache placement."""

    def __init__(
        self,
        elastic_profile: ElasticPDProfile,
        endpoint_profile: EndpointServiceProfile,
        *,
        pair_count: int = 2,
        allow_service_proxy: bool = False,
        service_proxy_policy: FrozenServiceProxyPolicy | None = None,
        mesh_enabled: bool = False,
    ) -> None:
        if not isinstance(elastic_profile, ElasticPDProfile):
            raise TypeError("elastic_profile must be ElasticPDProfile")
        if not isinstance(endpoint_profile, EndpointServiceProfile):
            raise TypeError("endpoint_profile must be EndpointServiceProfile")
        if type(pair_count) is not int or pair_count <= 0:
            raise ValueError("pair_count must be positive")
        if (
            endpoint_profile.elastic_profile_fingerprint_sha256
            != elastic_profile.fingerprint_sha256
        ):
            raise ValueError("candidate profiles have different elastic identity")
        self.elastic_profile = elastic_profile
        self.endpoint_profile = endpoint_profile
        self.pair_count = pair_count
        if type(allow_service_proxy) is not bool:
            raise TypeError("allow_service_proxy must be bool")
        if service_proxy_policy is not None and not isinstance(
            service_proxy_policy, FrozenServiceProxyPolicy
        ):
            raise TypeError("service_proxy_policy must be FrozenServiceProxyPolicy")
        if allow_service_proxy and service_proxy_policy is not None:
            raise ValueError(
                "legacy service proxy flag cannot be combined with frozen policy")
        self.allow_service_proxy = allow_service_proxy
        self.service_proxy_policy = service_proxy_policy
        if type(mesh_enabled) is not bool:
            raise TypeError("mesh_enabled must be bool")
        self.mesh_enabled = mesh_enabled

    @staticmethod
    def _prefill_resident(residency: CacheResidency) -> bool:
        return residency in {CacheResidency.P_ONLY, CacheResidency.BOTH}

    @staticmethod
    def _decoder_resident(residency: CacheResidency) -> bool:
        return residency in {CacheResidency.D_ONLY, CacheResidency.BOTH}

    def _remote_edge_residency(
        self,
        source: PairCacheState,
        destination: PairCacheState,
    ) -> CacheResidency | None:
        """Return the causal cache state visible to one P_i -> D_j edge.

        UNKNOWN never becomes a hit.  A destination that already owns the
        decoder prefix is represented by its local candidate instead of a
        redundant transfer.  A proven source P copy turns the cross edge into
        P_ONLY; otherwise only an explicit MISS may launch fresh prefill.
        """

        if (
            source.residency is CacheResidency.UNKNOWN
            or destination.residency is CacheResidency.UNKNOWN
            or self._decoder_resident(destination.residency)
        ):
            return None
        if self._prefill_resident(source.residency):
            return CacheResidency.P_ONLY
        if source.residency is CacheResidency.MISS:
            return CacheResidency.MISS
        return None

    def _proxy_enabled_for(
        self, *, prompt_tokens: int, output_tokens: int,
        residency: CacheResidency,
    ) -> bool:
        if self.service_proxy_policy is not None:
            return (
                self.service_proxy_policy.allows_geometry(
                    prompt_tokens, output_tokens)
                and self.service_proxy_policy.allows_residency(residency.value)
            )
        return self.allow_service_proxy

    def _service_row(
        self, *, prompt_tokens: int, output_tokens: int,
        residency: CacheResidency, route: GlobalRoute,
    ):
        cold_unknown = residency is CacheResidency.UNKNOWN
        try:
            return self.endpoint_profile.exact_row(
                prompt_tokens,
                output_tokens,
                residency,
                cold_unknown_as_miss=cold_unknown,
            )
        except ValueError:
            if not self._proxy_enabled_for(
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                residency=residency,
            ):
                raise
            proxy = self.endpoint_profile.external_credit_proxy(
                prompt_tokens,
                output_tokens,
                residency,
                route=(
                    EndpointRoute.LOCAL
                    if route is GlobalRoute.LOCAL else EndpointRoute.REMOTE),
                cold_unknown_as_miss=cold_unknown,
            )
            if (
                self.service_proxy_policy is not None
                and not self.service_proxy_policy.allows_lookup_mode(
                    proxy.lookup_mode)
            ):
                raise ValueError(
                    "endpoint service proxy lookup mode is not frozen-policy allowlisted"
                )
            return proxy.row

    def _minimum_ttft(
        self, prompt_tokens: int, output_tokens: int, *, route: GlobalRoute
    ) -> float:
        rows = [
            row for row in self.endpoint_profile.rows
            if row.prompt_tokens == prompt_tokens
            and row.output_tokens == output_tokens
        ]
        if not rows and (
            self.allow_service_proxy
            or (
                self.service_proxy_policy is not None
                and self.service_proxy_policy.allows_geometry(
                    prompt_tokens, output_tokens)
            )
        ):
            rows = [
                row for row in self.endpoint_profile.rows
                if row.prompt_tokens == prompt_tokens
                and row.output_tokens >= output_tokens
            ]
        if not rows:
            raise ValueError("endpoint profile lacks the request geometry")
        return min(
            row.local_ttft_prior_ms
            if route is GlobalRoute.LOCAL
            else row.remote_ttft_prior_ms
            for row in rows
        )

    def build(
        self,
        *,
        request_id: str,
        tenant_id: str,
        arrival_ns: int,
        deadline_ns: int,
        prompt_tokens: int,
        output_tokens: int,
        cache_states: tuple[PairCacheState, ...],
        cache_group_key: str | None = None,
    ) -> GlobalRequest:
        if tuple(item.pair_index for item in cache_states) != tuple(
            range(self.pair_count)
        ):
            raise ValueError("cache states must cover every pair in order")
        row = self.elastic_profile.exact_row(prompt_tokens, output_tokens)
        if row is None:
            raise ValueError("elastic profile lacks the exact request geometry")
        minimum_local_ttft = self._minimum_ttft(
            prompt_tokens, output_tokens, route=GlobalRoute.LOCAL)
        minimum_remote_ttft = self._minimum_ttft(
            prompt_tokens, output_tokens, route=GlobalRoute.REMOTE)
        candidates = []
        for state in cache_states:
            local_service = self._service_row(
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                residency=state.residency,
                route=GlobalRoute.LOCAL,
            )
            local_e2e = (
                row.local_upper_bound_ms
                + max(
                    0.0,
                    local_service.local_ttft_prior_ms - minimum_local_ttft,
                )
            )
            candidates.append(RouteCandidate(
                pair_index=state.pair_index,
                route=GlobalRoute.LOCAL,
                work=ResourceVector(
                    decode_tokens=output_tokens,
                    active_sequences=1,
                    endpoint_requests=1,
                    local_prefill_token_ms=local_service.local_token_ms,
                ),
                predicted_e2e_ms=local_e2e,
                predicted_ttft_ms=local_service.local_ttft_prior_ms,
                uncertainty_ms=row.uncertainty_ms,
                cache_affinity=state.residency in {
                    CacheResidency.D_ONLY, CacheResidency.BOTH,
                },
                prefill_index=state.pair_index,
                decoder_index=state.pair_index,
            ))
        remote_edges = (
            tuple(
                (source, destination)
                for destination in cache_states
                for source in cache_states
            )
            if self.mesh_enabled
            else tuple((state, state) for state in cache_states)
        )
        for source, destination in remote_edges:
            edge_residency = self._remote_edge_residency(
                source, destination)
            remote_allowed = (
                row.evidence_safe
                and edge_residency in {
                    CacheResidency.MISS, CacheResidency.P_ONLY,
                }
                and (
                    self.service_proxy_policy is None
                    or self.service_proxy_policy.allows_remote_residency(
                        edge_residency.value)
                )
            )
            if remote_allowed and edge_residency is not None:
                remote_service = self._service_row(
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    residency=edge_residency,
                    route=GlobalRoute.REMOTE,
                )
                remote_e2e = (
                    row.remote_upper_bound_ms
                    + max(
                        0.0,
                        remote_service.remote_ttft_prior_ms
                        - minimum_remote_ttft,
                    )
                )
                candidates.append(RouteCandidate(
                    pair_index=destination.pair_index,
                    route=GlobalRoute.REMOTE,
                    work=ResourceVector(
                        decode_tokens=output_tokens,
                        active_sequences=1,
                        endpoint_requests=1,
                        remote_prefill_token_ms=(
                            remote_service.remote_prefill_token_ms),
                        remote_kv_bytes=row.remote_kv_bytes,
                        remote_semantic_ops=1,
                    ),
                    predicted_e2e_ms=remote_e2e,
                    predicted_ttft_ms=remote_service.remote_ttft_prior_ms,
                    uncertainty_ms=row.uncertainty_ms,
                    cache_affinity=(
                        edge_residency is CacheResidency.P_ONLY),
                    prefill_index=source.pair_index,
                    decoder_index=destination.pair_index,
                ))
        candidates.sort(key=lambda candidate: (
            int(candidate.decoder_index),
            0 if candidate.route is GlobalRoute.LOCAL else 1,
            int(candidate.prefill_index),
        ))
        return GlobalRequest(
            request_id=request_id,
            tenant_id=tenant_id,
            arrival_ns=arrival_ns,
            deadline_ns=deadline_ns,
            candidates=tuple(candidates),
            cache_group_key=cache_group_key,
        )


__all__ = ["GlobalCandidateBuilder", "PairCacheState"]
