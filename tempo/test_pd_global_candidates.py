from __future__ import annotations

from dataclasses import replace

import pytest

from eval.sota_4node.test_tempo_pd_endpoint_feedback_router import _load_elastic
from tempo.pd_elastic_controller import CacheResidency
from tempo.pd_endpoint_controller import EndpointAdmissionConfig
from tempo.pd_endpoint_profile import (
    EndpointServiceProfile,
    EndpointServiceRow,
    SCHEMA_V1,
)
from tempo.pd_global_candidates import GlobalCandidateBuilder, PairCacheState
from tempo.pd_global_orchestrator import GlobalRoute
from tempo.pd_global_profile import (
    FrozenServiceProxyPolicy,
    SERVICE_PROXY_POLICY_ID,
)


def endpoint_profile(elastic_sha: str) -> EndpointServiceProfile:
    rows = tuple(sorted((
        EndpointServiceRow(
            prompt_tokens=10,
            output_tokens=64,
            cache_residency=residency,
            local_ttft_prior_ms=local_ttft,
            remote_ttft_prior_ms=remote_ttft,
            local_token_ms=int(local_ttft * 10),
            remote_prefill_token_ms=int(remote_ttft * 10),
            samples_local=3,
            samples_remote=3,
            outputs_equivalent=True,
            evidence_valid=True,
        )
        for residency, local_ttft, remote_ttft in (
            (CacheResidency.MISS, 8.0, 12.0),
            (CacheResidency.P_ONLY, 7.0, 4.0),
            (CacheResidency.D_ONLY, 2.0, 12.0),
            (CacheResidency.BOTH, 1.0, 4.0),
        )
    ), key=lambda row: row.cache_residency.value))
    return EndpointServiceProfile(
        profile_id="global-candidate-test",
        elastic_profile_fingerprint_sha256=elastic_sha,
        workload_manifest_sha256="a" * 64,
        deployment_scope="calibration_only",
        default_e2e_deadline_ms=1_000.0,
        controller=EndpointAdmissionConfig(
            local_token_ms_window=1_000,
            remote_prefill_token_ms_window=1_000,
            remote_kv_bytes_window=1_000_000,
            remote_semantic_ops_window=4,
            feedback_history=4,
            feedback_quantile=0.9,
            minimum_feedback=1,
            route_margin_ms=1.0,
            feedback_fresh_ns=1_000,
            probe_after_ns=1_000,
            denied_probe_after_ns=2_000,
        ),
        rows=rows,
        fingerprint_sha256="b" * 64,
        schema=SCHEMA_V1,
    )


def test_candidates_are_pair_route_specific_and_cache_causal(tmp_path) -> None:
    elastic = _load_elastic(tmp_path, local_ms=20.0, remote_ms=25.0)
    builder = GlobalCandidateBuilder(elastic, endpoint_profile(
        elastic.fingerprint_sha256))
    value = builder.build(
        request_id="global-candidate",
        tenant_id="latency",
        arrival_ns=10,
        deadline_ns=1_000_000_000,
        prompt_tokens=10,
        output_tokens=64,
        cache_states=(
            PairCacheState(
                0, CacheResidency.P_ONLY,
                "completed_frontend_affinity_evidence"),
            PairCacheState(
                1, CacheResidency.D_ONLY,
                "completed_frontend_affinity_evidence"),
        ),
    )
    keys = [(item.pair_index, item.route) for item in value.candidates]
    assert keys == [
        (0, GlobalRoute.LOCAL),
        (0, GlobalRoute.REMOTE),
        (1, GlobalRoute.LOCAL),
    ]
    remote = value.candidates[1]
    assert remote.cache_affinity is True
    assert remote.work.remote_semantic_ops == 1
    assert remote.work.remote_kv_bytes == 10 * 100
    local_d = value.candidates[2]
    assert local_d.cache_affinity is True
    assert local_d.predicted_ttft_ms == 2.0


def test_unknown_cache_state_is_local_only_and_uses_miss_bound(tmp_path) -> None:
    elastic = _load_elastic(tmp_path)
    builder = GlobalCandidateBuilder(elastic, endpoint_profile(
        elastic.fingerprint_sha256))
    value = builder.build(
        request_id="unknown",
        tenant_id="batch",
        arrival_ns=10,
        deadline_ns=1_000_000_000,
        prompt_tokens=10,
        output_tokens=64,
        cache_states=tuple(
            PairCacheState(
                pair, CacheResidency.UNKNOWN, "unknown_fail_closed")
            for pair in range(2)
        ),
    )
    assert all(item.route is GlobalRoute.LOCAL for item in value.candidates)
    assert all(item.predicted_ttft_ms == 8.0 for item in value.candidates)


def test_service_proxy_geometry_ceiling_supports_missing_output_row(tmp_path) -> None:
    base = _load_elastic(tmp_path)
    elastic = replace(
        base,
        rows=(replace(base.rows[0], output_tokens=32),),
    )
    builder = GlobalCandidateBuilder(
        elastic,
        endpoint_profile(elastic.fingerprint_sha256),
        allow_service_proxy=True,
    )
    value = builder.build(
        request_id="proxy-geometry",
        tenant_id="batch",
        arrival_ns=10,
        deadline_ns=1_000_000_000,
        prompt_tokens=10,
        output_tokens=32,
        cache_states=tuple(
            PairCacheState(
                pair, CacheResidency.P_ONLY,
                "completed_frontend_affinity_evidence",
            )
            for pair in range(2)
        ),
    )
    assert value.candidates
    assert all(item.predicted_ttft_ms > 0.0 for item in value.candidates)


def test_frozen_service_proxy_requires_allowlisted_geometry_and_mode(tmp_path) -> None:
    base = _load_elastic(tmp_path)
    elastic = replace(
        base,
        rows=(replace(base.rows[0], output_tokens=32),),
    )
    policy = FrozenServiceProxyPolicy(
        policy_id=SERVICE_PROXY_POLICY_ID,
        endpoint_profile_id="global-candidate-test",
        endpoint_profile_fingerprint_sha256="b" * 64,
        calibration_receipt_sha256="e" * 64,
        allowed_lookup_modes=(
            "exact", "same_residency_geometry_ceiling",
            "miss_via_prefill_only_geometry_ceiling"),
        allowed_cache_residencies=("confirmed_miss", "prefill_only"),
        allowed_remote_cache_residencies=("prefill_only",),
        allowed_geometries=((10, 32),),
        proxy_is_not_exact=True,
        numeric_rows_unchanged=True,
        performance_claim_allowed=False,
    )
    builder = GlobalCandidateBuilder(
        elastic,
        endpoint_profile(elastic.fingerprint_sha256),
        service_proxy_policy=policy,
    )
    request = builder.build(
        request_id="frozen-proxy",
        tenant_id="batch",
        arrival_ns=10,
        deadline_ns=1_000_000_000,
        prompt_tokens=10,
        output_tokens=32,
        cache_states=tuple(
            PairCacheState(pair, CacheResidency.MISS, "explicit_cache_reset_miss")
            for pair in range(2)
        ),
    )
    assert request.candidates
    assert all(item.route is GlobalRoute.LOCAL for item in request.candidates)

    disallowed = replace(
        policy, allowed_geometries=((10, 64),),
    )
    with pytest.raises(ValueError, match="endpoint profile lacks the request geometry"):
        GlobalCandidateBuilder(
            elastic,
            endpoint_profile(elastic.fingerprint_sha256),
            service_proxy_policy=disallowed,
        ).build(
            request_id="frozen-proxy-denied",
            tenant_id="batch",
            arrival_ns=10,
            deadline_ns=1_000_000_000,
            prompt_tokens=10,
            output_tokens=32,
            cache_states=tuple(
                PairCacheState(pair, CacheResidency.MISS,
                                "explicit_cache_reset_miss")
                for pair in range(2)
            ),
        )


def test_profile_identity_and_complete_pair_state_fail_closed(tmp_path) -> None:
    elastic = _load_elastic(tmp_path)
    profile = endpoint_profile(elastic.fingerprint_sha256)
    with pytest.raises(ValueError, match="different elastic identity"):
        GlobalCandidateBuilder(
            elastic,
            replace(profile, elastic_profile_fingerprint_sha256="c" * 64),
        )
    builder = GlobalCandidateBuilder(elastic, profile)
    with pytest.raises(ValueError, match="cover every pair"):
        builder.build(
            request_id="missing-pair",
            tenant_id="batch",
            arrival_ns=10,
            deadline_ns=1_000,
            prompt_tokens=10,
            output_tokens=64,
            cache_states=(PairCacheState(
                0, CacheResidency.MISS, "explicit_cache_reset_miss"),),
        )


def test_cache_state_sources_cannot_relabel_unknown_as_hit() -> None:
    with pytest.raises(ValueError, match="UNKNOWN"):
        PairCacheState(
            0, CacheResidency.P_ONLY, "unknown_fail_closed")
    with pytest.raises(ValueError, match="MISS"):
        PairCacheState(
            0, CacheResidency.P_ONLY, "explicit_cache_reset_miss")


def test_mesh_builder_materializes_unique_pxd_edges(tmp_path) -> None:
    elastic = _load_elastic(tmp_path, local_ms=20.0, remote_ms=25.0)
    builder = GlobalCandidateBuilder(
        elastic,
        endpoint_profile(elastic.fingerprint_sha256),
        mesh_enabled=True,
    )
    value = builder.build(
        request_id="mesh-candidate",
        tenant_id="latency",
        arrival_ns=10,
        deadline_ns=1_000_000_000,
        prompt_tokens=10,
        output_tokens=64,
        cache_states=(
            PairCacheState(
                0,
                CacheResidency.P_ONLY,
                "completed_frontend_affinity_evidence",
            ),
            PairCacheState(
                1,
                CacheResidency.MISS,
                "completed_frontend_affinity_evidence",
            ),
        ),
    )
    assert [item.identity_key for item in value.candidates] == [
        (0, 0, GlobalRoute.LOCAL),
        (0, 0, GlobalRoute.REMOTE),
        (1, 0, GlobalRoute.REMOTE),
        (1, 1, GlobalRoute.LOCAL),
        (0, 1, GlobalRoute.REMOTE),
        (1, 1, GlobalRoute.REMOTE),
    ]
    assert [item.edge_id for item in value.candidates] == [
        "local:d0",
        "remote:p0->d0",
        "remote:p1->d0",
        "local:d1",
        "remote:p0->d1",
        "remote:p1->d1",
    ]
    assert value.candidates[4].cache_affinity is True


def test_mesh_builder_never_promotes_unknown_source_to_remote_hit(tmp_path) -> None:
    elastic = _load_elastic(tmp_path)
    builder = GlobalCandidateBuilder(
        elastic,
        endpoint_profile(elastic.fingerprint_sha256),
        mesh_enabled=True,
    )
    value = builder.build(
        request_id="mesh-unknown",
        tenant_id="batch",
        arrival_ns=10,
        deadline_ns=1_000_000_000,
        prompt_tokens=10,
        output_tokens=64,
        cache_states=(
            PairCacheState(0, CacheResidency.UNKNOWN, "unknown_fail_closed"),
            PairCacheState(
                1,
                CacheResidency.MISS,
                "completed_frontend_affinity_evidence",
            ),
        ),
    )
    assert all(
        item.prefill_index != 0
        for item in value.candidates
        if item.route is GlobalRoute.REMOTE
    )
