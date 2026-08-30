"""Invariants for Candidate M's observed-pressure pair spill."""

from __future__ import annotations

from tempo.pd_global_orchestrator import (
    GlobalRoute,
    PairCapacity,
    ResourceVector,
    TenantPolicy,
)


def test_candidate_m_profile_mechanism_is_observed_pressure_only() -> None:
    # This test is intentionally source-level: the native campaign contract
    # binds the same existing method and changes only its frozen profile knob.
    from tempo.pd_global_orchestrator import GlobalOrchestratorConfig

    config = GlobalOrchestratorConfig(
        capacities=(
            PairCapacity(0, ResourceVector(
                decode_tokens=100, active_sequences=10,
                endpoint_requests=10, local_prefill_token_ms=100,
                remote_prefill_token_ms=100, remote_kv_bytes=1_000,
                remote_semantic_ops=2)),
            PairCapacity(1, ResourceVector(
                decode_tokens=100, active_sequences=10,
                endpoint_requests=10, local_prefill_token_ms=100,
                remote_prefill_token_ms=100, remote_kv_bytes=1_000,
                remote_semantic_ops=2)),
        ),
        tenants=(
            TenantPolicy("latency", 2.0, admission_priority=800,
                         protected_capacity_fraction=0.2),
            TenantPolicy("background", 0.5, admission_priority=0,
                         pair_spread_limit=1),
        ),
        telemetry_fresh_ns=1_000_000_000,
        queue_capacity=32,
        maximum_active_pairs=2,
        business_clean_pair_pressure_fraction=0.5,
    )
    assert config.business_clean_pair_pressure_fraction == 0.5
    assert config.priority_service_lane_mode == "disabled"
    assert config.service_feasibility_mode == "disabled"
    assert config.shared_fabric_control_mode == "disabled"
    # The candidate does not add an oracle route or a new transport path.
    assert GlobalRoute.LOCAL.value == "decoder_local_chunked_prefill"
    assert GlobalRoute.REMOTE.value == "official_lmcache_remote_prefill"
