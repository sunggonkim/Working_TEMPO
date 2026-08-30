from __future__ import annotations

import copy
import os
from unittest.mock import patch

import pytest

from eval.sota_4node import analyze_tempo_pd_c4_semantic_epoch_screen as semantic
from eval.sota_4node import build_tempo_pd_c4_semantic_integration_run_contract as contract
from eval.sota_4node import run_tempo_pd_c4_adaptive_screen_client as client
from eval.sota_4node import run_tempo_pd_c4_fixed_phase_client as c4
from eval.sota_4node import vllm_lmcache_pd_c4_adaptive_screen_node as node
from eval.sota_4node.test_analyze_tempo_pd_c4_semantic_epoch_screen import (
    _decision as semantic_decision,
)
from eval.sota_4node.test_run_tempo_pd_c4_adaptive_screen_client import (
    _decision as adaptive_decision,
)
from tempo.pd_contention_workload import CacheState, ForegroundArm, Tenant


def _tempo_metadata():
    return {
        "tenant": Tenant.FOREGROUND.value,
        "arm": ForegroundArm.TEMPO.value,
        "prompt_tokens": 512,
        "output_tokens": 16,
        "cache_state": CacheState.MISS.value,
        "terminal_item": 3,
    }


def _semantic_contract():
    return {
        "semantic_credit_contract": dict(
            contract.profile_builder.SEMANTIC_ROUTING_POLICY),
        "endpoint_service_profile": {
            "fingerprint_sha256": "a" * 64,
        },
    }


def test_runtime_contract_binding_is_mutually_exclusive():
    with patch.dict(os.environ, {
        client.SEMANTIC_RUN_CONTRACT_ENV: "/result/semantic.json",
        client.SEMANTIC_RUN_CONTRACT_SHA_ENV: "a" * 64,
    }, clear=True):
        path, digest, is_semantic = client._runtime_contract_binding()
    assert path == "/result/semantic.json"
    assert digest == "a" * 64
    assert is_semantic is True

    with patch.dict(os.environ, {
        client.RUN_CONTRACT_ENV: "/result/adaptive.json",
        client.RUN_CONTRACT_SHA_ENV: "b" * 64,
        client.SEMANTIC_RUN_CONTRACT_ENV: "/result/semantic.json",
        client.SEMANTIC_RUN_CONTRACT_SHA_ENV: "a" * 64,
    }, clear=True):
        with pytest.raises(ValueError, match="exactly one"):
            client._runtime_contract_binding()


def test_semantic_tempo_decision_uses_the_profile_bound_epoch_latch():
    metadata = _tempo_metadata()
    decision = adaptive_decision(
        metadata, route=c4._REMOTE_ROUTE, block_arm=ForegroundArm.TEMPO)
    decision.update(semantic_decision())
    decision["endpoint_feedback_accepted"] = True
    route = client._validate_dynamic_decision(
        decision,
        metadata,
        block_arm=ForegroundArm.TEMPO,
        request_id="semantic-tempo-request",
        semantic_contract=_semantic_contract(),
    )
    assert route == c4._REMOTE_ROUTE

    drifted = copy.deepcopy(decision)
    drifted["semantic_epoch_confirmation_requests"] = 3
    with pytest.raises(ValueError, match="confirmation"):
        client._validate_dynamic_decision(
            drifted,
            metadata,
            block_arm=ForegroundArm.TEMPO,
            request_id="semantic-tempo-request",
            semantic_contract=_semantic_contract(),
        )


def test_route_pinned_background_request_releases_passive_credit():
    metadata = {
        "tenant": Tenant.REMOTE_HOT.value,
        "arm": ForegroundArm.REMOTE.value,
        "prompt_tokens": 512,
        "output_tokens": 16,
        "cache_state": CacheState.MISS.value,
        "terminal_item": 2,
    }
    decision = adaptive_decision(
        metadata, route=c4._REMOTE_ROUTE, block_arm=ForegroundArm.LOCAL)
    decision.update({
        "endpoint_passive_feedback_enabled": True,
        "endpoint_passive_registered": True,
        "endpoint_external_credit_registered": True,
        "endpoint_feedback_passive": True,
        "endpoint_feedback_event": (
            "external_credit_passive_first_response_chunk"),
        "admission_credit_scope": None,
        "admission_credit_release_event": None,
        "endpoint_passive_service_lookup_mode": (
            "miss_via_prefill_only_geometry_ceiling"),
        "endpoint_passive_service_source_prompt_tokens": 2048,
        "endpoint_passive_service_source_output_tokens": 128,
        "endpoint_passive_service_source_cache_residency": "prefill_only",
    })
    route = client._validate_dynamic_decision(
        decision,
        metadata,
        block_arm=ForegroundArm.LOCAL,
        request_id="epd-remote-endpoint-observed-background",
        semantic_contract=_semantic_contract(),
    )
    assert route == c4._REMOTE_ROUTE

    decision["endpoint_feedback_event"] = None
    with pytest.raises(ValueError, match="external credit"):
        client._validate_dynamic_decision(
            decision,
            metadata,
            block_arm=ForegroundArm.LOCAL,
            request_id="epd-remote-endpoint-observed-background",
            semantic_contract=_semantic_contract(),
        )


def test_semantic_node_prestart_environment_is_exact():
    values = dict(node._SEMANTIC_FIXED_RUNTIME_ENVIRONMENT)
    values.update({
        client.SEMANTIC_RUN_CONTRACT_ENV: "/repo/results/semantic.json",
        client.SEMANTIC_RUN_CONTRACT_SHA_ENV: "a" * 64,
        "TEMPO_PD_C4_READINESS_S": "3600",
    })
    with patch.dict(os.environ, values, clear=True):
        assert node._validate_prestart_environment() is True

    values[client.RUN_CONTRACT_ENV] = "/repo/results/adaptive.json"
    values[client.RUN_CONTRACT_SHA_ENV] = "b" * 64
    with patch.dict(os.environ, values, clear=True):
        with pytest.raises(ValueError, match="exactly one"):
            node._validate_prestart_environment()


def test_semantic_runtime_environment_does_not_alias_candidate_a():
    environment = contract.SEMANTIC_FIXED_RUNTIME_ENVIRONMENT
    assert "TEMPO_PD_C4_ADAPTIVE_APPROVED" not in environment
    assert environment[
        "TEMPO_PD_C4_SEMANTIC_INTEGRATION_APPROVED"] == "YES"
    assert environment["TEMPO_PD_ENDPOINT_ROUTING_POLICY"] == (
        "semantic_epoch_v1")
    assert environment["TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK"] == "1"
