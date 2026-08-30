from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval.sota_4node import (
    build_tempo_pd_c4_adaptive_screen_manifest as manifest_builder,
)
from eval.sota_4node import run_tempo_pd_c4_adaptive_screen_client as client
from eval.sota_4node import run_tempo_pd_c4_fixed_phase_client as c4
from tempo.pd_cache_state_protocol import build_cache_preparation_plan
from tempo.pd_contention_workload import CacheState, ForegroundArm, Tenant


class _Tokenizer:
    def __init__(self):
        self.values = {}
        self.unknown = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in self.values:
            return list(self.values[text])
        if text not in self.unknown:
            base = 70_000 + 2 * len(self.unknown)
            self.unknown[text] = (base, base + 1)
        return list(self.unknown[text])

    def decode(
        self, token_ids, *, skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        value = "tokens:" + ",".join(str(item) for item in token_ids)
        self.values[value] = tuple(token_ids)
        return value


def _manifest() -> dict[str, object]:
    return {
        "background_rates_per_s": {
            "decoder_hot": 1.0,
            "cold_remote_hot": 1.0,
            "kv_remote_hot": 1.0,
        },
        "foreground_rate_per_s": 1.0,
        "phase_duration_ms": 1000.0,
        "arm_order_by_replicate": [
            list(values)
            for values in manifest_builder.ARM_ORDER_BY_REPLICATE
        ],
    }


def _decision(
    metadata: dict[str, object], *, route: str,
    block_arm: ForegroundArm,
) -> dict[str, object]:
    state = CacheState(str(metadata["cache_state"]))
    prompt_tokens = int(metadata["prompt_tokens"])
    skipped = state in {CacheState.MISS, CacheState.P_ONLY}
    local_cached = (
        0 if skipped else c4.full_prefix_hit_tokens(prompt_tokens))
    total_cached = (
        prompt_tokens if route == c4._REMOTE_ROUTE else local_cached)
    value: dict[str, object] = {
        "route": route,
        "request_cache_contract": state.value,
        "decision_cache_residency": c4._DECISION_STATE[state],
        "decoder_prefix_read_skipped": skipped,
        "decoder_prefix_cached_tokens": local_cached,
        "decoder_total_cached_tokens": total_cached,
        "decoder_external_cached_tokens": total_cached - local_cached,
        "decoder_prefix_usage_prompt_tokens": (
            prompt_tokens + int(route == c4._REMOTE_ROUTE)),
        "decoder_prefix_expected_full_hit_tokens": (
            c4.full_prefix_hit_tokens(prompt_tokens)),
        "decoder_prefix_full_hit_observed": not skipped,
        "decoder_prefix_cache_evidence_source": (
            c4.DECODER_CACHE_EVIDENCE_SOURCE),
        "frontend_pair_policy": "item_modulo_v1",
        "frontend_pair_index": int(metadata["terminal_item"]) % 2,
        "lmcache_source_cached_tokens": (
            prompt_tokens
            if state in {CacheState.P_ONLY, CacheState.BOTH} else 0),
        "endpoint_policy_applied": False,
        "endpoint_decision_attempts": 0,
        "endpoint_feedback_event": None,
        "admission_credit_release_event": None,
        "admission_credit_released_ns": None,
    }
    tenant = Tenant(str(metadata["tenant"]))
    if tenant is Tenant.FOREGROUND:
        if block_arm is ForegroundArm.LOCAL:
            value.update({"arm": "always_local", "reason": "fixed_always_local"})
        elif block_arm is ForegroundArm.REMOTE:
            value.update({
                "arm": "official_lmcache_remote",
                "reason": "fixed_official_lmcache_remote",
            })
        elif block_arm is ForegroundArm.PREDICTOR:
            value.update({
                "arm": "predictor",
                "reason": (
                    "predictor_decoder_residency_local"
                    if state in {CacheState.D_ONLY, CacheState.BOTH}
                    else (
                        "predictor_local_safe"
                        if route == c4._LOCAL_ROUTE
                        else "predictor_remote_lower_bound"
                    )
                ),
            })
        else:
            value.update({
                "arm": "tempo",
                "reason": "endpoint_feedback_test",
                "endpoint_feedback_mode": "adaptive",
                "endpoint_policy_applied": True,
                "endpoint_decision_attempts": 1,
                "endpoint_decision_history": [{"route": route}],
                "endpoint_decision_route": route,
                "endpoint_request_local_allowed": True,
                "endpoint_request_remote_allowed": (
                    state not in {CacheState.D_ONLY, CacheState.BOTH}),
                "admission_credit_release_event": "first_response_chunk",
                "admission_credit_released_ns": 123,
                "endpoint_feedback_event": "first_response_chunk",
                "endpoint_feedback_accepted": True,
            })
    return value


class _Response:
    def __init__(self, value: dict[str, object]):
        self.status = 200
        self._encoded = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        return False

    def read(self):
        return self._encoded


def _controller_state(generation: int, *, completed: int = 0):
    return {
        "success": True,
        "controller_generation": generation,
        "queued_requests": 0,
        "controller": {
            "inflight": 0,
            "completed": completed,
            "resources": {
                "local_token_ms": 0,
                "remote_prefill_token_ms": 0,
                "remote_kv_bytes": 0,
                "remote_semantic_ops": 0,
            },
        },
    }


class C4AdaptiveScreenClientTest(unittest.TestCase):
    def test_materializes_exact_eight_blocks_and_all_six_geometries(self):
        tokenizer = _Tokenizer()
        factory = c4._PromptFactory(tokenizer, {
            length: tuple(range(length)) for length in (512, 2048, 4094)
        })
        manifest = _manifest()
        order = client._block_order(manifest)
        self.assertEqual(
            tuple((arm.value, replicate) for arm, replicate in order),
            tuple(
                (arm, replicate)
                for replicate, values in enumerate(
                    manifest_builder.ARM_ORDER_BY_REPLICATE)
                for arm in values
            ),
        )
        blocks = [
            c4._materialize_block(
                sequence=sequence,
                arm=arm,
                replicate=replicate,
                manifest=manifest,
                factory=factory,
            )
            for sequence, (arm, replicate) in enumerate(order)
        ]
        self.assertEqual(len(blocks), 8)
        expected_geometries = {
            (512, 16, "miss"),
            (2048, 128, "p_only"),
            (4094, 256, "d_only"),
            (512, 128, "both"),
            (2048, 256, "miss"),
            (4094, 16, "p_only"),
        }
        for block in blocks:
            foreground = [
                row for row in block["request_index"].values()
                if row["tenant"] == Tenant.FOREGROUND.value
            ]
            self.assertEqual({
                (row["prompt_tokens"], row["output_tokens"], row["cache_state"])
                for row in foreground
            }, expected_geometries)
        for replicate in range(2):
            selected = [
                block for block in blocks if block["replicate"] == replicate]
            self.assertEqual(
                len({block["schedule_sha256"] for block in selected}), 1)
            paired_prompts = [{
                row["pair_key"]: row["prompt_token_sha256"]
                for row in block["request_index"].values()
                if row["tenant"] == Tenant.FOREGROUND.value
            } for block in selected]
            self.assertTrue(all(
                value == paired_prompts[0] for value in paired_prompts[1:]))
        plan = build_cache_preparation_plan(
            item for block in blocks for item in block["items"])
        self.assertGreater(len(plan.source_probe_rows), 0)
        self.assertGreater(len(plan.decoder_prepare_rows), 0)

    def test_dynamic_policy_and_cache_evidence_are_exact_for_every_state(self):
        for block_arm in client.ARMS:
            for index, state in enumerate(CacheState):
                route = {
                    ForegroundArm.LOCAL: c4._LOCAL_ROUTE,
                    ForegroundArm.REMOTE: c4._REMOTE_ROUTE,
                }.get(block_arm)
                if route is None:
                    route = (
                        c4._LOCAL_ROUTE
                        if state in {CacheState.D_ONLY, CacheState.BOTH}
                        or index % 2 else c4._REMOTE_ROUTE)
                metadata = {
                    "tenant": Tenant.FOREGROUND.value,
                    "arm": block_arm.value,
                    "prompt_tokens": 512,
                    "output_tokens": 16,
                    "cache_state": state.value,
                    "terminal_item": 3,
                }
                decision = _decision(
                    metadata, route=route, block_arm=block_arm)
                with self.subTest(arm=block_arm.value, state=state.value):
                    self.assertEqual(
                        client._validate_dynamic_decision(
                            decision, metadata, block_arm=block_arm),
                        route,
                    )

        background = {
            "tenant": Tenant.REMOTE_HOT.value,
            "arm": ForegroundArm.REMOTE.value,
            "prompt_tokens": 512,
            "output_tokens": 16,
            "cache_state": CacheState.MISS.value,
            "terminal_item": 2,
        }
        remote = _decision(
            background,
            route=c4._REMOTE_ROUTE,
            block_arm=ForegroundArm.LOCAL,
        )
        client._validate_dynamic_decision(
            remote, background, block_arm=ForegroundArm.LOCAL)
        local = _decision(
            background,
            route=c4._LOCAL_ROUTE,
            block_arm=ForegroundArm.LOCAL,
        )
        with self.assertRaisesRegex(ValueError, "contention tenant escaped"):
            client._validate_dynamic_decision(
                local, background, block_arm=ForegroundArm.LOCAL)

    def test_controller_reset_and_quiescence_are_generation_bound(self):
        state = _controller_state(7)
        with patch.object(client, "urlopen", return_value=_Response(state)):
            observed = client._controller_reset("http://pair0")
        self.assertEqual(observed["controller_generation"], 7)
        self.assertTrue(client._controllers_quiescent([state, state]))
        busy = _controller_state(7)
        busy["controller"]["resources"]["remote_semantic_ops"] = 1
        self.assertFalse(client._controllers_quiescent([state, busy]))
        with patch.object(client, "urlopen", return_value=_Response(busy)):
            with self.assertRaisesRegex(ValueError, "reset evidence"):
                client._controller_reset("http://pair0")

    def test_four_arm_paired_gate_requires_exact_inventory_and_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            contracts = {}
            order = client._block_order(_manifest())
            for sequence, (arm, replicate) in enumerate(order):
                key = f"{sequence:02d}_{arm.value}_r{replicate}"
                request_id = f"request-{sequence}"
                path = root / f"{key}.json"
                path.write_text(json.dumps({
                    "requests": [{
                        "request_id": request_id,
                        "output_text_sha256": "a" * 64,
                    }],
                    "router_decisions": [{
                        "request_id": request_id,
                        "route": (
                            c4._LOCAL_ROUTE
                            if replicate == 0 else c4._REMOTE_ROUTE),
                    }],
                }), encoding="utf-8")
                paths[key] = path
                contracts[key] = {
                    "schema": client.BLOCK_SCHEMA,
                    "sequence": sequence,
                    "arm": arm.value,
                    "replicate": replicate,
                    "semantic_schedule_sha256": (
                        "b" * 64 if replicate == 0 else "c" * 64),
                    "request_index": {request_id: {
                        "tenant": Tenant.FOREGROUND.value,
                        "pair_key": f"r{replicate}:paired-request",
                        "prompt_token_sha256": "d" * 64,
                    }},
                    "all_requests_valid": True,
                    "completion_cache_evidence_exact": True,
                    "phase_aligned_endpoint_evidence": True,
                    "controller_reset_before_block_exact": True,
                    "controller_quiescent_after_block": True,
                }
            gate = client._paired_gate(
                block_paths=paths, contracts=contracts)
            self.assertEqual(gate["paired_foreground_requests"], 2)
            self.assertTrue(gate["tempo_both_routes_exercised"])

            changed = json.loads(paths["01_remote_r0"].read_text(
                encoding="utf-8"))
            changed["requests"][0]["output_text_sha256"] = "e" * 64
            paths["01_remote_r0"].write_text(
                json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "paired prompt/output"):
                client._paired_gate(block_paths=paths, contracts=contracts)

            contracts.pop("07_local_r1")
            with self.assertRaisesRegex(ValueError, "artifact inventory"):
                client._paired_gate(block_paths=paths, contracts=contracts)


if __name__ == "__main__":
    unittest.main()
