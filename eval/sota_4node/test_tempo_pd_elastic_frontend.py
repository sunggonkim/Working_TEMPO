import asyncio
import json
from types import SimpleNamespace
import unittest

import httpx

from eval.sota_4node import tempo_pd_elastic_frontend as frontend
from tempo.pd_global_orchestrator import GlobalRoute


class CanonicalFrontendTest(unittest.TestCase):
    def test_mesh_remote_failure_scope_resolves_global_route(self):
        request = httpx.Request("POST", "http://router/v1/completions")
        response = httpx.Response(500, request=request)
        error = httpx.HTTPStatusError(
            "upstream failed", request=request, response=response)
        decision = SimpleNamespace(
            route=GlobalRoute.REMOTE,
            prefill_index=0,
            decoder_index=1,
        )
        self.assertEqual(
            frontend.tempo_global_failure_scope(
                error, decision=decision, mesh_enabled=True),
            "edge",
        )

    def test_wire_schema_and_pair_policy_are_canonical(self):
        self.assertEqual(frontend.ROUTER_SCHEMA,
                         "tempo-elastic-pd-router-canonical")
        self.assertEqual(frontend.FRONTEND_SCHEMA,
                         "tempo-elastic-pd-frontend-canonical-semantic-pressure-4")
        self.assertEqual(
            frontend.PAIR_AFFINITY_POLICY,
            "warm-prompt-sha256-owner-set-v2")
        self.assertEqual(
            frontend.PAIR_POLICY,
            "tempo-min-outstanding-decode-tokens-v1")
        self.assertEqual(
            frontend.BUCKET_ROTATION_PAIR_POLICY,
            "tempo-cache-stable-log2-decode-bucket-rotation-v1")

    def test_request_arm_and_decode_tokens_are_explicit(self):
        self.assertEqual(
            frontend.request_arm("epd-tempo-r1-measured-item-03"), "tempo")
        self.assertEqual(
            frontend.request_arm("epd-remote-r0-warm-item-00"), "remote")
        self.assertEqual(
            frontend.request_arm("epd-local-r0-warm-seed-item-00"), "local")
        self.assertEqual(
            frontend.request_arm("epd-queue_gpu-r0-measured-item-00"),
            "queue_gpu")
        seed_id = "epd-tempo-r0-warm-seed-o256-item-00"
        self.assertEqual(frontend.placement_decode_tokens(seed_id, 2), 256)
        self.assertEqual(
            frontend.placement_decode_tokens(
                "epd-tempo-r0-measured-item-00", 128), 128)
        self.assertEqual(
            frontend._decode_tokens(
                json.dumps(
                    {"max_tokens": 256, "prompt": "cache me"}).encode()),
            256)
        tokens, key = frontend._completion_shape(
            json.dumps({"max_tokens": 16, "prompt": "cache me"}).encode())
        self.assertEqual(tokens, 16)
        self.assertEqual(len(key), 64)
        for request_id in ("", "unscoped-item-00"):
            with self.assertRaises(ValueError):
                frontend.request_arm(request_id)
        self.assertTrue(frontend.c4_physical_pair_pin(
            "epd-tempo-c4-cache-p-only-warm-physical-g00-item-000000",
            "tempo",
        ))
        self.assertTrue(frontend.c4_physical_pair_pin(
            ("epd-tempo-c4-cache-p-only-warm-seed-o8-physical-"
             "g00-item-000000"),
            "tempo",
        ))
        self.assertTrue(frontend.c4_physical_pair_pin(
            ("epd-queue_gpu-c4-cache-p-only-warm-seed-o8-physical-"
             "g00-item-000000"),
            "queue_gpu",
        ))
        self.assertFalse(frontend.c4_physical_pair_pin(
            "epd-tempo-c4-cache-p-only-measured-item-000000", "tempo"))
        for payload in (b"{}", b'{"max_tokens": 0}', b'{"max_tokens": true}'):
            with self.assertRaises(ValueError):
                frontend._decode_tokens(payload)

    def test_upstream_status_preserves_bounded_contract_body(self):
        request = httpx.Request("POST", "http://router/v1/completions")
        response = httpx.Response(
            400, request=request,
            content=b'{"detail":"TEMPO-GO pair/router identity mismatch"}',
        )
        with self.assertRaises(httpx.HTTPStatusError) as raised:
            asyncio.run(frontend._raise_upstream_status_with_body(response))
        self.assertIn("TEMPO-GO pair/router identity mismatch", str(raised.exception))

    def test_cold_mode_disables_only_measured_tempo_warm_affinity(self):
        measured = "epd-tempo-r0-measured-item-00"
        self.assertTrue(frontend.requires_warm_pair_affinity(
            measured, "tempo", cold_measured=False))
        self.assertFalse(frontend.requires_warm_pair_affinity(
            measured, "tempo", cold_measured=True))
        p_only = (
            "epd-tempo-r0-cache-p-only-measured-item-00")
        self.assertTrue(frontend.requires_warm_pair_affinity(
            p_only, "tempo", cold_measured=False))
        self.assertTrue(frontend.requires_warm_pair_affinity(
            p_only, "tempo", cold_measured=True))
        for request_id in (
            "epd-tempo-r0-cache-d-only-measured-item-00",
            "epd-tempo-r0-cache-both-measured-item-00",
            "epd-tempo-r0-warm-cache-d-probe-item-00",
        ):
            with self.subTest(request_id=request_id):
                self.assertTrue(frontend.requires_warm_pair_affinity(
                    request_id, "tempo", cold_measured=True))
        self.assertFalse(frontend.requires_warm_pair_affinity(
            "epd-tempo-r0-warm-cache-d-seed-item-00",
            "tempo", cold_measured=True))
        for request_id, arm in (
            ("epd-tempo-r0-warm-item-00", "tempo"),
            ("epd-local-r0-measured-item-00", "local"),
            ("epd-remote-r0-measured-item-00", "remote"),
            ("epd-predictor-r0-measured-item-00", "predictor"),
        ):
            with self.subTest(request_id=request_id):
                self.assertFalse(frontend.requires_warm_pair_affinity(
                    request_id, arm, cold_measured=False))
        self.assertFalse(frontend.requires_warm_pair_affinity(
            "epd-queue_gpu-r0-measured-item-00",
            "queue_gpu", cold_measured=False))
        with self.assertRaises(TypeError):
            frontend.requires_warm_pair_affinity(
                measured, "tempo", cold_measured=1)
    def test_affinity_shadow_id_preserves_warm_contract(self):
        request_id = "epd-tempo-r0-warm-seed-o256-item-03"
        shadow = frontend.affinity_shadow_request_id(request_id, 1)
        self.assertEqual(
            shadow,
            "epd-tempo-r0-warm-seed-o256-affinity-shadow-p1-item-03")
        self.assertEqual(frontend.request_arm(shadow), "tempo")
        self.assertIn("-warm-seed-", shadow)
        with self.assertRaises(ValueError):
            frontend.affinity_shadow_request_id(
                "epd-tempo-r0-measured-item-03", 1)


    def test_completion_prompt_is_required_for_cache_ownership(self):
        for payload in (
            b'{"max_tokens": 16}',
            b'{"max_tokens": 16, "prompt": ""}',
            b'{"max_tokens": 16, "prompt": [1, 2]}',
        ):
            with self.assertRaises(ValueError):
                frontend._completion_shape(payload)

    def test_explicit_cache_contract_controls_decoder_affinity(self):
        common = {
            "decoder_prefix_caching": True,
            "affinity_required": True,
            "decoder_reuse_items": None,
        }
        self.assertFalse(frontend.uses_decoder_affinity(
            "epd-tempo-r0-cache-p-only-measured-item-00", **common))
        self.assertFalse(frontend.uses_decoder_affinity(
            "epd-tempo-r0-cache-miss-measured-item-00", **common))
        for request_id in (
            "epd-tempo-r0-warm-cache-d-seed-item-00",
            "epd-tempo-r0-warm-cache-d-probe-item-00",
            "epd-tempo-r0-cache-d-only-measured-item-00",
            "epd-tempo-r0-cache-both-measured-item-00",
        ):
            with self.subTest(request_id=request_id):
                self.assertTrue(frontend.uses_decoder_affinity(
                    request_id, **common))


class PairLoadLedgerTest(unittest.IsolatedAsyncioTestCase):
    async def test_global_rejection_explicitly_records_no_commit(self):
        app = SimpleNamespace(state=SimpleNamespace(
            tempo_go_rejection_lock=asyncio.Lock(),
            tempo_go_rejections={},
        ))
        await frontend._record_tempo_go_rejection(
            app,
            "reject-r0",
            decision={
                "schema": "tempo-go-global-orchestrator-v1",
                "kind": "reject",
                "request_id": "reject-r0",
                "reason": "global_telemetry_refresh_timeout",
                "decided_ns": 20,
            },
            decision_sha256="a" * 64,
            tokenizer_ms=1.0,
            admission_arrival_ns=10,
        )
        row = app.state.tempo_go_rejections["reject-r0"]
        self.assertFalse(row["frontend_pair_global_commit"])
        self.assertIs(row["tempo_go_global_commit_applied"], False)

    async def test_queue_gpu_pair_selection_uses_running_waiting_and_kv_gauges(self):
        class Response:
            def __init__(self, running, waiting, kv):
                self.value = {
                    "vllm_scheduler": {
                        "decision_mode": "observe_only",
                        "source": "router_local_vllm_prometheus_observe_only",
                        "num_requests_running": running,
                        "num_requests_waiting": waiting,
                        "kv_cache_usage_fraction": kv,
                    },
                }

            def raise_for_status(self):
                return None

            def json(self):
                return self.value

        class Client:
            def __init__(self, response):
                self.response = response

            async def get(self, _path):
                return self.response

        selected, observation = await frontend._queue_gpu_pair_selection([
            Client(Response(8, 1, 0.2)),
            Client(Response(3, 0, 0.9)),
        ])
        self.assertEqual(selected, 1)
        self.assertEqual(observation["selected_pair"], 1)
        self.assertEqual(len(observation["pairs"]), 2)

    async def test_queue_gpu_selection_is_recorded_as_distinct_dynamic_policy(self):
        ledger = frontend.PairLoadLedger(2)
        row = await ledger.reserve(
            "epd-queue_gpu-r0-measured-item-0", 16, 0,
            dynamic=True,
            queue_gpu_pair=1,
            queue_gpu_observation={
                "schema": "tempo-go-vllm-scheduler-pair-selection-v1",
                "selected_pair": 1,
            },
        )
        self.assertEqual(row["frontend_pair_index"], 1)
        self.assertEqual(
            row["frontend_pair_policy"],
            "queue-gpu-vllm-scheduler-observe-only-v1",
        )
        self.assertTrue(row["frontend_pair_queue_gpu_selection"])

    async def test_active_counts_are_pair_local_and_live_until_release(self):
        ledger = frontend.PairLoadLedger(2)
        first = await ledger.reserve(
            "epd-local-r0-measured-item-0", 256, 0, dynamic=False)
        self.assertEqual(
            first["frontend_pair_active_requests_before"], [0, 0])
        self.assertEqual(
            first["frontend_pair_active_requests_after_reserve"], [1, 0])

        second = await ledger.reserve(
            "epd-remote-r0-measured-item-1", 16, 1, dynamic=False)
        self.assertEqual(
            second["frontend_pair_active_requests_before"], [1, 0])
        self.assertEqual(
            second["frontend_pair_active_requests_after_reserve"], [1, 1])
        snapshot = await ledger.snapshot()
        self.assertEqual(snapshot["active"], 2)
        self.assertEqual(snapshot["active_by_pair"], [1, 1])

        self.assertTrue(await ledger.release(first["request_id"]))
        snapshot = await ledger.snapshot()
        self.assertEqual(snapshot["active"], 1)
        self.assertEqual(snapshot["active_by_pair"], [0, 1])
        released = snapshot["rows"][first["request_id"]]
        self.assertEqual(
            released["frontend_pair_active_requests_after_release"], [0, 1])
        self.assertFalse(await ledger.release(first["request_id"]))

    async def test_decision_get_retries_one_stale_keepalive_close(self):
        sentinel = object()

        class FlakyClient:
            def __init__(self):
                self.calls = 0

            async def get(self, path):
                self.calls += 1
                self.assert_path = path
                if self.calls == 1:
                    raise frontend.httpx.RemoteProtocolError(
                        "stale keep-alive")
                return sentinel

        client = FlakyClient()
        response, retries = await frontend._get_pair_decisions(client)
        self.assertIs(response, sentinel)
        self.assertEqual(retries, 1)
        self.assertEqual(client.calls, 2)
        self.assertEqual(client.assert_path, "/tempo/decisions")

    async def test_warm_prompt_owner_is_sticky_and_measured_miss_fails(self):
        ledger = frontend.PairLoadLedger(2)
        key = "a" * 64
        busy = await ledger.reserve("busy", 256, 0, dynamic=False)
        self.assertEqual(busy["frontend_pair_index"], 0)

        seed = await ledger.reserve(
            "tempo-warm-seed", 2, 0, dynamic=True,
            affinity_key=key, affinity_seed=True)
        self.assertEqual(seed["frontend_pair_index"], 1)
        self.assertTrue(seed["frontend_pair_affinity_created"])
        self.assertFalse(seed["frontend_pair_affinity_hit"])
        self.assertTrue(await ledger.release("tempo-warm-seed"))

        probe = await ledger.reserve(
            "tempo-warm-probe", 16, 0, dynamic=True,
            affinity_key=key, affinity_seed=True)
        self.assertEqual(probe["frontend_pair_index"], 1)
        self.assertTrue(probe["frontend_pair_affinity_hit"])
        self.assertFalse(probe["frontend_pair_affinity_created"])
        self.assertTrue(await ledger.release("tempo-warm-probe"))

        measured = await ledger.reserve(
            "tempo-measured", 128, 0, dynamic=True,
            affinity_key=key, affinity_required=True)
        self.assertEqual(measured["frontend_pair_index"], 1)
        self.assertTrue(measured["frontend_pair_affinity_hit"])
        self.assertTrue(measured["frontend_pair_affinity_required"])
        self.assertEqual(
            (await ledger.snapshot())["pair_affinity_entries"], 1)

        with self.assertRaisesRegex(ValueError, "lacks warm pair affinity"):
            await ledger.reserve(
                "tempo-measured-miss", 16, 0, dynamic=True,
                affinity_key="b" * 64, affinity_required=True)

    async def test_replicated_affinity_uses_only_proven_owners(self):
        ledger = frontend.PairLoadLedger(2)
        key = "c" * 64
        self.assertEqual(
            await ledger.register_affinity_replicas(key, {0}), [0])
        with self.assertRaisesRegex(ValueError, "lacks replicated pair affinity"):
            await ledger.reserve(
                "tempo-measured-one-owner", 16, 0, dynamic=True,
                affinity_key=key, affinity_required=True,
                affinity_owner_count_required=2)

        self.assertEqual(
            await ledger.register_affinity_replicas(
                key, {0, 1}, evidence_request_ids={
                    0: "epd-tempo-r0-warm-item-00",
                    1: ("epd-tempo-r0-warm-affinity-shadow-p1-"
                        "item-00"),
                }),
            [0, 1])
        await ledger.reserve("busy-pair-zero", 256, 0, dynamic=False)
        measured = await ledger.reserve(
            "tempo-measured-two-owners", 128, 0, dynamic=True,
            affinity_key=key, affinity_required=True,
            affinity_owner_count_required=2)
        self.assertEqual(measured["frontend_pair_index"], 1)
        self.assertEqual(
            measured["frontend_pair_affinity_owner_indices"], [0, 1])
        self.assertEqual(
            measured["frontend_pair_affinity_replica_count"], 2)
        self.assertEqual(
            measured["frontend_pair_affinity_registration_source"],
            "completed_warm_probe_eof")
        self.assertEqual(
            len(measured[
                "frontend_pair_affinity_evidence_request_ids"]), 2)
        snapshot = await ledger.snapshot()
        self.assertEqual(snapshot["pair_affinity_entries"], 1)
        self.assertEqual(snapshot["pair_affinity_replicas"], 2)


    async def test_completed_measured_prompt_pins_decoder_owner(self):
        ledger = frontend.PairLoadLedger(2)
        key = "d" * 64
        await ledger.register_affinity_replicas(key, {0, 1})
        first = await ledger.reserve(
            "epd-tempo-r0-measured-item-00", 128, 0, dynamic=True,
            affinity_key=key, affinity_required=True,
            affinity_owner_count_required=2,
            prefer_decode_affinity=True)
        self.assertEqual(first["frontend_pair_index"], 0)
        self.assertFalse(first["frontend_pair_decode_affinity_hit"])
        self.assertTrue(
            await ledger.release("epd-tempo-r0-measured-item-00"))
        await ledger.register_decode_affinity(
            key, 0,
            evidence_request_id="epd-tempo-r0-measured-item-00")
        await ledger.reserve("busy-decoder-owner", 256, 0, dynamic=False)
        repeated = await ledger.reserve(
            "epd-tempo-r1-measured-item-00", 128, 1, dynamic=True,
            affinity_key=key, affinity_required=True,
            affinity_owner_count_required=2,
            prefer_decode_affinity=True)
        self.assertEqual(repeated["frontend_pair_index"], 0)
        self.assertTrue(repeated["frontend_pair_decode_affinity_hit"])
        self.assertEqual(
            (await ledger.snapshot())["decode_affinity_entries"], 1)


    async def test_decoder_cache_reset_clears_only_quiescent_decode_affinity(self):
        ledger = frontend.PairLoadLedger(2)
        key = "e" * 64
        await ledger.register_affinity_replicas(key, {0, 1})
        await ledger.register_decode_affinity(
            key, 1,
            evidence_request_id="epd-tempo-r0-measured-item-00")
        await ledger.reserve("active", 16, 0, dynamic=False)
        with self.assertRaisesRegex(RuntimeError, "requires a quiescent"):
            await ledger.clear_decode_affinity_for_cache_reset()
        self.assertTrue(await ledger.release("active"))
        self.assertEqual(
            await ledger.clear_decode_affinity_for_cache_reset(), 1)
        snapshot = await ledger.snapshot()
        self.assertEqual(snapshot["decode_affinity_entries"], 0)
        self.assertEqual(snapshot["pair_affinity_entries"], 1)

    async def test_d_cache_probe_requires_seeded_decoder_owner(self):
        ledger = frontend.PairLoadLedger(2)
        key = "f" * 64
        seed_id = "epd-tempo-r0-warm-cache-d-seed-item-00"
        seed = await ledger.reserve(
            seed_id, 2, 1, dynamic=True,
            affinity_key=key, affinity_seed=True)
        self.assertEqual(seed["frontend_pair_index"], 1)
        self.assertTrue(await ledger.release(seed_id))

        probe_id = "epd-tempo-r0-warm-cache-d-probe-item-00"
        with self.assertRaisesRegex(
            ValueError, "lacks proven decoder pair affinity",
        ):
            await ledger.reserve(
                probe_id, 2, 0, dynamic=True,
                affinity_key=key, affinity_required=True,
                affinity_owner_count_required=None,
                prefer_decode_affinity=True,
                decode_affinity_required=True)

        await ledger.register_decode_affinity(
            key, 1, evidence_request_id=seed_id)
        probe = await ledger.reserve(
            probe_id, 2, 0, dynamic=True,
            affinity_key=key, affinity_required=True,
            affinity_owner_count_required=None,
            prefer_decode_affinity=True,
            decode_affinity_required=True)
        self.assertEqual(probe["frontend_pair_index"], 1)
        self.assertTrue(probe["frontend_pair_decode_affinity_hit"])
        self.assertTrue(probe["frontend_pair_decode_affinity_required"])


    async def test_bucket_rotation_is_cache_stable_and_shape_balanced(self):
        ledger = frontend.PairLoadLedger(
            2, tempo_policy=frontend.BUCKET_ROTATION_PAIR_POLICY)
        expected = {
            16: (0, 1), 32: (1, 0), 64: (0, 1),
            128: (1, 0), 256: (0, 1),
        }
        for output_tokens, pair_indices in expected.items():
            for preferred, selected in enumerate(pair_indices):
                request_id = f"tempo-{output_tokens}-{preferred}"
                row = await ledger.reserve(
                    request_id, output_tokens, preferred, dynamic=True)
                self.assertEqual(row["frontend_pair_index"], selected)
                self.assertEqual(
                    row["frontend_pair_placement_decode_tokens"],
                    output_tokens)
                self.assertTrue(await ledger.release(request_id))

        seed = await ledger.reserve(
            "tempo-seed", 2, 0, dynamic=True, placement_tokens=256)
        self.assertEqual(seed["frontend_pair_index"], 0)
        self.assertEqual(seed["frontend_decode_tokens_reserved"], 2)
        self.assertEqual(seed["frontend_pair_placement_decode_tokens"], 256)

    async def test_tempo_chooses_minimum_load_with_deterministic_tie(self):
        ledger = frontend.PairLoadLedger(2)
        first = await ledger.reserve(
            "tempo-0", 256, 0, dynamic=True)
        second = await ledger.reserve(
            "tempo-1", 256, 0, dynamic=True)
        third = await ledger.reserve(
            "tempo-2", 128, 1, dynamic=True)
        self.assertEqual(first["frontend_pair_index"], 0)
        self.assertEqual(second["frontend_pair_index"], 1)
        self.assertEqual(third["frontend_pair_index"], 1)
        self.assertEqual(
            (await ledger.snapshot())["loads"], [256, 384])

        self.assertTrue(await ledger.release("tempo-1"))
        self.assertFalse(await ledger.release("tempo-1"))
        self.assertTrue(await ledger.release("tempo-0"))
        self.assertTrue(await ledger.release("tempo-2"))
        snapshot = await ledger.snapshot()
        self.assertEqual(snapshot["loads"], [0, 0])
        self.assertEqual(snapshot["active"], 0)
        self.assertTrue(
            snapshot["rows"]["tempo-2"]["frontend_pair_released"])

    async def test_c4_physical_seed_pin_is_preferred_and_registers_affinity(self):
        ledger = frontend.PairLoadLedger(2)
        key = "e" * 64
        await ledger.reserve("busy-pair-zero", 256, 0, dynamic=False)
        seed = await ledger.reserve(
            ("epd-tempo-c4-cache-p-only-warm-seed-o8-physical-"
             "g00-item-000000"),
            2,
            0,
            dynamic=True,
            placement_tokens=8,
            affinity_key=key,
            affinity_seed=True,
            pair_pin_preferred=True,
        )
        self.assertEqual(seed["frontend_pair_index"], 0)
        self.assertTrue(seed["frontend_pair_physical_seed_pin"])
        self.assertTrue(seed["frontend_pair_affinity_created"])
        self.assertTrue(await ledger.release(
            "epd-tempo-c4-cache-p-only-warm-seed-o8-physical-"
            "g00-item-000000"))
        measured = await ledger.reserve(
            "epd-tempo-c4-cache-p-only-measured-item-000000",
            8,
            1,
            dynamic=True,
            affinity_key=key,
            affinity_required=True,
        )
        self.assertEqual(measured["frontend_pair_index"], 0)

    async def test_queue_gpu_cannot_override_physical_seed_owner_pin(self):
        ledger = frontend.PairLoadLedger(2)
        with self.assertRaisesRegex(
            ValueError, "queue-GPU pair requires unpinned dynamic routing"
        ):
            await ledger.reserve(
                "epd-queue_gpu-c4-cache-p-only-warm-physical-item-000000",
                2,
                0,
                dynamic=True,
                pair_pin_preferred=True,
                queue_gpu_pair=1,
            )

    async def test_baseline_keeps_static_pair_even_when_loaded(self):
        ledger = frontend.PairLoadLedger(2)
        first = await ledger.reserve(
            "local-0", 256, 0, dynamic=False)
        second = await ledger.reserve(
            "local-2", 16, 0, dynamic=False)
        self.assertEqual(first["frontend_pair_index"], 0)
        self.assertEqual(second["frontend_pair_index"], 0)
        self.assertEqual(
            (await ledger.snapshot())["loads"], [272, 0])

    async def test_duplicate_reservation_is_rejected(self):
        ledger = frontend.PairLoadLedger(2)
        await ledger.reserve("tempo-0", 16, 0, dynamic=True)
        with self.assertRaises(ValueError):
            await ledger.reserve("tempo-0", 16, 1, dynamic=True)

    async def test_global_pair_commit_overrides_local_load_balancer(self):
        ledger = frontend.PairLoadLedger(2)
        await ledger.reserve("busy-pair-zero", 256, 0, dynamic=False)
        committed = await ledger.reserve(
            "tempo-go", 64, 0, dynamic=True,
            affinity_key="a" * 64,
            committed_pair=0,
        )
        self.assertEqual(committed["frontend_pair_index"], 0)
        self.assertTrue(committed["frontend_pair_global_commit"])
        self.assertEqual(
            committed["frontend_pair_policy"],
            "tempo-go-global-committed-pair-v1",
        )
        await ledger.record_global_decision(
            "tempo-go",
            decision={"pair_index": 0, "route": "local", "decided_ns": 20},
            decision_sha256="b" * 64,
            tokenizer_ms=1.25,
            admission_arrival_ns=10,
        )
        row = (await ledger.snapshot())["rows"]["tempo-go"]
        self.assertEqual(row["frontend_tempo_go_decision_sha256"], "b" * 64)
        self.assertEqual(row["frontend_tempo_go_tokenizer_ms"], 1.25)
        self.assertEqual(row["frontend_tempo_go_admission_arrival_ns"], 10)
        self.assertEqual(row["frontend_tempo_go_admission_wait_ns"], 10)

    async def test_cache_states_use_only_completed_affinity_evidence(self):
        ledger = frontend.PairLoadLedger(2)
        key = "c" * 64
        seed = await ledger.reserve(
            "tempo-seed", 2, 0, dynamic=True,
            affinity_key=key, affinity_seed=True)
        self.assertEqual(seed["frontend_pair_index"], 0)
        unknown = await ledger.cache_states(
            key, explicit_cache_reset_miss=False)
        self.assertEqual(
            [item.residency for item in unknown],
            [frontend.CacheResidency.UNKNOWN, frontend.CacheResidency.UNKNOWN],
        )
        await ledger.release("tempo-seed")
        await ledger.register_affinity_replicas(
            key, {0}, evidence_request_ids={
                0: "epd-tempo-r0-warm-item-00"})
        proven = await ledger.cache_states(
            key, explicit_cache_reset_miss=False)
        self.assertEqual(
            [item.residency for item in proven],
            [frontend.CacheResidency.P_ONLY, frontend.CacheResidency.MISS],
        )
        cold = await ledger.cache_states(
            "d" * 64, explicit_cache_reset_miss=True)
        self.assertEqual(
            [item.residency for item in cold],
            [frontend.CacheResidency.MISS, frontend.CacheResidency.MISS],
        )


if __name__ == "__main__":
    unittest.main()
