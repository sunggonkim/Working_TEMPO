import asyncio
import httpx
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from eval.sota_4node import analyze_tempo_pd_elastic as analyzer
from eval.sota_4node import tempo_pd_elastic_router as router
from eval.sota_4node import tempo_pd_decoder_selecting_proxy as selecting_proxy

from eval.sota_4node import run_tempo_pd_elastic as client
from eval.sota_4node import vllm_lmcache_elastic_pd_node as node
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as perf
from eval.sota_4node.test_tempo_pd_elastic_router_v445 import config, profile_payload
from tempo.pd_elastic_profile import load_elastic_profile


def _decoder_usage_stream(
    *, prompt_tokens, completion_tokens, cached_tokens,
    local_cached_tokens=None, external_cached_tokens=None,
):
    if local_cached_tokens is None:
        local_cached_tokens = cached_tokens
    if external_cached_tokens is None:
        external_cached_tokens = cached_tokens - local_cached_tokens
    payload = {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {
                "cached_tokens": cached_tokens,
                "tempo_cache_breakdown_schema": (
                    "tempo-vllm-prefill-cache-breakdown-v1"),
                "tempo_local_cached_tokens": local_cached_tokens,
                "tempo_external_cached_tokens": external_cached_tokens,
            },
        },
    }
    return (
        "data: " + json.dumps(payload, separators=(",", ":"))
        + "\n\ndata: [DONE]\n\n"
    ).encode()


class CanonicalRouterIntegrationTest(unittest.TestCase):
    def test_frontend_semantic_load_is_strict_pair_local_eof_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {
                    "TEMPO_PD_LOCAL_DECODER_INDEX": "1",
                    "TEMPO_VLLM_MAX_NUM_SEQS": "16",
                },
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)

        request_id = "epd-local-r0-measured-item-semantic"
        evidence = core.prepare_frontend_semantic_load(
            request_id=request_id,
            pair_index="1",
            decode_tokens_before="512",
            active_requests_before="4",
            max_num_seqs="16",
        )
        self.assertEqual(evidence["schema"],
                         router.FRONTEND_SEMANTIC_LOAD_SCHEMA)
        self.assertEqual(evidence["occupancy_ratio_before"], 0.25)
        core.decide(
            request_id=request_id,
            prompt_tokens=512,
            output_tokens=16,
        )
        row = next(
            value for value in core.records()
            if value["request_id"] == request_id)
        self.assertEqual(row["frontend_semantic_pair_index"], 1)
        self.assertEqual(row["frontend_semantic_decode_tokens_before"], 512)
        self.assertEqual(row["frontend_semantic_active_requests_before"], 4)
        self.assertEqual(row["frontend_semantic_max_num_seqs"], 16)
        self.assertEqual(row["frontend_semantic_occupancy_ratio_before"], 0.25)

        self.assertIsNone(core.prepare_frontend_semantic_load(
            request_id="maintenance-no-headers",
            pair_index=None,
            decode_tokens_before=None,
            active_requests_before=None,
            max_num_seqs=None,
        ))
        invalid_cases = (
            {
                "pair_index": "0", "decode_tokens_before": "512",
                "active_requests_before": "4", "max_num_seqs": "16",
            },
            {
                "pair_index": "1", "decode_tokens_before": "0512",
                "active_requests_before": "4", "max_num_seqs": "16",
            },
            {
                "pair_index": "1", "decode_tokens_before": "512",
                "active_requests_before": None, "max_num_seqs": "16",
            },
            {
                "pair_index": "1", "decode_tokens_before": "512",
                "active_requests_before": "4", "max_num_seqs": "8",
            },
        )
        for index, values in enumerate(invalid_cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                core.prepare_frontend_semantic_load(
                    request_id=f"invalid-semantic-{index}", **values)
        with self.assertRaisesRegex(ValueError, "recorded twice"):
            core.prepare_frontend_semantic_load(
                request_id=request_id,
                pair_index="1",
                decode_tokens_before="512",
                active_requests_before="4",
                max_num_seqs="16",
            )

    def test_proxy_kv_control_overlap_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {"TEMPO_PD_PROXY_KV_CONTROL_OVERLAP": "1"},
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)
            self.assertTrue(core.proxy_kv_control_overlap)
            self.assertEqual(
                router.TRANSFER_EVIDENCE_OVERLAPPED,
                "eof_complete_after_control_overlap",
            )

    def test_app_endpoint_schemas_stay_canonical_after_build(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            app = router.build_app(config(), profile, allow_screen_profile=True)

            async def exercise_app():
                async with app.router.lifespan_context(app):
                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                    ) as http:
                        health = await http.get("/health")
                        decisions = await http.get("/tempo/decisions")
                        return health, decisions

            health, decisions = asyncio.run(exercise_app())
            self.assertEqual(health.json()["schema"], router.ROUTER_SCHEMA)
            self.assertIsNot(app.state.vllm_metrics, app.state.local)
            self.assertEqual(
                decisions.json()["schema"], router.ROUTER_SCHEMA)

    def test_fixed_contention_geometry_is_profile_independent_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            core = router.ElasticPDRouterCore(
                config(), profile, allow_screen_profile=True)

        local = core.decide(
            request_id="epd-local-ct-measured-local",
            prompt_tokens=4094,
            output_tokens=2,
        )
        remote = core.decide(
            request_id="epd-remote-ct-measured-remote",
            prompt_tokens=4094,
            output_tokens=2,
        )
        self.assertEqual(local.route, router.ElasticRoute.LOCAL)
        self.assertEqual(remote.route, router.ElasticRoute.REMOTE)
        self.assertEqual(local.reason, "fixed_always_local")
        self.assertEqual(remote.reason, "fixed_official_lmcache_remote")
        with self.assertRaisesRegex(ValueError, "no exact elastic profile row"):
            core.decide(
                request_id="epd-tempo-ct-measured-unprofiled",
                prompt_tokens=4094,
                output_tokens=2,
            )

    @staticmethod
    def _vllm_load_metrics():
        return (
            'vllm:num_requests_running{model_name="other",engine="0"} 99\n'
            'vllm:num_requests_running{model_name="served",engine="0"} 3\n'
            'vllm:num_requests_running{model_name="served",engine="1"} 2\n'
            'vllm:num_requests_waiting{model_name="served",engine="0"} 1\n'
            'vllm:num_requests_waiting{model_name="served",engine="1"} 0\n'
            'vllm:kv_cache_usage_perc{model_name="served",engine="0"} 0.25\n'
            'vllm:kv_cache_usage_perc{model_name="served",engine="1"} 0.40\n'
        )

    def test_vllm_load_metric_parser_is_engine_complete_and_fail_closed(self):
        payload = self._vllm_load_metrics()
        snapshot = router.parse_vllm_load_metrics(
            payload, served_model_name="served")
        self.assertEqual(snapshot["schema"], router.VLLM_LOAD_SNAPSHOT_SCHEMA)
        self.assertEqual(snapshot["decision_mode"], "observe_only")
        self.assertEqual(snapshot["engine_indices"], [0, 1])
        self.assertEqual(snapshot["num_requests_running"], 5)
        self.assertEqual(snapshot["num_requests_waiting"], 1)
        self.assertEqual(snapshot["kv_cache_usage_perc"], 0.40)

        duplicate = payload + (
            'vllm:num_requests_running{model_name="served",engine="0"} 3\n')
        missing_engine = payload.replace(
            'vllm:kv_cache_usage_perc{model_name="served",engine="1"} 0.40\n',
            "",
        )
        nonfinite = payload.replace(" 0.40\n", " NaN\n")
        nonintegral = payload.replace(
            'vllm:num_requests_running{model_name="served",engine="1"} 2\n',
            'vllm:num_requests_running{model_name="served",engine="1"} 2.5\n',
        )
        invalid_kv = payload.replace(" 0.40\n", " 1.01\n")
        invalid_engine = payload.replace('engine="1"', 'engine="01"')
        for name, invalid in (
            ("empty", ""),
            ("duplicate", duplicate),
            ("missing_engine", missing_engine),
            ("nonfinite", nonfinite),
            ("nonintegral", nonintegral),
            ("invalid_kv", invalid_kv),
            ("invalid_engine", invalid_engine),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                router.parse_vllm_load_metrics(
                    invalid, served_model_name="served")

    def test_analyzer_explicit_cold_contract_is_route_exact(self):
        contract = {
            "phase": "measured",
            "cache_keys_disjoint_across_blocks": True,
            "cache_keys_stable_across_warm_and_measured": False,
            "cache_key_isolation_scope": "phase_arm_replicate_and_item",
            "warm_preparation": "unmeasured_only_no_measured_key_reuse",
            "measured_cache_residency": "cold_disjoint_prompt_keys",
        }
        self.assertTrue(
            analyzer._explicit_cold_artifact_contract_valid(contract))
        self.assertFalse(analyzer._explicit_cold_artifact_contract_valid({
            **contract,
            "cache_keys_stable_across_warm_and_measured": True,
        }))

        local = {
            "benchmark_cold_measured": True,
            "decision_cache_residency": "unknown",
            "route": "decoder_local_chunked_prefill",
            "cache_residency": "confirmed_miss",
            "completion_cache_residency": "confirmed_miss",
            "lmcache_source_cached_tokens": None,
            "lmcache_source_full_hit_observed": None,
        }
        remote = {
            **local,
            "route": "official_lmcache_remote_prefill",
            "cache_residency": "prefill_only",
            "completion_cache_residency": "prefill_only",
            "lmcache_source_cached_tokens": 0,
            "lmcache_source_full_hit_observed": False,
        }
        self.assertTrue(analyzer._explicit_cold_completion_valid(local))
        self.assertTrue(analyzer._explicit_cold_completion_valid(remote))
        self.assertFalse(analyzer._explicit_cold_completion_valid({
            **remote,
            "lmcache_source_cached_tokens": 1,
        }))

        cold_affinity = {
            "frontend_pair_affinity_policy":
                "warm-prompt-sha256-owner-set-v2",
            "frontend_pair_affinity_required": False,
            "frontend_pair_affinity_owner_count_required": 1,
            "frontend_pair_affinity_hit": False,
            "frontend_pair_affinity_created": False,
            "frontend_pair_affinity_owner_indices": [],
            "frontend_pair_affinity_replica_count": 0,
            "frontend_pair_affinity_evidence_request_ids": [],
            "frontend_pair_affinity_registration_source":
                "reservation_or_unproven",
        }
        self.assertTrue(analyzer._tempo_pair_affinity_matches_mode(
            cold_affinity, cold_measured=True))
        self.assertFalse(analyzer._tempo_pair_affinity_matches_mode(
            {
                **cold_affinity,
                "frontend_pair_affinity_owner_indices": [0],
            },
            cold_measured=True,
        ))
    def test_vllm_load_snapshot_is_request_start_observe_only_evidence(self):
        class FakeResponse:
            def __init__(self, text):
                self.text = text
                self.status_checked = False

            def raise_for_status(self):
                self.status_checked = True

        class FakeClient:
            def __init__(self, text):
                self.response = FakeResponse(text)
                self.paths = []

            async def get(self, path):
                self.paths.append(path)
                return self.response

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {router.VLLM_LOAD_SNAPSHOT_MODE_ENV: "observe_only"},
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)
        request_id = "epd-tempo-r0-measured-item-0"
        local_client = FakeClient(self._vllm_load_metrics())
        snapshot = asyncio.run(core.prepare_vllm_load_snapshot(
            request_id, local_client))
        self.assertEqual(local_client.paths, ["/metrics"])
        self.assertTrue(local_client.response.status_checked)
        self.assertEqual(snapshot["endpoint"], "/metrics")
        self.assertEqual(snapshot["source"],
                         "local_decoder_prometheus_request_start")
        self.assertEqual(snapshot["decision_mode"], "observe_only")
        self.assertGreater(snapshot["sampled_ns"], 0)
        self.assertGreaterEqual(snapshot["fetch_ms"], 0)
        self.assertEqual(core.vllm_load_snapshot(request_id), snapshot)
        serialized = {
            "vllm_load_snapshot_schema": snapshot["schema"],
            "vllm_load_snapshot_source": snapshot["source"],
            "vllm_load_decision_mode": snapshot["decision_mode"],
            "vllm_load_endpoint": snapshot["endpoint"],
            "vllm_load_model_name": snapshot["model_name"],
            "vllm_load_engine_indices": snapshot["engine_indices"],
            "vllm_load_sampled_ns": snapshot["sampled_ns"],
            "vllm_load_fetch_ms": snapshot["fetch_ms"],
            "vllm_num_requests_running": snapshot["num_requests_running"],
            "vllm_num_requests_waiting": snapshot["num_requests_waiting"],
            "vllm_kv_cache_usage_perc": snapshot["kv_cache_usage_perc"],
        }
        self.assertTrue(analyzer._valid_vllm_load_snapshot(serialized))
        for field, invalid in (
            ("vllm_load_decision_mode", "policy_input"),
            ("vllm_num_requests_waiting", -1),
            ("vllm_kv_cache_usage_perc", 1.01),
        ):
            self.assertFalse(analyzer._valid_vllm_load_snapshot(
                {**serialized, field: invalid}))
        with self.assertRaises(ValueError):
            asyncio.run(core.prepare_vllm_load_snapshot(
                request_id, local_client))

    def test_vllm_load_snapshot_disabled_mode_has_no_request_rpc(self):
        class NoRequestClient:
            def __init__(self):
                self.called = False

            async def get(self, _path):
                self.called = True
                raise AssertionError("disabled load snapshot made an RPC")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {router.VLLM_LOAD_SNAPSHOT_MODE_ENV: "disabled"},
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)
        request_id = "epd-tempo-r0-measured-item-1"
        local_client = NoRequestClient()
        snapshot = asyncio.run(core.prepare_vllm_load_snapshot(
            request_id, local_client))
        self.assertFalse(local_client.called)
        self.assertEqual(snapshot["source"],
                         "explicitly_disabled_no_request_rpc")
        self.assertEqual(snapshot["decision_mode"], "disabled")
        self.assertEqual(snapshot["engine_indices"], [])
        self.assertEqual(snapshot["fetch_ms"], 0)
        self.assertIsNone(snapshot["num_requests_running"])
        serialized = {
            "vllm_load_snapshot_schema": snapshot["schema"],
            "vllm_load_snapshot_source": snapshot["source"],
            "vllm_load_decision_mode": snapshot["decision_mode"],
            "vllm_load_endpoint": snapshot["endpoint"],
            "vllm_load_model_name": snapshot["model_name"],
            "vllm_load_engine_indices": snapshot["engine_indices"],
            "vllm_load_sampled_ns": snapshot["sampled_ns"],
            "vllm_load_fetch_ms": snapshot["fetch_ms"],
            "vllm_num_requests_running": snapshot["num_requests_running"],
            "vllm_num_requests_waiting": snapshot["num_requests_waiting"],
            "vllm_kv_cache_usage_perc": snapshot["kv_cache_usage_perc"],
        }
        self.assertTrue(analyzer._valid_vllm_load_snapshot(serialized))
        self.assertFalse(analyzer._valid_vllm_load_snapshot({
            **serialized,
            "vllm_num_requests_running": 0,
        }))
        with patch.dict(
            "os.environ",
            {router.VLLM_LOAD_SNAPSHOT_MODE_ENV: "policy_input"},
            clear=False,
        ), self.assertRaises(ValueError):
            router.ElasticPDRouterCore(
                config(), profile, allow_screen_profile=True)

    def test_child_process_uses_canonical_stream_wrapper(self):
        command = [
            "python", "-m",
            "eval.sota_4node.run_tempo_pd_elastic_stream_metrics_v445",
        ]
        rewritten = client._canonicalize_child_command(command)
        self.assertEqual(
            rewritten[-1], "eval.sota_4node.run_tempo_pd_elastic_stream_metrics"
        )
        self.assertEqual(command[-1],
                         "eval.sota_4node.run_tempo_pd_elastic_stream_metrics_v445")

    def test_actual_vllm_batch_ceiling_fits_canonical_long_prompt_batch(self):
        command = node._vllm_command(
            Path("/repo/.vllm_venv/bin/vllm"),
            Path("/repo/models/Qwen2.5-7B-Instruct"),
            is_prefill=True,
            mode="tempo_elastic_pd_v445",
            pair=0,
            ports={
                "prefill_api": 12000,
                "decode_api": 14000,
                "proxy_http": 16000,
                "proxy_notify": 18000,
                "decoder_init": 20000,
                "decoder_alloc": 22000,
            },
        )
        index = command.index("--max-num-batched-tokens") + 1
        self.assertEqual(command[index], "32768")
        seq_index = command.index("--max-num-seqs") + 1
        self.assertEqual(command[seq_index], "8")
        policy_index = command.index("--scheduling-policy") + 1
        self.assertEqual(command[policy_index], "fcfs")
        connector = json.loads(
            command[command.index("--kv-transfer-config") + 1])
        self.assertTrue(
            connector["kv_connector_extra_config"][
                "lmcache.extra_config"][
                    "enable_cache_usage_details_in_response"])

    def test_actual_vllm_async_scheduling_is_explicitly_selectable(self):
        with patch.dict(
            "os.environ", {"TEMPO_VLLM_ASYNC_SCHEDULING": "1"}, clear=False
        ):
            command = node._vllm_command(
                Path("/repo/.vllm_venv/bin/vllm"),
                Path("/repo/models/Qwen2.5-7B-Instruct"),
                is_prefill=False,
                mode="tempo_elastic_pd_v445",
                pair=0,
                ports={
                    "prefill_api": 12000,
                    "decode_api": 14000,
                    "proxy_http": 16000,
                    "proxy_notify": 18000,
                    "decoder_init": 20000,
                    "decoder_alloc": 22000,
                },
            )
        self.assertIn("--async-scheduling", command)
        self.assertNotIn("--no-async-scheduling", command)

    def test_remote_decode_placement_is_explicit_and_bounded(self):
        hosts = ["prefill-0", "decode-0", "prefill-1", "decode-1"]
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                perf._decode_hosts(hosts, 0),
                ("paired", "decode-0", "decode-0"),
            )
        with patch.dict(
            "os.environ",
            {"TEMPO_PD_REMOTE_DECODE_PLACEMENT": "cross"},
            clear=False,
        ):
            self.assertEqual(
                perf._decode_hosts(hosts, 0),
                ("cross", "decode-0", "decode-1"),
            )
            self.assertEqual(
                perf._decode_hosts(hosts, 1),
                ("cross", "decode-1", "decode-0"),
            )
        with patch.dict(
            "os.environ",
            {"TEMPO_PD_REMOTE_DECODE_PLACEMENT": "long_decode_cross"},
            clear=False,
        ):
            self.assertEqual(
                perf._decode_hosts(hosts, 0),
                ("long_decode_cross", "decode-0", "decode-0"),
            )
            self.assertEqual(
                perf._decode_hosts(hosts, 1),
                ("long_decode_cross", "decode-1", "decode-1"),
            )
        with patch.dict(
            "os.environ",
            {"TEMPO_PD_REMOTE_DECODE_PLACEMENT": "arbitrary"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                perf._decode_hosts(hosts, 0)

    def test_long_decode_cross_proxy_and_header_selection_are_strict(self):
        hosts = ["prefill-0", "decode-0", "prefill-1", "decode-1"]
        original = [
            "python", "/repo/disagg_proxy_server.py",
            "--decoder-host", "decode-0",
            "--num-decoders", "1",
        ]
        command = perf._multi_decoder_proxy_command(
            original, hosts, Path("/repo/selector.py"))
        self.assertEqual(original[3], "decode-0")
        self.assertEqual(command[1], "/repo/selector.py")
        self.assertEqual(command[3], "decode-0,decode-1")
        self.assertEqual(command[5], "2")
        self.assertEqual(
            selecting_proxy.requested_decoder_index(
                {"x-tempo-pd-decoder-index": "1"}, 2, required=True),
            1,
        )
        with self.assertRaises(ValueError):
            selecting_proxy.requested_decoder_index({}, 2, required=True)
        with self.assertRaises(ValueError):
            selecting_proxy.requested_decoder_index(
                {"x-tempo-pd-decoder-index": "2"}, 2, required=True)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {
                    "TEMPO_PD_REMOTE_DECODE_PLACEMENT": "long_decode_cross",
                    "TEMPO_PD_LOCAL_DECODER_INDEX": "0",
                },
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)
            short = SimpleNamespace(
                request_id="epd-tempo-r0-measured-item-12",
                route=router.ElasticRoute.REMOTE,
                output_tokens=128,
            )
            long = SimpleNamespace(
                request_id="epd-tempo-r0-measured-item-16",
                route=router.ElasticRoute.REMOTE,
                output_tokens=256,
            )
            base_headers = {"Content-Type": "application/json"}
            short_headers = core.prepare_upstream_headers(
                short, base_headers)
            long_headers = core.prepare_upstream_headers(
                long, base_headers)
            self.assertEqual(
                short_headers[router.DECODER_INDEX_HEADER], "0")
            self.assertEqual(
                long_headers[router.DECODER_INDEX_HEADER], "1")
            self.assertNotIn(router.DECODER_INDEX_HEADER, base_headers)

    def test_actual_vllm_sequence_capacity_is_explicitly_selectable(self):
        with patch.dict(
            "os.environ", {"TEMPO_VLLM_MAX_NUM_SEQS": "16"}, clear=False
        ):
            command = node._vllm_command(
                Path("/repo/.vllm_venv/bin/vllm"),
                Path("/repo/models/Qwen2.5-7B-Instruct"),
                is_prefill=False,
                mode="tempo_elastic_pd_v445",
                pair=0,
                ports={
                    "prefill_api": 12000,
                    "decode_api": 14000,
                    "proxy_http": 16000,
                    "proxy_notify": 18000,
                    "decoder_init": 20000,
                    "decoder_alloc": 22000,
                },
            )
        seq_index = command.index("--max-num-seqs") + 1
        policy_index = command.index("--scheduling-policy") + 1
        self.assertEqual(command[seq_index], "16")
        self.assertEqual(command[policy_index], "fcfs")

    def test_evidence_positive_remote_priority_excludes_externality_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {
                    "TEMPO_VLLM_SCHEDULING_POLICY": "priority",
                    "TEMPO_PD_REMOTE_CATCHUP_PRIORITY": "0",
                    "TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY": "-1",
                },
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)
        def record(request_id, prompt_tokens, output_tokens):
            return SimpleNamespace(
                request_id=request_id,
                arm=router.ElasticExperimentArm.TEMPO,
                route=router.ElasticRoute.REMOTE,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        short = record(
            "epd-tempo-r0-measured-item-12", 512, 128)
        long_prompt = record(
            "epd-tempo-r0-measured-item-20", 4094, 16)
        externality = record(
            "epd-tempo-r0-measured-item-14", 2048, 128)
        self.assertEqual(
            core.prepare_upstream_payload(short, {})["priority"], -1)
        self.assertEqual(
            core.prepare_upstream_payload(long_prompt, {})["priority"], -1)
        self.assertNotIn(
            "priority", core.prepare_upstream_payload(externality, {}))
        self.assertEqual(
            core._request_upstream_priorities[short.request_id][
                "priority_class"],
            "strong_remote_catchup",
        )

    def test_remote_long_decode_can_receive_bounded_catchup_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {
                    "TEMPO_VLLM_SCHEDULING_POLICY": "priority",
                    "TEMPO_PD_REMOTE_CATCHUP_PRIORITY": "-1",
                    "TEMPO_PD_REMOTE_CATCHUP_MIN_OUTPUT_TOKENS": "256",
                },
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)
                command = node._vllm_command(
                    Path("/repo/.vllm_venv/bin/vllm"),
                    Path("/repo/models/Qwen2.5-7B-Instruct"),
                    is_prefill=False,
                    mode="tempo_elastic_pd_v445",
                    pair=0,
                    ports={
                        "prefill_api": 12000,
                        "decode_api": 14000,
                        "proxy_http": 16000,
                        "proxy_notify": 18000,
                        "decoder_init": 20000,
                        "decoder_alloc": 22000,
                    },
                )
            record = SimpleNamespace(
                request_id="epd-tempo-r0-measured-item-17",
                arm=router.ElasticExperimentArm.TEMPO,
                route=router.ElasticRoute.REMOTE,
                output_tokens=256,
            )
            payload = {"model": "model", "max_tokens": 256}
            prepared = core.prepare_upstream_payload(record, payload)
            self.assertNotIn("priority", payload)
            self.assertEqual(prepared["priority"], -1)
            self.assertEqual(
                command[command.index("--scheduling-policy") + 1], "priority")

            short = SimpleNamespace(
                request_id="epd-tempo-r0-measured-item-16",
                arm=router.ElasticExperimentArm.TEMPO,
                route=router.ElasticRoute.REMOTE,
                output_tokens=128,
            )
            self.assertNotIn(
                "priority", core.prepare_upstream_payload(short, payload))

    def test_long_remote_priority_precedes_general_remote_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {
                    "TEMPO_VLLM_SCHEDULING_POLICY": "priority",
                    "TEMPO_PD_REMOTE_CATCHUP_PRIORITY": "-1",
                    "TEMPO_PD_REMOTE_CATCHUP_MIN_OUTPUT_TOKENS": "128",
                    "TEMPO_PD_LONG_REMOTE_CATCHUP_PRIORITY": "-2",
                    "TEMPO_PD_LONG_REMOTE_CATCHUP_MIN_PROMPT_TOKENS": "2048",
                },
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)

        medium = SimpleNamespace(
            request_id="epd-tempo-r0-measured-item-12",
            arm=router.ElasticExperimentArm.TEMPO,
            route=router.ElasticRoute.REMOTE,
            output_tokens=128,
        )
        long = SimpleNamespace(
            request_id="epd-tempo-r0-measured-item-16",
            arm=router.ElasticExperimentArm.TEMPO,
            route=router.ElasticRoute.REMOTE,
            output_tokens=256,
            prompt_tokens=2048,
        )
        short_prompt_long = SimpleNamespace(
            request_id="epd-tempo-r0-measured-item-17",
            arm=router.ElasticExperimentArm.TEMPO,
            route=router.ElasticRoute.REMOTE,
            output_tokens=256,
            prompt_tokens=1230,
        )
        local_long = SimpleNamespace(
            request_id="epd-tempo-r0-measured-item-18",
            arm=router.ElasticExperimentArm.TEMPO,
            route=router.ElasticRoute.LOCAL,
            output_tokens=256,
            prompt_tokens=2048,
        )
        payload = {"model": "model"}
        self.assertEqual(
            core.prepare_upstream_payload(medium, payload)["priority"], -1)
        self.assertEqual(
            core.prepare_upstream_payload(long, payload)["priority"], -2)
        self.assertEqual(
            core.prepare_upstream_payload(
                short_prompt_long, payload)["priority"], -1)
        self.assertEqual(
            core._request_upstream_priorities[short_prompt_long.request_id][
                "priority_class"], "remote_catchup")
        self.assertNotIn(
            "priority", core.prepare_upstream_payload(local_long, payload))
        self.assertEqual(
            core._request_upstream_priorities[long.request_id][
                "priority_class"],
            "long_remote_catchup",
        )

    def test_adaptive_fabric_congestion_suppresses_remote_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {
                    "TEMPO_VLLM_SCHEDULING_POLICY": "priority",
                    "TEMPO_PD_LONG_REMOTE_CATCHUP_PRIORITY": "-1",
                    "TEMPO_PD_LONG_REMOTE_CATCHUP_MIN_PROMPT_TOKENS": "1230",
                },
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)
        core.pressure_mode = router.PRESSURE_ADAPTIVE_MODE
        congested = SimpleNamespace(
            request_id="epd-tempo-r0-measured-item-18",
            arm=router.ElasticExperimentArm.TEMPO,
            route=router.ElasticRoute.REMOTE,
            output_tokens=256,
            prompt_tokens=2048,
        )
        idle = SimpleNamespace(
            request_id="epd-tempo-r0-measured-item-19",
            arm=router.ElasticExperimentArm.TEMPO,
            route=router.ElasticRoute.REMOTE,
            output_tokens=256,
            prompt_tokens=2048,
        )
        core._request_pressure_snapshots[congested.request_id] = {
            "fabric_congested": True,
        }
        core._request_pressure_snapshots[idle.request_id] = {
            "fabric_congested": False,
        }
        prepared = core.prepare_upstream_payload(congested, {"model": "model"})
        self.assertNotIn("priority", prepared)
        evidence = core._request_upstream_priorities[congested.request_id]
        self.assertTrue(evidence["fabric_congested"])
        self.assertTrue(evidence["fabric_congestion_suppressed"])
        self.assertEqual(evidence["priority_class"], "request_default")
        self.assertEqual(
            core.prepare_upstream_payload(idle, {"model": "model"})["priority"],
            -1,
        )
        self.assertFalse(
            core._request_upstream_priorities[idle.request_id][
                "fabric_congestion_suppressed"])

    def test_remote_catchup_priority_is_strictly_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {
                    "TEMPO_VLLM_SCHEDULING_POLICY": "priority",
                    "TEMPO_PD_REMOTE_CATCHUP_PRIORITY": "-2",
                },
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)
            self.assertEqual(core.remote_catchup_priority, -2)
            with patch.dict(
                "os.environ",
                {"TEMPO_PD_REMOTE_CATCHUP_PRIORITY": "-3"},
                clear=False,
            ):
                with self.assertRaises(ValueError):
                    router.ElasticPDRouterCore(
                        config(), profile, allow_screen_profile=True)

    def test_medium_remote_catchup_precedes_general_remote_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {
                    "TEMPO_VLLM_SCHEDULING_POLICY": "priority",
                    "TEMPO_PD_REMOTE_CATCHUP_PRIORITY": "-1",
                    "TEMPO_PD_REMOTE_CATCHUP_MIN_OUTPUT_TOKENS": "256",
                    "TEMPO_PD_MEDIUM_REMOTE_CATCHUP_PRIORITY": "-2",
                },
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)
            medium = SimpleNamespace(
                request_id="epd-tempo-r0-measured-item-13",
                arm=router.ElasticExperimentArm.TEMPO,
                route=router.ElasticRoute.REMOTE,
                prompt_tokens=1230,
                output_tokens=128,
            )
            prepared = core.prepare_upstream_payload(
                medium, {"model": "model", "max_tokens": 128})
            self.assertEqual(prepared["priority"], -2)
            self.assertEqual(
                core._request_upstream_priorities[medium.request_id][
                    "priority_class"],
                "medium_remote_catchup",
            )
            for prompt_tokens in (512, 2048):
                sibling = SimpleNamespace(
                    request_id=(
                        f"epd-tempo-r0-measured-p{prompt_tokens}"),
                    arm=router.ElasticExperimentArm.TEMPO,
                    route=router.ElasticRoute.REMOTE,
                    prompt_tokens=prompt_tokens,
                    output_tokens=128,
                )
                self.assertEqual(
                    core.prepare_upstream_payload(
                        sibling,
                        {"model": "model", "max_tokens": 128},
                    )["priority"],
                    -2,
                )
            late_4k = SimpleNamespace(
                request_id="epd-tempo-r0-measured-item-22",
                arm=router.ElasticExperimentArm.TEMPO,
                route=router.ElasticRoute.REMOTE,
                prompt_tokens=4094,
                output_tokens=128,
            )
            self.assertNotIn(
                "priority",
                core.prepare_upstream_payload(
                    late_4k, {"model": "model", "max_tokens": 128}),
            )


    def test_output64_median_guard_precedes_remote_catchup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {
                    "TEMPO_VLLM_SCHEDULING_POLICY": "priority",
                    "TEMPO_PD_REMOTE_CATCHUP_PRIORITY": "0",
                    "TEMPO_PD_MEDIAN_GUARD_PRIORITY": "-2",
                },
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)
            median = SimpleNamespace(
                request_id="epd-tempo-r0-measured-item-08",
                arm=router.ElasticExperimentArm.TEMPO,
                route=router.ElasticRoute.LOCAL,
                output_tokens=64,
            )
            payload = {"model": "model", "max_tokens": 64}
            self.assertEqual(
                core.prepare_upstream_payload(median, payload)["priority"], -2)
            nonmedian = SimpleNamespace(
                request_id="epd-tempo-r0-measured-item-07",
                arm=router.ElasticExperimentArm.TEMPO,
                route=router.ElasticRoute.LOCAL,
                output_tokens=32,
            )
            self.assertNotIn(
                "priority", core.prepare_upstream_payload(nonmedian, payload))

    def test_prefiller_retains_local_cpu_cache_only(self):
        ports = {
            "prefill_api": 12000,
            "decode_api": 14000,
            "proxy_http": 16000,
            "proxy_notify": 18000,
            "decoder_init": 20000,
            "decoder_alloc": 22000,
        }
        prefill = node._config_text(
            is_prefill=True, prefill_host="prefill",
            decode_host="decode", ports=ports)
        decoder = node._config_text(
            is_prefill=False, prefill_host="prefill",
            decode_host="decode", ports=ports)
        self.assertIn("local_cpu: True", prefill)
        self.assertIn("max_local_cpu_size: 16", prefill)
        self.assertIn(
            'retrieve_locations: ["LocalCPUBackend"]', prefill)
        self.assertIn("save_unfull_chunk: true", prefill)
        self.assertIn("local_cpu: False", decoder)
        self.assertNotIn("LocalCPUBackend", decoder)

    def test_prompt_key_is_stable_across_warm_and_measured(self):
        class FakeTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return list(text.encode("ascii"))

            @staticmethod
            def decode(
                token_ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ):
                del skip_special_tokens, clean_up_tokenization_spaces
                return bytes(token_ids).decode("ascii")

        old_tokenizer = client._prior._TOKENIZER
        client._prior._TOKENIZER = FakeTokenizer()
        try:
            rows = [{
                "request_id": "base",
                "prompt": "x" * 400,
                "max_tokens": 64,
            }]
            warm = client._derive(
                rows, arm="tempo", replicate=0,
                phase="warm", offset=400)
            measured = client._derive(
                rows, arm="tempo", replicate=1,
                phase="measured", offset=100)
            other_arm = client._derive(
                rows, arm="remote", replicate=0,
                phase="warm", offset=500)
        finally:
            client._prior._TOKENIZER = old_tokenizer
        self.assertEqual(warm[0]["prompt"], measured[0]["prompt"])
        self.assertNotEqual(warm[0]["request_id"], measured[0]["request_id"])
        self.assertNotEqual(warm[0]["prompt"], other_arm[0]["prompt"])

    def test_real_source_hit_establishes_and_preserves_p_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            core = router.ElasticPDRouterCore(
                config(), profile, allow_screen_profile=True)

            def complete(
                request_id, *, max_tokens, prompt_key,
                cached_tokens=None,
            ):
                core.prepare_prompt_namespace(request_id, prompt_key)
                record = core.decide(
                    request_id=request_id,
                    prompt_tokens=10,
                    output_tokens=max_tokens,
                )
                core.mark_upstream_started(request_id)
                core.mark_first_response_chunk(request_id)
                if record.route is router.ElasticRoute.REMOTE:
                    headers = {
                        "X-Tempo-LMCache-PD-Transfer": "complete",
                        "X-Tempo-LMCache-PD-Prompt-Tokens": "11",
                        "X-Tempo-LMCache-PD-Cached-Tokens": str(cached_tokens),
                        "X-Tempo-LMCache-PD-KV-Bytes": "1100",
                        "X-Tempo-LMCache-PD-Request-Id": request_id,
                    }
                else:
                    headers = {}
                event = core.observe_backend_completion(
                    request_id, route=record.route.value,
                    upstream_headers=headers)
                core.complete(request_id)
                return record, event

            seed, seed_event = complete(
                "epd-remote-r0-warm-seed-item-00",
                max_tokens=2, prompt_key="remote-key",
                cached_tokens=0)
            self.assertIs(seed.route, router.ElasticRoute.REMOTE)
            self.assertIsNone(seed_event)
            _, probe_event = complete(
                "epd-remote-r0-warm-item-00",
                max_tokens=64, prompt_key="remote-key",
                cached_tokens=10)
            self.assertIs(
                probe_event.residency, router.CacheResidency.P_ONLY)
            measured, measured_event = complete(
                "epd-remote-r0-measured-item-00",
                max_tokens=64, prompt_key="remote-key",
                cached_tokens=10)
            self.assertIs(
                measured.cache_residency, router.CacheResidency.P_ONLY)
            self.assertIs(
                measured_event.residency, router.CacheResidency.P_ONLY)

            complete(
                "epd-local-r0-warm-seed-item-00",
                max_tokens=2, prompt_key="local-key",
                cached_tokens=0)
            complete(
                "epd-local-r0-warm-item-00",
                max_tokens=64, prompt_key="local-key",
                cached_tokens=10)
            local, local_event = complete(
                "epd-local-r0-measured-item-00",
                max_tokens=64, prompt_key="local-key")
            self.assertIs(local.route, router.ElasticRoute.LOCAL)
            self.assertIs(
                local.cache_residency, router.CacheResidency.P_ONLY)
            self.assertIs(
                local_event.residency, router.CacheResidency.P_ONLY)

            records = {
                row["request_id"]: row for row in core.records()}
            self.assertEqual(
                records[measured.request_id][
                    "lmcache_source_cached_tokens"],
                10)
            self.assertTrue(
                records[measured.request_id][
                    "lmcache_source_full_hit_observed"])
            self.assertIsNone(
                records[local.request_id][
                    "lmcache_source_cached_tokens"])

    def test_explicit_p_only_measurement_overrides_global_cold_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ", {router.COLD_MEASURED_ENV: "1"}, clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)

        def complete(request_id, *, prompt_key, cached_tokens):
            core.prepare_prompt_namespace(request_id, prompt_key)
            record = core.decide(
                request_id=request_id, prompt_tokens=10, output_tokens=64)
            core.mark_upstream_started(request_id)
            core.mark_first_response_chunk(request_id)
            event = core.observe_backend_completion(
                request_id,
                route=record.route.value,
                upstream_headers={
                    "X-Tempo-LMCache-PD-Transfer": "complete",
                    "X-Tempo-LMCache-PD-Prompt-Tokens": "11",
                    "X-Tempo-LMCache-PD-Cached-Tokens": str(cached_tokens),
                    "X-Tempo-LMCache-PD-KV-Bytes": "1100",
                    "X-Tempo-LMCache-PD-Request-Id": request_id,
                },
            )
            core.complete(request_id)
            return record, event

        _, seed_event = complete(
            "epd-remote-r0-warm-seed-item-00",
            prompt_key="p-only-key",
            cached_tokens=0,
        )
        self.assertIsNone(seed_event)
        complete(
            "epd-remote-r0-warm-item-00",
            prompt_key="p-only-key",
            cached_tokens=10,
        )
        measured, event = complete(
            "epd-remote-r0-cache-p-only-measured-item-00",
            prompt_key="p-only-key",
            cached_tokens=10,
        )
        self.assertEqual(measured.reason, "fixed_official_lmcache_remote")
        self.assertIs(
            measured.cache_residency, router.CacheResidency.P_ONLY)
        self.assertIs(event.residency, router.CacheResidency.P_ONLY)
        row = {
            value["request_id"]: value for value in core.records()
        }[measured.request_id]
        self.assertEqual(row["request_cache_contract"], "p_only")
        self.assertEqual(row["decision_cache_residency"], "prefill_only")

        missing_id = "epd-remote-r0-cache-p-only-measured-item-01"
        core.prepare_prompt_namespace(missing_id, "missing-p-only-key")
        with self.assertRaisesRegex(ValueError, "lacks completed cache evidence"):
            core.decide(
                request_id=missing_id, prompt_tokens=10, output_tokens=64)

    def test_explicit_miss_is_unseen_at_commit_and_zero_hit_at_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            core = router.ElasticPDRouterCore(
                config(), profile, allow_screen_profile=True)

        request_id = "epd-local-r0-cache-miss-measured-item-00"
        core.prepare_prompt_namespace(request_id, "fresh-explicit-miss")
        record = core.decide(
            request_id=request_id, prompt_tokens=10, output_tokens=64)
        self.assertIs(record.cache_residency, router.CacheResidency.MISS)
        core.mark_upstream_started(request_id)
        core.mark_first_response_chunk(request_id)
        event = core.observe_backend_completion(
            request_id, route=record.route.value, upstream_headers={})
        self.assertIs(event.residency, router.CacheResidency.MISS)
        core.complete(request_id)

        repeated_id = "epd-local-r1-cache-miss-measured-item-00"
        core.prepare_prompt_namespace(repeated_id, "fresh-explicit-miss")
        with self.assertRaisesRegex(ValueError, "previously observed"):
            core.decide(
                request_id=repeated_id, prompt_tokens=10, output_tokens=64)


    def test_explicit_cold_measured_accepts_disjoint_local_and_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_payload()))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {router.COLD_MEASURED_ENV: "1"},
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)

            def complete(request_id, *, prompt_key):
                core.prepare_prompt_namespace(request_id, prompt_key)
                record = core.decide(
                    request_id=request_id,
                    prompt_tokens=10,
                    output_tokens=64,
                )
                self.assertIs(
                    record.cache_residency, router.CacheResidency.UNKNOWN)
                core.mark_upstream_started(request_id)
                core.mark_first_response_chunk(request_id)
                headers = {}
                if record.route is router.ElasticRoute.REMOTE:
                    headers = {
                        "X-Tempo-LMCache-PD-Transfer": "complete",
                        "X-Tempo-LMCache-PD-Prompt-Tokens": "11",
                        "X-Tempo-LMCache-PD-Cached-Tokens": "0",
                        "X-Tempo-LMCache-PD-KV-Bytes": "1100",
                        "X-Tempo-LMCache-PD-Request-Id": request_id,
                    }
                event = core.observe_backend_completion(
                    request_id, route=record.route.value,
                    upstream_headers=headers)
                core.complete(request_id)
                return record, event

            local, local_event = complete(
                "epd-local-r0-measured-item-00",
                prompt_key="cold-local-key")
            remote, remote_event = complete(
                "epd-remote-r0-measured-item-01",
                prompt_key="cold-remote-key")
            self.assertIs(local.route, router.ElasticRoute.LOCAL)
            self.assertIs(
                local_event.residency, router.CacheResidency.MISS)
            self.assertIs(remote.route, router.ElasticRoute.REMOTE)
            self.assertIs(
                remote_event.residency, router.CacheResidency.P_ONLY)
            tempo, tempo_event = complete(
                "epd-tempo-r0-measured-item-02",
                prompt_key="cold-tempo-key")
            self.assertIs(tempo.route, router.ElasticRoute.REMOTE)
            self.assertIs(
                tempo_event.residency, router.CacheResidency.P_ONLY)

            records = {
                row["request_id"]: row for row in core.records()}
            for record in (local, remote, tempo):
                self.assertTrue(
                    records[record.request_id]["benchmark_cold_measured"])
                self.assertEqual(
                    records[record.request_id]["profile_remote_backend"],
                    "lmcache-ucx")
            for record in (local, remote):
                self.assertIsNone(
                    records[record.request_id][
                        "cold_unknown_remote_candidate"])
                self.assertIsNone(
                    records[record.request_id][
                        "cold_unknown_remote_admitted"])
            self.assertTrue(
                records[tempo.request_id]["cold_unknown_remote_candidate"])
            self.assertTrue(
                records[tempo.request_id]["cold_unknown_remote_admitted"])
            self.assertEqual(
                records[remote.request_id]["lmcache_source_cached_tokens"], 0)
            self.assertFalse(
                records[remote.request_id][
                    "lmcache_source_full_hit_observed"])
    def test_selected_decoder_reuse_records_p_only_to_both_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            payload = profile_payload()
            payload["controller"]["remote_kv_budget_bytes"] = 4000
            payload["rows"][0].update({
                "prompt_tokens": 32,
                "remote_kv_bytes": 3200,
            })
            path.write_text(json.dumps(payload))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {
                    "TEMPO_VLLM_DECODER_PREFIX_CACHING": "1",
                    "TEMPO_PD_DECODER_REUSE_ITEMS": "0",
                },
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)

            def complete(
                request_id, *, max_tokens, source_cached_tokens=None,
                decoder_cached_tokens=None,
            ):
                core.prepare_prompt_namespace(request_id, "a" * 64)
                record = core.decide(
                    request_id=request_id,
                    prompt_tokens=32,
                    output_tokens=max_tokens,
                )
                core.prepare_upstream_payload(record, {
                    "stream": True,
                    "stream_options": {"include_usage": True},
                })
                core.mark_upstream_started(request_id)
                core.mark_first_response_chunk(request_id)
                headers = {}
                if record.route is router.ElasticRoute.REMOTE:
                    headers = {
                        "X-Tempo-LMCache-PD-Transfer": "complete",
                        "X-Tempo-LMCache-PD-Prompt-Tokens": "33",
                        "X-Tempo-LMCache-PD-Cached-Tokens": str(
                            source_cached_tokens),
                        "X-Tempo-LMCache-PD-KV-Bytes": "3300",
                        "X-Tempo-LMCache-PD-Request-Id": request_id,
                    }
                    core.observe_backend_stream_chunk(
                        request_id,
                        route=record.route.value,
                        chunk=_decoder_usage_stream(
                            prompt_tokens=33,
                            completion_tokens=max_tokens - 1,
                            cached_tokens=32,
                            local_cached_tokens=0,
                            external_cached_tokens=32,
                        ),
                    )
                else:
                    core.observe_backend_stream_chunk(
                        request_id,
                        route=record.route.value,
                        chunk=_decoder_usage_stream(
                            prompt_tokens=32,
                            completion_tokens=max_tokens,
                            cached_tokens=decoder_cached_tokens,
                        ),
                    )
                event = core.observe_backend_completion(
                    request_id,
                    route=record.route.value,
                    upstream_headers=headers,
                )
                core.complete(request_id)
                return record, event

            complete(
                "epd-local-r0-warm-seed-item-00",
                max_tokens=2,
                source_cached_tokens=0,
            )
            complete(
                "epd-local-r0-warm-item-00",
                max_tokens=64,
                source_cached_tokens=32,
            )
            measured, event = complete(
                "epd-local-r0-measured-item-00",
                max_tokens=64,
                decoder_cached_tokens=16,
            )
            self.assertIs(
                measured.cache_residency, router.CacheResidency.P_ONLY)
            self.assertIs(event.residency, router.CacheResidency.BOTH)

            row = {
                value["request_id"]: value for value in core.records()
            }[measured.request_id]
            self.assertEqual(
                row["decision_cache_residency"], "prefill_only")
            self.assertEqual(
                row["completion_cache_residency"], "prefill_and_decode")
            self.assertTrue(row["decoder_prefix_caching"])
            self.assertTrue(
                row["decoder_cache_reuse_enabled_for_request"])
            self.assertEqual(row["decoder_cache_reuse_items"], [0])
            self.assertEqual(
                row["upstream_cache_salt"],
                router.cache_reuse.namespace_cache_salt(
                    arm="always_local", prompt_key="a" * 64),
            )
            self.assertEqual(row["decoder_prefix_cached_tokens"], 16)
            self.assertTrue(row["decoder_prefix_full_hit_observed"])
            self.assertFalse(row["decoder_prefix_read_skipped"])

    def test_d_only_and_both_require_exact_vllm_apc_hit_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            payload = profile_payload()
            payload["controller"]["remote_kv_budget_bytes"] = 4000
            payload["rows"][0].update({
                "prompt_tokens": 32,
                "remote_kv_bytes": 3200,
            })
            path.write_text(json.dumps(payload))
            profile = load_elastic_profile(path)
            with patch.dict(
                "os.environ",
                {
                    "TEMPO_VLLM_DECODER_PREFIX_CACHING": "1",
                    "TEMPO_PD_DECODER_REUSE_ITEMS": "all",
                },
                clear=False,
            ):
                core = router.ElasticPDRouterCore(
                    config(), profile, allow_screen_profile=True)

        def complete(
            request_id, *, prompt_key, decoder_cached=None,
            source_cached=None, remote_decoder_local=None,
        ):
            core.prepare_prompt_namespace(request_id, prompt_key)
            record = core.decide(
                request_id=request_id,
                prompt_tokens=32,
                output_tokens=64,
            )
            prepared = core.prepare_upstream_payload(record, {
                "stream": True,
                "stream_options": {"include_usage": True},
            })
            core.mark_upstream_started(request_id)
            core.mark_first_response_chunk(request_id)
            headers = {}
            if record.route is router.ElasticRoute.LOCAL:
                core.observe_backend_stream_chunk(
                    request_id,
                    route=record.route.value,
                    chunk=_decoder_usage_stream(
                        prompt_tokens=32,
                        completion_tokens=64,
                        cached_tokens=decoder_cached,
                    ),
                )
            else:
                if remote_decoder_local is None:
                    remote_decoder_local = (
                        router.full_prefix_hit_tokens(32)
                        if any(marker in request_id for marker in (
                            "-cache-d-only-measured-",
                            "-cache-both-measured-",
                        )) else 0
                    )
                headers = {
                    "X-Tempo-LMCache-PD-Transfer": "complete",
                    "X-Tempo-LMCache-PD-Prompt-Tokens": "33",
                    "X-Tempo-LMCache-PD-Cached-Tokens": str(source_cached),
                    "X-Tempo-LMCache-PD-KV-Bytes": "3300",
                    "X-Tempo-LMCache-PD-Request-Id": request_id,
                }
                core.observe_backend_stream_chunk(
                    request_id,
                    route=record.route.value,
                    chunk=_decoder_usage_stream(
                        prompt_tokens=33,
                        completion_tokens=63,
                        cached_tokens=32,
                        local_cached_tokens=remote_decoder_local,
                        external_cached_tokens=32 - remote_decoder_local,
                    ),
                )
            event = core.observe_backend_completion(
                request_id,
                route=record.route.value,
                upstream_headers=headers,
            )
            core.complete(request_id)
            return record, event, prepared

        _, d_seed, seed_payload = complete(
            "epd-tempo-r0-warm-cache-d-seed-item-00",
            prompt_key="b" * 64,
            decoder_cached=0,
        )
        self.assertIsNone(d_seed)
        self.assertEqual(
            seed_payload["vllm_xargs"][
                router.VLLM_SKIP_LOCAL_PREFIX_READ_XARG], 1)
        _, d_probe, probe_payload = complete(
            "epd-tempo-r0-warm-cache-d-probe-item-00",
            prompt_key="b" * 64,
            decoder_cached=16,
        )
        self.assertIs(d_probe.residency, router.CacheResidency.D_ONLY)
        self.assertEqual(
            probe_payload["vllm_xargs"][
                router.VLLM_SKIP_LOCAL_PREFIX_READ_XARG], 0)
        d_measured, d_event, _ = complete(
            "epd-tempo-r0-cache-d-only-measured-item-00",
            prompt_key="b" * 64,
            decoder_cached=16,
        )
        self.assertIs(d_measured.route, router.ElasticRoute.LOCAL)
        self.assertIs(d_event.residency, router.CacheResidency.D_ONLY)

        complete(
            "epd-tempo-r0-warm-seed-item-01",
            prompt_key="c" * 64,
            source_cached=0,
        )
        _, p_event, _ = complete(
            "epd-tempo-r0-warm-item-01",
            prompt_key="c" * 64,
            source_cached=32,
        )
        self.assertIs(p_event.residency, router.CacheResidency.P_ONLY)
        # The live protocol resets decoder APC here while preserving LMCache.
        complete(
            "epd-tempo-r0-warm-cache-d-seed-item-01",
            prompt_key="c" * 64,
            decoder_cached=0,
        )
        _, both_probe, _ = complete(
            "epd-tempo-r0-warm-cache-d-probe-item-01",
            prompt_key="c" * 64,
            decoder_cached=16,
        )
        self.assertIs(both_probe.residency, router.CacheResidency.BOTH)
        both_measured, both_event, _ = complete(
            "epd-tempo-r0-cache-both-measured-item-01",
            prompt_key="c" * 64,
            decoder_cached=16,
        )
        self.assertIs(both_measured.route, router.ElasticRoute.LOCAL)
        self.assertIs(both_event.residency, router.CacheResidency.BOTH)

        rows = {row["request_id"]: row for row in core.records()}
        self.assertEqual(
            rows[d_measured.request_id]["request_cache_contract"],
            "d_only",
        )
        self.assertEqual(
            rows[both_measured.request_id]["request_cache_contract"],
            "both",
        )
        self.assertTrue(
            rows[both_measured.request_id][
                "decoder_prefix_full_hit_observed"])

        # A fixed remote arm is the counterfactual that exposes why stock
        # cached_tokens is insufficient: total=32 in all cases, but D_ONLY and
        # BOTH are real only when decoder-local APC contributes the exact
        # preparation-proven full(P)=16 prefix.
        complete(
            "epd-remote-r0-warm-cache-d-seed-item-02",
            prompt_key="d" * 64,
            decoder_cached=0,
        )
        complete(
            "epd-remote-r0-warm-cache-d-probe-item-02",
            prompt_key="d" * 64,
            decoder_cached=16,
        )
        remote_d, _, remote_d_payload = complete(
            "epd-remote-r0-cache-d-only-measured-item-02",
            prompt_key="d" * 64,
            source_cached=0,
        )
        self.assertIs(remote_d.route, router.ElasticRoute.REMOTE)
        self.assertNotIn("vllm_xargs", remote_d_payload)
        self.assertEqual(
            remote_d_payload[
                router.PROXY_DECODER_SKIP_LOCAL_PREFIX_READ_FIELD],
            0,
        )

        complete(
            "epd-remote-r0-warm-seed-item-03",
            prompt_key="e" * 64,
            source_cached=0,
        )
        complete(
            "epd-remote-r0-warm-item-03",
            prompt_key="e" * 64,
            source_cached=32,
        )
        complete(
            "epd-remote-r0-warm-cache-d-seed-item-03",
            prompt_key="e" * 64,
            decoder_cached=0,
        )
        complete(
            "epd-remote-r0-warm-cache-d-probe-item-03",
            prompt_key="e" * 64,
            decoder_cached=16,
        )
        remote_both, _, _ = complete(
            "epd-remote-r0-cache-both-measured-item-03",
            prompt_key="e" * 64,
            source_cached=32,
        )
        rows = {row["request_id"]: row for row in core.records()}
        for measured in (remote_d, remote_both):
            row = rows[measured.request_id]
            self.assertEqual(row["decoder_prefix_cached_tokens"], 16)
            self.assertEqual(row["decoder_total_cached_tokens"], 32)
            self.assertEqual(row["decoder_external_cached_tokens"], 16)
            self.assertEqual(row["decoder_prefix_usage_prompt_tokens"], 33)
            self.assertTrue(row["decoder_prefix_full_hit_observed"])



if __name__ == "__main__":
    unittest.main()
