from __future__ import annotations

import copy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from eval.sota_4node import analyze_tempo_pd_independent_validation as analyzer
from eval.sota_4node import build_tempo_pd_independent_validation_manifest as manifest_builder
from eval.sota_4node import build_tempo_pd_independent_validation_run_contract as contract_builder
from eval.sota_4node import promote_tempo_pd_profiles_for_independent_validation as promotion
from eval.sota_4node import run_tempo_pd_independent_validation_client as client
from eval.sota_4node import vllm_lmcache_pd_independent_validation_node as node
from tempo.pd_contention_workload import (
    ForegroundArm,
    Tenant,
    VALIDATION_FOREGROUND_GEOMETRIES,
)


class _Factory:
    def prompt(self, key, prompt_tokens):
        prompt = f"held-out-{prompt_tokens}-{key}"
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return prompt, digest


class IndependentValidationTest(unittest.TestCase):
    def setUp(self):
        self.prereg_path = manifest_builder.DEFAULT_PREREGISTRATION
        self.prereg_sha = hashlib.sha256(
            self.prereg_path.read_bytes()).hexdigest()
        self.prereg = json.loads(
            self.prereg_path.read_text(encoding="utf-8"))

    def test_preregistration_freezes_original_gates(self):
        loaded = manifest_builder._load_preregistration(
            self.prereg_path, expected_sha256=self.prereg_sha)
        self.assertEqual(loaded, self.prereg)
        gates = loaded["success_gates"]
        self.assertEqual(
            gates["minimum_pooled_median_e2e_gain_vs_strongest_fixed"],
            0.10)
        self.assertEqual(
            gates["minimum_pooled_median_e2e_gain_vs_predictor"], 0.05)
        self.assertEqual(
            gates["minimum_request_goodput_gain_vs_strongest_fixed"], 0.05)
        self.assertEqual(
            gates[
                "minimum_overall_paired_win_fraction_vs_strongest_fixed"],
            0.75)
        self.assertEqual(
            gates[
                "minimum_group_paired_win_fraction_vs_group_strongest_fixed"],
            0.60)
        self.assertEqual(loaded["workload"]["replicate_ids"], [2, 3, 4, 5])
        self.assertEqual(
            loaded["workload"]["paired_foreground_samples_per_group"], 16)

    def test_mutated_threshold_fails_even_with_matching_digest(self):
        value = copy.deepcopy(self.prereg)
        value["success_gates"][
            "minimum_pooled_median_e2e_gain_vs_strongest_fixed"] = 0.09
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "prereg.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "thresholds differ"):
                manifest_builder._load_preregistration(
                    path, expected_sha256=digest)

    def _manifest(self):
        workload = self.prereg["workload"]
        return {
            "phase_order": workload["phase_order"],
            "phase_duration_ms": workload["phase_duration_ms"],
            "foreground_rate_per_s": workload["foreground_rate_per_s"],
            "background_rates_per_s": {
                "decoder_hot": 1.0,
                "cold_remote_hot": 1.0,
                "kv_remote_hot": 1.0,
            },
            "replicate_ids": workload["replicate_ids"],
            "arm_order_by_replicate": [
                {
                    "replicate": replicate,
                    "arms": workload["arm_order_by_replicate"][str(replicate)],
                }
                for replicate in workload["replicate_ids"]
            ],
            "cooldown_s": workload["cooldown_s"],
            "paired_foreground_samples_per_group": 16,
            "measurement": self.prereg["measurement"],
            "success_gates": self.prereg["success_gates"],
        }

    def test_held_out_burst_schedule_has_exact_groups(self):
        manifest = self._manifest()
        order = client._block_order(manifest)
        self.assertEqual(len(order), 16)
        self.assertEqual(tuple((arm.value, rep) for arm, rep in order), node._ORDER)
        block = client._materialize_block(
            sequence=0,
            arm=ForegroundArm.LOCAL,
            replicate=2,
            manifest=manifest,
            factory=_Factory(),
        )
        foreground = [
            metadata for metadata in block["request_index"].values()
            if metadata["tenant"] == Tenant.FOREGROUND.value
        ]
        self.assertEqual(len(foreground), 144)
        groups = Counter(
            (
                row["phase"], row["prompt_tokens"], row["output_tokens"],
                row["cache_state"],
            ) for row in foreground
        )
        self.assertEqual(len(groups), 36)
        self.assertEqual(set(groups.values()), {4})
        first_phase = sorted(
            row["arrival_offset_ms"] for row in foreground
            if row["phase"] == "c0_cool")
        self.assertEqual(first_phase[:2], [62.5, 187.5])
        self.assertTrue(all(
            metadata["pair_key"].startswith("r2:")
            for metadata in foreground))

    def test_profile_promotion_is_metadata_only(self):
        source_elastic = {
            "schema": "elastic",
            "profile_id": "calibrated",
            "deployment_scope": "screen_only",
            "identity": {"model": "qwen"},
            "controller": {"window": 17, "margin": 5.0},
            "rows": [{"prompt": 512, "bound": 123.0}],
        }
        source_endpoint = {
            "schema": "endpoint",
            "profile_id": "calibrated-endpoint",
            "elastic_profile_fingerprint_sha256": "0" * 64,
            "workload_manifest_sha256": "1" * 64,
            "deployment_scope": "calibration_only",
            "default_e2e_deadline_ms": 16000.0,
            "controller": {"window": 23, "quantile": 0.9},
            "routing_policy": {
                "policy": "semantic_epoch_v1",
                "epoch_confirmation_requests": 2,
            },
            "rows": [{"cache": "miss", "prior": 44.0}],
            "fingerprint_sha256": "2" * 64,
        }
        original_elastic = copy.deepcopy(source_elastic)
        original_endpoint = copy.deepcopy(source_endpoint)
        elastic, endpoint, elastic_fingerprint = (
            promotion._promote_raw_profiles(
                source_elastic,
                source_endpoint,
                manifest_sha256="a" * 64,
            )
        )
        self.assertEqual(source_elastic, original_elastic)
        self.assertEqual(source_endpoint, original_endpoint)
        self.assertEqual(elastic["identity"], source_elastic["identity"])
        self.assertEqual(elastic["controller"], source_elastic["controller"])
        self.assertEqual(elastic["rows"], source_elastic["rows"])
        self.assertEqual(endpoint["controller"], source_endpoint["controller"])
        self.assertEqual(
            endpoint["routing_policy"], source_endpoint["routing_policy"])
        self.assertEqual(endpoint["rows"], source_endpoint["rows"])
        self.assertEqual(elastic["deployment_scope"], "replicated")
        self.assertEqual(endpoint["deployment_scope"], "frozen_validation")
        self.assertEqual(
            endpoint["elastic_profile_fingerprint_sha256"],
            elastic_fingerprint)
        self.assertEqual(endpoint["workload_manifest_sha256"], "a" * 64)

    def _samples(self, *, tempo_e2e=80.0):
        samples = []
        ordinal = 0
        for replicate in (2, 3, 4, 5):
            for phase in self.prereg["workload"]["phase_order"]:
                for geometry in VALIDATION_FOREGROUND_GEOMETRIES:
                    for item in range(4):
                        tempo_route = (
                            client.c4._LOCAL_ROUTE
                            if ordinal % 2 == 0
                            else client.c4._REMOTE_ROUTE
                        )
                        samples.append({
                            "pair_key": f"r{replicate}:{phase}:{ordinal}:{item}",
                            "replicate": replicate,
                            "phase": phase,
                            "arrival_offset_ms": float(ordinal),
                            "prompt_tokens": geometry.prompt_tokens,
                            "output_tokens": geometry.output_tokens,
                            "cache_state": geometry.cache_state.value,
                            "ordinal": ordinal,
                            "arms": {
                                "local": {
                                    "route": client.c4._LOCAL_ROUTE,
                                    "e2e_ms": 100.0,
                                    "ttft_ms": 50.0,
                                    "tpot_ms": 10.0,
                                },
                                "remote": {
                                    "route": client.c4._REMOTE_ROUTE,
                                    "e2e_ms": 120.0,
                                    "ttft_ms": 60.0,
                                    "tpot_ms": 11.0,
                                },
                                "predictor": {
                                    "route": client.c4._LOCAL_ROUTE,
                                    "e2e_ms": 90.0,
                                    "ttft_ms": 45.0,
                                    "tpot_ms": 10.0,
                                },
                                "tempo": {
                                    "route": tempo_route,
                                    "e2e_ms": tempo_e2e,
                                    "ttft_ms": 40.0,
                                    "tpot_ms": 9.0,
                                },
                            },
                        })
                        ordinal += 1
        self.assertEqual(len(samples), 576)
        return samples

    @staticmethod
    def _goodput():
        durations = {"local": 10.0, "remote": 12.0,
                     "predictor": 9.0, "tempo": 8.0}
        return [
            {
                "block_key": f"{arm}-{replicate}",
                "sequence": sequence,
                "arm": arm,
                "replicate": replicate,
                "foreground_requests": 144,
                "foreground_output_tokens": 14400,
                "dispatch_to_stream_end_s": durations[arm],
            }
            for sequence, (arm, replicate) in enumerate(
                (value, replicate)
                for replicate in (2, 3, 4, 5)
                for value in ("local", "remote", "predictor", "tempo")
            )
        ]

    def test_final_metric_gates_use_median_and_all_groups(self):
        metrics, groups = analyzer._performance_metrics(
            self._samples(), self._goodput(), self._manifest())
        self.assertEqual(
            metrics["strongest_fixed_arm_authoritative"], "local")
        self.assertAlmostEqual(
            metrics["pooled_median_e2e_gain_vs_strongest_fixed"], 0.20)
        self.assertGreaterEqual(
            metrics["pooled_median_e2e_gain_vs_predictor"], 0.05)
        self.assertGreaterEqual(
            metrics["request_goodput_gain_vs_strongest_fixed"], 0.05)
        self.assertEqual(len(groups), 36)
        self.assertTrue(all(row["paired_requests"] == 16 for row in groups))
        self.assertTrue(all(row["group_pass"] for row in groups))
        self.assertTrue(all(metrics["final_gates"].values()))
        self.assertTrue(metrics["all_performance_gates_pass"])

    def test_final_metric_gate_does_not_relax_predictor_failure(self):
        metrics, _ = analyzer._performance_metrics(
            self._samples(tempo_e2e=97.0), self._goodput(), self._manifest())
        self.assertFalse(metrics["final_gates"][
            "pooled_median_e2e_gain_vs_predictor_at_least_5pct"])
        self.assertFalse(metrics["all_performance_gates_pass"])

    def test_runtime_contract_requires_replicated_profile(self):
        environment = contract_builder.INDEPENDENT_FIXED_RUNTIME_ENVIRONMENT
        self.assertEqual(
            environment["TEMPO_ELASTIC_PD_PROFILE_SCOPE"], "replicated")
        self.assertEqual(
            environment["TEMPO_PD_INDEPENDENT_VALIDATION_APPROVED"], "YES")
        self.assertEqual(
            environment["TEMPO_PD_FRONTEND_PAIR_POLICY"],
            "tempo-min-outstanding-decode-tokens-v1",
        )
        self.assertEqual(
            environment["TEMPO_PD_ENDPOINT_ROUTING_POLICY"],
            "instant_score_v1",
        )
        self.assertNotIn("TEMPO_PD_C4_ADAPTIVE_APPROVED", environment)

    def test_semantic_candidate_runtime_is_exact_and_mutually_exclusive(self):
        candidate = {
            "kind": "candidate_b_semantic_epoch_v1",
            "endpoint_routing_policy": "semantic_epoch_v1",
            "passive_external_credit": True,
            "implementation_entry": (
                "semantic_integration_implementation_contract"),
        }
        environment = contract_builder.independent_runtime_environment(
            candidate)
        self.assertEqual(
            environment["TEMPO_PD_INDEPENDENT_VALIDATION_APPROVED"], "YES")
        self.assertEqual(
            environment["TEMPO_PD_ENDPOINT_ROUTING_POLICY"],
            "semantic_epoch_v1",
        )
        self.assertEqual(
            environment["TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK"], "1")
        self.assertEqual(
            environment["TEMPO_ELASTIC_PD_PROFILE_SCOPE"], "replicated")
        self.assertNotIn(
            "TEMPO_PD_C4_SEMANTIC_INTEGRATION_APPROVED", environment)
        self.assertNotIn("TEMPO_PD_C4_ADAPTIVE_APPROVED", environment)
        with self.assertRaisesRegex(ValueError, "runtime policy is unsupported"):
            contract_builder.independent_runtime_environment({
                **candidate,
                "passive_external_credit": False,
            })

    def test_node_prestart_derives_semantic_environment_from_bound_contract(self):
        candidate = {
            "kind": "candidate_b_semantic_epoch_v1",
            "endpoint_routing_policy": "semantic_epoch_v1",
            "passive_external_credit": True,
            "implementation_entry": (
                "semantic_integration_implementation_contract"),
        }
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "run-contract.json"
            path.write_text(
                json.dumps({"candidate": candidate}), encoding="utf-8")
            environment = contract_builder.independent_runtime_environment(
                candidate)
            process_environment = {
                **environment,
                client.RUN_CONTRACT_ENV: str(path.resolve()),
                client.RUN_CONTRACT_SHA_ENV: hashlib.sha256(
                    path.read_bytes()).hexdigest(),
            }
            with mock.patch.dict(os.environ, process_environment, clear=True):
                self.assertEqual(
                    node._validate_prestart_environment(), environment)
                os.environ["TEMPO_PD_ENDPOINT_ROUTING_POLICY"] = (
                    "instant_score_v1")
                with self.assertRaisesRegex(
                    ValueError, "requires TEMPO_PD_ENDPOINT_ROUTING_POLICY"
                ):
                    node._validate_prestart_environment()

    def test_independent_block_conversion_preserves_semantic_contract(self):
        semantic_child = {
            "schema": client.adaptive.SEMANTIC_BLOCK_SCHEMA,
            "endpoint_routing_policy": "semantic_epoch_v1",
            "semantic_credit_contract": {"policy": "semantic_epoch_v1"},
            "passive_external_endpoint_credit": True,
            "semantic_decisions_exact": True,
            "external_credit_lifecycle_exact": True,
            "external_route_pinned_requests": 7,
            "passive_completions": 7,
        }
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "block.json"
            path.write_text(
                json.dumps({
                    "c4_adaptive_screen_contract": semantic_child,
                    "requests": [],
                }),
                encoding="utf-8",
            )
            with mock.patch.object(
                client.adaptive,
                "_validate_block",
                return_value=(semantic_child, {"validated": True}),
            ):
                contract, summary = client._validate_block(
                    path,
                    {},
                    {},
                    controller_reset=[],
                    controller_before=[],
                    controller_after=[],
                    semantic_contract={"candidate": "b"},
                )
            self.assertEqual(contract["schema"], client.BLOCK_SCHEMA)
            self.assertEqual(
                contract["endpoint_routing_policy"], "semantic_epoch_v1")
            self.assertTrue(contract["passive_external_endpoint_credit"])
            self.assertEqual(contract["external_route_pinned_requests"], 7)
            self.assertEqual(contract["passive_completions"], 7)
            self.assertTrue(contract["held_out_burst_workload"])
            self.assertFalse(contract["calibration_only"])
            self.assertEqual(summary, {"validated": True})
            rewritten = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("c4_adaptive_screen_contract", rewritten)
            self.assertEqual(
                rewritten["independent_validation_contract"], contract)

    def test_semantic_candidate_exercise_requires_full_epoch_lifecycle(self):
        request_index = {
            "tempo-local": {"tenant": Tenant.FOREGROUND.value},
            "tempo-remote": {"tenant": Tenant.FOREGROUND.value},
            "external-local": {"tenant": Tenant.DECODER_HOT.value},
            "external-remote": {"tenant": Tenant.KV_REMOTE_HOT.value},
        }
        decisions = [{
            "request_id": request_id,
            "frontend_semantic_load_schema": analyzer.semantic_policy.LOAD_SCHEMA,
            "frontend_semantic_load_source": analyzer.semantic_policy.LOAD_SOURCE,
        } for request_id in request_index]
        raw = {
            "independent_validation_contract": {
                "request_index": request_index,
            },
            "router_decisions": decisions,
        }
        with mock.patch.object(
            analyzer.semantic_policy,
            "_validate_semantic_decision",
            side_effect=[
                (
                    analyzer.semantic_policy.LOCAL_ROUTE,
                    "semantic_epoch_open_remote_high_water",
                    1,
                ),
                (
                    analyzer.semantic_policy.REMOTE_ROUTE,
                    "semantic_epoch_close_decoder_low_water",
                    2,
                ),
            ],
        ), mock.patch.object(
            analyzer.semantic_policy,
            "_validate_external_decision",
            side_effect=[
                (analyzer.semantic_policy.LOCAL_ROUTE, "local_service_proxy"),
                (analyzer.semantic_policy.REMOTE_ROUTE, "remote_service_proxy"),
            ],
        ):
            exercise = analyzer._candidate_exercise(
                [(raw, ForegroundArm.TEMPO)],
                semantic_contract={"candidate": "b"},
            )
        self.assertTrue(exercise["all_pass"])
        self.assertTrue(all(exercise["gates"].values()))
        self.assertEqual(exercise["maximum_epoch_generation"], 2)
        self.assertEqual(exercise["external_route_pinned_requests"], 2)

        instant = analyzer._candidate_exercise(
            [], semantic_contract=None)
        self.assertTrue(instant["all_pass"])
        self.assertFalse(instant["semantic_epoch_required"])

    def test_semantic_summary_observes_empty_pair_cell_without_forcing_it(self):
        value = analyzer._semantic_cell_summary([], max_num_seqs=16)
        self.assertEqual(value["requests"], 0)
        self.assertEqual(value["max_num_seqs"], 16)
        self.assertIsNone(value["active_requests_before"]["median"])
        self.assertIsNone(
            value["capacity_event_fraction"]["at_least_half"])
        self.assertEqual(value["pair_counts"], {})


if __name__ == "__main__":
    unittest.main()
