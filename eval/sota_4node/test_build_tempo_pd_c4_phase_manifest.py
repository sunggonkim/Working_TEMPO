from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from eval.sota_4node import build_tempo_pd_c4_phase_manifest as builder


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(
            json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class C4PhaseManifestBuilderTest(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        source = _write(root / "results/source.jsonl", "{}\n")
        profile = _write(root / "eval/profile.json", "{}\n")

        cold_raw = _write(root / "results/cold/raw.json", "{}\n")
        cold_manifest = _write(root / "eval/cold-manifest.json", {
            "schema": builder.FROZEN_COLD_MANIFEST_SCHEMA,
            "foreground_rate_per_s": 2.0,
            "load": {
                "decoder_offered_rate_per_s": 22.4,
                "remote_offered_rate_per_s": 4.76,
            },
        })
        cold_gate = {
            "schema": "tempo-pd-contention-crossover-gate-v4",
            "workload_valid_for_controller_tuning": True,
        }
        cold_result = _write(root / "results/cold/result.json", {
            "schema": builder.COLD_RESULT_SCHEMA,
            "controller_tuning_allowed": True,
            "performance_claim_allowed": False,
            "crossover_gate": cold_gate,
            "raw": str(cold_raw),
            "frozen_workload_manifest": str(cold_manifest),
            "frozen_workload_manifest_sha256": _sha(cold_manifest),
            "source_workload": str(source),
            "profile": str(profile),
        })
        cold_characterization = _write(
            root / "results/cold/characterization.json", {
                "schema": builder.COLD_CHARACTERIZATION_SCHEMA,
                "source": str(cold_raw),
                "crossover_gate": cold_gate,
            })

        p_only_raw = _write(root / "results/p-only/raw.json", "{}\n")
        p_only_result = _write(root / "results/p-only/result.json", {
            "schema": builder.P_ONLY_RESULT_SCHEMA,
            "performance_claim_allowed": False,
            "physical_switch_bottleneck_claim_allowed": False,
            "stopped_after_first_invalid_block": None,
            "raw": str(p_only_raw),
            "source_workload": str(source),
            "source_workload_sha256": _sha(source),
            "profile": str(profile),
            "profile_sha256": _sha(profile),
        })
        p_only_characterization = _write(
            root / "results/p-only/characterization.json", {
                "schema": builder.P_ONLY_CHARACTERIZATION_SCHEMA,
                "source": str(p_only_raw),
                "all_measured_requests_valid": True,
                "first_rate_with_2x_remote_foreground_median": 12.0,
                "first_rate_with_over_10pct_remote_drain": 12.0,
                "invariants": {
                    "background_full_source_hits_exact": True,
                    "preseed_outside_measurement_window": True,
                    "decoder_prefix_caching": False,
                    "synthetic_network_background": False,
                },
            })

        c3_manifest = _write(root / "eval/c3-manifest.json", {
            "schema": builder.C3_MANIFEST_SCHEMA,
            "performance_claim_allowed": False,
            "arm_order_policy": "paired_abba",
            "within_rate_block_order": [
                "local", "remote", "remote", "local"],
            "p_only_rates_per_s": [0, 4, 8, 12],
            "decoder_hot_rate_per_s": 22.4,
            "foreground_rate_per_s": 2.0,
            "phase_duration_ms": 8000.0,
            "source_workload": {
                "path": str(source.relative_to(root)),
                "sha256": _sha(source),
            },
            "profile": {
                "path": str(profile.relative_to(root)),
                "sha256": _sha(profile),
            },
            "transport": "LMCacheConnectorV1:UCX",
        })
        c3_result = _write(root / "results/c3/result.json", {
            "schema": builder.C3_RESULT_SCHEMA,
            "performance_claim_allowed": False,
            "physical_switch_bottleneck_claim_allowed": False,
        })
        c3_characterization = _write(
            root / "results/c3/characterization.json", {
                "schema": builder.C3_CHARACTERIZATION_SCHEMA,
                "all_measured_requests_valid": True,
            })
        c3_gate = _write(root / "results/c3/gate.json", {
            "schema": builder.C3_GATE_SCHEMA,
            "c3_coupled_characterization_valid": True,
            "authorizes_c4_phase_trace": True,
            "performance_claim_allowed": False,
            "physical_switch_bottleneck_claim_allowed": False,
            "remote_control_replicate_direction_correct": [True, True],
            "remote_control_median_gain": 0.10,
            "local_overload_replicate_direction_correct": [True, True],
            "local_overload_median_gain": 0.20,
            "manifest": str(c3_manifest),
            "manifest_sha256": _sha(c3_manifest),
            "result": str(c3_result),
            "result_sha256": _sha(c3_result),
            "characterization": str(c3_characterization),
            "characterization_sha256": _sha(c3_characterization),
        })
        return {
            "repo_root": root,
            "c3_gate_path": c3_gate,
            "cold_result_path": cold_result,
            "cold_characterization_path": cold_characterization,
            "p_only_result_path": p_only_result,
            "p_only_characterization_path": p_only_characterization,
        }

    def test_passed_parents_build_exact_six_phase_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._fixture(Path(directory))
            value = builder.build_manifest(**inputs)
            self.assertEqual(value["schema"], builder.SCHEMA)
            self.assertEqual(value["phase_order"], [
                "c0_cool",
                "c1_decoder_hot",
                "c2_remote_hot",
                "c2_kv_remote_hot",
                "c3_both_hot",
                "recovery",
            ])
            self.assertEqual(value["phase_duration_ms"], 8000.0)
            self.assertEqual(value["background_rates_per_s"], {
                "decoder_hot": 22.4,
                "cold_remote_hot": 4.76,
                "kv_remote_hot": 12.0,
            })
            protocol = value["cache_state_protocol"]
            self.assertEqual(
                protocol["schema"], builder.CACHE_STATE_PROTOCOL_SCHEMA)
            self.assertEqual(
                protocol["fixed_arm_pair_placement"],
                "terminal_item_modulo_two_pairs",
            )
            self.assertEqual(protocol["preparation_order"], [
                "remote_seed_and_full_hit_probe_for_p_only_and_both",
                "quiescent_all_decoder_apc_reset_preserving_external_lmcache",
                "local_miss_seed_and_full_hit_probe_for_d_only_and_both",
                "measured_trace",
            ])
            self.assertFalse(
                protocol["request_id_labels_without_completion_evidence_allowed"])
            self.assertTrue(protocol["decoder_usage_breakdown_required"])
            self.assertFalse(
                protocol[
                    "stock_cached_tokens_without_source_breakdown_allowed"])
            self.assertEqual(
                protocol["state_contracts"]["d_only"][
                    "local_probe_decoder_cached_tokens"],
                "floor((prompt_tokens-1)/16)*16",
            )
            remote = protocol["measured_decoder_route_contracts"][
                "official_lmcache_remote_prefill"]
            self.assertEqual(remote["usage_prompt_tokens"], "P+1")
            self.assertEqual(remote["total_cached_tokens"], "P")
            self.assertEqual(
                remote["decoder_residency_basis"],
                "exact_local_preparation_hit_on_original_P_token_prompt",
            )
            self.assertEqual(
                remote["local_cached_tokens_by_state"]["d_only"],
                "floor((P-1)/16)*16",
            )
            self.assertEqual(
                remote["external_cached_tokens"],
                "P-local_cached_tokens",
            )
            endpoint = value["endpoint_evidence_contract"]
            self.assertEqual(
                endpoint["schema"],
                builder.ENDPOINT_EVIDENCE_CONTRACT_SCHEMA,
            )
            self.assertTrue(endpoint["measurement_start_marker_required"])
            self.assertTrue(
                endpoint["publisher_pid_matches_measured_child"])
            self.assertEqual(
                endpoint["sampling_policy"],
                "workload_start_boundary_midpoint_and_end_boundary",
            )
            self.assertEqual(endpoint["phase_boundary_samples"], 7)
            self.assertEqual(endpoint["phase_midpoint_samples"], 6)
            self.assertEqual(
                endpoint["cassini_phase_windows"],
                "two_nonoverlapping_half_phase_endpoint_deltas",
            )
            self.assertEqual(
                value["fixed_runtime_environment"],
                dict(sorted(
                    builder.C4_FIXED_RUNTIME_ENVIRONMENT.items())),
            )
            self.assertFalse(value["performance_claim_allowed"])
            self.assertFalse(value["controller_tuning_allowed"])
            self.assertEqual(
                value["fingerprint_sha256"],
                builder.manifest_fingerprint(value),
            )
            self.assertEqual(
                set(value["parent_evidence"]), {
                    "cold_c1_c2_result",
                    "cold_c1_c2_characterization",
                    "p_only_result",
                    "p_only_characterization",
                    "c3_abba_gate",
                    "c3_abba_manifest",
                    "c3_abba_result",
                    "c3_abba_characterization",
                })

    def test_failed_c3_gate_cannot_build_c4(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._fixture(Path(directory))
            gate_path = inputs["c3_gate_path"]
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["authorizes_c4_phase_trace"] = False
            _write(gate_path, gate)
            with self.assertRaisesRegex(ValueError, "does not authorize C4"):
                builder.build_manifest(**inputs)

    def test_p_only_knee_drift_cannot_build_c4(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._fixture(Path(directory))
            path = inputs["p_only_characterization_path"]
            value = json.loads(path.read_text(encoding="utf-8"))
            value["first_rate_with_2x_remote_foreground_median"] = 8.0
            _write(path, value)
            with self.assertRaisesRegex(ValueError, "2x service knee differs"):
                builder.build_manifest(**inputs)

    def test_parent_digest_drift_cannot_build_c4(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._fixture(Path(directory))
            gate_path = inputs["c3_gate_path"]
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["result_sha256"] = "0" * 64
            _write(gate_path, gate)
            with self.assertRaisesRegex(ValueError, "result digest mismatch"):
                builder.build_manifest(**inputs)


if __name__ == "__main__":
    unittest.main()
