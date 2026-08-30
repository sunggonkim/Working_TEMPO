from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval.sota_4node import analyze_tempo_pd_c4_fixed_phase as analyzer
from eval.sota_4node import build_tempo_pd_c4_adaptive_run_contract as builder
from eval.sota_4node import build_tempo_pd_c4_adaptive_screen_manifest as screen
from eval.sota_4node import build_tempo_pd_c4_calibrated_profiles as profiles
from eval.sota_4node import replay_tempo_pd_c4_calibrated_controller as replay
from eval.sota_4node.test_analyze_tempo_pd_c4_fixed_phase import (
    _Fixture,
    _sha,
    _write,
)


class C4AdaptiveRunContractTest(unittest.TestCase):
    def _inputs(self, root: Path):
        fixture = _Fixture(root)
        analysis_value = analyzer.analyze(
            fixture.result, expected_result_sha256=_sha(fixture.result))
        analysis_path = _write(root / "analysis.json", analysis_value)
        analysis_sha = _sha(analysis_path)
        manifest_value = screen.build_manifest(
            analysis_path, expected_analysis_sha256=analysis_sha)
        manifest_path = _write(root / "manifest.json", manifest_value)
        manifest_sha = _sha(manifest_path)
        elastic, endpoint, receipt = profiles.build_profiles(
            analysis_path=analysis_path,
            expected_analysis_sha256=analysis_sha,
            workload_manifest_path=manifest_path,
            expected_workload_manifest_sha256=manifest_sha,
            elastic_profile_id="synthetic-adaptive-elastic",
            endpoint_profile_id="synthetic-adaptive-endpoint",
        )
        elastic_path = _write(root / "elastic.json", elastic)
        endpoint_path = _write(root / "endpoint.json", endpoint)
        receipt_path = _write(root / "receipt.json", receipt)
        replay_inputs = {
            "analysis_path": analysis_path,
            "analysis_sha256": analysis_sha,
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha,
            "elastic_path": elastic_path,
            "elastic_sha256": _sha(elastic_path),
            "endpoint_path": endpoint_path,
            "endpoint_sha256": _sha(endpoint_path),
            "receipt_path": receipt_path,
            "receipt_sha256": _sha(receipt_path),
        }
        replay_value = replay.replay(**replay_inputs)
        replay_path = _write(root / "replay.json", replay_value)
        implementation_path = _write(root / "implementation.json", {})
        return {
            **replay_inputs,
            "replay_path": replay_path,
            "replay_sha256": _sha(replay_path),
            "implementation_path": implementation_path,
            "implementation_sha256": _sha(implementation_path),
            "repo_root": Path(__file__).resolve().parents[2],
        }, replay_value

    def test_contract_binds_only_a_reproducibly_authorized_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, replay_value = self._inputs(Path(directory))
            replay_value["screen_gates"] = {
                name: True for name in replay_value["screen_gates"]
            }
            replay_value["live_adaptive_screen_authorized"] = True
            replay_value["fingerprint_sha256"] = replay.replay_fingerprint(
                replay_value)
            inputs["replay_path"] = _write(
                Path(directory) / "authorized-replay.json", replay_value)
            inputs["replay_sha256"] = _sha(inputs["replay_path"])
            implementation_value = {
                "fingerprint_sha256": "f" * 64,
                "files": [],
            }
            with (
                patch.object(
                    builder.replay_module, "replay", return_value=replay_value),
                patch.object(
                    builder.implementation,
                    "verify_contract",
                    return_value=implementation_value,
                ),
            ):
                value = builder.build_run_contract(**inputs)
            self.assertEqual(value["schema"], builder.SCHEMA)
            self.assertEqual(
                value["fingerprint_sha256"],
                builder.contract_fingerprint(value),
            )
            self.assertTrue(value["offline_replay_authorized"])
            self.assertEqual(
                value["fixed_runtime_environment"][
                    "TEMPO_PD_ENDPOINT_FEEDBACK_MODE"],
                "adaptive",
            )
            self.assertEqual(
                value["fixed_runtime_environment"][
                    "TEMPO_PD_ENDPOINT_ROUTING_POLICY"],
                "instant_score_v1",
            )
            self.assertFalse(value["performance_claim_allowed"])

    def test_non_authorizing_replay_fails_before_live_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, _ = self._inputs(Path(directory))
            with self.assertRaisesRegex(ValueError, "does not reproducibly authorize"):
                builder.build_run_contract(**inputs)


if __name__ == "__main__":
    unittest.main()
