from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from eval.sota_4node import analyze_tempo_pd_c4_fixed_phase as analyzer
from eval.sota_4node import build_tempo_pd_c4_adaptive_screen_manifest as screen
from eval.sota_4node import build_tempo_pd_c4_calibrated_profiles as profiles
from eval.sota_4node import replay_tempo_pd_c4_calibrated_controller as replay
from eval.sota_4node.test_analyze_tempo_pd_c4_fixed_phase import (
    _Fixture,
    _sha,
    _write,
)


class C4CalibratedControllerReplayTest(unittest.TestCase):
    def _inputs(self, root: Path):
        fixture = _Fixture(root)
        analysis_value = analyzer.analyze(
            fixture.result, expected_result_sha256=_sha(fixture.result))
        analysis_path = _write(root / "analysis.json", analysis_value)
        analysis_sha = _sha(analysis_path)
        manifest_value = screen.build_manifest(
            analysis_path, expected_analysis_sha256=analysis_sha)
        manifest_path = _write(root / "adaptive-manifest.json", manifest_value)
        manifest_sha = _sha(manifest_path)
        elastic, endpoint, receipt = profiles.build_profiles(
            analysis_path=analysis_path,
            expected_analysis_sha256=analysis_sha,
            workload_manifest_path=manifest_path,
            expected_workload_manifest_sha256=manifest_sha,
            elastic_profile_id="synthetic-c4-elastic",
            endpoint_profile_id="synthetic-c4-endpoint",
        )
        elastic_path = _write(root / "elastic.json", elastic)
        endpoint_path = _write(root / "endpoint.json", endpoint)
        receipt_path = _write(root / "receipt.json", receipt)
        return {
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

    def test_replay_drains_all_first_response_credits(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._inputs(Path(directory))
            value = replay.replay(**inputs)
            self.assertEqual(value["paired_requests"], len(value["decisions"]))
            self.assertEqual(len(value["phase_summaries"]), 6)
            self.assertTrue(value["screen_gates"]["all_requests_replayed"])
            self.assertTrue(value["screen_gates"]["all_resources_released"])
            self.assertFalse(value["performance_claim_allowed"])
            # The synthetic fixture makes local uniformly faster, so route
            # diversity correctly prevents authorization of a live screen.
            self.assertFalse(value["screen_gates"]["both_routes_exercised"])
            self.assertFalse(value["live_adaptive_screen_authorized"])

    def test_endpoint_profile_hash_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._inputs(Path(directory))
            expected = inputs["endpoint_sha256"]
            inputs["endpoint_path"].write_text(
                inputs["endpoint_path"].read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            inputs["endpoint_sha256"] = expected
            with self.assertRaisesRegex(ValueError, "endpoint profile digest"):
                replay.replay(**inputs)


if __name__ == "__main__":
    unittest.main()
