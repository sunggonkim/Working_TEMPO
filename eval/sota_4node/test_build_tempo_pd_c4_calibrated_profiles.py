from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from eval.sota_4node import analyze_tempo_pd_c4_fixed_phase as analyzer
from eval.sota_4node import build_tempo_pd_c4_adaptive_screen_manifest as screen
from eval.sota_4node import build_tempo_pd_c4_calibrated_profiles as builder
from eval.sota_4node.test_analyze_tempo_pd_c4_fixed_phase import (
    _Fixture,
    _sha,
    _write,
)
from tempo.pd_elastic_controller_v443 import CacheResidency
from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_endpoint_profile import load_endpoint_service_profile


class C4CalibratedProfilesTest(unittest.TestCase):
    def _inputs(self, root: Path):
        fixture = _Fixture(root)
        analysis_value = analyzer.analyze(
            fixture.result, expected_result_sha256=_sha(fixture.result))
        analysis_path = _write(root / "analysis.json", analysis_value)
        analysis_sha = _sha(analysis_path)
        manifest_value = screen.build_manifest(
            analysis_path, expected_analysis_sha256=analysis_sha)
        manifest_path = _write(root / "adaptive-manifest.json", manifest_value)
        return fixture, analysis_path, analysis_sha, manifest_path

    def test_six_c4_rows_include_missing_old_profile_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture, analysis_path, analysis_sha, manifest_path = self._inputs(root)
            elastic, endpoint, receipt = builder.build_profiles(
                analysis_path=analysis_path,
                expected_analysis_sha256=analysis_sha,
                workload_manifest_path=manifest_path,
                expected_workload_manifest_sha256=_sha(manifest_path),
                elastic_profile_id="synthetic-c4-elastic",
                endpoint_profile_id="synthetic-c4-endpoint",
            )
            elastic_path = _write(root / "elastic.json", elastic)
            endpoint_path = _write(root / "endpoint.json", endpoint)
            loaded_elastic = load_elastic_profile(elastic_path)
            loaded_endpoint = load_endpoint_service_profile(endpoint_path)
            self.assertEqual(len(loaded_elastic.rows), 6)
            self.assertEqual(len(loaded_endpoint.rows), 6)
            self.assertIsNotNone(loaded_elastic.exact_row(4094, 256))
            row = loaded_endpoint.exact_row(
                4094, 256, CacheResidency.D_ONLY)
            self.assertEqual(row.cache_residency, CacheResidency.D_ONLY)
            self.assertTrue(receipt["includes_4094_256_d_only"])
            self.assertFalse(
                receipt["remote_admission_for_d_only_or_both_allowed"])
            self.assertFalse(
                receipt["formula_contract"]["controller_parameter_search"])
            self.assertEqual(
                endpoint["elastic_profile_fingerprint_sha256"],
                loaded_elastic.fingerprint_sha256,
            )
            self.assertEqual(
                endpoint["workload_manifest_sha256"], _sha(manifest_path))

    def test_live_manifest_slo_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture, analysis_path, analysis_sha, manifest_path = self._inputs(root)
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            value["measurement"]["tpot_slo_ms"] = 999.0
            value["fingerprint_sha256"] = screen.manifest_fingerprint(value)
            _write(manifest_path, value)
            with self.assertRaisesRegex(ValueError, "SLO contract differs"):
                builder.build_profiles(
                    analysis_path=analysis_path,
                    expected_analysis_sha256=analysis_sha,
                    workload_manifest_path=manifest_path,
                    expected_workload_manifest_sha256=_sha(manifest_path),
                    elastic_profile_id="synthetic-c4-elastic",
                    endpoint_profile_id="synthetic-c4-endpoint",
                )

    def test_missing_c0_geometry_fails_profile_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _Fixture(root)
            analysis = analyzer.analyze(
                fixture.result, expected_result_sha256=_sha(fixture.result))
            analysis["foreground_paired_samples"] = [
                row for row in analysis["foreground_paired_samples"]
                if not (
                    row["phase"] == "c0_cool"
                    and row["prompt_tokens"] == 4094
                    and row["output_tokens"] == 256
                    and row["cache_state"] == "d_only"
                )
            ]
            with self.assertRaisesRegex(ValueError, "geometry/state inventory"):
                builder._paired_c0_groups(analysis)


if __name__ == "__main__":
    unittest.main()
