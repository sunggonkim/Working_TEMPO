import json
from pathlib import Path
import tempfile
import unittest

from tempo.pd_elastic_profile_v444 import load_elastic_profile


def payload():
    return {
        "schema": "tempo-elastic-pd-profile-444",
        "profile_id": "fixture",
        "deployment_scope": "screen_only",
        "identity": {
            "model_id": "qwen", "model_revision": "sha", "topology_id": "2x-tp4",
            "remote_backend": "lmcache-ucx", "classifier_version": "exact-v1",
            "kv_bytes_per_token": 100,
        },
        "controller": {
            "local_compute_budget_us": 1000, "remote_kv_budget_bytes": 2000,
            "arrival_window": 4, "enter_high_gap_ns": 39000000,
            "exit_high_gap_ns": 78000000, "exit_consecutive_windows": 2,
            "route_margin_ms": 5.0, "spill_regression_budget_ms": 5.0,
        },
        "rows": [{
            "prompt_tokens": 10, "output_tokens": 64,
            "local_upper_bound_ms": 30.0, "remote_upper_bound_ms": 20.0,
            "uncertainty_ms": 2.0, "local_tbt_safe": True,
            "remote_evidence_valid": True, "local_compute_cost_us": 400,
            "remote_kv_bytes": 1000, "samples_local": 3, "samples_remote": 3,
            "outputs_equivalent": True, "remote_transfer_failures": 0,
        }],
    }


class ElasticProfileTest(unittest.TestCase):
    def _load(self, value):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(value))
            return load_elastic_profile(path)

    def test_exact_profile_loads_and_estimates(self):
        profile = self._load(payload())
        row = profile.exact_row(10, 64)
        self.assertIsNotNone(row)
        self.assertTrue(row.evidence_safe)
        self.assertTrue(row.estimate(100).remote_evidence_valid)
        self.assertEqual(len(profile.fingerprint_sha256), 64)
        profile.validate_identity(
            model_id="qwen", model_revision="sha", topology_id="2x-tp4",
            remote_backend="lmcache-ucx", classifier_version="exact-v1",
            kv_bytes_per_token=100,
        )

    def test_geometry_and_identity_fail_closed(self):
        bad = payload()
        bad["rows"][0]["remote_kv_bytes"] = 999
        with self.assertRaisesRegex(ValueError, "exact prompt geometry"):
            self._load(bad)
        profile = self._load(payload())
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            profile.validate_identity(
                model_id="other", model_revision="sha", topology_id="2x-tp4",
                remote_backend="lmcache-ucx", classifier_version="exact-v1",
                kv_bytes_per_token=100,
            )

    def test_unknown_or_weak_rows_never_create_remote_evidence(self):
        value = payload()
        value["rows"][0]["outputs_equivalent"] = False
        profile = self._load(value)
        self.assertFalse(profile.exact_row(10, 64).estimate(100).remote_evidence_valid)
        self.assertIsNone(profile.exact_row(11, 64))


if __name__ == "__main__":
    unittest.main()
