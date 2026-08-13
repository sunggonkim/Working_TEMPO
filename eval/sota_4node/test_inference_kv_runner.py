from __future__ import annotations

import unittest

from eval.sota_4node.inference_kv_runner import (
    build_kv_matrix,
    build_manifest,
    validate_kv_manifest,
    validate_kv_matrix,
)


class InferenceKVRunnerTests(unittest.TestCase):
    def test_matrix_has_local_fabric_and_persistent_paths(self) -> None:
        runs = build_kv_matrix()
        validate_kv_matrix(runs)
        self.assertEqual([run.mode for run in runs], [
            "fg_only", "open_combined", "d2h_only", "remote_fabric",
            "persistent_tier", "combined",
        ])
        self.assertEqual(runs[0].route, ())
        self.assertIn("nic_fabric", runs[3].route)
        self.assertIn("slingshot_fabric", runs[3].route)
        self.assertEqual(runs[4].route[0], "persistent_endpoint")
        self.assertEqual(runs[4].route[-1], "gpu_local")

    def test_manifest_is_design_only_and_exact(self) -> None:
        manifest = build_manifest()
        validate_kv_manifest(manifest)
        self.assertEqual((manifest["world_size"], manifest["nodes"]), (1, 1))
        self.assertFalse(manifest["live_backend"])
        self.assertFalse(manifest["slurm_submitted"])
        self.assertEqual(manifest["evidence_state"], "design_only")
        self.assertEqual(
            manifest["domain_footprints"]["remote_fabric"]["shared_domains"],
            ["gpu_local"],
        )
        self.assertEqual(manifest["domain_footprints"]["fg_only"]["shared_domains"], [])
        self.assertIn("max_domain_exposure_ns", manifest["metric_contract"]["required_fields"])
        self.assertIn("domain_exposure_ns", manifest["metric_contract"]["required_fields"])

    def test_manifest_rejects_live_or_nonpositive_geometry(self) -> None:
        candidate = build_manifest()
        candidate["live_backend"] = True
        with self.assertRaisesRegex(ValueError, "live backend"):
            validate_kv_manifest(candidate)
        with self.assertRaisesRegex(ValueError, "positive int"):
            build_manifest(kv_bytes_per_request=0)

    def test_manifest_rejects_route_or_contract_edits(self) -> None:
        candidate = build_manifest()
        candidate["runs"] = [dict(item) for item in candidate["runs"]]
        candidate["runs"][3]["route"] = ["gpu_local", "pcie_host"]
        with self.assertRaisesRegex(ValueError, "frozen adapter matrix"):
            validate_kv_manifest(candidate)
        candidate = build_manifest()
        candidate["correctness_contract"] = dict(candidate["correctness_contract"])
        candidate["correctness_contract"]["stale_version_rejection"] = 1
        with self.assertRaisesRegex(ValueError, "correctness contract"):
            validate_kv_manifest(candidate)
        candidate = build_manifest()
        candidate["domain_footprints"] = {
            mode: dict(value) for mode, value in candidate["domain_footprints"].items()
        }
        candidate["domain_footprints"]["remote_fabric"]["shared_domains"] = []
        with self.assertRaisesRegex(ValueError, "domain footprints"):
            validate_kv_manifest(candidate)
        candidate = build_manifest()
        candidate["metric_contract"] = dict(candidate["metric_contract"])
        candidate["metric_contract"]["promotion_rule"] = "latency_only"
        with self.assertRaisesRegex(ValueError, "metric contract"):
            validate_kv_manifest(candidate)


if __name__ == "__main__":
    unittest.main()
