from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tempo.pd_admission import (
    PDCalibrationProfile, PDEvidenceLevel, PDPolicyConfig, PDWorkloadClass,
)
from tempo.pd_policy_manifest import PDPolicyManifest, load_manifest, write_manifest


def _profile(evidence: PDEvidenceLevel = PDEvidenceLevel.SCREEN) -> PDCalibrationProfile:
    return PDCalibrationProfile(
        workload=PDWorkloadClass(
            model_id="qwen", model_revision="sha", topology_id="2x-tp4",
            remote_backend="lmcache-ucx", prompt_bucket="c1:prompt:1024",
            output_bucket="c1:output:32", decoder_load_bucket="c1:load:0",
            kv_bytes_bucket="c1:kv:1",
        ),
        evidence_level=evidence,
        local_samples=3, remote_samples=3,
        local_latency_p50_ms=100.0, remote_latency_p50_ms=80.0,
        local_latency_lower_bound_ms=95.0, remote_latency_upper_bound_ms=85.0,
        outputs_equivalent=True, remote_transfer_failures=0,
        valid_from_epoch=4, valid_through_epoch=4,
    )


class PDPolicyManifestTests(unittest.TestCase):
    def test_round_trip_is_content_bound_and_screen_is_explicit(self) -> None:
        manifest = PDPolicyManifest(
            classifier_version="c1", policy_epoch=4, deployment_scope="screen_only",
            config=PDPolicyConfig(minimum_samples_per_route=3,
                                  require_replicated_evidence=False),
            profiles=(_profile(),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            write_manifest(path, manifest)
            loaded = load_manifest(path)
        self.assertEqual(loaded.manifest_id, manifest.manifest_id)
        with self.assertRaisesRegex(ValueError, "explicit allow_screen_profiles"):
            loaded.build_policy()
        self.assertEqual(loaded.build_policy(allow_screen_profiles=True).profile_count, 1)

    def test_unknown_key_fails_closed(self) -> None:
        manifest = PDPolicyManifest(
            classifier_version="c1", policy_epoch=4, deployment_scope="screen_only",
            config=PDPolicyConfig(require_replicated_evidence=False),
            profiles=(_profile(),),
        ).canonical_dict()
        manifest["surprise"] = True
        with self.assertRaisesRegex(ValueError, "manifest keys mismatch"):
            PDPolicyManifest.from_dict(manifest)

    def test_production_rejects_screen_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "screen-only evidence"):
            PDPolicyManifest(
                classifier_version="c1", policy_epoch=4, deployment_scope="production",
                config=PDPolicyConfig(), profiles=(_profile(),),
            )

    def test_refuses_overwrite(self) -> None:
        manifest = PDPolicyManifest(
            classifier_version="c1", policy_epoch=4, deployment_scope="screen_only",
            config=PDPolicyConfig(require_replicated_evidence=False),
            profiles=(_profile(),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps({}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                write_manifest(path, manifest)


if __name__ == "__main__":
    unittest.main()
