from __future__ import annotations

import unittest

from eval.sota_4node.tempo_pd_router_v1 import (
    RouterConfig, RouterMode, TempoPDRouterCore,
)
from tempo.pd_admission import (
    PDCalibrationProfile, PDEvidenceLevel, PDPolicyConfig, PDRoute,
)
from tempo.pd_policy_manifest import PDPolicyManifest


def _config(mode: RouterMode) -> RouterConfig:
    return RouterConfig(
        mode=mode,
        local_url="http://local:8000",
        remote_url="http://remote:8001",
        tokenizer_url="http://local:8000",
        served_model_name="served",
        model_id="qwen",
        model_revision="sha",
        topology_id="2x-tp4",
        remote_backend="lmcache-ucx",
        classifier_version="c1",
        decoder_load_bucket="streams:7",
        kv_bytes_per_token=100,
    )


def _manifest(*, remote_win: bool = True) -> PDPolicyManifest:
    probe = TempoPDRouterCore(_config(RouterMode.FIXED_LOCAL))
    workload, _ = probe.classify(prompt_tokens=10, output_tokens=2)
    profile = PDCalibrationProfile(
        workload=workload,
        evidence_level=PDEvidenceLevel.SCREEN,
        local_samples=3,
        remote_samples=3,
        local_latency_p50_ms=100.0,
        remote_latency_p50_ms=80.0 if remote_win else 99.0,
        local_latency_lower_bound_ms=95.0,
        remote_latency_upper_bound_ms=85.0 if remote_win else 96.0,
        outputs_equivalent=True,
        remote_transfer_failures=0,
        valid_from_epoch=1,
        valid_through_epoch=1,
    )
    return PDPolicyManifest(
        classifier_version="c1",
        policy_epoch=1,
        deployment_scope="screen_only",
        config=PDPolicyConfig(
            minimum_samples_per_route=3,
            require_replicated_evidence=False,
        ),
        profiles=(profile,),
    )


class TempoPDRouterCoreTests(unittest.TestCase):
    def test_fixed_modes_share_exact_classifier(self) -> None:
        local = TempoPDRouterCore(_config(RouterMode.FIXED_LOCAL))
        remote = TempoPDRouterCore(_config(RouterMode.LMCACHE_ALWAYS_REMOTE))
        left = local.decide(request_id="l", prompt_tokens=10, output_tokens=2)
        right = remote.decide(request_id="r", prompt_tokens=10, output_tokens=2)
        self.assertEqual(left.workload, right.workload)
        self.assertEqual(left.route, PDRoute.DECODER_LOCAL)
        self.assertEqual(right.route, PDRoute.REMOTE_PREFILL)
        self.assertEqual(left.potential_kv_bytes, 1000)

    def test_tempo_uses_frozen_profile_and_lifecycle(self) -> None:
        core = TempoPDRouterCore(
            _config(RouterMode.TEMPO_AUTO), _manifest(), allow_screen_profiles=True
        )
        record = core.decide(request_id="r", prompt_tokens=10, output_tokens=2)
        self.assertEqual(record.route, PDRoute.REMOTE_PREFILL)
        self.assertIsNotNone(record.profile_id)
        core.mark_upstream_started("r")
        core.mark_response_started("r")
        core.complete("r")
        self.assertEqual(core.records()[0]["phase"], "complete")

    def test_weak_profile_selects_local(self) -> None:
        core = TempoPDRouterCore(
            _config(RouterMode.TEMPO_AUTO), _manifest(remote_win=False),
            allow_screen_profiles=True,
        )
        record = core.decide(request_id="r", prompt_tokens=10, output_tokens=2)
        self.assertEqual(record.route, PDRoute.DECODER_LOCAL)
        self.assertEqual(record.reason, "remote_benefit_lower_bound_below_margin")

    def test_missing_class_and_duplicate_id_fail_closed(self) -> None:
        core = TempoPDRouterCore(
            _config(RouterMode.TEMPO_AUTO), _manifest(), allow_screen_profiles=True
        )
        missing = core.decide(request_id="r", prompt_tokens=11, output_tokens=2)
        self.assertEqual(missing.route, PDRoute.DECODER_LOCAL)
        self.assertEqual(missing.reason, "no_exact_workload_profile")
        with self.assertRaisesRegex(ValueError, "duplicate request_id"):
            core.decide(request_id="r", prompt_tokens=11, output_tokens=2)

    def test_screen_and_classifier_version_are_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "allow_screen_profiles"):
            TempoPDRouterCore(_config(RouterMode.TEMPO_AUTO), _manifest())
        bad = _config(RouterMode.TEMPO_AUTO)
        bad = RouterConfig(**{**bad.__dict__, "classifier_version": "c2"})
        with self.assertRaisesRegex(ValueError, "classifier_version mismatch"):
            TempoPDRouterCore(bad, _manifest(), allow_screen_profiles=True)


if __name__ == "__main__":
    unittest.main()
