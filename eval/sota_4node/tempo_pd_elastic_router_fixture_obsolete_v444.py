import json
from pathlib import Path
import tempfile
import unittest

from eval.sota_4node.tempo_pd_elastic_router_v444 import (
    ElasticExperimentArm, ElasticPDRouterCore,
)
from eval.sota_4node.tempo_pd_router_v1 import RouterConfig, RouterMode
from tempo.pd_elastic_controller_v443 import CacheResidency, ElasticRoute
from tempo.pd_elastic_profile_v444 import load_elastic_profile


def profile_payload():
    return {
        "schema": "tempo-elastic-pd-profile-444", "profile_id": "fixture",
        "deployment_scope": "screen_only",
        "identity": {
            "model_id": "qwen", "model_revision": "sha", "topology_id": "2x-tp4",
            "remote_backend": "lmcache-ucx", "classifier_version": "exact-v1",
            "kv_bytes_per_token": 100,
        },
        "controller": {
            "local_compute_budget_us": 400, "remote_kv_budget_bytes": 1000,
            "arrival_window": 4, "enter_high_gap_ns": 39000000,
            "exit_high_gap_ns": 78000000, "exit_consecutive_windows": 2,
            "route_margin_ms": 5.0, "spill_regression_budget_ms": 5.0,
        },
        "rows": [{
            "prompt_tokens": 10, "output_tokens": 64,
            "local_upper_bound_ms": 30.0, "remote_upper_bound_ms": 20.0,
            "uncertainty_ms": 1.0, "local_tbt_safe": True,
            "remote_evidence_valid": True, "local_compute_cost_us": 400,
            "remote_kv_bytes": 1000, "samples_local": 3, "samples_remote": 3,
            "outputs_equivalent": True, "remote_transfer_failures": 0,
        }],
    }


def config():
    return RouterConfig(
        mode=RouterMode.TEMPO_AUTO, local_url="http://local:1",
        remote_url="http://remote:2", tokenizer_url="http://local:1",
        served_model_name="served", model_id="qwen", model_revision="sha",
        topology_id="2x-tp4", remote_backend="lmcache-ucx",
        classifier_version="exact-v1", decoder_load_bucket="exact",
        kv_bytes_per_token=100,
    )


class ElasticRouterCoreTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "profile.json"
        path.write_text(json.dumps(profile_payload()))
        self.profile = load_elastic_profile(path)

    def core(self):
        return ElasticPDRouterCore(
            config(), self.profile, allow_screen_profile=True,
            cache_residency=lambda _request_id: CacheResidency.MISS,
        )

    def test_four_arms_share_geometry_and_commit_before_start(self):
        core = self.core()
        rows = [
            core.decide(request_id=f"epd-{arm}-x", prompt_tokens=10, output_tokens=64)
            for arm in ("local", "remote", "predictor", "tempo")
        ]
        self.assertEqual({row.potential_kv_bytes for row in rows}, {1000})
        self.assertEqual(rows[0].route, ElasticRoute.LOCAL)
        self.assertEqual(rows[1].route, ElasticRoute.REMOTE)
        self.assertEqual(rows[2].route, ElasticRoute.REMOTE)
        self.assertEqual(rows[3].route, ElasticRoute.LOCAL)
        for row in rows:
            self.assertEqual(row.phase, "route_committed" if row.arm is not ElasticExperimentArm.TEMPO else "local_reserved")

    def test_credit_queue_retry_and_exact_release(self):
        core = self.core()
        first = core.decide(request_id="epd-tempo-first", prompt_tokens=10, output_tokens=64)
        queued = core.decide(request_id="epd-tempo-second", prompt_tokens=10, output_tokens=64)
        self.assertEqual(first.route, ElasticRoute.LOCAL)
        self.assertEqual(queued.route, ElasticRoute.QUEUE)
        core.mark_upstream_started(first.request_id)
        core.mark_response_started(first.request_id)
        core.complete(first.request_id)
        retried = core.retry(queued.request_id, 1000)
        self.assertEqual(retried.route, ElasticRoute.LOCAL)
        self.assertEqual(retried.attempt, 2)

    def test_missing_profile_and_queued_start_fail_closed(self):
        core = self.core()
        with self.assertRaisesRegex(ValueError, "no exact"):
            core.decide(request_id="epd-tempo-missing", prompt_tokens=11, output_tokens=64)
        core.decide(request_id="epd-tempo-first", prompt_tokens=10, output_tokens=64)
        queued = core.decide(request_id="epd-tempo-second", prompt_tokens=10, output_tokens=64)
        self.assertEqual(queued.route, ElasticRoute.QUEUE)
        with self.assertRaisesRegex(ValueError, "queued request"):
            core.mark_upstream_started(queued.request_id)

    def test_identity_and_explicit_arm_are_mandatory(self):
        core = self.core()
        with self.assertRaisesRegex(ValueError, "explicit epd arm"):
            core.decide(request_id="opaque", prompt_tokens=10, output_tokens=64)
        bad = config()
        bad = RouterConfig(**{**bad.__dict__, "model_revision": "other"})
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            ElasticPDRouterCore(bad, self.profile, allow_screen_profile=True)


if __name__ == "__main__":
    unittest.main()
