import json
from pathlib import Path
import tempfile
import unittest

from eval.sota_4node.build_tempo_pd_elastic_profile import build_profile
from eval.sota_4node.build_tempo_pd_elastic_profile_v445 import _mad


class ElasticProfileBuilderTest(unittest.TestCase):
    def test_mad_is_deterministic_and_robust(self):
        self.assertEqual(_mad([1.0, 2.0, 3.0]), 1.0)
        self.assertEqual(_mad([1.0, 1.0, 100.0]), 0.0)

    @staticmethod
    def artifact(arm, replicate, latency_ms):
        remote = arm == "remote"
        return {
            "validation": {"performance_claim_allowed": True},
            "run": {"run_id": f"{arm}-r{replicate}"},
            "requests": [{
                "request_id": (
                    f"epd-{arm}-r{replicate}-measured-item-00"),
                "valid": True,
                "dispatch_offset_ns": 0,
                "stream_end_offset_ns": int(latency_ms * 1_000_000),
                "token_arrival_offsets_ns": [1_000_000],
                "requested_max_tokens": 16,
                "output_text_sha256": "same-output",
                "output_token_proofs": (
                    ["official_lmcache_proxy_single_prefill_token"]
                    if remote else []),
                "usage": {
                    "prompt_tokens": 513 if remote else 512,
                    "total_tokens": 529 if remote else 528,
                },
                "router": {"route": (
                    "official_lmcache_remote_prefill" if remote
                    else "decoder_local_chunked_prefill")},
            }],
        }

    def test_uncertainty_preserves_worst_paired_route_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, (arm, replicate, latency) in enumerate((
                ("local", 0, 100.0), ("remote", 0, 60.0),
                ("local", 1, 110.0), ("remote", 1, 80.0),
            )):
                path = root / f"{index}.json"
                path.write_text(json.dumps(
                    self.artifact(arm, replicate, latency)))
                paths.append(path)
            payload = build_profile(
                paths,
                deployment_scope="screen_only",
                profile_id="paired-gap-test",
                model_id="model",
                model_revision="revision",
                topology_id="topology",
                remote_backend="backend",
                classifier_version="classifier",
                kv_bytes_per_token=1,
                local_capacity_equivalent=1,
                remote_capacity_equivalent=1,
                latency_estimator="median",
                spill_regression_budget_ms=5.0,
            )
        row = payload["rows"][0]
        self.assertEqual(row["local_upper_bound_ms"], 105.0)
        self.assertEqual(row["remote_upper_bound_ms"], 70.0)
        self.assertEqual(row["uncertainty_ms"], 5.0)
        self.assertEqual(
            row["local_upper_bound_ms"]
            - row["remote_upper_bound_ms"]
            - row["uncertainty_ms"],
            30.0,
        )

    def test_replicated_profile_uses_cross_cohort_paired_lower_quartile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for cohort, gap in enumerate((-10.0, 20.0, 30.0, 40.0)):
                cohort_root = root / f"cohort-{cohort}"
                cohort_root.mkdir()
                for arm, latency in (
                    ("local", 100.0), ("remote", 100.0 - gap),
                ):
                    path = cohort_root / f"{arm}.json"
                    path.write_text(json.dumps(
                        self.artifact(arm, 0, latency)))
                    paths.append(path)
            payload = build_profile(
                paths,
                deployment_scope="replicated",
                paired_gap_lower_quantile=0.25,
                profile_id="paired-q25-test",
                model_id="model",
                model_revision="revision",
                topology_id="topology",
                remote_backend="backend",
                classifier_version="paired-gap-q25",
                kv_bytes_per_token=1,
                local_capacity_equivalent=1,
                remote_capacity_equivalent=1,
                latency_estimator="median",
                spill_regression_budget_ms=5.0,
            )
        row = payload["rows"][0]
        self.assertEqual(row["samples_local"], 4)
        self.assertEqual(row["samples_remote"], 4)
        self.assertEqual(row["local_upper_bound_ms"], 100.0)
        self.assertEqual(row["remote_upper_bound_ms"], 75.0)
        self.assertEqual(row["uncertainty_ms"], 12.5)
        self.assertEqual(
            row["local_upper_bound_ms"]
            - row["remote_upper_bound_ms"]
            - row["uncertainty_ms"],
            12.5,
        )


if __name__ == "__main__":
    unittest.main()
