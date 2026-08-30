import json
from pathlib import Path
import tempfile
import unittest

from eval.sota_4node import tempo_pd_elastic_router_v444 as wire
from eval.sota_4node.tempo_pd_elastic_router_v449 import ElasticPDRouterCore
from eval.sota_4node.test_tempo_pd_elastic_router_v445 import config, profile_payload
from tempo.pd_elastic_controller_v443 import CacheResidency, ElasticRoute
from tempo.pd_elastic_profile_v444 import load_elastic_profile


class FirstResponseCreditReleaseTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "profile.json"
        path.write_text(json.dumps(profile_payload()))
        profile = load_elastic_profile(path)
        self.core = ElasticPDRouterCore(
            config(), profile, allow_screen_profile=True,
            cache_residency=lambda _request_id: CacheResidency.MISS,
        )

    def test_first_response_releases_credit_before_stream_completion(self):
        first = self.core.decide(
            request_id="epd-tempo-first", prompt_tokens=10, output_tokens=64)
        queued = self.core.decide(
            request_id="epd-tempo-second", prompt_tokens=10, output_tokens=64)
        self.assertEqual((first.route, queued.route),
                         (ElasticRoute.REMOTE, ElasticRoute.QUEUE))
        self.core.mark_upstream_started(first.request_id)
        self.core.mark_first_response_chunk(first.request_id)
        retried = self.core.retry(queued.request_id, 1_000_000.0)
        self.assertEqual(retried.route, ElasticRoute.REMOTE)
        self.assertEqual(self.core.elastic.remote_kv_used_bytes, 1000)
        self.core.complete(first.request_id)
        self.assertEqual(self.core.elastic.remote_kv_used_bytes, 1000)
        row = next(value for value in self.core.records()
                   if value["request_id"] == first.request_id)
        self.assertEqual(row["admission_credit_release_event"],
                         "first_response_chunk")
        self.assertIsNotNone(row["admission_credit_released_ns"])

    def test_stream_failure_after_release_does_not_double_release(self):
        row = self.core.decide(
            request_id="epd-tempo-fail", prompt_tokens=10, output_tokens=64)
        self.core.mark_upstream_started(row.request_id)
        self.core.mark_first_response_chunk(row.request_id)
        self.core.fail(row.request_id, "late stream failure")
        self.assertEqual(self.core.elastic.remote_kv_used_bytes, 0)
        with self.assertRaisesRegex(ValueError, "twice"):
            self.core.mark_first_response_chunk(row.request_id)

    def test_frozen_public_wire_schema_is_preserved(self):
        from eval.sota_4node import tempo_pd_elastic_router_v449 as launch
        self.assertIs(launch.runtime._headers, wire._headers)


if __name__ == "__main__":
    unittest.main()
