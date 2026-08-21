from __future__ import annotations

import copy
import unittest

from eval.sota_4node.build_tempo_pd_endpoint_service_profile import (
    collect_service_rows,
)


def _artifact(arm: str, replicate: int, ttft_ms: int) -> dict[str, object]:
    request_id = f"epd-{arm}-r{replicate}-measured-item-00"
    route = (
        "decoder_local_chunked_prefill"
        if arm == "local"
        else "official_lmcache_remote_prefill"
    )
    return {
        "validation": {
            "all_streams_valid": True,
            "performance_claim_allowed": True,
            "router_decisions_exact": True,
        },
        "requests": [{
            "dispatch_offset_ns": 1_000_000,
            "output_text_sha256": "a" * 64,
            "request_id": request_id,
            "requested_max_tokens": 16,
            "token_arrival_offsets_ns": [1_000_000 + ttft_ms * 1_000_000],
            "valid": True,
        }],
        "router_decisions": [{
            "cache_residency": "prefill_only",
            "output_tokens": 16,
            "prompt_tokens": 512,
            "request_id": request_id,
            "route": route,
        }],
    }


class EndpointServiceProfileBuilderTest(unittest.TestCase):
    def test_paired_medians_and_token_ms_are_exact(self) -> None:
        rows = collect_service_rows([
            _artifact("local", 0, 10),
            _artifact("remote", 0, 30),
            _artifact("remote", 1, 50),
            _artifact("local", 1, 20),
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cache_residency"], "prefill_only")
        self.assertEqual(rows[0]["local_ttft_prior_ms"], 15.0)
        self.assertEqual(rows[0]["remote_ttft_prior_ms"], 40.0)
        self.assertEqual(rows[0]["local_token_ms"], 7_680)
        self.assertEqual(rows[0]["remote_prefill_token_ms"], 20_480)
        self.assertEqual(rows[0]["samples_local"], 2)
        self.assertEqual(rows[0]["samples_remote"], 2)

    def test_mismatched_paired_output_is_rejected(self) -> None:
        artifacts = [
            _artifact("local", 0, 10),
            _artifact("remote", 0, 30),
            _artifact("remote", 1, 50),
            _artifact("local", 1, 20),
        ]
        changed = copy.deepcopy(artifacts[2])
        changed["requests"][0]["output_text_sha256"] = "b" * 64
        artifacts[2] = changed
        with self.assertRaisesRegex(ValueError, "not equivalent"):
            collect_service_rows(artifacts)

    def test_unknown_cache_residency_is_rejected(self) -> None:
        artifact = _artifact("local", 0, 10)
        artifact["router_decisions"][0]["cache_residency"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown cache"):
            collect_service_rows([artifact])


if __name__ == "__main__":
    unittest.main()
