import unittest

from eval.sota_4node.build_tempo_pd_elastic_profile_v446 import normalize_artifact


class ProxyPromptNormalizationTest(unittest.TestCase):
    def test_only_proven_remote_head_is_normalized(self):
        artifact = {"requests": [
            {"router": {"route": "decoder_local_recompute_or_cache"},
             "usage": {"prompt_tokens": 10, "total_tokens": 12}},
            {"router": {"route": "remote_prefill_live_kv"},
             "output_token_proofs": ["official_lmcache_proxy_single_prefill_token"],
             "usage": {"prompt_tokens": 11, "total_tokens": 13}},
        ]}
        normalized, count = normalize_artifact(artifact)
        self.assertEqual(count, 1)
        self.assertEqual(normalized["requests"][0]["usage"]["prompt_tokens"], 10)
        self.assertEqual(normalized["requests"][1]["usage"]["prompt_tokens"], 10)

    def test_remote_without_exact_proof_fails_closed(self):
        artifact = {"requests": [{
            "router": {"route": "remote_prefill_live_kv"},
            "output_token_proofs": [],
            "usage": {"prompt_tokens": 11, "total_tokens": 13},
        }]}
        with self.assertRaisesRegex(ValueError, "head-token proof"):
            normalize_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
