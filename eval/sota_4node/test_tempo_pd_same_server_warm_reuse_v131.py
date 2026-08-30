from __future__ import annotations

import unittest

from eval.sota_4node import run_tempo_pd_same_server_warm_reuse_client_v131 as client


class WarmReuseClientTest(unittest.TestCase):
    def test_prompt_keys_are_stable_within_arm(self) -> None:
        rows = [{"request_id": "r0", "prompt": "nonce 001. payload", "max_tokens": 16}]
        warm = client._derive(rows, prefix="ssb-tempo-r0-warm-", offset=300)
        first = client._derive(rows, prefix="ssb-tempo-r0-measured-", offset=500)
        second = client._derive(rows, prefix="ssb-tempo-r1-measured-", offset=800)
        self.assertEqual(warm[0]["prompt"], first[0]["prompt"])
        self.assertEqual(first[0]["prompt"], second[0]["prompt"])
        self.assertIn("nonce 201.", warm[0]["prompt"])
        self.assertEqual(len({warm[0]["request_id"], first[0]["request_id"],
                              second[0]["request_id"]}), 3)

    def test_prompt_keys_are_isolated_across_arms(self) -> None:
        rows = [{"request_id": "r0", "prompt": "nonce 001. payload", "max_tokens": 16}]
        prompts = {
            client._derive(rows, prefix=f"ssb-{arm}-r0-warm-", offset=0)[0]["prompt"]
            for arm in ("local", "tempo", "remote")
        }
        self.assertEqual(len(prompts), 3)


if __name__ == "__main__":
    unittest.main()
