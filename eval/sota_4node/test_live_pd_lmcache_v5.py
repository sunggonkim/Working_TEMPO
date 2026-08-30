from __future__ import annotations

import unittest

from eval.sota_4node import live_pd_controller_lmcache_v5 as client
from eval.sota_4node import vllm_lmcache_live_pd_node_v5 as node


class LivePDV5Tests(unittest.TestCase):
    def test_counts_batched_logprob_tokens(self) -> None:
        self.assertEqual(client._choice_token_count({
            "text": "ab", "logprobs": {"tokens": ["a", "b"]}
        }), 2)

    def test_finish_event_with_token_is_not_dropped(self) -> None:
        self.assertEqual(client._choice_token_count({
            "text": "x", "finish_reason": "length", "logprobs": {"tokens": ["x"]}
        }), 1)

    def test_empty_finish_event_is_zero(self) -> None:
        self.assertEqual(client._choice_token_count({
            "text": "", "finish_reason": "length", "logprobs": {"tokens": []}
        }), 0)

    def test_node_routes_client_v5(self) -> None:
        captured = []
        old = node._ORIGINAL_RUN
        try:
            node._ORIGINAL_RUN = lambda command, *args, **kwargs: captured.append(command)
            node._run(["python", "-m", "eval.sota_4node.live_pd_controller_lmcache_v3"])
        finally:
            node._ORIGINAL_RUN = old
        self.assertIn("eval.sota_4node.live_pd_controller_lmcache_v5", captured[0])


if __name__ == "__main__":
    unittest.main()
