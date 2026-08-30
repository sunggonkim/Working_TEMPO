from __future__ import annotations

import unittest

from eval.sota_4node import live_pd_controller_lmcache_v6 as client
from eval.sota_4node import vllm_lmcache_live_pd_node_v7 as node


class LivePDV6Tests(unittest.TestCase):
    def test_largest_bucket_is_below_failed_repetition(self) -> None:
        self.assertEqual(client.BUCKET_REPETITIONS, (16, 64, 96))
        self.assertLess(max(client.BUCKET_REPETITIONS), 128)

    def test_node_routes_production_client(self) -> None:
        captured = []
        old = node._ORIGINAL_RUN
        try:
            node._ORIGINAL_RUN = lambda command, *args, **kwargs: captured.append(command)
            node._run(["python", "-m", "eval.sota_4node.live_pd_controller_lmcache_v3"])
        finally:
            node._ORIGINAL_RUN = old
        self.assertIn("eval.sota_4node.live_pd_controller_lmcache_v6", captured[0])


if __name__ == "__main__":
    unittest.main()
