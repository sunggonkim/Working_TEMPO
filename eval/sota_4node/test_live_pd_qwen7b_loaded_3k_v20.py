from __future__ import annotations

import unittest

from eval.sota_4node import live_pd_controller_lmcache_v16_qwen7b_loaded_3k as controller
from eval.sota_4node import vllm_lmcache_live_pd_node_v20 as node


class Loaded3KCompositionTest(unittest.TestCase):
    def test_frozen_buckets(self) -> None:
        self.assertEqual(controller.LOADED_BUCKET_REPETITIONS, (64, 192, 256))

    def test_controller_mapping(self) -> None:
        old = node._ORIGINAL_RUN
        node._ORIGINAL_RUN = lambda command, *a, **k: command
        try:
            mapped = node._run([
                "python", "-m",
                "eval.sota_4node.live_pd_controller_lmcache_v15_qwen7b_long_loaded",
            ])
        finally:
            node._ORIGINAL_RUN = old
        self.assertEqual(
            mapped[-1],
            "eval.sota_4node.live_pd_controller_lmcache_v16_qwen7b_loaded_3k",
        )


if __name__ == "__main__":
    unittest.main()
