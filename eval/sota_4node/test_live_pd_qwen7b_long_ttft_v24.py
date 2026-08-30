from __future__ import annotations

import unittest

from eval.sota_4node import live_pd_controller_lmcache_v20_qwen7b_long_ttft as controller
from eval.sota_4node import vllm_lmcache_live_pd_node_v24 as node


class LongTtftCompositionTest(unittest.TestCase):
    def test_frozen_workload(self) -> None:
        self.assertEqual(controller.LONG_BUCKET_REPETITIONS, (64, 256, 512))
        self.assertEqual(controller.FOREGROUND_OUTPUT_TOKENS, 2)

    def test_controller_mapping(self) -> None:
        old = node._ORIGINAL_RUN
        node._ORIGINAL_RUN = lambda command, *a, **k: command
        try:
            mapped = node._run([
                "python", "-m",
                "eval.sota_4node.live_pd_controller_lmcache_v19_qwen7b_loaded_saturated",
            ])
        finally:
            node._ORIGINAL_RUN = old
        self.assertEqual(
            mapped[-1],
            "eval.sota_4node.live_pd_controller_lmcache_v20_qwen7b_long_ttft",
        )


if __name__ == "__main__":
    unittest.main()
