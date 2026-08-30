from __future__ import annotations

import unittest

from eval.sota_4node import live_pd_controller_lmcache_v17_qwen7b_loaded_short as controller
from eval.sota_4node import vllm_lmcache_live_pd_node_v21 as node


class LoadedShortCompositionTest(unittest.TestCase):
    def test_three_distinct_bucket_labels_share_length(self) -> None:
        self.assertEqual(controller.SHORT_BUCKET_REPETITIONS, (64, 64, 64))
        prompts = [controller.loaded._prompt("validation", i, 64) for i in range(3)]
        self.assertEqual(len(set(prompts)), 3)

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
            "eval.sota_4node.live_pd_controller_lmcache_v17_qwen7b_loaded_short",
        )


if __name__ == "__main__":
    unittest.main()
