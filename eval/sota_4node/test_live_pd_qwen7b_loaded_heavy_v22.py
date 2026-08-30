from __future__ import annotations

import unittest

from eval.sota_4node import live_pd_controller_lmcache_v18_qwen7b_loaded_heavy as controller
from eval.sota_4node import vllm_lmcache_live_pd_node_v22 as node


class HeavyLoadedCompositionTest(unittest.TestCase):
    def test_three_background_streams(self) -> None:
        self.assertEqual(controller.BACKGROUND_STREAMS, 3)

    def test_server_capacity_is_four(self) -> None:
        old = node._ORIGINAL_COMMAND
        node._ORIGINAL_COMMAND = lambda *a, **k: [
            "vllm", "serve", "model", "--max-num-seqs", "2"
        ]
        try:
            command = node._vllm_command()
        finally:
            node._ORIGINAL_COMMAND = old
        self.assertEqual(command[-1], "4")

    def test_heavy_controller_mapping(self) -> None:
        old = node._ORIGINAL_RUN
        node._ORIGINAL_RUN = lambda command, *a, **k: command
        try:
            mapped = node._run([
                "python", "-m",
                "eval.sota_4node.live_pd_controller_lmcache_v17_qwen7b_loaded_short",
            ])
        finally:
            node._ORIGINAL_RUN = old
        self.assertEqual(
            mapped[-1],
            "eval.sota_4node.live_pd_controller_lmcache_v18_qwen7b_loaded_heavy",
        )


if __name__ == "__main__":
    unittest.main()
