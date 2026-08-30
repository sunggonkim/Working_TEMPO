from __future__ import annotations

import unittest

from eval.sota_4node import vllm_lmcache_live_pd_node_v17 as node


class SinglePrefillCommandTest(unittest.TestCase):
    def test_frozen_batch_limit_is_appended_once(self) -> None:
        old = node._ORIGINAL_COMMAND
        node._ORIGINAL_COMMAND = lambda *a, **k: ["vllm", "serve", "model"]
        try:
            command = node._vllm_command()
        finally:
            node._ORIGINAL_COMMAND = old
        self.assertEqual(command.count("--max-num-batched-tokens"), 1)
        self.assertEqual(command[-2:], ["--max-num-batched-tokens", "8192"])

    def test_existing_batch_limit_is_replaced(self) -> None:
        old = node._ORIGINAL_COMMAND
        node._ORIGINAL_COMMAND = lambda *a, **k: [
            "vllm", "serve", "model", "--max-num-batched-tokens", "2048"
        ]
        try:
            command = node._vllm_command()
        finally:
            node._ORIGINAL_COMMAND = old
        self.assertEqual(command[-1], "8192")
        self.assertEqual(command.count("--max-num-batched-tokens"), 1)


if __name__ == "__main__":
    unittest.main()
