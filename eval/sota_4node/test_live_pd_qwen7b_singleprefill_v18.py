from __future__ import annotations

import unittest

from eval.sota_4node import vllm_lmcache_live_pd_node_v18 as node


class FinalHookSinglePrefillTest(unittest.TestCase):
    def test_final_hook_appends_limit(self) -> None:
        old = node._ORIGINAL_FINAL_COMMAND
        node._ORIGINAL_FINAL_COMMAND = lambda *a, **k: [
            "vllm", "serve", "model", "--max-model-len", "8192"
        ]
        try:
            command = node._vllm_command()
        finally:
            node._ORIGINAL_FINAL_COMMAND = old
        self.assertEqual(command.count("--max-num-batched-tokens"), 1)
        self.assertEqual(command[-2:], ["--max-num-batched-tokens", "8192"])

    def test_main_installs_final_hook(self) -> None:
        self.assertIsNot(node._vllm_command, node._ORIGINAL_FINAL_COMMAND)
        self.assertEqual(node._vllm_command.__module__, node.__name__)


if __name__ == "__main__":
    unittest.main()
