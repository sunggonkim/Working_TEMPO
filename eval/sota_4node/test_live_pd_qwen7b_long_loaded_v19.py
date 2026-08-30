from __future__ import annotations

import unittest

from eval.sota_4node import live_pd_controller_lmcache_v15_qwen7b_long_loaded as controller
from eval.sota_4node import vllm_lmcache_live_pd_node_v19 as node


class LongLoadedCompositionTest(unittest.TestCase):
    def test_calibration_prompts_are_identical(self) -> None:
        self.assertEqual(
            controller._prompt("calibration-remote", 2, 512),
            controller._prompt("calibration-direct", 2, 512),
        )

    def test_final_command_has_loaded_and_long_contract(self) -> None:
        old = node._ORIGINAL_COMMAND
        node._ORIGINAL_COMMAND = lambda *a, **k: [
            "vllm", "serve", "model", "--max-model-len", "2048",
            "--max-num-seqs", "2",
        ]
        try:
            command = node._vllm_command()
        finally:
            node._ORIGINAL_COMMAND = old
        self.assertEqual(command[command.index("--max-model-len") + 1], "8192")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "2")
        self.assertEqual(command[command.index("--max-num-batched-tokens") + 1], "8192")

    def test_pd_limit_matches_context(self) -> None:
        old = node._ORIGINAL_CONFIG
        node._ORIGINAL_CONFIG = lambda *a, **k: "pd_max_prefill_len: 2048\n"
        try:
            text = node._config_text()
        finally:
            node._ORIGINAL_CONFIG = old
        self.assertEqual(text, "pd_max_prefill_len: 8192\n")


if __name__ == "__main__":
    unittest.main()
