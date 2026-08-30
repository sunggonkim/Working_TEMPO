from __future__ import annotations

import unittest

from eval.sota_4node import live_pd_controller_lmcache_v13_qwen7b_sameprompt as client
from eval.sota_4node import vllm_lmcache_live_pd_node_v15 as node


class SamePromptTests(unittest.TestCase):
    def test_calibration_prompts_identical(self) -> None:
        remote = client._prompt("calibration-remote", 1, 64)
        direct = client._prompt("calibration-direct", 1, 64)
        self.assertEqual(remote, direct)

    def test_non_calibration_prompt_unchanged(self) -> None:
        self.assertEqual(
            client._prompt("validation", 0, 16),
            client._ORIGINAL_PROMPT("validation", 0, 16),
        )

    def test_routes_sameprompt_client(self) -> None:
        captured = []
        old = node._ORIGINAL_RUN
        try:
            node._ORIGINAL_RUN = lambda command, *args, **kwargs: captured.append(command)
            node._run(["python", "-m", "eval.sota_4node.live_pd_controller_lmcache_v12_qwen7b_unloaded"])
        finally:
            node._ORIGINAL_RUN = old
        self.assertIn("eval.sota_4node.live_pd_controller_lmcache_v13_qwen7b_sameprompt", captured[0])


if __name__ == "__main__":
    unittest.main()
