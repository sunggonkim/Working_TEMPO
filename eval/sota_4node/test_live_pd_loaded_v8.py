from __future__ import annotations

import unittest

from eval.sota_4node import live_pd_controller_lmcache_v8_loaded as client
from eval.sota_4node import vllm_lmcache_live_pd_node_v10 as node


class LoadedLivePDTests(unittest.TestCase):
    def test_background_contract(self) -> None:
        self.assertEqual(client.BACKGROUND_TOKENS, 128)
        self.assertEqual(client.BACKGROUND_HEADSTART_S, 0.15)

    def test_max_num_seqs_two(self) -> None:
        old = node.cli_compatible._vllm_command
        try:
            node.cli_compatible._vllm_command = lambda *args, **kwargs: [
                "vllm", "--max-num-seqs", "1"
            ]
            command = node._vllm_command()
        finally:
            node.cli_compatible._vllm_command = old
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "2")

    def test_routes_loaded_client(self) -> None:
        captured = []
        old = node._ORIGINAL_RUN
        try:
            node._ORIGINAL_RUN = lambda command, *args, **kwargs: captured.append(command)
            node._run(["python", "-m", "eval.sota_4node.live_pd_controller_lmcache_v3"])
        finally:
            node._ORIGINAL_RUN = old
        self.assertIn("eval.sota_4node.live_pd_controller_lmcache_v8_loaded", captured[0])


if __name__ == "__main__":
    unittest.main()
