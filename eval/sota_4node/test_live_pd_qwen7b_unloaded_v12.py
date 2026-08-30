from __future__ import annotations

import unittest

from eval.sota_4node import live_pd_controller_lmcache_v12_qwen7b_unloaded as client
from eval.sota_4node import vllm_lmcache_live_pd_node_v14 as node


class Qwen7BUnloadedTests(unittest.TestCase):
    def test_kv_geometry(self) -> None:
        self.assertEqual(
            client._potential_kv_bytes_tp4(1)["logical_bytes"],
            28 * 4 * 128 * 2 * 2,
        )

    def test_routes_unloaded_client(self) -> None:
        captured = []
        old = node._ORIGINAL_RUN
        try:
            node._ORIGINAL_RUN = lambda command, *args, **kwargs: captured.append(command)
            node._run(["python", "-m", "eval.sota_4node.live_pd_controller_lmcache_v7"])
        finally:
            node._ORIGINAL_RUN = old
        self.assertIn("eval.sota_4node.live_pd_controller_lmcache_v12_qwen7b_unloaded", captured[0])


if __name__ == "__main__":
    unittest.main()
