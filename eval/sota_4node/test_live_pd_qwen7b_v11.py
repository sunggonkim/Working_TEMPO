from __future__ import annotations

import unittest
from pathlib import Path

from eval.sota_4node import live_pd_controller_lmcache_v11_qwen7b as client
from eval.sota_4node import vllm_lmcache_live_pd_node_v13 as node


class Qwen7BLivePDTests(unittest.TestCase):
    def test_qwen_tp4_kv_geometry(self) -> None:
        value = client._potential_kv_bytes_tp4(1)
        self.assertEqual(value["logical_bytes"], 28 * 4 * 128 * 2 * 2)
        self.assertEqual(value["logical_bytes"], value["tp4_physical_bytes"])

    def test_model_path_substitution_is_exact(self) -> None:
        root = Path("/workspace")
        self.assertEqual(
            str(node._div(root, node._TINY_RELATIVE)),
            "/workspace/models/Qwen2.5-7B-Instruct",
        )
        self.assertEqual(str(node._div(root, "other")), "/workspace/other")

    def test_routes_qwen_client(self) -> None:
        captured = []
        old = node._ORIGINAL_RUN
        try:
            node._ORIGINAL_RUN = lambda command, *args, **kwargs: captured.append(command)
            node._run(["python", "-m", "eval.sota_4node.live_pd_controller_lmcache_v10_streamsync"])
        finally:
            node._ORIGINAL_RUN = old
        self.assertIn("eval.sota_4node.live_pd_controller_lmcache_v11_qwen7b", captured[0])


if __name__ == "__main__":
    unittest.main()
