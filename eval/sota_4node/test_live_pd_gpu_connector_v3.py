from __future__ import annotations

import unittest

from eval.sota_4node import vllm_lmcache_live_pd_node_v8 as node


class GPUConnectorV3Tests(unittest.TestCase):
    def test_config_selects_v3_once(self) -> None:
        old = node._ORIGINAL_CONFIG_TEXT
        try:
            node._ORIGINAL_CONFIG_TEXT = lambda *args, **kwargs: "chunk_size: 64\n"
            text = node._config_text()
        finally:
            node._ORIGINAL_CONFIG_TEXT = old
        self.assertEqual(text.count("use_gpu_connector_v3: True"), 1)

    def test_routes_token_accurate_model_valid_client(self) -> None:
        captured = []
        old = node._ORIGINAL_RUN
        try:
            node._ORIGINAL_RUN = lambda command, *args, **kwargs: captured.append(command)
            node._run(["python", "-m", "eval.sota_4node.live_pd_controller_lmcache_v3"])
        finally:
            node._ORIGINAL_RUN = old
        self.assertIn("eval.sota_4node.live_pd_controller_lmcache_v6", captured[0])


if __name__ == "__main__":
    unittest.main()
