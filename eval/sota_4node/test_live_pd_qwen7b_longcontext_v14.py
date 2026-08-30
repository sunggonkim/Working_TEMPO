from __future__ import annotations

import unittest

from eval.sota_4node import live_pd_controller_lmcache_v14_qwen7b_longcontext as client
from eval.sota_4node import vllm_lmcache_live_pd_node_v16 as node


class QwenLongContextTests(unittest.TestCase):
    def test_long_buckets(self) -> None:
        self.assertEqual(client.LONG_BUCKET_REPETITIONS, (64, 256, 512))

    def test_max_model_len(self) -> None:
        old = node._ORIGINAL_COMMAND
        try:
            node._ORIGINAL_COMMAND = lambda *args, **kwargs: [
                "vllm", "--max-model-len", "2048"
            ]
            command = node._vllm_command()
        finally:
            node._ORIGINAL_COMMAND = old
        self.assertEqual(command[-1], "8192")

    def test_pd_limit(self) -> None:
        old = node._ORIGINAL_CONFIG
        try:
            node._ORIGINAL_CONFIG = lambda *args, **kwargs: "pd_max_prefill_len: 2048\nuse_gpu_connector_v3: True\n"
            text = node._config_text()
        finally:
            node._ORIGINAL_CONFIG = old
        self.assertIn("pd_max_prefill_len: 8192", text)
        self.assertIn("use_gpu_connector_v3: True", text)


if __name__ == "__main__":
    unittest.main()
