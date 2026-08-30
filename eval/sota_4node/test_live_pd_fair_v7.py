from __future__ import annotations

import unittest

from eval.sota_4node import live_pd_controller_lmcache_v7 as client
from eval.sota_4node import vllm_lmcache_live_pd_node_v9 as node


class FairLivePDTests(unittest.TestCase):
    def test_tp4_bytes_are_not_replicated(self) -> None:
        value = client._potential_kv_bytes_tp4(100)
        self.assertEqual(value["logical_bytes"], value["tp4_physical_bytes"])
        self.assertNotIn("tp8_physical_bytes", value)

    def test_node_routes_fair_client(self) -> None:
        captured = []
        old = node._ORIGINAL_RUN
        try:
            node._ORIGINAL_RUN = lambda command, *args, **kwargs: captured.append(command)
            node._run(["python", "-m", "eval.sota_4node.live_pd_controller_lmcache_v3"])
        finally:
            node._ORIGINAL_RUN = old
        self.assertIn("eval.sota_4node.live_pd_controller_lmcache_v7", captured[0])

    def test_gpu_connector_v3_is_retained(self) -> None:
        text = node.gpu_v3._config_text(
            is_prefill=True,
            prefill_host="p",
            decode_host="d",
            ports={"proxy_notify": 1},
        )
        self.assertIn("use_gpu_connector_v3: True", text)


if __name__ == "__main__":
    unittest.main()
