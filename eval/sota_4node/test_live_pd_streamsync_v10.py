from __future__ import annotations

import unittest

from eval.sota_4node import live_pd_controller_lmcache_v10_streamsync as client
from eval.sota_4node import vllm_lmcache_live_pd_node_v12 as node


class StreamSyncTests(unittest.TestCase):
    def test_node_routes_stream_sync_client(self) -> None:
        captured = []
        old = node._ORIGINAL_RUN
        try:
            node._ORIGINAL_RUN = lambda command, *args, **kwargs: captured.append(command)
            node._run(["python", "-m", "eval.sota_4node.live_pd_controller_lmcache_v8_loaded"])
        finally:
            node._ORIGINAL_RUN = old
        self.assertIn("eval.sota_4node.live_pd_controller_lmcache_v10_streamsync", captured[0])

    def test_first_token_sync_is_not_fixed_sleep(self) -> None:
        self.assertNotIn("BACKGROUND_HEADSTART_S", client._with_background.__code__.co_names)


if __name__ == "__main__":
    unittest.main()
