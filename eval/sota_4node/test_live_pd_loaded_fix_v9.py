from __future__ import annotations

import unittest

from eval.sota_4node import live_pd_controller_lmcache_v9_loaded_fix as client
from eval.sota_4node import vllm_lmcache_live_pd_node_v11 as node


class LoadedFixTests(unittest.TestCase):
    def test_nested_callback_is_normalized(self) -> None:
        old = client._ORIGINAL_WITH_BACKGROUND
        try:
            client._ORIGINAL_WITH_BACKGROUND = (
                lambda *args, foreground, **kwargs: foreground()
            )
            value = client._with_background("d", 0, 0, "t", lambda: lambda: {"ok": True})
        finally:
            client._ORIGINAL_WITH_BACKGROUND = old
        self.assertEqual(value, {"ok": True})

    def test_node_routes_fixed_client(self) -> None:
        captured = []
        old = node._ORIGINAL_RUN
        try:
            node._ORIGINAL_RUN = lambda command, *args, **kwargs: captured.append(command)
            node._run(["python", "-m", "eval.sota_4node.live_pd_controller_lmcache_v8_loaded"])
        finally:
            node._ORIGINAL_RUN = old
        self.assertIn("eval.sota_4node.live_pd_controller_lmcache_v9_loaded_fix", captured[0])


if __name__ == "__main__":
    unittest.main()
