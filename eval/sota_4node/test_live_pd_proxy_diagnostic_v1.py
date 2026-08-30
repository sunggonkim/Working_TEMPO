from __future__ import annotations

import unittest

from eval.sota_4node import vllm_lmcache_live_pd_node_v6 as node


class DiagnosticProxyTests(unittest.TestCase):
    def test_proxy_command_preserves_arguments(self) -> None:
        old = node._ORIGINAL_PROXY_COMMAND
        try:
            node._ORIGINAL_PROXY_COMMAND = lambda *args, **kwargs: ["python", "/upstream.py", "--port", "1"]
            command = node._proxy_command()
        finally:
            node._ORIGINAL_PROXY_COMMAND = old
        self.assertEqual(command[:3], ["python", "-m", "eval.sota_4node.lmcache_disagg_proxy_diagnostic_v1"])
        self.assertEqual(command[3:], ["--port", "1"])


if __name__ == "__main__":
    unittest.main()
