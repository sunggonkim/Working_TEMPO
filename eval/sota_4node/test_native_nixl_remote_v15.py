from __future__ import annotations

import json
from pathlib import Path
import unittest

from eval.sota_4node import native_nixl_pd_proxy_v15 as proxy


class NativeNixlTests(unittest.TestCase):
    def test_native_connector_and_proxy_contract(self):
        root = Path(__file__).resolve().parent
        source = (root / "vllm_native_nixl_remote_node_v15.py").read_text()
        self.assertIn('"kv_connector": "NixlConnector"', source)
        self.assertIn('"kv_role": role', source)
        self.assertIn('"backends": ["UCX"]', source)
        self.assertIn("native_nixl_pd_proxy_v15", source)
        self.assertEqual(proxy.SCHEMA, "tempo-native-nixl-pd-proxy-15")

    def test_launcher_is_one_bounded_step(self):
        root = Path(__file__).resolve().parent
        launcher = (root / "run_native_nixl_remote_v15_in_allocation.sh").read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertNotIn("salloc", launcher)


if __name__ == "__main__":
    unittest.main()
