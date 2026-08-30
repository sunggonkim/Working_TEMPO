from __future__ import annotations

from pathlib import Path
import unittest


class NativeNixlV17Tests(unittest.TestCase):
    def test_internal_id_proxy_and_bounded_launcher(self):
        root = Path(__file__).resolve().parent
        proxy = (root / "native_nixl_pd_proxy_v17.py").read_text()
        self.assertIn("external_id or", proxy)
        self.assertIn("uuid.uuid4().hex", proxy)
        node = (root / "vllm_native_nixl_remote_node_v17.py").read_text()
        self.assertIn("base._proxy_command = _proxy_command", node)
        launcher = (root / "run_native_nixl_remote_v17_in_allocation.sh").read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertNotIn("salloc", launcher)


if __name__ == "__main__":
    unittest.main()
