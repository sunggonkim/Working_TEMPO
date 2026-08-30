from __future__ import annotations

from pathlib import Path
import unittest


class LibfabricTests(unittest.TestCase):
    def test_explicit_backend_and_provider(self):
        root = Path(__file__).resolve().parent
        source = (root / "vllm_lmcache_libfabric_node_v5.py").read_text()
        self.assertIn('"nixl_backends: [LIBFABRIC]"', source)
        self.assertIn('"FI_PROVIDER": "cxi"', source)
        self.assertIn(".nixl_cu12.mesonpy.libs", source)
        launcher = (root / "run_lmcache_libfabric_v5_in_allocation.sh").read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertNotIn("salloc", launcher)


if __name__ == "__main__":
    unittest.main()
