from __future__ import annotations

from pathlib import Path
import unittest


class PollingTests(unittest.TestCase):
    def test_frozen_single_factor_and_sender_only_telemetry(self):
        root = Path(__file__).resolve().parent
        source = (root / "vllm_lmcache_nixl_polling_snapshot_node_v3.py").read_text()
        self.assertIn('"TEMPO_NIXL_CACHE_CAPACITY": "0"', source)
        self.assertIn('"TEMPO_NIXL_YIELD_POLLS": "4096"', source)
        self.assertIn("if args.node_index % 2 == 0", source)
        launcher = (root / "run_lmcache_nixl_polling_v3_in_allocation.sh").read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertNotIn("salloc", launcher)


if __name__ == "__main__":
    unittest.main()
