from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from eval.sota_4node import vllm_lmcache_nixl_polling_balanced_node_v4 as balanced


class BalancedTests(unittest.TestCase):
    def test_only_yield_budget_changes(self):
        with patch.object(balanced, "_ORIGINAL_ENVIRONMENT", return_value={
            "TEMPO_NIXL_CACHE_CAPACITY": "0",
            "TEMPO_NIXL_YIELD_POLLS": "4096",
            "TEMPO_NIXL_SLEEP_US": "100",
        }):
            env = balanced._environment()
        self.assertEqual(env["TEMPO_NIXL_CACHE_CAPACITY"], "0")
        self.assertEqual(env["TEMPO_NIXL_YIELD_POLLS"], "16")
        self.assertEqual(env["TEMPO_NIXL_SLEEP_US"], "100")

    def test_launcher_is_bounded(self):
        root = Path(__file__).resolve().parent
        source = (root / "run_lmcache_nixl_polling_balanced_v4_in_allocation.sh").read_text()
        self.assertEqual(source.count("srun "), 1)
        self.assertNotIn("salloc", source)


if __name__ == "__main__":
    unittest.main()
