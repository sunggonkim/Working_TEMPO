from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval/sota_4node"


class LauncherTests(unittest.TestCase):
    def test_contract_and_launcher_are_frozen(self):
        contract = json.loads((EVAL / "lmcache_nixl_hotpath_ab_contract_v1.json").read_text())
        self.assertEqual(contract["routes"], ["stock_lmcache_remote", "tempo_nixl_remote"])
        self.assertEqual(contract["output_tokens"], 32)
        launcher = (EVAL / "run_lmcache_nixl_hotpath_ab_v1_in_allocation.sh").read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertIn("--nodes=4 --ntasks=4", launcher)
        self.assertNotIn("sbatch", launcher)
        self.assertNotIn("salloc", launcher)

    def test_node_uses_same_remote_policy_and_only_optimized_env(self):
        source = (EVAL / "vllm_lmcache_nixl_hotpath_ab_node_v1.py").read_text()
        self.assertIn('router_mode="lmcache_always_remote"', source)
        self.assertIn('if mode == "tempo_nixl_remote"', source)
        self.assertIn('"TEMPO_LMCACHE_NIXL_HOTPATH": "1"', source)
        self.assertIn("run_tempo_pd_stream_metrics_v3", (
            EVAL / "vllm_lmcache_tempo_pd_perf_node_v4.py").read_text())


if __name__ == "__main__":
    unittest.main()
