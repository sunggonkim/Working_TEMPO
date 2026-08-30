from __future__ import annotations

import json
from pathlib import Path
import unittest


class ContinuationTests(unittest.TestCase):
    def test_reuses_exact_validation_and_runs_three_new_stages(self):
        root = Path(__file__).resolve().parent
        source = (root / "vllm_lmcache_remote_crossover_continue_node_v10.py").read_text()
        self.assertIn('args.scout_root / "crossover_local/raw.json"', source)
        self.assertIn('args.scout_root / "crossover_remote/raw.json"', source)
        self.assertIn('(\"calibration_local\", \"fixed_local\")', source)
        self.assertIn('(\"calibration_remote\", \"lmcache_always_remote\")', source)
        self.assertIn('router_mode="tempo_auto"', source)
        self.assertIn("base._config_text = chunk256._config_text", source)
        contract = json.loads(
            (root / "tempo_pd_remote_crossover_continue_contract_v10.json").read_text()
        )
        self.assertEqual(contract["request_rate_per_s"], 8.0)
        self.assertEqual(contract["chunk_size_tokens"], 256)

    def test_launcher_is_one_bounded_step(self):
        root = Path(__file__).resolve().parent
        launcher = (
            root / "run_tempo_pd_remote_crossover_continue_v10_in_allocation.sh"
        ).read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertNotIn("salloc", launcher)
        self.assertIn("--time=01:59:00", launcher)


if __name__ == "__main__":
    unittest.main()
