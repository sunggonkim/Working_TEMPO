from __future__ import annotations

import json
from pathlib import Path
import unittest


class RemoteCrossoverTests(unittest.TestCase):
    def test_exact_two_arm_chunk256_contract(self):
        root = Path(__file__).resolve().parent
        source = (root / "vllm_lmcache_remote_crossover_node_v9.py").read_text()
        self.assertIn('("crossover_local", "fixed_local")', source)
        self.assertIn('("crossover_remote", "lmcache_always_remote")', source)
        self.assertIn("base._config_text = chunk256._config_text", source)
        self.assertIn("legacy._proxy_command = chunk256._proxy_command", source)
        contract = json.loads(
            (root / "tempo_pd_remote_crossover_contract_v9.json").read_text()
        )
        self.assertEqual(contract["chunk_size_tokens"], 256)

    def test_launcher_is_bounded_and_parameterized(self):
        root = Path(__file__).resolve().parent
        launcher = (root / "run_tempo_pd_remote_crossover_v9_in_allocation.sh").read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertNotIn("salloc", launcher)
        self.assertIn('"${RATE}" "${WORKERS}" 32 3', launcher)
        self.assertIn("--time=01:29:00", launcher)


if __name__ == "__main__":
    unittest.main()
