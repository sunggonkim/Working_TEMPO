from __future__ import annotations

import json
from pathlib import Path
import unittest


class ControllerTests(unittest.TestCase):
    def test_controller_and_baseline_share_chunk256(self):
        root = Path(__file__).resolve().parent
        source = (root / "vllm_lmcache_tempo_pd_chunk256_node_v8.py").read_text()
        self.assertIn("base._config_text = chunk256._config_text", source)
        self.assertIn("legacy._proxy_command = chunk256._proxy_command", source)
        contract = json.loads((root / "tempo_pd_performance_contract_chunk256_v8.json").read_text())
        self.assertEqual(contract["chunk_size_tokens"], 256)
        launcher = (root / "run_tempo_pd_chunk256_v8_in_allocation.sh").read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertNotIn("salloc", launcher)


if __name__ == "__main__":
    unittest.main()
