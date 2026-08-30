from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eval.sota_4node import vllm_lmcache_same_server_hybrid_cold_node_v178 as node


class HybridColdV179Test(unittest.TestCase):
    def test_only_obsolete_capacity_analyzer_is_bypassed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            command = ["python", "-m", "eval.sota_4node.analyze_tempo_pd_capacity_v13",
                       "--output", str(output)]
            result = node._CapacityAnalysisBypass.run(command)
            self.assertEqual(result.returncode, 0)
            self.assertIn("obsolete-capacity-analysis-bypass", output.read_text())

    def test_launcher_is_one_bounded_step(self) -> None:
        root = Path(__file__).resolve().parent
        source = (root / "run_tempo_pd_same_server_hybrid_cold_v179_in_allocation.sh").read_text()
        self.assertEqual(source.count(" srun "), 1)
        self.assertIn("same_server_hybrid_cold_node_entry_v178.sh", source)
        self.assertNotIn("crossover_local/raw.json", source)


if __name__ == "__main__":
    unittest.main()
