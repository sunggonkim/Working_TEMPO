from __future__ import annotations

import unittest
from pathlib import Path

from eval.sota_4node import vllm_lmcache_same_server_hybrid_cold_node_v175 as node


class HybridColdV176Test(unittest.TestCase):
    def test_client_replacement(self) -> None:
        command = node._client_command(
            Path("python"), base_url="http://x", model=Path("m"),
            workload=Path("w"), output=Path("o"), mode="tempo_auto",
            run_id="r", request_rate=1.0, max_workers=1,
        )
        self.assertIn("eval.sota_4node.run_tempo_pd_same_server_cold_prewarm_client_v174", command)

    def test_launcher_is_bounded(self) -> None:
        root = Path(__file__).resolve().parent
        source = (root / "run_tempo_pd_same_server_hybrid_cold_v176_in_allocation.sh").read_text()
        self.assertEqual(source.count(" srun "), 1)
        self.assertIn("same_server_hybrid_cold_node_entry_v175.sh", source)
        self.assertIn("crossover_local/raw.json", source)


if __name__ == "__main__":
    unittest.main()
