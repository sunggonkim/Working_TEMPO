from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "eval/sota_4node/run_tempo_pd_perf_v5_in_allocation.sh"
CONTRACT = ROOT / "eval/sota_4node/tempo_pd_performance_contract_v3.json"


class TempoPDPerfLauncherV5Tests(unittest.TestCase):
    def test_exact_output_window_and_tpot_samples_are_frozen(self) -> None:
        value = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(value["workload"]["output_tokens"], 32)
        self.assertGreaterEqual(value["workload"]["output_tokens"] - 1, 31)
        self.assertLessEqual(
            max(value["workload"]["expected_prompt_tokens_for_frozen_model"]) + 32,
            value["model_max_length"],
        )

    def test_launcher_uses_coalescing_aware_node_and_one_step(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertEqual(text.count(" srun "), 1)
        self.assertIn("tempo_pd_perf_node_entry_v4.sh", text)
        self.assertIn(" 2.0 4 32 3 1000 100 3000", text)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("salloc", text)


if __name__ == "__main__":
    unittest.main()
