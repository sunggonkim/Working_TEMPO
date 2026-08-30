from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "eval/sota_4node/run_tempo_pd_perf_v6_loaded_in_allocation.sh"
CONTRACT = ROOT / "eval/sota_4node/tempo_pd_performance_contract_v4.json"


class TempoPDPerfLauncherV6Tests(unittest.TestCase):
    def test_only_offered_load_changes(self) -> None:
        value = json.loads(CONTRACT.read_text(encoding="utf-8"))
        workload = value["workload"]
        self.assertEqual(workload["request_rate_per_second"], 4.0)
        self.assertEqual(workload["max_workers"], 8)
        self.assertEqual(workload["output_tokens"], 32)
        self.assertEqual(workload["prompt_repetitions"], [64, 192, 384])

    def test_launcher_is_bounded_one_step(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertEqual(text.count(" srun "), 1)
        self.assertIn(" 4.0 8 32 3 2000 150 5000", text)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("salloc", text)


if __name__ == "__main__":
    unittest.main()
