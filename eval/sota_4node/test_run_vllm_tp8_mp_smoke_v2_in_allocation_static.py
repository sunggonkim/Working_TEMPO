from __future__ import annotations

import ast
import re
import subprocess
import unittest
from pathlib import Path


HERE = Path(__file__).parent
LAUNCHER = HERE / "run_vllm_tp8_mp_smoke_v2_in_allocation.sh"
ENTRY = HERE / "vllm_tp8_mp_smoke_node_entry_v2.sh"
DRIVER = HERE / "vllm_tp8_mp_smoke_node_v2.py"


class VllmTp8MpSmokeV2StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = LAUNCHER.read_text()
        cls.entry = ENTRY.read_text()
        cls.driver = DRIVER.read_text()

    def test_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        subprocess.run(["bash", "-n", str(ENTRY)], check=True)
        ast.parse(self.driver)

    def test_one_bounded_srun_and_no_scheduler_mutation(self) -> None:
        self.assertEqual(len(re.findall(r"\bsrun\b", self.launcher)), 1)
        self.assertIn("timeout --foreground --signal=TERM --kill-after=15s 600s", self.launcher)
        self.assertIn("--nodes=2 --ntasks=2 --ntasks-per-node=1", self.launcher)
        self.assertIn("--gpus-per-task=4 --gpu-bind=none", self.launcher)
        self.assertIsNone(re.search(r"(?m)^\s*(?:salloc|sbatch|scancel)\b", self.launcher))

    def test_v2_is_wired_end_to_end(self) -> None:
        self.assertIn("vllm_tp8_mp_smoke_node_entry_v2.sh", self.launcher)
        self.assertIn("vllm_tp8_mp_smoke_node_v2.py", self.entry)

    def test_exact_16_non_final_event_contract(self) -> None:
        self.assertIn('"min_tokens": EXPECTED_TOKEN_CHUNKS', self.driver)
        self.assertIn('"ignore_eos": True', self.driver)
        self.assertIn('choice.get("finish_reason") is not None', self.driver)
        self.assertIn("chunks += 1", self.driver)
        self.assertIn("chunks != EXPECTED_TOKEN_CHUNKS", self.driver)
        self.assertLess(self.driver.index('choice.get("finish_reason")'), self.driver.index("chunks += 1"))
        self.assertNotIn("if text:", self.driver)

    def test_tp8_offline_node_local_contract(self) -> None:
        self.assertIn('"--tensor-parallel-size",\n        "8"', self.driver)
        self.assertIn('"--distributed-executor-backend",\n        "mp"', self.driver)
        self.assertIn("HF_HUB_OFFLINE=1", self.entry)
        self.assertIn("FLASHINFER_WORKSPACE_BASE", self.entry)
        self.assertIn("/tmp/tempo-vllm-", self.entry)


if __name__ == "__main__":
    unittest.main()
