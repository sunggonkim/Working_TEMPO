from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_inference_epoch_in_allocation.sh")


class InferenceEpochAllocationLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_requires_existing_two_node_allocation_and_approval(self) -> None:
        self.assertIn("SLURM_JOB_ID", self.text)
        self.assertIn("SLURM_JOB_NUM_NODES", self.text)
        self.assertIn("TEMPO_INFERENCE_EPOCH_APPROVED", self.text)
        self.assertIn("module load pytorch/2.8.0", self.text)
        self.assertIsNone(re.search(r"(?m)^\s*(?:salloc|sbatch|scancel)\b", self.text))

    def test_exactly_one_bounded_srun(self) -> None:
        self.assertEqual(len(re.findall(r"\bsrun\b", self.text)), 1)
        self.assertIn("timeout --signal=TERM --kill-after=5s 240s", self.text)
        self.assertIn("--nodes=2", self.text)
        self.assertIn("--ntasks=8", self.text)
        self.assertIn("--ntasks-per-node=4", self.text)
        self.assertIn("--distribution=block:block", self.text)
        self.assertIn("--gpu-bind=none", self.text)

    def test_compiles_then_runs_signed_epoch_plan(self) -> None:
        compiler = "python -m eval.sota_4node.compile_inference_epoch_plan"
        runner = "python -m eval.sota_4node.run_inference_epoch_2node"
        self.assertIn(compiler, self.text)
        self.assertIn(runner, self.text)
        self.assertLess(self.text.index(compiler), self.text.index(runner))
        self.assertIn('export TEMPO_EPOCH_PLAN="${PLAN_PATH}"', self.text)
        self.assertIn("--deadline-tokens 10", self.text)
        self.assertIn("--token-slack-ms 1x4,3x6,0x6", self.text)
        self.assertIn("--width-penalty-ms 0:0,1:1,2:3,4:9", self.text)

    def test_no_retry_or_monitor_loop(self) -> None:
        for forbidden in ("retry", "squeue", "sacct", "while ", "until ", "find "):
            self.assertNotIn(forbidden, self.text)


if __name__ == "__main__":
    unittest.main()
