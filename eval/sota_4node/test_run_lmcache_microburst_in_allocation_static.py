from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


SCRIPT = Path(__file__).with_name("run_lmcache_microburst_in_allocation.sh")


class LMCacheMicroburstLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_shell_and_existing_allocation_contract(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        self.assertIn("TEMPO_LMCACHE_MICROBURST_APPROVED", self.text)
        self.assertIn("SLURM_JOB_ID:?run inside an existing allocation", self.text)
        self.assertNotRegex(self.text, r"\b(?:sbatch|salloc|scancel|retry|while|sleep)\b")

    def test_one_bounded_step(self) -> None:
        self.assertEqual(len(re.findall(r"^\s*srun\b", self.text, re.MULTILINE)), 1)
        self.assertIn("timeout --foreground --signal=TERM --kill-after=5s 240s", self.text)
        self.assertIn("--nodes=2", self.text)
        self.assertIn("--ntasks=8", self.text)

    def test_frozen_microburst_candidate(self) -> None:
        for literal in (
            "--total-quanta 64",
            "--deadline-tokens 34",
            "--token-slack-ms 1x4,3x60",
            "--kv-kib 4096",
            "--chunk-kib 256",
            "--tokens 64",
            "--layers 8",
            "lmcache_microburst_bootstrap",
        ):
            self.assertIn(literal, self.text)
        self.assertNotRegex(self.text, r"\b(?:pip|cmake|make|git clone|git fetch)\b")


if __name__ == "__main__":
    unittest.main()
