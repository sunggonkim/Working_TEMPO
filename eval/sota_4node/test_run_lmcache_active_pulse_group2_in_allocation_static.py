from __future__ import annotations

from pathlib import Path
import re
import unittest


SCRIPT = Path(__file__).with_name(
    "run_lmcache_active_pulse_group2_in_allocation.sh"
).read_text(encoding="utf-8")


class Group2LauncherStaticTests(unittest.TestCase):
    def test_exactly_one_bounded_approved_srun(self) -> None:
        self.assertEqual(len(re.findall(r"\bsrun\b", SCRIPT)), 1)
        self.assertIn("TEMPO_LMCACHE_ACTIVE_GROUP2_APPROVED", SCRIPT)
        self.assertIn("timeout --foreground --signal=TERM --kill-after=5s 240s", SCRIPT)
        self.assertIn("--nodes=2 --ntasks=8 --ntasks-per-node=4", SCRIPT)
        self.assertIn("--gpus-per-node=4", SCRIPT)

    def test_compile_precedes_exact_workload(self) -> None:
        self.assertLess(
            SCRIPT.index("compile_lmcache_active_pulse_group2_plan"),
            SCRIPT.index("srun --exact"),
        )
        self.assertIn("TEMPO_ACTIVE_GROUP2_SERVICE_PLAN", SCRIPT)
        self.assertIn("--requests 2 --kv-kib 8192 --chunk-kib 512", SCRIPT)
        self.assertIn("--tokens 64 --layers 8", SCRIPT)

    def test_no_allocation_submission_install_build_or_retry(self) -> None:
        for forbidden in (
            "salloc", "sbatch", "scancel", "pip install", "git clone",
            "cmake", "make -j", "while ", "until ",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, SCRIPT)


if __name__ == "__main__":
    unittest.main()
