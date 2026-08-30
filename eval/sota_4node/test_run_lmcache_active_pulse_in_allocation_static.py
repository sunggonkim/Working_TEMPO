from __future__ import annotations

from pathlib import Path
import re
import unittest


SCRIPT = Path(__file__).with_name(
    "run_lmcache_active_pulse_in_allocation.sh"
).read_text(encoding="utf-8")


class ActivePulseAllocationLauncherStaticTests(unittest.TestCase):
    def test_requires_explicit_existing_allocation_approval(self) -> None:
        self.assertIn("SLURM_JOB_ID:?run inside an existing allocation", SCRIPT)
        self.assertIn('"${SLURM_JOB_NUM_NODES:-}" == 2', SCRIPT)
        self.assertIn("TEMPO_LMCACHE_ACTIVE_PULSE_APPROVED", SCRIPT)
        self.assertIn("== YES", SCRIPT)

    def test_has_exactly_one_bounded_two_node_eight_rank_srun(self) -> None:
        self.assertEqual(len(re.findall(r"\bsrun\b", SCRIPT)), 1)
        self.assertIn("timeout --foreground --signal=TERM --kill-after=5s 240s", SCRIPT)
        self.assertIn("--nodes=2 --ntasks=8 --ntasks-per-node=4", SCRIPT)
        self.assertIn("--gpus-per-node=4", SCRIPT)
        self.assertIn("--time=00:04:00", SCRIPT)

    def test_compiles_signed_active_plan_before_srun(self) -> None:
        compile_position = SCRIPT.index(
            "eval.sota_4node.compile_lmcache_active_pulse_plan"
        )
        srun_position = SCRIPT.index("srun --exact")
        self.assertLess(compile_position, srun_position)
        self.assertIn('export TEMPO_ACTIVE_SERVICE_PLAN="${ACTIVE_PLAN_PATH}"', SCRIPT)

    def test_exact_8mib_workload_and_bootstrap_are_pinned(self) -> None:
        self.assertIn("--requests 2 --kv-kib 8192 --chunk-kib 512", SCRIPT)
        self.assertIn("--tokens 64 --layers 8", SCRIPT)
        self.assertIn("pytorch/2.8.0", SCRIPT)
        self.assertIn(
            "227d13f5c9fdb52ddb933641d34331f678de03a0", SCRIPT
        )
        self.assertIn("eval.sota_4node.lmcache_active_pulse_bootstrap", SCRIPT)

    def test_does_not_allocate_submit_install_build_poll_or_retry(self) -> None:
        forbidden = (
            "salloc",
            "sbatch",
            "scancel",
            "pip install",
            "git clone",
            "cmake",
            "make -j",
            "while ",
            "until ",
        )
        for item in forbidden:
            with self.subTest(item=item):
                self.assertNotIn(item, SCRIPT)


if __name__ == "__main__":
    unittest.main()
