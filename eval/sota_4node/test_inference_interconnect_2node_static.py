from __future__ import annotations

import re
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_inference_interconnect_2node.slurm")


class InferenceInterconnectTwoNodeStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_exact_bounded_allocation(self) -> None:
        for directive in (
            "#SBATCH --nodes=2",
            "#SBATCH --ntasks-per-node=4",
            "#SBATCH --gpus-per-node=4",
            "#SBATCH --cpus-per-task=32",
            "#SBATCH --constraint=gpu&hbm80g",
            "#SBATCH --qos=debug_preempt",
            "#SBATCH --time=00:05:00",
            "#SBATCH --no-requeue",
            "#SBATCH --mail-type=NONE",
        ):
            self.assertIn(directive, self.text)
        self.assertIn('[[ "${SLURM_JOB_NUM_NODES:-}" == 2 ]] || exit 2', self.text)
        self.assertIn('[[ "${SLURM_NTASKS:-}" == 8 ]] || exit 2', self.text)

    def test_requires_explicit_approval_and_known_runtime(self) -> None:
        self.assertIn("TEMPO_INFERENCE_2NODE_APPROVED", self.text)
        self.assertIn("module load pytorch/2.8.0", self.text)
        self.assertIn("timeout --signal=TERM --kill-after=5s 240s", self.text)
        self.assertIn("--time=00:04:00", self.text)

    def test_one_srun_invokes_the_research_runner(self) -> None:
        self.assertEqual(len(re.findall(r"\bsrun\b", self.text)), 1)
        self.assertIn("run_inference_interconnect_2node.py", self.text)
        self.assertIn('--output-dir "${RESULT_DIR}"', self.text)
        self.assertIn("--ntasks=8 --ntasks-per-node=4", self.text)
        self.assertIn("--distribution=block:block", self.text)
        for argument in (
            "--requests-per-block 2",
            "--tokens 16",
            "--layers 4",
            "--hidden-size 1024",
            "--context 128",
            "--kv-mib 128",
            "--chunk-mib 32",
        ):
            self.assertIn(argument, self.text)

    def test_result_scope_and_no_old_experiment_machinery(self) -> None:
        self.assertIn(
            'RESULT_DIR="${REPO_ROOT}/results/inference_interconnect_2node_job_${SLURM_JOB_ID}"',
            self.text,
        )
        for forbidden in (
            "find ",
            "sha256sum",
            "setup_baselines",
            "capture_g1_domain_counters",
            "build_g2_fabric_observation",
            "retry",
            "sacct",
            "squeue",
        ):
            self.assertNotIn(forbidden, self.text)
        self.assertIsNone(re.search(r"(?m)^\s*(?:sbatch|salloc)\b", self.text))


if __name__ == "__main__":
    unittest.main()
