from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


HERE = Path(__file__).parent
LAUNCHER = HERE / "run_vllm_tp8_mp_in_allocation.sh"
NODE_ENTRY = HERE / "vllm_tp8_mp_node.sh"


class VllmTp8MpAllocationLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")
        cls.node_entry = NODE_ENTRY.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        subprocess.run(["bash", "-n", str(NODE_ENTRY)], check=True)

    def test_existing_allocation_and_no_scheduler_mutation(self) -> None:
        self.assertIn("SLURM_JOB_ID", self.launcher)
        self.assertIn("SLURM_JOB_NUM_NODES", self.launcher)
        self.assertIn("TEMPO_VLLM_TP8_APPROVED", self.launcher)
        self.assertIsNone(
            re.search(r"(?m)^\s*(?:salloc|sbatch|scancel)\b", self.launcher)
        )

    def test_exactly_one_bounded_node_major_srun(self) -> None:
        self.assertEqual(len(re.findall(r"\bsrun\b", self.launcher)), 1)
        self.assertIn(
            "timeout --foreground --signal=TERM --kill-after=15s 600s",
            self.launcher,
        )
        self.assertIn("--nodes=2 --ntasks=2 --ntasks-per-node=1", self.launcher)
        self.assertIn("--distribution=block:block", self.launcher)
        self.assertIn("--gpus-per-task=4 --gpu-bind=none", self.launcher)
        self.assertIn("SLURM_NODEID", self.node_entry)

    def test_real_vllm_tp8_native_mp_contract(self) -> None:
        self.assertIn(".vllm_venv/bin/vllm", self.node_entry)
        self.assertIn("--tensor-parallel-size 8", self.node_entry)
        self.assertIn("--distributed-executor-backend mp", self.node_entry)
        self.assertIn("--nnodes 2", self.node_entry)
        self.assertIn('--node-rank "${SLURM_NODEID}"', self.node_entry)
        self.assertIn("VLLM_ARGS+=(--headless)", self.node_entry)

    def test_local_model_offline_and_node_local_caches(self) -> None:
        self.assertIn("models/TinyLlama-1.1B-Chat-v1.0", self.node_entry)
        self.assertIn("HF_HUB_OFFLINE=1", self.node_entry)
        self.assertIn("TRANSFORMERS_OFFLINE=1", self.node_entry)
        self.assertIn(
            'NODE_CACHE="/tmp/tempo-vllm-${SLURM_JOB_ID}-node${SLURM_NODEID}"',
            self.node_entry,
        )
        self.assertIn("FLASHINFER_WORKSPACE_BASE", self.node_entry)

    def test_no_retry_or_monitor_loop(self) -> None:
        combined = self.launcher + self.node_entry
        for forbidden in ("retry", "squeue", "sacct", "while ", "until ", "find "):
            self.assertNotIn(forbidden, combined)


if __name__ == "__main__":
    unittest.main()
