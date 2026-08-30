from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


SCRIPT = Path(__file__).with_name("run_lmcache_epoch_in_allocation.sh")


class LMCacheEpochAllocationLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_requires_existing_explicitly_approved_two_node_allocation(self) -> None:
        self.assertIn("SLURM_JOB_ID:?run inside an existing allocation", self.text)
        self.assertIn("SLURM_JOB_NUM_NODES", self.text)
        self.assertIn("TEMPO_LMCACHE_EPOCH_APPROVED", self.text)
        self.assertNotRegex(self.text, r"\b(?:sbatch|salloc|scancel)\b")

    def test_uses_one_bounded_eight_rank_step_without_retry(self) -> None:
        self.assertEqual(len(re.findall(r"^\s*srun\b", self.text, re.MULTILINE)), 1)
        self.assertIn("timeout --foreground --signal=TERM --kill-after=5s 240s", self.text)
        self.assertIn("--time=00:04:00", self.text)
        self.assertIn("--nodes=2", self.text)
        self.assertIn("--ntasks=8", self.text)
        self.assertIn("--ntasks-per-node=4", self.text)
        self.assertIn("--distribution=block:block", self.text)
        self.assertNotRegex(self.text, r"\b(?:retry|while|sleep)\b")

    def test_preserves_module_torch_and_appends_dependency_site(self) -> None:
        self.assertIn("module load pytorch/2.8.0", self.text)
        self.assertIn('MODULE_PYTHON=$(command -v python)', self.text)
        self.assertIn("TEMPO_LMCACHE_EXTRA_SITE", self.text)
        self.assertNotIn('.sota_venv/bin/python', self.text)
        self.assertIn("lmcache_epoch_bootstrap", self.text)
        self.assertIn("runtime_preflight.json", self.text)

    def test_freezes_candidate_and_transport_scope(self) -> None:
        self.assertIn("227d13f5c9fdb52ddb933641d34331f678de03a0", self.text)
        self.assertIn("NCCL_NET=Socket", self.text)
        self.assertIn("NCCL_SOCKET_IFNAME=hsn", self.text)
        self.assertIn("--total-quanta 16", self.text)
        self.assertIn("--deadline-tokens 10", self.text)
        self.assertIn("--requests 2", self.text)
        self.assertIn("--kv-mib 32", self.text)
        self.assertIn("--chunk-mib 8", self.text)
        self.assertNotRegex(self.text, r"\b(?:pip|cmake|make|git clone|git fetch)\b")


if __name__ == "__main__":
    unittest.main()
