from __future__ import annotations

from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name(
    "run_tempo_go_c6_nccl_victim_abba_in_allocation.sh"
)


class C6NCCLVictimABBAStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_requires_approved_existing_interactive_allocation(self) -> None:
        self.assertIn("TEMPO_GO_C6_QUALIFICATION_APPROVED", self.text)
        self.assertIn("SLURM_JOB_ID", self.text)
        self.assertIn("SLURM_JOB_NUM_NODES", self.text)
        self.assertNotIn("sbatch", self.text)
        self.assertNotIn("scancel", self.text)

    def test_uses_frozen_abba_contract_and_native_component(self) -> None:
        self.assertIn("tempo_go_c6_qualification_contract_v1.json", self.text)
        self.assertIn("run_lmcache_nixl_contention_2node_in_allocation.sh", self.text)
        self.assertIn("TEMPO_GO_CROSS_LAYER_NO_BACKGROUND_TRANSFER", self.text)
        self.assertIn("MINIMUM_ACTIVE_DURATION_S", self.text)
        self.assertIn("analyze_tempo_go_c6_nccl_victim_abba", self.text)

    def test_contains_no_privileged_or_container_path(self) -> None:
        for forbidden in ("sudo", "udiRoot", "CAP_NET_ADMIN", "--image"):
            self.assertNotIn(forbidden, self.text)


if __name__ == "__main__":
    unittest.main()
