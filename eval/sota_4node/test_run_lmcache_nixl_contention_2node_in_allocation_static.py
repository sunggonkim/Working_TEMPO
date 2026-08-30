from __future__ import annotations

from pathlib import Path
import re
import unittest


SCRIPT = Path(__file__).with_name(
    "run_lmcache_nixl_contention_2node_in_allocation.sh"
)


class LMCacheNixlContentionAllocationStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_requires_existing_approved_four_node_allocation(self) -> None:
        self.assertIn("TEMPO_GO_CROSS_LAYER_COMPONENT_APPROVED", self.text)
        self.assertIn('[[ "${SLURM_JOB_NUM_NODES:-}" == 4 ]]', self.text)
        self.assertIn('[[ ! -e "${RESULT_DIR}/result.json" ]] || exit 2', self.text)
        self.assertIn('[[ ! -e "${RESULT_DIR}/nccl_observer.json" ]] || exit 2', self.text)

    def test_sets_rank_bootstrap_and_observer_contract(self) -> None:
        for value in (
            "export WORLD_SIZE=8",
            "export MASTER_ADDR=",
            "export MASTER_PORT=",
            "TEMPO_GO_CROSS_LAYER_EPOCH",
            "TEMPO_GO_NCCL_COMMUNICATOR_ID",
            "TEMPO_GO_NCCL_TIMEOUT_SECONDS",
            "--observer-output",
            "--process-group-timeout-s",
        ):
            self.assertIn(value, self.text)
        self.assertIn("run_lmcache_nixl_contention_2node", self.text)
        self.assertIn("TEMPO_GO_CROSS_LAYER_MASTER_PORT", self.text)
        self.assertIn("TEMPO_GO_CROSS_LAYER_NIXL_PORT_BASE", self.text)
        self.assertIn("TEMPO_GO_CROSS_LAYER_TRAFFIC_PATTERN", self.text)
        self.assertIn('--traffic-pattern "${COJOB_TRAFFIC_PATTERN}"', self.text)

    def test_has_one_bounded_two_node_srun_and_no_privilege_path(self) -> None:
        self.assertEqual(len(re.findall(r"(?m)^\s*/usr/bin/srun\b", self.text)), 1)
        self.assertIn("--nodes=2 --ntasks=8 --ntasks-per-node=4", self.text)
        self.assertIn('--nodelist="${COJOB_NODELIST}"', self.text)
        for forbidden in ("sudo", "su ", "udiRoot", "CAP_NET_ADMIN", "--image"):
            self.assertNotIn(forbidden, self.text)

    def test_does_not_mislabel_mpi_counter_reports_as_nccL_evidence(self) -> None:
        self.assertIn("MPI-only TEMPO_GO_CXI_COUNTER_REPORT", self.text)
        self.assertIn("unset MPICH_OFI_CXI_COUNTER_REPORT", self.text)
        self.assertIn('"mpich_ofi_cxi_counter_report": ""', self.text)
        self.assertIn("FI_CXI_TELEMETRY", self.text)
        self.assertIn("per_cxi_interface_domain_delta", self.text)


if __name__ == "__main__":
    unittest.main()
