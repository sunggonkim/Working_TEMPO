from __future__ import annotations

import copy
import unittest

from eval.sota_4node.p2p_nvlink_plan import build_nvlink_p2p_plan, validate_nvlink_p2p_plan


class NvlinkP2PPlanTests(unittest.TestCase):
    def test_plan_is_design_only_and_explicit(self) -> None:
        plan = build_nvlink_p2p_plan()
        self.assertFalse(plan["slurm_submitted"])
        self.assertEqual(plan["required_domains"], ["gpu_local", "nvlink_p2p"])
        self.assertEqual(len(plan["pairs"]), 12)
        self.assertTrue(plan["pair_contract"]["required_path_record_per_pair"])
        self.assertEqual(plan["evidence_state"], "design_only")
        self.assertEqual(
            {name: record["path_status"] for name, record in plan["path_contract"].items()},
            {"gpu_local": "not_traversed", "nvlink_p2p": "not_traversed"},
        )
        self.assertEqual(
            plan["promotion_requires"]["intervention"],
            "p2p_enabled_vs_matched_open",
        )

    def test_plan_rejects_topology_or_counter_relabeling(self) -> None:
        plan = build_nvlink_p2p_plan()
        changed = copy.deepcopy(plan)
        changed["path_contract"]["nvlink_p2p"]["counter_family"] = "pcie_tx_rx_bytes"
        with self.assertRaises(ValueError):
            validate_nvlink_p2p_plan(changed)

    def test_plan_rejects_incomplete_pair_coverage(self) -> None:
        plan = build_nvlink_p2p_plan()
        plan["pairs"] = plan["pairs"][:-1]
        with self.assertRaises(ValueError):
            validate_nvlink_p2p_plan(plan)

    def test_plan_rejects_submission_marker(self) -> None:
        plan = build_nvlink_p2p_plan()
        plan["slurm_submitted"] = True
        with self.assertRaises(ValueError):
            validate_nvlink_p2p_plan(plan)


if __name__ == "__main__":
    unittest.main()
