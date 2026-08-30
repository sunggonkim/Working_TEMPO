from __future__ import annotations

import json
from pathlib import Path
import threading
import unittest
from unittest import mock

from eval.sota_4node import run_vllm_lmcache_tp16_deadline_e11_sleep025_entry as runner
from eval.sota_4node import vllm_lmcache_tp16_deadline_e11_sleep025_node as node


ROOT = Path(__file__).resolve().parents[2]


class DeadlineE11Sleep025Tests(unittest.TestCase):
    def test_contract_is_exact_and_records_branch(self) -> None:
        payload = json.loads(
            (ROOT / "eval/sota_4node/real_tp16_deadline_e11_sleep025.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(payload, runner._expected_contract())
        controller = payload["deadline_controller"]
        self.assertEqual(controller["decode_progress_sleep_ms"], 0.25)
        self.assertEqual(controller["branch_rule"], runner.BRANCH_RULE)
        self.assertEqual(
            controller["decision_evidence_results"],
            list(runner.DECISION_EVIDENCE_RESULTS),
        )

    def test_actual_worker_sleeps_quarter_ms(self) -> None:
        class Agent:
            def __init__(self):
                self.states = iter(("PROC", "DONE"))
            def transfer(self, handle):
                return "PROC"
            def check_xfer_state(self, handle):
                return next(self.states)
        class Base:
            def tempo_prepare(self, objects, transfer_spec):
                return "handle"
        channel = runner._deadline_sleep025_channel_class(Base)()
        channel.nixl_agent = Agent()
        with mock.patch.object(runner.time, "sleep") as sleep:
            result = channel.tempo_adaptive_write(
                [object()], {"receiver_id": "rank-8"}, threading.Event()
            )
        sleep.assert_called_once_with(0.00025)
        self.assertEqual(result["polls"], result["low_priority_sleeps"] + 1)
        self.assertEqual(result["boost_polls"], 0)
        self.assertEqual(result["configured_low_priority_sleep_ns"], 250_000)

    def test_latin_order_and_hard_gates(self) -> None:
        for prompt in range(3):
            modes = [mode for row_prompt, mode in runner.BLOCKS if row_prompt == prompt]
            self.assertCountEqual(
                modes, [runner.old.FG, runner.old.LMCACHE, runner.CANDIDATE_MODE]
            )
        contract = runner._expected_contract()
        gates = contract["campaign"]["candidate_gates"]
        self.assertEqual(gates["all_candidate_observed_slack_ge_ms"], 200.0)
        self.assertEqual(gates["paired_service_delta_median_le_ms"], -5.0)
        self.assertEqual(gates["paired_service_win_max_delta_ms"], -5.0)
        self.assertEqual(gates["paired_service_win_min_prompts"], 2)
        self.assertEqual(gates["candidate_e2e_p50_le_fg_ratio"], 1.05)
        self.assertEqual(gates["candidate_tpot_p99_le_lmcache_ratio"], 1.10)

    def test_node_and_launcher_target_sleep025(self) -> None:
        launcher = (
            ROOT
            / "eval/sota_4node/run_vllm_lmcache_tp16_deadline_e11_sleep025_in_allocation.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(launcher.count("srun --exact"), 1)
        self.assertIn("deadline_e11_sleep025_node.py", launcher)
        self.assertIn("real_tp16_deadline_e11_sleep025.json", launcher)
        self.assertNotIn("salloc", launcher)
        self.assertNotIn("sbatch", launcher)
        self.assertIn("deadline_e11_sleep025_entry", node.base.RUNNER_MODULE)


if __name__ == "__main__":
    unittest.main()
