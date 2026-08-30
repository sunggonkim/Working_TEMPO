from __future__ import annotations

import json
from pathlib import Path
import threading
import unittest
from unittest import mock

from eval.sota_4node import run_vllm_lmcache_tp16_deadline_d10_entry as runner
from eval.sota_4node import vllm_lmcache_tp16_deadline_d10_node as node


ROOT = Path(__file__).resolve().parents[2]


class DeadlineD10Tests(unittest.TestCase):
    def test_contract_is_exact_and_hardened(self) -> None:
        payload = json.loads(
            (ROOT / "eval/sota_4node/real_tp16_deadline_d10.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload, runner._expected_contract())
        controller = payload["deadline_controller"]
        gates = payload["campaign"]["candidate_gates"]
        self.assertEqual(controller["prior_observed_min_slack_ms"], 263.210418)
        self.assertEqual(controller["decode_progress_sleep_ms"], 2.0)
        self.assertEqual(gates["all_candidate_observed_slack_ge_ms"], 200.0)
        self.assertEqual(gates["paired_service_delta_median_le_ms"], -5.0)
        self.assertEqual(gates["paired_service_win_min_prompts"], 2)
        self.assertEqual(gates["candidate_e2e_p50_le_fg_ratio"], 1.05)
        self.assertEqual(gates["candidate_tpot_p99_le_lmcache_ratio"], 1.10)

    def test_actual_worker_sleeps_two_ms_and_accounts_polls(self) -> None:
        class Agent:
            def __init__(self) -> None:
                self.states = iter(("PROC", "DONE"))

            def transfer(self, handle):
                return "PROC"

            def check_xfer_state(self, handle):
                return next(self.states)

        class Base:
            def tempo_prepare(self, objects, transfer_spec):
                return "prepared-handle"

        channel = runner._deadline_sleep2_channel_class(Base)()
        channel.nixl_agent = Agent()
        with mock.patch.object(runner.time, "sleep") as sleep:
            result = channel.tempo_adaptive_write(
                [object()], {"receiver_id": "rank-8"}, threading.Event()
            )
        sleep.assert_called_once_with(0.002)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["polls"], 2)
        self.assertEqual(result["low_priority_sleeps"], 1)
        self.assertEqual(result["boost_polls"], 0)
        self.assertEqual(result["yields"], 0)
        self.assertEqual(result["configured_low_priority_sleep_ns"], 2_000_000)

    def test_candidate_mode_keeps_latin_order(self) -> None:
        self.assertEqual(len(runner.BLOCKS), 9)
        for prompt in range(3):
            modes = [mode for row_prompt, mode in runner.BLOCKS if row_prompt == prompt]
            self.assertCountEqual(
                modes,
                [runner.old.FG, runner.old.LMCACHE, runner.CANDIDATE_MODE],
            )

    def test_node_and_launcher_target_deadline_d10(self) -> None:
        launcher = (
            ROOT / "eval/sota_4node/run_vllm_lmcache_tp16_deadline_d10_in_allocation.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(launcher.count("srun --exact"), 1)
        self.assertIn("tp16_deadline_d10_node.py", launcher)
        self.assertIn("real_tp16_deadline_d10.json", launcher)
        self.assertNotIn("salloc", launcher)
        self.assertNotIn("sbatch", launcher)
        self.assertIn("tp16_deadline_d10_entry", node.base.RUNNER_MODULE)
        self.assertIn("real_tp16_deadline_d10.json", str(node.base.PLAN_RELATIVE))

    def test_no_measured_hook_and_exact_poll_gates_are_wired(self) -> None:
        source = (
            ROOT / "eval/sota_4node/run_vllm_lmcache_tp16_deadline_d10_entry.py"
        ).read_text(encoding="utf-8")
        self.assertIn("time.sleep(LOW_PRIORITY_SLEEP_S)", source)
        self.assertNotIn("boost.set()", source)
        self.assertIn('"all_candidate_poll_accounting_exact"', source)
        self.assertIn('"all_candidate_configured_sleep_exact_2ms"', source)
        self.assertIn('"candidate_service_makespan_median_le_minus_5ms_and_2of3"', source)


if __name__ == "__main__":
    unittest.main()
