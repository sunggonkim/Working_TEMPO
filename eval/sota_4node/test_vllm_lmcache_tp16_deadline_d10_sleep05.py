from __future__ import annotations

import json
from pathlib import Path
import threading
import unittest
from unittest import mock

from eval.sota_4node import run_vllm_lmcache_tp16_deadline_d10_sleep05_entry as runner
from eval.sota_4node import vllm_lmcache_tp16_deadline_d10_sleep05_node as node


ROOT = Path(__file__).resolve().parents[2]


class DeadlineD10Sleep05Tests(unittest.TestCase):
    def test_contract_is_exact_and_hardened(self) -> None:
        payload = json.loads(
            (ROOT / "eval/sota_4node/real_tp16_deadline_d10_sleep05.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload, runner._expected_contract())
        controller = payload["deadline_controller"]
        gates = payload["campaign"]["candidate_gates"]
        self.assertEqual(controller["prior_observed_min_slack_ms"], 263.210418)
        self.assertEqual(controller["decode_progress_sleep_ms"], 0.5)
        self.assertEqual(gates["all_candidate_observed_slack_ge_ms"], 200.0)
        self.assertEqual(gates["paired_service_delta_median_le_ms"], -5.0)
        self.assertEqual(gates["paired_service_win_max_delta_ms"], -5.0)
        self.assertEqual(gates["paired_service_win_min_prompts"], 2)

    def test_actual_worker_sleeps_half_ms_and_accounts_polls(self) -> None:
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

        channel = runner._deadline_sleep05_channel_class(Base)()
        channel.nixl_agent = Agent()
        with mock.patch.object(runner.time, "sleep") as sleep:
            result = channel.tempo_adaptive_write(
                [object()], {"receiver_id": "rank-8"}, threading.Event()
            )
        sleep.assert_called_once_with(0.0005)
        self.assertEqual(result["polls"], 2)
        self.assertEqual(result["low_priority_sleeps"], 1)
        self.assertEqual(result["boost_polls"], 0)
        self.assertEqual(result["configured_low_priority_sleep_ns"], 500_000)

    def test_candidate_mode_keeps_latin_order(self) -> None:
        self.assertEqual(len(runner.BLOCKS), 9)
        for prompt in range(3):
            modes = [mode for row_prompt, mode in runner.BLOCKS if row_prompt == prompt]
            self.assertCountEqual(
                modes,
                [runner.old.FG, runner.old.LMCACHE, runner.CANDIDATE_MODE],
            )

    def test_node_and_launcher_target_sleep05(self) -> None:
        launcher = (
            ROOT
            / "eval/sota_4node/run_vllm_lmcache_tp16_deadline_d10_sleep05_in_allocation.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(launcher.count("srun --exact"), 1)
        self.assertIn("deadline_d10_sleep05_node.py", launcher)
        self.assertIn("real_tp16_deadline_d10_sleep05.json", launcher)
        self.assertNotIn("salloc", launcher)
        self.assertNotIn("sbatch", launcher)
        self.assertIn("deadline_d10_sleep05_entry", node.base.RUNNER_MODULE)

    def test_hard_gate_and_actual_sleep_are_wired(self) -> None:
        source = (
            ROOT
            / "eval/sota_4node/run_vllm_lmcache_tp16_deadline_d10_sleep05_entry.py"
        ).read_text(encoding="utf-8")
        self.assertIn("time.sleep(LOW_PRIORITY_SLEEP_S)", source)
        self.assertNotIn("boost.set()", source)
        self.assertIn("sum(delta <= SERVICE_DELTA_MAX_MS", source)
        self.assertIn("all_candidate_configured_sleep_exact_0_5ms", source)


if __name__ == "__main__":
    unittest.main()
