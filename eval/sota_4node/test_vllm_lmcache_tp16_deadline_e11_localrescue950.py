from __future__ import annotations

import json
from pathlib import Path
import threading
import unittest
from unittest import mock

from eval.sota_4node import run_vllm_lmcache_tp16_deadline_e11_localrescue950_entry as runner
from eval.sota_4node import vllm_lmcache_tp16_deadline_e11_localrescue950_node as node


ROOT = Path(__file__).resolve().parents[2]


class ImmediateTimer:
    def __init__(self, delay, callback):
        self.delay = delay
        self.callback = callback
        self.daemon = False
        self.name = ""
        self._alive = False
    def start(self):
        self._alive = True
        self.callback()
        self._alive = False
    def cancel(self):
        self._alive = False
    def join(self, timeout=None):
        return None
    def is_alive(self):
        return self._alive


class DeadlineE11LocalRescue950Tests(unittest.TestCase):
    def setUp(self) -> None:
        with runner._RESCUE_RECORDS_LOCK:
            runner._RESCUE_RECORDS.clear()

    def test_contract_is_exact_and_hardened(self) -> None:
        payload = json.loads(
            (ROOT / "eval/sota_4node/real_tp16_deadline_e11_localrescue950.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(payload, runner._expected_contract())
        rescue = payload["local_rescue"]
        self.assertEqual(rescue["trigger_ms_from_source_worker_start"], 950.0)
        self.assertEqual(rescue["global_rescue_collectives"], 0)
        self.assertEqual(rescue["basis_result"], runner.C9_RESULT)
        gates = payload["campaign"]["candidate_gates"]
        self.assertEqual(gates["all_candidate_observed_slack_ge_ms"], 200.0)
        self.assertEqual(gates["paired_service_delta_median_le_ms"], -5.0)
        self.assertEqual(gates["paired_service_win_max_delta_ms"], -5.0)
        self.assertEqual(gates["paired_service_win_min_prompts"], 2)

    def test_timer_arms_only_unfinished_source(self) -> None:
        class Channel:
            def tempo_adaptive_write(self, objects, spec, boost):
                self.saw_boost = boost.is_set()
                return {"completed": 1, "polls": 2, "low_priority_sleeps": 1,
                        "boost_polls": 1, "yields": 0}
        channel = Channel()
        state = {"error": None}
        entered, done, boost = threading.Event(), threading.Event(), threading.Event()
        with mock.patch.object(runner.threading, "Timer", ImmediateTimer):
            runner._local_rescue_transfer_worker(
                channel=channel, obj=object(), receiver_id="rank-8",
                mode=runner.CANDIDATE_MODE, boost=boost, entered=entered,
                done=done, state=state,
            )
        record = runner._take_rescue_record(threading.current_thread().name)
        self.assertEqual(ImmediateTimer(runner.LOCAL_RESCUE_TRIGGER_S, lambda: None).delay, 0.950)
        self.assertTrue(channel.saw_boost)
        self.assertTrue(record["unfinished_at_trigger"])
        self.assertTrue(record["rescue_armed"])
        self.assertFalse(record["completed_before_rescue"])
        self.assertTrue(record["timer_joined"])

    def test_completed_source_is_never_boosted(self) -> None:
        class Channel:
            def tempo_adaptive_write(self, objects, spec, boost):
                self.saw_boost = boost.is_set()
                return {"completed": 1, "polls": 1, "low_priority_sleeps": 0,
                        "boost_polls": 0, "yields": 0}
        channel = Channel()
        state = {"error": None}
        entered, done, boost = threading.Event(), threading.Event(), threading.Event()
        runner._local_rescue_transfer_worker(
            channel=channel, obj=object(), receiver_id="rank-8",
            mode=runner.CANDIDATE_MODE, boost=boost, entered=entered,
            done=done, state=state,
        )
        record = runner._take_rescue_record(threading.current_thread().name)
        self.assertFalse(channel.saw_boost)
        self.assertFalse(record["rescue_armed"])
        self.assertTrue(record["completed_before_rescue"])
        self.assertTrue(record["timer_joined"])

    def test_latin_order_node_and_launcher(self) -> None:
        for prompt in range(3):
            modes = [mode for row_prompt, mode in runner.BLOCKS if row_prompt == prompt]
            self.assertCountEqual(
                modes, [runner.old.FG, runner.old.LMCACHE, runner.CANDIDATE_MODE]
            )
        launcher = (
            ROOT
            / "eval/sota_4node/run_vllm_lmcache_tp16_deadline_e11_localrescue950_in_allocation.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(launcher.count("srun --exact"), 1)
        self.assertNotIn("salloc", launcher)
        self.assertNotIn("sbatch", launcher)
        self.assertIn("deadline_e11_localrescue950_node.py", launcher)
        self.assertIn("deadline_e11_localrescue950_entry", node.base.RUNNER_MODULE)


if __name__ == "__main__":
    unittest.main()
