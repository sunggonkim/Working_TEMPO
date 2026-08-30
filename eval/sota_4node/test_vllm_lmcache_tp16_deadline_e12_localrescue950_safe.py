from __future__ import annotations
import json
from pathlib import Path
import threading
import unittest
from unittest import mock
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_e12_localrescue950_safe_entry as runner
from eval.sota_4node import vllm_lmcache_tp16_deadline_e12_localrescue950_safe_node as node

ROOT = Path(__file__).resolve().parents[2]

class SafeLocalRescueTests(unittest.TestCase):
    def setUp(self):
        with runner.e11._RESCUE_RECORDS_LOCK:
            runner.e11._RESCUE_RECORDS.clear()

    def _run(self, states, times):
        class Agent:
            def __init__(self): self.states = iter(states)
            def transfer(self, handle): return "PROC"
            def check_xfer_state(self, handle): return next(self.states)
        class Channel:
            def __init__(self): self.nixl_agent = Agent()
            def tempo_prepare(self, objects, spec): return "handle"
        state = {"error": None}
        entered, done, boost = threading.Event(), threading.Event(), threading.Event()
        with mock.patch.object(runner.time, "perf_counter_ns", side_effect=times):
            runner._safe_local_rescue_transfer_worker(
                channel=Channel(), obj=object(), receiver_id="rank-8",
                mode=runner.CANDIDATE_MODE, boost=boost, entered=entered,
                done=done, state=state,
            )
        record = runner.e11._take_rescue_record(threading.current_thread().name)
        return state, boost, record

    def test_contract_exact_and_no_timer(self):
        payload = json.loads(
            (ROOT / "eval/sota_4node/real_tp16_deadline_e12_localrescue950_safe.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(payload, runner._expected_contract())
        self.assertEqual(payload["local_rescue"]["timer_threads_created"], 0)
        self.assertEqual(payload["local_rescue"]["arm_condition"],
                         "observed_PROC_and_worker_elapsed_ge_950ms")

    def test_proc_at_950_arms_and_worker_observes(self):
        state, boost, record = self._run(
            ("PROC", "DONE"), (0, 950_000_000, 960_000_000)
        )
        self.assertTrue(boost.is_set())
        self.assertTrue(record["rescue_armed"])
        self.assertEqual(record["status_at_arm"], "PROC")
        self.assertTrue(record["boost_observed_by_worker"])
        self.assertEqual(state["boost_polls"], 1)
        self.assertEqual(state["polls"], state["boost_polls"] + 1)

    def test_done_before_trigger_never_arms(self):
        state, boost, record = self._run(("DONE",), (0, 500_000_000))
        self.assertFalse(boost.is_set())
        self.assertFalse(record["rescue_armed"])
        self.assertTrue(record["completed_before_rescue"])
        self.assertEqual(state["boost_polls"], 0)

    def test_wiring_and_no_timer_symbol(self):
        source = (ROOT / "eval/sota_4node/run_vllm_lmcache_tp16_deadline_e12_localrescue950_safe_entry.py").read_text(encoding="utf-8")
        self.assertNotIn("threading.Timer", source)
        self.assertIn('status_at_arm"] = "PROC"', source)
        launcher = (ROOT / "eval/sota_4node/run_vllm_lmcache_tp16_deadline_e12_localrescue950_safe_in_allocation.sh").read_text(encoding="utf-8")
        self.assertEqual(launcher.count("srun --exact"), 1)
        self.assertNotIn("salloc", launcher)
        self.assertNotIn("sbatch", launcher)
        self.assertIn("deadline_e12_localrescue950_safe_entry", node.base.RUNNER_MODULE)

if __name__ == "__main__": unittest.main()
