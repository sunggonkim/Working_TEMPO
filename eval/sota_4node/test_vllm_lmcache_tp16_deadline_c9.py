from __future__ import annotations

import json
from pathlib import Path
import unittest

from eval.sota_4node import run_vllm_lmcache_tp16_deadline_c9_entry as runner
from eval.sota_4node import vllm_lmcache_tp16_deadline_c9_node as node


ROOT = Path(__file__).resolve().parents[2]


class DeadlineC9Tests(unittest.TestCase):
    def test_contract_is_exact_and_predeclares_defer(self) -> None:
        payload = json.loads(
            (ROOT / "eval/sota_4node/real_tp16_deadline_c9.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload, runner._expected_contract())
        controller = payload["deadline_controller"]
        self.assertEqual(controller["decision"], "defer/no_rescue")
        self.assertGreaterEqual(
            controller["prior_observed_min_slack_ms"],
            controller["defer_threshold_ms"],
        )
        self.assertFalse(controller["measured_candidate_request_marked"])
        self.assertEqual(controller["token31_hook_events_per_candidate"], 0)
        self.assertEqual(controller["gate_collectives_per_candidate"], 0)
        self.assertEqual(controller["rescue_armed_sources"], 0)

    def test_candidate_mode_keeps_latin_order(self) -> None:
        self.assertEqual(len(runner.BLOCKS), 9)
        for prompt in range(3):
            modes = [mode for row_prompt, mode in runner.BLOCKS if row_prompt == prompt]
            self.assertCountEqual(
                modes,
                [runner.old.FG, runner.old.LMCACHE, runner.CANDIDATE_MODE],
            )

    def test_hot_path_has_no_listener_gate_or_promotion(self) -> None:
        source = (
            ROOT / "eval/sota_4node/run_vllm_lmcache_tp16_deadline_c9_entry.py"
        ).read_text(encoding="utf-8")
        hot_path = source[source.index("def _run_block("):source.index("def _validate_trace(")]
        self.assertIn('f"control-{args.allocation_id}', hot_path)
        self.assertNotIn("GateListener", hot_path)
        self.assertNotIn("listener.accept", hot_path)
        self.assertNotIn("boost.set()", hot_path)
        self.assertNotIn("dist.gather(", hot_path)
        self.assertIn('"candidate_hook_invocations": 0', hot_path)
        self.assertIn('"rescue_armed_sources": 0', hot_path)

    def test_node_and_launcher_target_deadline_c9(self) -> None:
        launcher = (
            ROOT / "eval/sota_4node/run_vllm_lmcache_tp16_deadline_c9_in_allocation.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(launcher.count("srun --exact"), 1)
        self.assertIn("tp16_deadline_c9_node.py", launcher)
        self.assertIn("real_tp16_deadline_c9.json", launcher)
        self.assertIn("NIXL_BACKEND=${1:-UCX}", launcher)
        self.assertNotIn("salloc", launcher)
        self.assertNotIn("sbatch", launcher)
        self.assertIn("tp16_deadline_c9_entry", node.base.RUNNER_MODULE)
        self.assertIn("real_tp16_deadline_c9.json", str(node.base.PLAN_RELATIVE))

    def test_aggregate_records_worker_progress(self) -> None:
        source = (
            ROOT / "eval/sota_4node/run_vllm_lmcache_tp16_deadline_c9_entry.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"low_priority_sleeps_sum"', source)
        self.assertIn('"boost_polls_sum"', source)
        self.assertIn('"observed_completion_slack_ms"', source)
        self.assertIn('"all_candidate_decisions_defer"', source)


if __name__ == "__main__":
    unittest.main()
