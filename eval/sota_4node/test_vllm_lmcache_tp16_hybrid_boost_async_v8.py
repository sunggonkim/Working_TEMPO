from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as runner
from eval.sota_4node import vllm_decode_quiescence_gate_launch_v3 as hook
from eval.sota_4node import vllm_lmcache_tp16_hybrid_boost_node_async_v8 as node
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol


ROOT = Path(__file__).resolve().parents[2]


class AsyncV8Tests(unittest.TestCase):
    def _event(self):
        return hook.ReadyEvent(0, "tempo-scout-test", 30, 31, 1, 2, 3, 3)

    def test_promotion_frame_is_truthful_and_round_trips(self) -> None:
        event = self._event()
        frame = protocol.ReleaseFrame.promotion(
            event, promotion_armed_sources=8
        )
        payload = frame.to_payload()
        self.assertEqual(payload["mode"], "tempo_async_promotion")
        self.assertEqual(payload["promotion_armed_sources"], 8)
        self.assertEqual(payload["completed_sources"], 0)
        self.assertEqual(payload["physical_descriptors"], 0)
        self.assertEqual(payload["completed_bytes"], 0)
        self.assertEqual(payload["source_elapsed_ns"], [])
        self.assertEqual(protocol.ReleaseFrame.from_payload(payload, event=event), frame)

    def test_promotion_rejects_false_completion(self) -> None:
        frame = protocol.ReleaseFrame.promotion(
            self._event(), promotion_armed_sources=8
        )
        with self.assertRaisesRegex(ValueError, "cannot claim transfer completion"):
            replace(frame, completed_bytes=1).validate()

    def test_contract_is_exact(self) -> None:
        payload = json.loads(
            (ROOT / "eval/sota_4node/real_tp16_hybrid_boost_async_v8.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(payload, runner._expected_contract())

    def test_node_and_launcher_target_async_v8(self) -> None:
        launcher = (
            ROOT
            / "eval/sota_4node/run_vllm_lmcache_tp16_hybrid_boost_async_v8_in_allocation.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(launcher.count("srun --exact"), 1)
        self.assertIn("hybrid_boost_node_async_v8.py", launcher)
        self.assertIn("NIXL_BACKEND=${1:-UCX}", launcher)
        self.assertNotIn("--plan", launcher)
        self.assertIn("hybrid_boost_async_v8_entry", node.base.RUNNER_MODULE)
        self.assertIn("real_tp16_hybrid_boost_async_v8.json", str(node.base.PLAN_RELATIVE))

    def test_hot_path_arms_then_releases_without_done_wait(self) -> None:
        source = (
            ROOT
            / "eval/sota_4node/run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry.py"
        ).read_text(encoding="utf-8")
        gate = source[source.index("    if marked:"):source.index("    client_control:")]
        self.assertIn("boost.set()", gate)
        self.assertIn("dist.all_reduce(armed", gate)
        self.assertIn("ReleaseFrame.promotion", gate)
        self.assertNotIn("done.wait", gate)


if __name__ == "__main__":
    unittest.main()
