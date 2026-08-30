from __future__ import annotations

from pathlib import Path
import unittest

from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_audit_v5 as fixture
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v7_entry as entry


def _records():
    records = fixture._valid_records()
    for item in records:
        for block in item["blocks"]:
            block["client_started_from_origin_ns"] = 2_000_000 if item["rank"] == 0 else 0
            block["client_finished_from_origin_ns"] = (
                int((block["client"]["request_e2e_ms"] + 2.0) * 1e6)
                if item["rank"] == 0
                else 0
            )
    return records


def _trace():
    events = []
    event_id = 1
    for index, (_prompt, mode) in enumerate(old.BLOCKS):
        if mode == old.TEMPO:
            events.append(
                {
                    "event_id": event_id,
                    "request_id": f"cmpl-tempo-scout-cpu-c0-b{index}-{mode}-0-x",
                    "mode": mode,
                    "fence_ms": 1.0,
                    "ready_to_release_ms": 18.0,
                    "release_to_next_step_ms": 1.0,
                    "total_gate_bubble_ms": 20.0,
                }
            )
            event_id += 1
    return {"validated": True, "events": events}


class HybridV7EntryTests(unittest.TestCase):
    def test_main_style_aggregate_rebinding_does_not_recurse(self) -> None:
        previous_old = old._aggregate
        previous_fixed = fixed._aggregate
        try:
            fixed._aggregate = entry._aggregate
            old._aggregate = entry._aggregate
            result = fixed._aggregate(_records(), _trace(), fixture._args())
        finally:
            old._aggregate = previous_old
            fixed._aggregate = previous_fixed
        self.assertEqual(result["schema_version"], fixed.RESULT_SCHEMA)
        self.assertEqual(result["screen_outcome"], "hybrid_candidate_pass")

    def test_main_style_validate_rebinding_delegates_to_original(self) -> None:
        previous_old = old._validate_trace
        previous_fixed = fixed._validate_trace
        sentinel = object()
        calls = []

        def fake_original(*args, **kwargs):
            calls.append((args, kwargs))
            return sentinel

        try:
            entry._ORIGINAL_VALIDATE_TRACE = fake_original
            fixed._validate_trace = entry._validate_trace
            old._validate_trace = entry._validate_trace
            with self.assertRaises((AttributeError, TypeError)):
                fixed._validate_trace(Path("/nonexistent"), [])
        finally:
            old._validate_trace = previous_old
            fixed._validate_trace = previous_fixed
        self.assertTrue(calls)

    def test_launcher_targets_recursion_safe_entry(self) -> None:
        text = Path(
            "eval/sota_4node/run_vllm_lmcache_tp16_hybrid_boost_v7_in_allocation.sh"
        ).read_text(encoding="utf-8")
        node = Path(
            "eval/sota_4node/vllm_lmcache_tp16_hybrid_boost_node_v7.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(text.count("srun --exact"), 1)
        self.assertIn("hybrid_boost_node_v7.py", text)
        self.assertIn("hybrid_boost_v7_entry", node)
        self.assertNotIn("--plan", text)


if __name__ == "__main__":
    unittest.main()
