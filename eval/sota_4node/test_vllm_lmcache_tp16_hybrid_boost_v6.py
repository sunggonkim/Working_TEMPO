from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_audit_v5 as fixture
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6_entry as entry
from eval.sota_4node import vllm_lmcache_tp16_hybrid_boost_node_v6 as node


ROOT = Path(__file__).resolve().parents[2]


def _records():
    records = fixture._valid_records()
    for item in records:
        for block in item["blocks"]:
            if item["rank"] == 0:
                block["client_started_from_origin_ns"] = 2_000_000
                block["client_finished_from_origin_ns"] = int(
                    (block["client"]["request_e2e_ms"] + 2.0) * 1e6
                )
            else:
                block["client_started_from_origin_ns"] = 0
                block["client_finished_from_origin_ns"] = 0
    return records


def _trace():
    events = []
    event_id = 1
    for index, (_prompt, mode) in enumerate(old.BLOCKS):
        if mode != old.TEMPO:
            continue
        events.append(
            {
                "event_id": event_id,
                "request_id": f"cmpl-tempo-scout-cpu-c0-b{index}-{old.TEMPO}-0-x",
                "mode": old.TEMPO,
                "fence_ms": 1.0,
                "ready_to_release_ms": 18.0,
                "release_to_next_step_ms": 1.0,
                "total_gate_bubble_ms": 20.0,
            }
        )
        event_id += 1
    return {"validated": True, "events": events}


class HybridV6Tests(unittest.TestCase):
    def test_contract_exact(self) -> None:
        payload = json.loads(
            (
                ROOT / "eval/sota_4node/real_tp16_hybrid_boost_v6.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(payload, fixed._expected_contract())

    def test_entry_binds_original_functions(self) -> None:
        self.assertIs(fixed._ORIGINAL_VALIDATE_TRACE, entry.old._validate_trace)
        self.assertIs(fixed._ORIGINAL_AGGREGATE, entry.old._aggregate)

    def test_valid_fixture_uses_trace_bubble_and_paired_gate(self) -> None:
        result = fixed._aggregate(_records(), _trace(), fixture._args())
        self.assertEqual(result["schema_version"], fixed.RESULT_SCHEMA)
        self.assertTrue(result["candidate_gates"]["total_gate_bubble_median_le_25ms"])
        self.assertTrue(result["candidate_gates"]["total_gate_bubble_max_le_30ms"])
        self.assertTrue(
            result["candidate_gates"]["tempo_service_makespan_beats_lmcache_paired"]
        )
        self.assertEqual(result["screen_outcome"], "hybrid_candidate_pass")

    def test_rank_block_identity_mismatch_fails_closed(self) -> None:
        records = _records()
        records[-1]["blocks"][0]["mode"] = "wrong-mode"
        with self.assertRaisesRegex(ValueError, "block identity mismatch"):
            fixed._aggregate(records, _trace(), fixture._args())

    def test_paired_gate_requires_two_prompt_wins(self) -> None:
        records = _records()
        tempo_indices = [
            index for index, (_prompt, mode) in enumerate(old.BLOCKS)
            if mode == old.TEMPO
        ]
        for index in tempo_indices[:2]:
            records[0]["blocks"][index]["client_finished_from_origin_ns"] = 160_000_000
        result = fixed._aggregate(records, _trace(), fixture._args())
        self.assertFalse(
            result["candidate_gates"]["tempo_service_makespan_beats_lmcache_paired"]
        )
        self.assertEqual(result["screen_outcome"], "hybrid_candidate_revise_or_stop")

    def test_launcher_and_node_target_v6(self) -> None:
        launcher = (
            ROOT
            / "eval/sota_4node/run_vllm_lmcache_tp16_hybrid_boost_v6_in_allocation.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(launcher.count("srun --exact"), 1)
        self.assertIn("timeout --foreground", launcher)
        self.assertIn("${#TEMPO_JOB_HOSTS[@]}", launcher)
        self.assertIn("vllm_lmcache_tp16_hybrid_boost_node_v6.py", launcher)
        self.assertNotIn('    --plan "${PLAN_PATH}"', launcher)
        self.assertEqual(
            node.base.RUNNER_MODULE,
            "eval.sota_4node.run_vllm_lmcache_tp16_hybrid_boost_v6_entry",
        )
        self.assertEqual(
            node.base.PLAN_RELATIVE,
            Path("eval/sota_4node/real_tp16_hybrid_boost_v6.json"),
        )

    def test_duplicate_trace_ids_are_explicitly_rejected(self) -> None:
        source = (
            ROOT / "eval/sota_4node/run_vllm_lmcache_tp16_hybrid_boost_v6.py"
        ).read_text(encoding="utf-8")
        self.assertIn("len(ids) != len(set(ids))", source)


if __name__ == "__main__":
    unittest.main()
