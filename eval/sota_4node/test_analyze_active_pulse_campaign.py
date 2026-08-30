from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node import analyze_active_pulse_campaign as analyzer


def _write_run(root: Path, greedy=(3.0, 3.0, 5.0), tempo=(2.0, 2.0, 4.0), *, lag=True) -> Path:
    root.mkdir()
    modes = ("fg_only", "lmcache_greedy", "lmcache_static_serial", "tempo_epoch") * 2
    config = {"tokens": 3, "requests": 1, "kv_bytes": 1024, "epoch_plan_signature": "runtime"}
    latency = {"fg_only": (1.0, 1.0, 1.0), "lmcache_greedy": greedy,
               "lmcache_static_serial": (4.0, 4.0, 4.0), "tempo_epoch": tempo}
    rank_records = []
    for rank in range(8):
        blocks = []
        for index, mode in enumerate(modes):
            source = rank < 4
            expected = 0 if mode == "fg_only" else 1024
            block = {
                "block_index": index, "mode": mode, "token_latency_ms": list(latency[mode]),
                "expected_source_bytes": expected if source else 0,
                "expected_receive_bytes": expected if not source else 0,
                "background_completed_bytes": expected if source else 0,
                "receiver_verified_bytes": expected if not source else 0,
                "correctness_met": True, "transfer_errors": [],
            }
            if mode == "tempo_epoch" and source:
                block.update({
                    "execution": "group2", "active_service_plan_signature": "same-schedule",
                    "schedule_start_adherence_met": True, "absolute_service_deadline_met": True,
                    "candidate_relative_token_deadline_used": False,
                    "absolute_service_deadline_ns": 10_000_000,
                    "background_finish_from_block_start_ms": 9.0,
                    "post_foreground_drain_ms": 0.0, "no_post_foreground_drain_met": True,
                    "lag_model_validated": lag,
                    "transfer_records": [{"started_within_scheduled_token": True,
                                          "finished_by_plan_deadline": True,
                                          "deadline_semantics": "absolute_from_block_start"}],
                })
            blocks.append(block)
        rank_records.append({"schema_version": analyzer.RANK_SCHEMA, "rank": rank,
                             "world_size": 8, "nodes": 2, "config": config, "blocks": blocks})
        (root / f"rank_{rank}.json").write_text(json.dumps(rank_records[-1]), encoding="utf-8")
    result_blocks = []
    for index, mode in enumerate(modes):
        expected = 0 if mode == "fg_only" else 4096
        result_blocks.append({"block_index": index, "mode": mode,
                              "expected_background_bytes": expected,
                              "background_completed_bytes": expected,
                              "receiver_verified_bytes": expected,
                              "correctness_met": True, "transfer_errors": []})
    result = {"schema_version": analyzer.RESULT_SCHEMA, "world_size": 8, "nodes": 2,
              "config": config, "block_sequence": list(modes), "blocks": result_blocks,
              "overall_correctness_met": True, "scheduler_semantics": {"name": "group2"}}
    path = root / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


class AnalyzeActivePulseCampaignTest(unittest.TestCase):
    def test_success_fraction_and_time_normalized_goodput_are_distinct(self):
        metric = analyzer.slo_metrics([1.0, 1.0, 10.0])
        self.assertAlmostEqual(metric["success_fraction"], 2 / 3)
        self.assertAlmostEqual(metric["time_normalized_goodput_tokens_per_s"], 2 * 1000 / 12)

    def test_two_service_valid_repeats_emit_conservative_signal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = analyzer.analyze_campaign([
                ("first", _write_run(root / "a")),
                ("second", _write_run(root / "b")),
            ])
        schedule = report["schedules"][0]
        self.assertEqual(schedule["outcome"], "repeatable_goodput_signal")
        self.assertTrue(schedule["repeatable_tail_signal"])
        self.assertEqual(schedule["repeat_count"], 2)

    def test_lag_validation_is_separate_from_service_validity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_run(Path(temporary) / "run", lag=False)
            run = analyzer.analyze_run(path)
        self.assertTrue(run["validation"]["service_execution_valid"])
        self.assertFalse(run["validation"]["lag_model_validated"])


if __name__ == "__main__":
    unittest.main()
