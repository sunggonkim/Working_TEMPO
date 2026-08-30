from __future__ import annotations

import copy
import unittest

from eval.sota_4node import run_lmcache_epoch_2node as runner
from tempo.inference_epoch import EpochProfile, WidthPoint, compile_epoch


def _plan():
    return compile_epoch(
        EpochProfile(
            total_quanta=16,
            deadline_tokens=10,
            token_slack_ns=(1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0),
            width_points=(
                WidthPoint(0, 0),
                WidthPoint(1, 1),
                WidthPoint(2, 3),
                WidthPoint(4, 9),
            ),
            max_width=2,
            protect_prefix_tokens=4,
            protect_prefix_max_width=1,
        )
    )


def _rank_records(*, requests: int = 2, kv_bytes: int = 32 << 20):
    config = {"requests": requests, "kv_bytes": kv_bytes, "tokens": 16}
    records = []
    for rank in range(runner.WORLD_SIZE):
        blocks = []
        for block_index, mode in enumerate(runner.BLOCK_MODES):
            background = mode != "fg_only"
            local_bytes = requests * kv_bytes if background else 0
            token_latency = [
                1.0 + token / 100.0 + rank / 1000.0
                for token in range(config["tokens"])
            ]
            blocks.append(
                {
                    "block_index": block_index,
                    "mode": mode,
                    "token_latency_ms": token_latency,
                    "decoder_latency_ms": [value - 0.01 for value in token_latency],
                    "background_batch_calls": 4 if background and rank < 4 else 0,
                    "background_completed_bytes": local_bytes if rank < 4 else 0,
                    "receiver_verified_bytes": local_bytes if rank >= 4 else 0,
                    "background_finish_from_block_start_ms": (
                        10.0 + rank if background and rank < 4 else 0.0
                    ),
                    "post_foreground_drain_ms": (
                        2.0 + rank / 10.0 if background and rank < 4 else 0.0
                    ),
                    "max_descriptor_start_lag_ms": (
                        0.2 + rank / 100.0 if background and rank < 4 else 0.0
                    ),
                    "schedule_start_adherence_met": True,
                    "plan_deadline_met": True,
                    "transfer_errors": [],
                    "correctness_met": True,
                }
            )
        records.append({"rank": rank, "config": config, "blocks": blocks})
    return records


class LMCacheEpochScheduleTest(unittest.TestCase):
    def test_latin_sequence_is_balanced(self) -> None:
        self.assertEqual(len(runner.BLOCK_MODES), 16)
        for row in runner.LATIN_ROWS:
            self.assertCountEqual(row, runner.MODE_ORDER)
        for column in range(4):
            self.assertCountEqual(
                [row[column] for row in runner.LATIN_ROWS],
                runner.MODE_ORDER,
            )
        for mode in runner.MODE_ORDER:
            self.assertEqual(runner.BLOCK_MODES.count(mode), 4)

    def test_each_background_policy_moves_the_same_quanta(self) -> None:
        plan = _plan()
        for mode in runner.MODE_ORDER[1:]:
            scheduled = [
                quantum
                for token in range(16)
                for quantum in runner.quantum_indices_for_token(plan, mode, token)
            ]
            self.assertEqual(sorted(scheduled), list(range(16)))
            self.assertEqual(len(scheduled), len(set(scheduled)))
        self.assertEqual(
            runner.quantum_indices_for_token(plan, "lmcache_greedy", 0),
            tuple(range(16)),
        )
        self.assertEqual(
            runner.quantum_indices_for_token(plan, "lmcache_static_serial", 15),
            (15,),
        )
        self.assertEqual(
            plan.width_by_token,
            (1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0),
        )

    def test_rank_mapping_covers_each_request_chunk_once(self) -> None:
        plan = _plan()
        for mode in runner.MODE_ORDER[1:]:
            for pair in range(runner.PAIR_COUNT):
                objects = [
                    index
                    for token in range(16)
                    for index in runner.object_indices_for_rank(
                        plan,
                        mode,
                        token,
                        pair_index=pair,
                        requests=2,
                    )
                ]
                self.assertEqual(sorted(objects), list(range(8)))
        self.assertEqual(
            runner.object_indices_for_rank(
                plan, "tempo_epoch", 0, pair_index=0, requests=2
            ),
            (0, 4),
        )
        self.assertEqual(
            runner.object_indices_for_rank(
                plan, "tempo_epoch", 4, pair_index=0, requests=2
            ),
            (1, 5),
        )
        with self.assertRaisesRegex(ValueError, "pair_index must be an int"):
            runner.object_indices_for_rank(
                plan, "tempo_epoch", 0, pair_index=True, requests=2
            )


class LMCacheEpochAggregationTest(unittest.TestCase):
    def test_aggregate_accepts_complete_executed_records(self) -> None:
        result = runner.aggregate_rank_records(_rank_records())
        self.assertTrue(result["overall_correctness_met"])
        self.assertTrue(result["tempo_epoch_execution_valid"])
        self.assertEqual(
            result["screen_outcome"],
            "valid_measurement_requires_performance_comparison",
        )
        self.assertEqual(len(result["blocks"]), 16)
        for mode in runner.MODE_ORDER:
            self.assertEqual(result["modes"][mode]["replicates"], 4)
        expected = runner.PAIR_COUNT * 2 * (32 << 20)
        for block in result["blocks"]:
            if block["mode"] == "fg_only":
                self.assertEqual(block["expected_background_bytes"], 0)
            else:
                self.assertEqual(block["background_completed_bytes"], expected)
                self.assertEqual(block["receiver_verified_bytes"], expected)

    def test_missing_bytes_fails_correctness(self) -> None:
        records = copy.deepcopy(_rank_records())
        block = next(
            item for item in records[0]["blocks"] if item["mode"] != "fg_only"
        )
        block["background_completed_bytes"] -= 8 << 20
        result = runner.aggregate_rank_records(records)
        self.assertFalse(result["overall_correctness_met"])
        self.assertEqual(result["screen_outcome"], "invalid_correctness")

    def test_calendar_backlog_kills_execution_claim_not_correctness(self) -> None:
        records = copy.deepcopy(_rank_records())
        tempo_block = next(
            item for item in records[0]["blocks"] if item["mode"] == "tempo_epoch"
        )
        tempo_block["schedule_start_adherence_met"] = False
        tempo_block["plan_deadline_met"] = False
        result = runner.aggregate_rank_records(records)
        self.assertTrue(result["overall_correctness_met"])
        self.assertFalse(result["tempo_epoch_execution_valid"])
        self.assertEqual(
            result["screen_outcome"],
            "kill_descriptor_calendar_service_mismatch",
        )

    def test_aggregate_rejects_rank_sequence_or_block_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact ranks"):
            runner.aggregate_rank_records(_rank_records()[:-1])
        records = _rank_records()
        records[1]["blocks"][0]["mode"] = "lmcache_greedy"
        with self.assertRaisesRegex(ValueError, "block sequences"):
            runner.aggregate_rank_records(records)
        records = _rank_records()
        records[1]["blocks"][0]["block_index"] = 9
        with self.assertRaisesRegex(ValueError, "block indices"):
            runner.aggregate_rank_records(records)


if __name__ == "__main__":
    unittest.main()
