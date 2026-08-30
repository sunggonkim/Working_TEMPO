from __future__ import annotations

import unittest

from eval.sota_4node.run_inference_interconnect_2node import (
    AOT_PAIR_CONCURRENCY_BY_MODE,
    BLOCK_MODES,
    CHUNKS_PER_REQUEST,
    LATIN_ROWS,
    MODE_ORDER,
    PAIR_COUNT,
    WORLD_SIZE,
    aggregate_rank_records,
    coalesced_transfer_groups,
    schedule_entries,
    schedule_summary,
    source_node_for,
)


class InferenceInterconnectScheduleTests(unittest.TestCase):
    def test_all_background_modes_move_equal_bytes(self) -> None:
        requests = 2
        expected = requests * PAIR_COUNT * CHUNKS_PER_REQUEST
        for mode in (
            "uncontrolled",
            "local",
            "global_static",
            "aot_uniform2",
            "aot_ramp2",
            "aot_ramp4",
            "aot_uniform2_coalesced",
            "aot_ramp2_coalesced",
            "aot_ramp4_coalesced",
            "tempo",
        ):
            self.assertEqual(
                schedule_summary(mode, requests_per_block=requests)["chunks"], expected
            )
        self.assertEqual(schedule_summary("fg_only", requests_per_block=requests)["chunks"], 0)

    def test_global_pair_concurrency_is_four_vs_one(self) -> None:
        self.assertEqual(schedule_summary("local", requests_per_block=2)["max_active_pairs"], 4)
        self.assertEqual(schedule_summary("global_static", requests_per_block=2)["max_active_pairs"], 1)
        self.assertEqual(schedule_summary("aot_uniform2", requests_per_block=2)["max_active_pairs"], 2)
        self.assertEqual(schedule_summary("aot_ramp2", requests_per_block=2)["max_active_pairs"], 2)
        self.assertEqual(schedule_summary("aot_ramp4", requests_per_block=2)["max_active_pairs"], 4)
        self.assertEqual(
            schedule_summary("aot_uniform2_coalesced", requests_per_block=2)["max_active_pairs"],
            2,
        )
        self.assertEqual(
            schedule_summary("aot_ramp2_coalesced", requests_per_block=2)["max_active_pairs"],
            2,
        )
        self.assertEqual(
            schedule_summary("aot_ramp4_coalesced", requests_per_block=2)["max_active_pairs"],
            4,
        )
        self.assertEqual(schedule_summary("tempo", requests_per_block=2)["max_active_pairs"], 1)
        self.assertEqual(schedule_summary("uncontrolled", requests_per_block=2)["max_active_pairs"], 4)

    def test_exact_schedule_and_alternating_sources(self) -> None:
        self.assertEqual(
            schedule_entries("local", 4, requests_per_block=1),
            ((0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1)),
        )
        self.assertEqual(schedule_entries("tempo", 7, requests_per_block=1), ((0, 3, 1),))
        self.assertEqual(
            schedule_entries("global_static", 7, requests_per_block=1),
            schedule_entries("tempo", 7, requests_per_block=1),
        )
        self.assertEqual(schedule_entries("tempo", 16, requests_per_block=1), ())
        self.assertEqual(
            schedule_entries("aot_uniform2", 0),
            ((0, 0, 0), (0, 1, 0)),
        )
        self.assertEqual(
            schedule_entries("aot_ramp2", 4),
            ((0, 0, 1), (0, 1, 1)),
        )
        self.assertEqual(
            schedule_entries("aot_ramp2", 9),
            ((0, 2, 3), (0, 3, 3)),
        )
        self.assertEqual(
            schedule_entries("aot_ramp4", 4),
            ((0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1)),
        )
        self.assertEqual(schedule_entries("aot_uniform2", 8), ())
        self.assertEqual(schedule_entries("aot_ramp2", 10), ())
        self.assertEqual(schedule_entries("aot_ramp4", 7), ())
        self.assertEqual(
            schedule_entries("aot_ramp4_coalesced", 4, requests_per_block=2),
            schedule_entries("aot_ramp4", 4, requests_per_block=2),
        )
        for widths in AOT_PAIR_CONCURRENCY_BY_MODE.values():
            self.assertEqual(sum(widths), PAIR_COUNT * CHUNKS_PER_REQUEST)
        self.assertEqual([source_node_for(3, request) for request in range(2)], [1, 0])

    def test_coalesced_groups_preserve_direction_chunk_and_request_order(self) -> None:
        entries = schedule_entries("aot_ramp4_coalesced", 0, requests_per_block=4)
        self.assertEqual(
            coalesced_transfer_groups(entries, block_index=0, pair_index=0),
            ((0, 0, (0, 2)), (1, 0, (1, 3))),
        )
        self.assertEqual(
            coalesced_transfer_groups(entries, block_index=1, pair_index=0),
            ((0, 0, (1, 3)), (1, 0, (0, 2))),
        )

    def test_latin_sequence_balances_modes_and_positions(self) -> None:
        self.assertEqual(len(LATIN_ROWS), len(MODE_ORDER))
        self.assertEqual(len(BLOCK_MODES), len(MODE_ORDER) ** 2)
        for row in LATIN_ROWS:
            self.assertEqual(set(row), set(MODE_ORDER))
        for column in range(len(MODE_ORDER)):
            self.assertEqual({row[column] for row in LATIN_ROWS}, set(MODE_ORDER))
        for mode in MODE_ORDER:
            self.assertEqual(BLOCK_MODES.count(mode), len(MODE_ORDER))


def _rank_records() -> list[dict[str, object]]:
    config = {
        "requests_per_block": 1,
        "tokens": 16,
        "layers": 2,
        "hidden_size": 128,
        "context": 16,
        "kv_bytes": 64,
        "chunk_bytes": 16,
        "chunks_per_request": 4,
        "replicates_per_mode": len(MODE_ORDER),
        "pair_warmup_directions": 2,
        "world_control_warmup": True,
    }
    records = []
    for rank in range(WORLD_SIZE):
        blocks = []
        for block_index, mode in enumerate(BLOCK_MODES):
            receives = mode != "fg_only" and (
                (rank // 4) != source_node_for(block_index, 0)
            )
            blocks.append({
                "block_index": block_index,
                "mode": mode,
                "source_nodes": [source_node_for(block_index, 0)],
                "token_latency_ms": [1.0 + block_index / 10 + rank / 100] * 16,
                "first_token_step_ms": 1.0 + block_index / 10 + rank / 100,
                "foreground_checksum": 3.5,
                "background_operations_participated": 0 if mode == "fg_only" else 4,
                "expected_receive_bytes": 64 if receives else 0,
                "background_completed_bytes": 64 if receives else 0,
                "post_foreground_drain_ms": 0.0 if mode == "fg_only" else 0.25 + rank / 100,
                "background_completion_upper_bound_ms": 0.0 if mode == "fg_only" else 10.0 + block_index / 10 + rank / 100,
                "block_data_plane_ms": 16.0 + block_index / 10 + rank / 100,
                "correctness_met": True,
                "controller_released": True,
                "background_stream": "dedicated_cuda",
                "admissions": ([{"admitted": True}] if rank == 0 and mode == "tempo" else []),
            })
        records.append({
            "schema_version": "tempo-inference-interconnect-rank-4",
            "rank": rank,
            "local_rank": rank % 4,
            "node_index": rank // 4,
            "hostname": f"node-{rank // 4}",
            "world_size": 8,
            "nodes": 2,
            "config": config,
            "blocks": blocks,
            "pair_warmup": {
                "source_nodes": [0, 1],
                "bytes_per_direction": 1,
                "background_stream": "dedicated_cuda",
                "correctness_met": True,
            },
            "control_warmup": {
                "source_rank": 0,
                "value": 137,
                "correctness_met": True,
            },
        })
    return records


class InferenceInterconnectAggregationTests(unittest.TestCase):
    def test_aggregate_preserves_balanced_replicates_and_bytes(self) -> None:
        result = aggregate_rank_records(_rank_records())
        self.assertTrue(result["overall_correctness_met"])
        self.assertEqual(result["block_sequence"], list(BLOCK_MODES))
        self.assertTrue(result["pair_warmup"]["correctness_met"])
        self.assertTrue(result["control_warmup"]["correctness_met"])
        self.assertAlmostEqual(result["blocks"][0]["global_first_token_step_ms"], 1.07)
        self.assertAlmostEqual(result["blocks"][0]["rank_first_token_step_p50_ms"], 1.035)
        for mode in MODE_ORDER:
            self.assertEqual(result["modes"][mode]["replicates"], len(MODE_ORDER))
        self.assertEqual(result["modes"]["fg_only"]["background_completed_bytes"], 0)
        self.assertEqual(
            result["modes"]["tempo"]["background_completed_bytes"],
            result["modes"]["tempo"]["expected_background_bytes"],
        )

    def test_aggregate_rejects_missing_rank_or_corrupt_completion(self) -> None:
        with self.assertRaisesRegex(ValueError, "ranks 0..7"):
            aggregate_rank_records(_rank_records()[:-1])
        records = _rank_records()
        records[0]["blocks"][1]["background_completed_bytes"] = 0
        result = aggregate_rank_records(records)
        self.assertFalse(result["blocks"][1]["correctness_met"])
        self.assertFalse(result["overall_correctness_met"])


if __name__ == "__main__":
    unittest.main()
