from __future__ import annotations

import json
from pathlib import Path
import unittest

from eval.sota_4node import run_vllm_lmcache_tp8_sidecar as screen


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = (
    REPO_ROOT
    / "results"
    / "lmcache_active_pulse_group2_job_56929977"
    / "active_pulse_group2_plan.json"
)


class VllmLmcacheTp8SidecarTests(unittest.TestCase):
    def test_frozen_artifact_and_schedule_are_exact(self) -> None:
        screen.validate_frozen_schedule()
        payload, signature = screen.load_frozen_plan(PLAN)
        self.assertEqual(signature, screen.EXPECTED_PLAN_SIGNATURE)
        self.assertEqual(
            payload["artifact_signature_sha256"],
            screen.EXPECTED_ARTIFACT_SIGNATURE,
        )
        self.assertEqual(
            tuple(payload["expected_width4_pulse_tokens"]),
            (4, 7, 10, 13, 17, 20, 23, 26),
        )

    def test_group2_rank_local_mapping_is_two_chunks_per_pulse(self) -> None:
        self.assertEqual(
            screen.schedule_object_indices(
                "tempo_group2", 4, pair_index=0
            ),
            (0, 16, 1, 17),
        )
        self.assertEqual(
            screen.schedule_object_indices(
                "tempo_group2", 7, pair_index=3
            ),
            (2, 18, 3, 19),
        )
        for pair in range(screen.PAIR_COUNT):
            flattened = [
                index
                for token in range(screen.TOKENS)
                for index in screen.schedule_object_indices(
                    "tempo_group2", token, pair_index=pair
                )
            ]
            self.assertEqual(sorted(flattened), list(range(32)))

    def test_foreground_and_greedy_schedules(self) -> None:
        for token in range(screen.TOKENS):
            self.assertEqual(
                screen.schedule_object_indices(
                    "fg_only", token, pair_index=0
                ),
                (),
            )
        self.assertEqual(
            screen.schedule_object_indices(
                "lmcache_greedy", 0, pair_index=2
            ),
            tuple(range(32)),
        )
        self.assertEqual(
            screen.schedule_object_indices(
                "lmcache_greedy", 1, pair_index=2
            ),
            (),
        )

    def test_latin_rows_balance_mode_and_position(self) -> None:
        self.assertEqual(len(screen.BLOCK_SPECS), 9)
        for mode in screen.MODES:
            positions = [
                position
                for _, position, observed in screen.BLOCK_SPECS
                if observed == mode
            ]
            self.assertEqual(sorted(positions), [0, 1, 2])

    def test_stream_parser_returns_delta_token_ids(self) -> None:
        lines = [
            b"event: ignored\n",
            b'data: {"choices":[{"text":"a","token_ids":[11]}]}\n',
            b'data: {"choices":[{"text":"bc","token_ids":[12,13]}]}\n',
            b'data: {"choices":[{"text":"","token_ids":[],"finish_reason":"length"}]}\n',
            b"data: [DONE]\n",
        ]
        ticks = iter((100, 200, 300))
        chunks = list(screen.iter_sse_chunks(lines, now_ns=lambda: next(ticks)))
        self.assertEqual(chunks[0], ([11], "a", 100))
        self.assertEqual(chunks[1], ([12, 13], "bc", 200))
        self.assertEqual(chunks[2], ([], "", 300))

    def test_bad_artifact_signature_is_rejected(self) -> None:
        payload = json.loads(PLAN.read_text(encoding="utf-8"))
        payload["artifact_signature_sha256"] = "0" * 64
        temporary = REPO_ROOT / "results" / "test_bad_group2_artifact.json"
        try:
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact signature changed"):
                screen.load_frozen_plan(temporary)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _records() -> list[dict[str, object]]:
        config = {"test": True}
        records: list[dict[str, object]] = []
        for rank in range(screen.WORLD_SIZE):
            source = rank < screen.RANKS_PER_NODE
            blocks = []
            for block_index, (prompt, position, mode) in enumerate(
                screen.BLOCK_SPECS
            ):
                active = mode != "fg_only"
                local_bytes = (
                    screen.REQUESTS * screen.KV_BYTES_PER_RANK if active else 0
                )
                blocks.append(
                    {
                        "block_index": block_index,
                        "prompt_index": prompt,
                        "latin_position": position,
                        "mode": mode,
                        "client": (
                            {
                                "ttft_ms": 10.0 + block_index,
                                "tpot_p50_ms": 2.0,
                                "tpot_p99_ms": 3.0,
                                "tpot_max_ms": 4.0,
                                "request_e2e_ms": 150.0,
                                "generated_tokens": screen.TOKENS,
                                "output_token_sha256": f"prompt-{prompt}",
                            }
                            if rank == 0
                            else None
                        ),
                        "background_completed_bytes": local_bytes if source else 0,
                        "receiver_verified_bytes": local_bytes if not source else 0,
                        "background_finish_from_request_start_ms": 80.0 if active else 0.0,
                        "post_foreground_drain_ms": 0.0,
                        "schedule_start_adherence_met": True,
                        "absolute_service_deadline_met": True,
                        "start_lag_cap_met": True,
                        "max_control_delivery_lag_ms": 0.2,
                        "max_descriptor_start_lag_ms": 0.3,
                        "transfer_errors": [],
                        "correctness_met": True,
                    }
                )
            records.append(
                {
                    "rank": rank,
                    "config": config,
                    "blocks": blocks,
                }
            )
        return records

    def test_aggregate_accepts_correct_balanced_screen(self) -> None:
        result = screen.aggregate_rank_records(self._records())
        self.assertTrue(result["overall_correctness_met"])
        self.assertTrue(result["output_equivalence_met"])
        self.assertEqual(
            result["screen_outcome"],
            "valid_component_screen_requires_performance_comparison",
        )
        self.assertEqual(result["modes"]["tempo_group2"]["replicates"], 3)

    def test_aggregate_rejects_output_divergence(self) -> None:
        records = self._records()
        records[0]["blocks"][1]["client"]["output_token_sha256"] = "different"
        result = screen.aggregate_rank_records(records)
        self.assertFalse(result["output_equivalence_met"])
        self.assertEqual(
            result["screen_outcome"],
            "invalid_output_or_transfer_correctness",
        )

    def test_aggregate_kills_group2_deadline_miss(self) -> None:
        records = self._records()
        group2_block = next(
            index
            for index, (_, _, mode) in enumerate(screen.BLOCK_SPECS)
            if mode == "tempo_group2"
        )
        records[0]["blocks"][group2_block]["absolute_service_deadline_met"] = False
        result = screen.aggregate_rank_records(records)
        self.assertEqual(
            result["screen_outcome"],
            "kill_absolute_service_deadline_miss",
        )


if __name__ == "__main__":
    unittest.main()
