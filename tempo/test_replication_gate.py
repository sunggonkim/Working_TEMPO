from __future__ import annotations

import unittest

from tempo.replication_gate import (
    InferenceReplicationBlock,
    TrainingReplicationBlock,
    evaluate_inference_replication,
    evaluate_training_replication,
)


class ReplicationGateTests(unittest.TestCase):
    SOURCE = "a" * 64
    WORKLOAD = "b" * 64

    def training_blocks(self, *, wins: int = 5) -> list[TrainingReplicationBlock]:
        return [
            TrainingReplicationBlock(
                f"b{i}", self.SOURCE, self.WORKLOAD,
                100, 90 if i < wins else 110, 100, 90 if i < wins else 110,
                True, True,
            )
            for i in range(5)
        ]

    def test_training_requires_four_of_five_complete_wins(self) -> None:
        result = evaluate_training_replication(self.training_blocks(wins=4))
        self.assertTrue(result.eligible)
        self.assertEqual((result.complete_blocks, result.wins), (5, 4))

    def test_training_rejects_one_block_or_metric_failure(self) -> None:
        self.assertFalse(evaluate_training_replication(self.training_blocks(wins=3)).eligible)
        incomplete = self.training_blocks(wins=5)
        incomplete[-1] = TrainingReplicationBlock("b4", self.SOURCE, self.WORKLOAD, 100, 90, 100, 90, False, True)
        result = evaluate_training_replication(incomplete)
        self.assertFalse(result.eligible)
        self.assertIn("deadline", " ".join(result.reasons))

    def test_training_rejects_duplicate_blocks_and_bad_thresholds(self) -> None:
        blocks = self.training_blocks()
        blocks[-1] = blocks[0]
        with self.assertRaises(ValueError):
            evaluate_training_replication(blocks)
        with self.assertRaises(ValueError):
            evaluate_training_replication([], minimum_blocks=4, required_wins=5)

    def test_training_rejects_mixed_source_or_workload_blocks(self) -> None:
        blocks = self.training_blocks()
        blocks[-1] = TrainingReplicationBlock("b4", "c" * 64, self.WORKLOAD, 100, 90, 100, 90, True, True)
        result = evaluate_training_replication(blocks)
        self.assertFalse(result.eligible)
        self.assertIn("use different source bundles", " ".join(result.reasons))
        blocks = self.training_blocks()
        blocks[-1] = TrainingReplicationBlock("b4", self.SOURCE, "d" * 64, 100, 90, 100, 90, True, True)
        result = evaluate_training_replication(blocks)
        self.assertFalse(result.eligible)
        self.assertIn("use different workload fingerprints", " ".join(result.reasons))

    def inference_blocks(self, *, wins: int = 5) -> list[InferenceReplicationBlock]:
        return [
            InferenceReplicationBlock(
                f"b{i}", self.SOURCE, self.WORKLOAD,
                100, 90 if i < wins else 110,
                100, 90 if i < wins else 110,
                900_000, 900_000 if i < wins else 890_000,
                True, True,
            )
            for i in range(5)
        ]

    def test_inference_requires_latency_win_and_goodput_preservation(self) -> None:
        result = evaluate_inference_replication(self.inference_blocks(wins=4))
        self.assertTrue(result.eligible)
        self.assertEqual(result.wins, 4)
        self.assertFalse(evaluate_inference_replication(self.inference_blocks(wins=3)).eligible)

    def test_inference_rejects_bad_goodput_or_types(self) -> None:
        blocks = self.inference_blocks()
        blocks[0] = InferenceReplicationBlock("b0", self.SOURCE, self.WORKLOAD, 100, 90, 100, 90, 900_000, 800_000, True, True)
        result = evaluate_inference_replication(blocks, required_wins=5)
        self.assertFalse(result.eligible)
        with self.assertRaises(TypeError):
            InferenceReplicationBlock("bad", self.SOURCE, self.WORKLOAD, 1, 1, 1, 1, 1, 1, 1, True)  # type: ignore[arg-type]

    def test_inference_rejects_mixed_source_blocks(self) -> None:
        blocks = self.inference_blocks()
        blocks[-1] = InferenceReplicationBlock("b4", "c" * 64, self.WORKLOAD, 100, 90, 100, 90, 900_000, 900_000, True, True)
        result = evaluate_inference_replication(blocks)
        self.assertFalse(result.eligible)
        self.assertIn("use different source bundles", " ".join(result.reasons))


if __name__ == "__main__":
    unittest.main()
