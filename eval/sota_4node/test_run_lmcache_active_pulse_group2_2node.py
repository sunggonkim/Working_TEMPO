from __future__ import annotations

from unittest import mock
import unittest

from eval.sota_4node import compile_lmcache_active_pulse_group2_plan as compiled
from eval.sota_4node import run_lmcache_active_pulse_group2_2node as runner


def _plans():
    payload = compiled.make_group2_experiment_artifact()
    profile, logical = compiled.load_group2_experiment_artifact(payload)
    _, runtime = runner._adapt_group2_plan(profile, logical)
    return profile, logical, runtime


class Group2RunnerTests(unittest.TestCase):
    def test_expansion_is_contiguous_width8_and_canonical(self) -> None:
        _, logical, runtime = _plans()
        self.assertEqual(runtime.width_by_token, compiled.EXPECTED_RUNTIME_WIDTH_BY_TOKEN)
        self.assertEqual(runtime.quantum_indices_by_token[4], tuple(range(8)))
        self.assertEqual(runtime.quantum_indices_by_token[7], tuple(range(8, 16)))
        self.assertEqual(
            tuple(i for token in runtime.quantum_indices_by_token for i in token),
            tuple(range(64)),
        )
        self.assertEqual(sum(logical.width_by_token), 32)
        self.assertEqual(runtime.completion_token_exclusive, 27)

    def test_service_and_lag_validity_are_separate(self) -> None:
        profile, logical, runtime = _plans()

        def fake(*args, plan, **kwargs):
            del args, kwargs
            self.assertEqual(plan.completion_token_exclusive, 64)
            return {
                "mode": "tempo_epoch",
                "background_finish_from_block_start_ms": 90.0,
                "post_foreground_drain_ms": 0.0,
                "max_descriptor_start_lag_ms": 3.0,
                "schedule_start_adherence_met": True,
                "plan_deadline_met": True,
                "correctness_met": True,
                "transfer_records": [
                    {"finished_ns": 1_090_000_000, "finished_by_plan_deadline": True}
                ],
            }

        with (
            mock.patch.object(runner, "_ORIGINAL_RUN_BLOCK", fake),
            mock.patch.object(runner, "_ACTIVE_PROFILE", profile),
            mock.patch.object(runner, "_ACTIVE_PLAN", logical),
        ):
            result = runner._run_group2_block(plan=runtime, tokens=64)
        self.assertTrue(result["service_execution_valid"])
        self.assertFalse(result["lag_model_validated"])
        self.assertFalse(result["promotion_valid"])


if __name__ == "__main__":
    unittest.main()
