from __future__ import annotations

from copy import deepcopy
from unittest import mock
import unittest

from eval.sota_4node import compile_lmcache_active_pulse_plan as compiled
from eval.sota_4node import run_lmcache_active_pulse_2node as runner


def _adapted_plan():
    artifact = compiled.make_active_pulse_experiment_artifact()
    profile, active_plan = compiled.load_active_pulse_experiment_artifact(artifact)
    _, runtime_plan = runner._adapt_active_plan(profile, active_plan)
    return profile, active_plan, runtime_plan


def _fake_block(*args, plan, **kwargs):
    del args, kwargs
    if plan.completion_token_exclusive != 64:
        raise AssertionError("legacy view must use the full token horizon")
    return {
        "mode": "tempo_epoch",
        "background_finish_from_block_start_ms": 90.0,
        "post_foreground_drain_ms": 0.0,
        "max_descriptor_start_lag_ms": 2.0,
        "schedule_start_adherence_met": True,
        "plan_deadline_met": True,
        "transfer_records": [
            {
                "finished_ns": 1_090_000_000,
                "finished_by_plan_deadline": True,
            }
        ],
    }


class ActivePulseRunnerTests(unittest.TestCase):
    def test_adapter_preserves_exact_assignments_and_issue_completion(self) -> None:
        _, active_plan, runtime_plan = _adapted_plan()
        self.assertEqual(runtime_plan.width_by_token, compiled.EXPECTED_WIDTH_BY_TOKEN)
        self.assertEqual(
            runtime_plan.quantum_indices_by_token,
            active_plan.quantum_indices_by_token,
        )
        self.assertEqual(runtime_plan.completion_token_exclusive, 29)
        self.assertEqual(runtime_plan.signature, active_plan.signature)

    def test_runtime_uses_absolute_block_start_deadline_and_lag_cap(self) -> None:
        profile, active_plan, runtime_plan = _adapted_plan()
        with (
            mock.patch.object(runner, "_ORIGINAL_RUN_BLOCK", _fake_block),
            mock.patch.object(runner, "_ACTIVE_PROFILE", profile),
            mock.patch.object(runner, "_ACTIVE_PLAN", active_plan),
        ):
            result = runner._run_active_pulse_block(
                plan=runtime_plan, tokens=64
            )
        self.assertTrue(result["absolute_service_deadline_met"])
        self.assertTrue(result["actual_start_lag_cap_met"])
        self.assertTrue(result["no_post_foreground_drain_met"])
        self.assertFalse(result["candidate_relative_token_deadline_used"])
        self.assertEqual(result["absolute_service_deadline_ns"], 91_257_744)
        self.assertEqual(
            result["transfer_records"][0]["deadline_semantics"],
            "absolute_from_block_start",
        )

    def test_absolute_deadline_miss_is_not_hidden_by_token_horizon(self) -> None:
        profile, active_plan, runtime_plan = _adapted_plan()

        def late_block(*args, plan, **kwargs):
            result = _fake_block(*args, plan=plan, **kwargs)
            result["background_finish_from_block_start_ms"] = 92.0
            result["transfer_records"][0]["finished_ns"] = 1_092_000_000
            return result

        with (
            mock.patch.object(runner, "_ORIGINAL_RUN_BLOCK", late_block),
            mock.patch.object(runner, "_ACTIVE_PROFILE", profile),
            mock.patch.object(runner, "_ACTIVE_PLAN", active_plan),
        ):
            result = runner._run_active_pulse_block(
                plan=runtime_plan, tokens=64
            )
        self.assertFalse(result["absolute_service_deadline_met"])
        self.assertFalse(result["plan_deadline_met"])

    def test_aggregate_requires_correctness_adherence_deadline_lag_and_no_drain(self) -> None:
        rank_records = []
        for rank in range(8):
            blocks = []
            for mode in runner.base.BLOCK_MODES:
                block = {
                    "mode": mode,
                    "correctness_met": True,
                    "schedule_start_adherence_met": True,
                }
                if mode == "tempo_epoch":
                    block.update(
                        {
                            "absolute_service_deadline_met": True,
                            "actual_start_lag_cap_met": True,
                            "no_post_foreground_drain_met": True,
                        }
                    )
                blocks.append(block)
            rank_records.append({"rank": rank, "blocks": blocks})

        aggregate = {
            "blocks": [{"mode": "tempo_epoch", "correctness_met": True}],
            "modes": {"tempo_epoch": {}},
            "scheduler_semantics": {},
            "overall_correctness_met": True,
            "tempo_epoch_execution_valid": True,
            "screen_outcome": "legacy",
        }
        with mock.patch.object(
            runner, "_ORIGINAL_AGGREGATE", return_value=deepcopy(aggregate)
        ):
            valid = runner._aggregate_active_pulse_records(rank_records)
        self.assertTrue(valid["active_pulse_execution_valid"])
        self.assertEqual(
            valid["screen_outcome"],
            "valid_measurement_requires_performance_comparison",
        )

        rank_records[0]["blocks"][3]["no_post_foreground_drain_met"] = False
        with mock.patch.object(
            runner, "_ORIGINAL_AGGREGATE", return_value=deepcopy(aggregate)
        ):
            invalid = runner._aggregate_active_pulse_records(rank_records)
        self.assertFalse(invalid["active_pulse_execution_valid"])
        self.assertEqual(invalid["screen_outcome"], "kill_post_foreground_drain")


if __name__ == "__main__":
    unittest.main()
