from __future__ import annotations

from dataclasses import replace
import unittest

from tempo.inference_epoch import (
    EpochPlan,
    EpochProfile,
    WidthPoint,
    compile_epoch,
    load_epoch_artifact,
    make_epoch_artifact,
    validate_epoch_plan,
)


def _ramp_profile() -> EpochProfile:
    return EpochProfile(
        total_quanta=16,
        deadline_tokens=10,
        token_slack_ns=(1,) * 4 + (3,) * 6 + (0,) * 6,
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


class InferenceEpochCompilerTests(unittest.TestCase):
    def test_compiles_expected_protected_ramp(self) -> None:
        profile = _ramp_profile()
        plan = compile_epoch(profile)
        self.assertTrue(plan.feasible)
        self.assertEqual(
            plan.width_by_token,
            (1,) * 4 + (2,) * 6 + (0,) * 6,
        )
        self.assertEqual(plan.completion_token_exclusive, 10)
        self.assertEqual(plan.total_predicted_penalty_ns, 22)
        self.assertEqual(plan.peak_predicted_penalty_ns, 3)
        self.assertEqual(
            [item for group in plan.quantum_indices_by_token for item in group],
            list(range(16)),
        )

    def test_compilation_is_deterministic_and_round_trips(self) -> None:
        profile = _ramp_profile()
        first = compile_epoch(profile)
        second = compile_epoch(profile)
        self.assertEqual(first, second)
        artifact = make_epoch_artifact(profile, first)
        loaded_profile, loaded_plan = load_epoch_artifact(artifact)
        self.assertEqual(loaded_profile, profile)
        self.assertEqual(loaded_plan, first)
        self.assertEqual(len(first.signature), 64)

    def test_infeasible_profile_fails_closed(self) -> None:
        profile = replace(
            _ramp_profile(),
            deadline_tokens=4,
            token_slack_ns=(1,) * 4 + (0,) * 12,
        )
        plan = compile_epoch(profile)
        self.assertFalse(plan.feasible)
        self.assertEqual(plan.reason, "deadline_capacity_shortfall")
        self.assertIsNone(plan.completion_token_exclusive)
        self.assertEqual(plan.width_by_token, (0,) * 16)
        self.assertTrue(all(not group for group in plan.quantum_indices_by_token))
        validate_epoch_plan(profile, plan)

    def test_tampering_is_rejected(self) -> None:
        profile = _ramp_profile()
        plan = compile_epoch(profile)
        with self.assertRaisesRegex(ValueError, "signature"):
            validate_epoch_plan(profile, replace(plan, signature="0" * 64))
        duplicated = list(plan.quantum_indices_by_token)
        duplicated[0] = (1,)
        tampered = EpochPlan(
            feasible=True,
            reason=plan.reason,
            width_by_token=plan.width_by_token,
            quantum_indices_by_token=tuple(duplicated),
            completion_token_exclusive=plan.completion_token_exclusive,
            total_predicted_penalty_ns=plan.total_predicted_penalty_ns,
            peak_predicted_penalty_ns=plan.peak_predicted_penalty_ns,
            signature=plan.signature,
        )
        with self.assertRaisesRegex(ValueError, "canonical quantum"):
            validate_epoch_plan(profile, tampered)

    def test_profile_rejects_bool_and_unordered_curve(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_quanta"):
            replace(_ramp_profile(), total_quanta=True)
        with self.assertRaisesRegex(ValueError, "increasing"):
            replace(
                _ramp_profile(),
                width_points=(WidthPoint(0, 0), WidthPoint(2, 3), WidthPoint(1, 1)),
            )


if __name__ == "__main__":
    unittest.main()
