from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import unittest

from tempo.inference_service import ServiceQuantum
from tempo.inference_service_active import (
    ActiveServiceProfile,
    compile_active_service,
    load_active_service_artifact,
    make_active_service_artifact,
    validate_active_service_plan,
)


def _profile(
    quanta: tuple[ServiceQuantum, ...],
    *,
    times: tuple[int, ...],
    penalties: tuple[int, ...],
    deadline: int,
    lag_cap: int = 0,
    max_width: int | None = None,
    prefix_tokens: int = 0,
    prefix_width: int = 0,
) -> ActiveServiceProfile:
    return ActiveServiceProfile(
        token_base_times_ns=times,
        active_lane_penalties_ns=penalties,
        deadline_ns=deadline,
        start_lag_cap_ns=lag_cap,
        max_issue_width=len(quanta) if max_width is None else max_width,
        protect_prefix_tokens=prefix_tokens,
        protect_prefix_max_width=prefix_width,
        quanta=quanta,
    )


class ActiveInferenceServiceCompilerTests(unittest.TestCase):
    def test_zero_issue_token_pays_for_resident_service(self) -> None:
        profile = _profile(
            (ServiceQuantum(lane=0, bytes=8, service_ns=10),),
            times=(0, 5),
            penalties=(0, 3),
            deadline=10,
        )

        plan = compile_active_service(profile)
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.width_by_token, (1, 0))
        self.assertEqual(plan.predicted_completion_ns, 10)
        self.assertEqual(plan.total_predicted_penalty_ns, 6)
        self.assertEqual(plan.peak_predicted_penalty_ns, 3)

    def test_compiler_avoids_long_residence_when_penalty_matters(self) -> None:
        profile = _profile(
            (ServiceQuantum(lane=0, bytes=8, service_ns=6),),
            times=(0, 2, 4),
            penalties=(0, 1),
            deadline=10,
        )

        active_plan = compile_active_service(profile)
        free_plan = compile_active_service(
            replace(profile, active_lane_penalties_ns=(0, 0))
        )
        self.assertEqual(free_plan.width_by_token, (1, 0, 0))
        self.assertEqual(active_plan.width_by_token, (0, 0, 1))
        self.assertEqual(active_plan.total_predicted_penalty_ns, 1)
        self.assertEqual(active_plan.predicted_completion_ns, 10)

    def test_parallel_lanes_charge_active_count_after_issue(self) -> None:
        profile = _profile(
            tuple(
                ServiceQuantum(lane=lane, bytes=4, service_ns=7)
                for lane in range(4)
            ),
            times=(0,),
            penalties=(0, 1, 2, 3, 4),
            deadline=7,
        )

        plan = compile_active_service(profile)
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.width_by_token, (4,))
        self.assertEqual(plan.predicted_completion_ns, 7)
        self.assertEqual(plan.total_predicted_penalty_ns, 4)

    def test_start_lag_and_deadline_are_replayed_exactly(self) -> None:
        profile = _profile(
            (
                ServiceQuantum(lane=0, bytes=8, service_ns=4),
                ServiceQuantum(lane=0, bytes=8, service_ns=4),
            ),
            times=(0, 3),
            penalties=(0, 0),
            deadline=8,
            lag_cap=1,
        )

        plan = compile_active_service(profile)
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.width_by_token, (1, 1))
        self.assertEqual(plan.predicted_completion_ns, 8)
        self.assertEqual(plan.predicted_max_start_lag_ns, 1)
        validate_active_service_plan(profile, plan)

        late_profile = replace(profile, deadline_ns=7)
        late_plan = compile_active_service(late_profile)
        self.assertFalse(late_plan.feasible)
        validate_active_service_plan(late_profile, late_plan)

    def test_artifact_signature_binds_active_penalties(self) -> None:
        profile = _profile(
            (ServiceQuantum(lane=0, bytes=8, service_ns=10),),
            times=(0, 5),
            penalties=(0, 1),
            deadline=10,
        )
        changed = replace(profile, active_lane_penalties_ns=(0, 2))
        plan = compile_active_service(profile)
        changed_plan = compile_active_service(changed)
        self.assertNotEqual(plan.signature, changed_plan.signature)

        artifact = make_active_service_artifact(profile, plan)
        loaded = load_active_service_artifact(artifact)
        self.assertEqual(loaded, (profile, plan))
        stale = deepcopy(artifact)
        stale["profile"]["active_lane_penalties_ns"][1] = 2
        with self.assertRaises(ValueError):
            load_active_service_artifact(stale)

    def test_active_penalty_curve_covers_every_lane(self) -> None:
        with self.assertRaisesRegex(ValueError, "every lane"):
            _profile(
                (
                    ServiceQuantum(lane=0, bytes=4, service_ns=1),
                    ServiceQuantum(lane=1, bytes=4, service_ns=1),
                ),
                times=(0,),
                penalties=(0, 1),
                deadline=1,
            )


if __name__ == "__main__":
    unittest.main()
