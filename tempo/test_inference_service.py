from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import unittest

from tempo.inference_service import (
    ServicePlan,
    ServiceProfile,
    ServiceQuantum,
    compile_service,
    load_service_artifact,
    make_service_artifact,
    validate_service_plan,
)


def _profile(
    quanta: tuple[ServiceQuantum, ...],
    *,
    times: tuple[int, ...],
    penalties: tuple[int, ...],
    deadline: int,
    lag_cap: int = 0,
    prefix_tokens: int = 0,
    prefix_width: int = 0,
) -> ServiceProfile:
    return ServiceProfile(
        token_base_times_ns=times,
        width_penalties_ns=penalties,
        deadline_ns=deadline,
        start_lag_cap_ns=lag_cap,
        max_width=len(penalties) - 1,
        protect_prefix_tokens=prefix_tokens,
        protect_prefix_max_width=prefix_width,
        quanta=quanta,
    )


class InferenceServiceCompilerTests(unittest.TestCase):
    def test_small_vs_coalesced_service_is_simulated(self) -> None:
        small = _profile(
            (
                ServiceQuantum(lane=0, bytes=8, service_ns=6),
                ServiceQuantum(lane=0, bytes=8, service_ns=6),
            ),
            times=(0, 5),
            penalties=(0, 0, 0),
            deadline=12,
            lag_cap=1,
        )
        coalesced = replace(
            small,
            quanta=(ServiceQuantum(lane=0, bytes=16, service_ns=10),),
        )

        small_plan = compile_service(small)
        coalesced_plan = compile_service(coalesced)
        self.assertTrue(small_plan.feasible)
        self.assertTrue(coalesced_plan.feasible)
        self.assertEqual(small_plan.width_by_token, (1, 1))
        self.assertEqual(small_plan.predicted_completion_ns, 12)
        self.assertEqual(small_plan.predicted_max_start_lag_ns, 1)
        self.assertEqual(coalesced_plan.width_by_token, (1, 0))
        self.assertEqual(coalesced_plan.predicted_completion_ns, 10)
        self.assertEqual(
            sum(quantum.bytes for quantum in small.quanta),
            sum(quantum.bytes for quantum in coalesced.quanta),
        )

    def test_four_lanes_complete_in_parallel(self) -> None:
        profile = _profile(
            tuple(
                ServiceQuantum(lane=lane, bytes=4, service_ns=7)
                for lane in range(4)
            ),
            times=(0,),
            penalties=(0, 1, 2, 3, 4),
            deadline=7,
        )
        plan = compile_service(profile)
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.width_by_token, (4,))
        self.assertEqual(plan.quantum_indices_by_token, ((0, 1, 2, 3),))
        self.assertEqual(plan.predicted_completion_ns, 7)
        self.assertEqual(plan.predicted_max_start_lag_ns, 0)

    def test_completion_exactly_at_deadline_is_feasible(self) -> None:
        profile = _profile(
            (ServiceQuantum(lane=3, bytes=32, service_ns=5),),
            times=(3,),
            penalties=(0, 1),
            deadline=8,
        )
        exact = compile_service(profile)
        late = compile_service(replace(profile, deadline_ns=7))
        self.assertTrue(exact.feasible)
        self.assertEqual(exact.predicted_completion_ns, 8)
        self.assertFalse(late.feasible)
        self.assertEqual(late.reason, "deadline_service_shortfall")
        validate_service_plan(replace(profile, deadline_ns=7), late)

    def test_payload_change_changes_signature_and_stale_plan_fails(self) -> None:
        profile = _profile(
            (ServiceQuantum(lane=0, bytes=16, service_ns=2),),
            times=(0,),
            penalties=(0, 1),
            deadline=2,
        )
        changed = replace(
            profile,
            quanta=(ServiceQuantum(lane=0, bytes=17, service_ns=2),),
        )
        original_plan = compile_service(profile)
        changed_plan = compile_service(changed)
        self.assertNotEqual(original_plan.signature, changed_plan.signature)
        with self.assertRaisesRegex(ValueError, "signature"):
            validate_service_plan(changed, original_plan)

        artifact = make_service_artifact(profile, original_plan)
        loaded_profile, loaded_plan = load_service_artifact(artifact)
        self.assertEqual((loaded_profile, loaded_plan), (profile, original_plan))
        stale = deepcopy(artifact)
        stale["profile"]["quanta"][0]["bytes"] = 17
        with self.assertRaisesRegex(ValueError, "signature"):
            load_service_artifact(stale)

    def test_lower_penalty_later_schedule_beats_earliest_issue(self) -> None:
        profile = _profile(
            (
                ServiceQuantum(lane=0, bytes=8, service_ns=1),
                ServiceQuantum(lane=1, bytes=8, service_ns=1),
            ),
            times=(0, 10),
            penalties=(0, 1, 9),
            deadline=20,
        )
        plan = compile_service(profile)
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.width_by_token, (1, 1))
        self.assertEqual(plan.total_predicted_penalty_ns, 2)
        self.assertEqual(plan.peak_predicted_penalty_ns, 1)
        self.assertEqual(plan.predicted_completion_ns, 12)
        self.assertGreater(plan.predicted_completion_ns, 1)

    def test_protected_prefix_is_a_hard_width_limit(self) -> None:
        profile = _profile(
            (
                ServiceQuantum(lane=0, bytes=4, service_ns=1),
                ServiceQuantum(lane=1, bytes=4, service_ns=1),
            ),
            times=(0,),
            penalties=(0, 1, 2),
            deadline=2,
            prefix_tokens=1,
            prefix_width=1,
        )
        self.assertFalse(compile_service(profile).feasible)

    def test_tamper_and_non_exact_payloads_are_rejected(self) -> None:
        profile = _profile(
            (
                ServiceQuantum(lane=0, bytes=4, service_ns=2),
                ServiceQuantum(lane=1, bytes=4, service_ns=2),
            ),
            times=(0,),
            penalties=(0, 1, 2),
            deadline=2,
        )
        plan = compile_service(profile)
        with self.assertRaisesRegex(ValueError, "signature"):
            validate_service_plan(profile, replace(plan, signature="0" * 64))

        tampered = ServicePlan(
            feasible=plan.feasible,
            reason=plan.reason,
            width_by_token=plan.width_by_token,
            quantum_indices_by_token=((1, 0),),
            predicted_completion_ns=plan.predicted_completion_ns,
            predicted_max_start_lag_ns=plan.predicted_max_start_lag_ns,
            total_predicted_penalty_ns=plan.total_predicted_penalty_ns,
            peak_predicted_penalty_ns=plan.peak_predicted_penalty_ns,
            signature=plan.signature,
        )
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate_service_plan(profile, tampered)

        artifact = make_service_artifact(profile, plan)
        artifact["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "fields are not exact"):
            load_service_artifact(artifact)
        with self.assertRaisesRegex(ValueError, "max_width"):
            replace(profile, max_width=True)


if __name__ == "__main__":
    unittest.main()
