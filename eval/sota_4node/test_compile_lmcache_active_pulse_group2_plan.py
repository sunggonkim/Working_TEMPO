from __future__ import annotations

from copy import deepcopy
import unittest

from eval.sota_4node import compile_lmcache_active_pulse_group2_plan as group2


class Group2CompilerTests(unittest.TestCase):
    def test_exact_fixed_replay(self) -> None:
        profile = group2.frozen_profile()
        plan = group2.replay_fixed_group2_calendar(profile)
        self.assertEqual(len(profile.quanta), 32)
        self.assertTrue(all(q.bytes == 2_097_152 for q in profile.quanta))
        self.assertTrue(all(q.service_ns == 8_733_599 for q in profile.quanta))
        self.assertEqual(set(plan.width_by_token), {0, 4})
        self.assertEqual(
            tuple(i for i, width in enumerate(plan.width_by_token) if width),
            (4, 7, 10, 13, 17, 20, 23, 26),
        )
        self.assertEqual(plan.predicted_completion_ns, 85_222_385)
        self.assertEqual(plan.predicted_max_start_lag_ns, 0)

    def test_estimate_and_initial_validation_state_are_explicit(self) -> None:
        artifact = group2.make_group2_experiment_artifact()
        group2.load_group2_experiment_artifact(artifact)
        provenance = artifact["calibration_provenance"]
        self.assertFalse(provenance["service_estimate"]["measured_at_two_mib"])
        self.assertEqual(
            provenance["service_estimate"]["status"],
            "pilot_derived_linear_estimate_not_2mib_measurement",
        )
        self.assertEqual(
            provenance["retry_scope"]["claim"],
            "fixed_calendar_replay_not_unconstrained_compiler_search",
        )

    def test_tamper_fails_closed(self) -> None:
        artifact = group2.make_group2_experiment_artifact()
        stale = deepcopy(artifact)
        stale["expected_width4_pulse_tokens"][0] = 5
        with self.assertRaises(ValueError):
            group2.load_group2_experiment_artifact(stale)


if __name__ == "__main__":
    unittest.main()
