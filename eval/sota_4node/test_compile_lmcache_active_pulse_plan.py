from __future__ import annotations

from copy import deepcopy
import unittest

from eval.sota_4node import compile_lmcache_active_pulse_plan as active_pulse


class ActivePulseCompilerTests(unittest.TestCase):
    def test_frozen_profile_is_exact_64_quantum_8mib_input(self) -> None:
        profile = active_pulse.frozen_profile()
        self.assertEqual(len(profile.token_base_times_ns), 64)
        self.assertEqual(profile.token_base_times_ns, active_pulse.TOKEN_BASE_TIMES_NS)
        self.assertEqual(profile.deadline_ns, 91_257_744)
        self.assertEqual(profile.start_lag_cap_ns, 2_272_580)
        self.assertEqual(profile.active_lane_penalties_ns, (0, 815_940, 815_940, 815_940, 815_940))
        self.assertEqual(len(profile.quanta), 64)
        self.assertEqual(
            tuple(quantum.lane for quantum in profile.quanta),
            tuple(index % 4 for index in range(64)),
        )
        self.assertTrue(all(quantum.bytes == 1_048_576 for quantum in profile.quanta))
        self.assertTrue(all(quantum.service_ns == 4_902_303 for quantum in profile.quanta))

    def test_fixed_replay_matches_expected_pulses_and_service_prediction(self) -> None:
        plan = active_pulse.replay_fixed_calendar(active_pulse.frozen_profile())
        self.assertTrue(plan.feasible)
        self.assertEqual(set(plan.width_by_token), {0, 4})
        self.assertEqual(
            tuple(index for index, width in enumerate(plan.width_by_token) if width == 4),
            (4, 5, 7, 8, 10, 12, 13, 15, 17, 18, 20, 21, 23, 25, 26, 28),
        )
        self.assertEqual(plan.predicted_completion_ns, 88_923_115)
        self.assertEqual(plan.predicted_max_start_lag_ns, 2_255_116)
        self.assertEqual(plan.total_predicted_penalty_ns, 21_214_440)
        self.assertEqual(plan.peak_predicted_penalty_ns, 815_940)

    def test_artifact_round_trip_labels_unmeasured_saturation(self) -> None:
        artifact = active_pulse.make_active_pulse_experiment_artifact()
        profile, plan = active_pulse.load_active_pulse_experiment_artifact(artifact)
        self.assertEqual(profile, active_pulse.frozen_profile())
        self.assertEqual(plan.width_by_token, active_pulse.EXPECTED_WIDTH_BY_TOKEN)
        entries = artifact["calibration_provenance"]["active_lane_penalty"]["entries"]
        self.assertEqual(
            [entry["active_lanes"] for entry in entries if entry["status"] == "unmeasured_saturation_assumption"],
            [1, 2, 3],
        )
        self.assertEqual(
            artifact["calibration_provenance"]["search_scope"]["claim"],
            "fixed_calendar_replay_not_unconstrained_compiler_search",
        )

    def test_envelope_and_inner_plan_tampering_fail_closed(self) -> None:
        artifact = active_pulse.make_active_pulse_experiment_artifact()
        stale_envelope = deepcopy(artifact)
        stale_envelope["expected_width4_pulse_tokens"][0] = 3
        with self.assertRaises(ValueError):
            active_pulse.load_active_pulse_experiment_artifact(stale_envelope)

        stale_inner = deepcopy(artifact)
        stale_inner["active_service_artifact"]["profile"]["deadline_ns"] += 1
        stale_inner["artifact_signature_sha256"] = active_pulse._envelope_signature(stale_inner)
        with self.assertRaises(ValueError):
            active_pulse.load_active_pulse_experiment_artifact(stale_inner)

    def test_replay_rejects_intermediate_width(self) -> None:
        widths = list(active_pulse.EXPECTED_WIDTH_BY_TOKEN)
        widths[4] = 3
        widths[6] = 1
        with self.assertRaisesRegex(ValueError, "outside"):
            active_pulse.replay_fixed_calendar(
                active_pulse.frozen_profile(), tuple(widths)
            )


if __name__ == "__main__":
    unittest.main()
