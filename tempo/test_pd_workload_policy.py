from __future__ import annotations

import unittest

from tempo.pd_workload_policy import FrozenPDPolicy


class FrozenPDPolicyTest(unittest.TestCase):
    def test_output_32_uses_eight_credits(self):
        controller = FrozenPDPolicy().controller(32)
        self.assertEqual(controller.high_local_inflight_cap, 8)
        self.assertEqual(controller.high_pair_interval_ns, 58_000_000)
        self.assertEqual(controller.calibration_requests, 3)

    def test_output_64_uses_nine_credits(self):
        controller = FrozenPDPolicy().controller(64)
        self.assertEqual(controller.high_local_inflight_cap, 9)
        self.assertEqual(controller.high_pair_interval_ns, 70_000_000)

    def test_unvalidated_output_length_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "GPU-validated"):
            FrozenPDPolicy().controller(128)

    def test_short_prompt_output64_uses_local_guard(self):
        policy = FrozenPDPolicy()
        self.assertTrue(policy.force_local(512, 64))
        self.assertFalse(policy.force_local(513, 64))
        self.assertFalse(policy.force_local(512, 32))

    def test_controller_prompt_bound_fails_closed(self):
        policy = FrozenPDPolicy()
        policy.validate_controller_workload(2048, 32)
        policy.validate_controller_workload(2048, 64)
        with self.assertRaisesRegex(ValueError, "prompts up to 2048"):
            policy.validate_controller_workload(2049, 32)
        with self.assertRaisesRegex(ValueError, "only the GPU-validated"):
            policy.validate_controller_workload(512, 128)

    def test_output16_direct_local_fast_path_is_bounded(self):
        policy = FrozenPDPolicy()
        self.assertTrue(policy.direct_local(512, 16))
        self.assertTrue(policy.direct_local(4096, 16))
        self.assertFalse(policy.direct_local(512, 32))
        with self.assertRaisesRegex(ValueError, "prompts up to 4096"):
            policy.direct_local(4097, 16)

    def test_local_guard_rejects_invalid_workload(self):
        with self.assertRaisesRegex(ValueError, "positive int"):
            FrozenPDPolicy().force_local(0, 64)
        with self.assertRaisesRegex(ValueError, "GPU-validated"):
            FrozenPDPolicy().force_local(512, 128)


if __name__ == "__main__":
    unittest.main()
