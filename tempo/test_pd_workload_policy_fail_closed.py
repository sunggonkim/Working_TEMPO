from __future__ import annotations

import unittest

from tempo.pd_workload_policy import FrozenPDPolicy


class FrozenPolicyFailClosedTest(unittest.TestCase):
    def test_output256_direct_local_is_validated_but_controller_is_not(self) -> None:
        policy = FrozenPDPolicy()
        self.assertTrue(policy.direct_local(512, 256))
        self.assertTrue(policy.direct_local(4096, 256))
        with self.assertRaisesRegex(ValueError, "GPU-validated"):
            policy.controller(256)

    def test_validated_direct_lengths_reject_oversized_prompts(self) -> None:
        policy = FrozenPDPolicy()
        for output_tokens in (16, 128):
            with self.assertRaisesRegex(ValueError, "prompts up to 4096"):
                policy.direct_local(4097, output_tokens)


if __name__ == "__main__":
    unittest.main()
