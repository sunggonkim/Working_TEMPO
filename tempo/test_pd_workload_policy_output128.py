from __future__ import annotations

import unittest

from tempo.pd_workload_policy import FrozenPDPolicy


class FrozenOutput128PolicyTest(unittest.TestCase):
    def test_output128_direct_local_fast_path_is_bounded(self) -> None:
        policy = FrozenPDPolicy()
        self.assertTrue(policy.direct_local(512, 128))
        self.assertTrue(policy.direct_local(1230, 128))
        self.assertTrue(policy.direct_local(4096, 128))
        with self.assertRaisesRegex(ValueError, "output128.*prompts up to 4096"):
            policy.direct_local(4097, 128)


if __name__ == "__main__":
    unittest.main()
