import unittest
from unittest import mock

from eval.sota_4node import tempo_pd_same_server_latched_microburst25_cap6_v401 as cap6
from eval.sota_4node import tempo_pd_same_server_latched_microburst25_v382 as base


class CapSixBindingTest(unittest.TestCase):
    def test_main_binds_and_restores_exact_policy_and_cap(self):
        original = base.POLICY_ID, base.LOCAL_INFLIGHT_CAP

        def observe(argv):
            return argv, base.POLICY_ID, base.LOCAL_INFLIGHT_CAP

        with mock.patch.object(base, "main", side_effect=observe):
            value = cap6.main(["sentinel"])
        self.assertEqual(
            value,
            (["sentinel"], "tempo-pd-latched-bypass-rolling-credit6-401", 6),
        )
        self.assertEqual((base.POLICY_ID, base.LOCAL_INFLIGHT_CAP), original)

    def test_main_restores_globals_after_failure(self):
        original = base.POLICY_ID, base.LOCAL_INFLIGHT_CAP
        with mock.patch.object(base, "main", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                cap6.main([])
        self.assertEqual((base.POLICY_ID, base.LOCAL_INFLIGHT_CAP), original)


if __name__ == "__main__":
    unittest.main()
