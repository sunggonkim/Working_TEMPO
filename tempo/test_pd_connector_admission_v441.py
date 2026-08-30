import unittest

from tempo.pd_connector_admission_v439 import ConnectorAdmissionState


class ConnectorAdmissionStateCorrectedTest(unittest.TestCase):
    def test_latched_local_bypass_credit_and_release(self):
        state = ConnectorAdmissionState(local_inflight_cap=6)
        now = 0
        for index in range(5):
            decision = state.decide(f"slow-{index}", now)
            now += 100_000_000
        self.assertEqual(decision.route, "remote_kv_pull")

        local = []
        # The first three burst gaps flush earlier 100 ms gaps from the
        # four-gap median. The following six requests consume local credits.
        for index in range(9):
            now += 14_000_000
            decision = state.decide(f"burst-{index}", now)
            if decision.route == "decoder_local_recompute":
                local.append(decision.request_id)
        self.assertEqual(len(local), 6)
        self.assertEqual(state.local_inflight, 6)

        now += 14_000_000
        capped = state.decide("burst-capped", now)
        self.assertEqual(capped.route, "remote_kv_pull")
        self.assertTrue(capped.local_capped)

        state.finish(local[0])
        now += 14_000_000
        admitted = state.decide("burst-after-release", now)
        self.assertEqual(admitted.route, "decoder_local_recompute")
        self.assertFalse(admitted.local_capped)

    def test_latch_remains_but_credit_disengages_after_sparse_gaps(self):
        state = ConnectorAdmissionState(local_inflight_cap=1)
        now = 0
        for index in range(4):
            state.decide(f"warm-{index}", now)
            now += 14_000_000
        owned = state.decide("owned", now)
        self.assertEqual(owned.route, "decoder_local_recompute")
        self.assertEqual(state.local_inflight, 1)

        for index in range(4):
            now += 100_000_000
            sparse = state.decide(f"sparse-{index}", now)
        self.assertTrue(sparse.high_load_latched)
        self.assertFalse(sparse.microburst_credit_active)
        self.assertEqual(sparse.route, "decoder_local_recompute")

    def test_duplicate_request_is_idempotent(self):
        state = ConnectorAdmissionState()
        first = state.decide("same", 1)
        second = state.decide("same", 999)
        self.assertIs(first, second)
        with self.assertRaises(ValueError):
            state.decide("next", 1)


if __name__ == "__main__":
    unittest.main()
