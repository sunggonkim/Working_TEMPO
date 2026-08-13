from __future__ import annotations

import unittest

from tempo.split_guard import (
    D2HCausalGuard,
    D2HGuardState,
    NodePFSLane,
    drain_requests,
    transition_deltas,
)


MIB = 1 << 20


class SplitGuardTests(unittest.TestCase):
    def test_d2h_is_closed_during_collective_and_one_quantum_residual(self) -> None:
        guard = D2HCausalGuard(quantum_bytes=MIB)
        self.assertEqual(guard.admit(8 * MIB), 0)
        guard.open_interval()
        first = guard.admit(8 * MIB)
        self.assertEqual(first, MIB)
        self.assertEqual(guard.admit(7 * MIB), 0)
        self.assertEqual(guard.snapshot().residual_bytes, MIB)
        guard.close_interval()
        self.assertEqual(guard.admit(7 * MIB), 0)
        guard.complete()
        self.assertEqual(guard.snapshot().completed_bytes, MIB)
        self.assertEqual(guard.state, D2HGuardState.CLOSED)

    def test_d2h_partial_completion_is_not_double_accounted(self) -> None:
        guard = D2HCausalGuard()
        guard.open_interval()
        guard.admit(2 * MIB)
        with self.assertRaises(ValueError):
            guard.complete(1)
        self.assertEqual(guard.snapshot().admitted_bytes, MIB)
        guard.complete()
        self.assertEqual(guard.snapshot().completed_bytes, MIB)

    def test_pfs_is_work_conserving_after_phase_changes(self) -> None:
        lane = NodePFSLane()
        a = lane.submit(rank=0, bytes=4 * MIB, deadline_ns=100, now_ns=0)
        b = lane.submit(rank=1, bytes=4 * MIB, deadline_ns=200, now_ns=0)
        grants = lane.grant_ready(now_ns=50)
        self.assertEqual({grant.request_id for grant in grants}, {a, b})
        self.assertEqual(lane.snapshot().inflight_bytes, 8 * MIB)
        self.assertEqual(drain_requests(lane, (a, b)), 8 * MIB)
        self.assertEqual(lane.snapshot().queued_bytes, 0)

    def test_pfs_deadline_priority_and_node_caps(self) -> None:
        lane = NodePFSLane(
            quantum_bytes=4 * MIB,
            max_inflight_bytes=16 * MIB,
            max_inflight_requests=4,
        )
        late = lane.submit(rank=0, bytes=4 * MIB, deadline_ns=500, now_ns=0)
        urgent = lane.submit(rank=1, bytes=4 * MIB, deadline_ns=100, now_ns=0)
        other = lane.submit(rank=2, bytes=4 * MIB, deadline_ns=200, now_ns=0)
        grants = lane.grant_ready(now_ns=50, limit=1)
        self.assertEqual(grants[0].request_id, urgent)
        self.assertNotEqual(grants[0].request_id, late)
        self.assertEqual(lane.snapshot().inflight_requests, 1)
        lane.complete(urgent)
        more = lane.grant_ready(now_ns=60, limit=2)
        self.assertEqual({item.request_id for item in more}, {late, other})
        self.assertEqual(lane.snapshot().inflight_requests, 2)

    def test_pfs_rejects_oversize_and_invalid_deadline(self) -> None:
        lane = NodePFSLane(quantum_bytes=4 * MIB)
        with self.assertRaises(ValueError):
            lane.submit(rank=0, bytes=4 * MIB + 1, deadline_ns=1, now_ns=0)
        with self.assertRaises(ValueError):
            lane.submit(rank=0, bytes=MIB, deadline_ns=0, now_ns=1)

    def test_split_transition_opens_pfs_once_and_d2h_only_after_collective(self) -> None:
        self.assertEqual(
            transition_deltas(
                phase_id=0,
                first_phase_id=0,
                is_compute=True,
                d2h_quantum_bytes=MIB,
                event_pfs_bytes=64 * MIB,
            ),
            (0, 64 * MIB),
        )
        self.assertEqual(
            transition_deltas(
                phase_id=1,
                first_phase_id=0,
                is_compute=False,
                d2h_quantum_bytes=MIB,
                event_pfs_bytes=64 * MIB,
            ),
            (0, 0),
        )
        self.assertEqual(
            transition_deltas(
                phase_id=2,
                first_phase_id=0,
                is_compute=True,
                d2h_quantum_bytes=MIB,
                event_pfs_bytes=64 * MIB,
            ),
            (MIB, 0),
        )


if __name__ == "__main__":
    unittest.main()
