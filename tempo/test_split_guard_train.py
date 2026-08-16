from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

from tempo import v4_controller as controller


MIB = 1 << 20


class _Engine:
    def __init__(self) -> None:
        self.prepared: list[tuple[int, int, int, int]] = []

    def prepare_credit_transition(
        self,
        token: int,
        plan_version: int,
        phase_id: int,
        d2h: int,
        pfs: int,
        cap: int,
        watchdog: int,
        not_before: int,
        expires: int,
        preserve_pfs_on_close: bool = False,
        preserve_d2h_on_close: bool = False,
        keep_d2h_active_on_close: bool = False,
    ) -> bool:
        del (
            cap,
            watchdog,
            not_before,
            expires,
            preserve_pfs_on_close,
            preserve_d2h_on_close,
            keep_d2h_active_on_close,
        )
        self.prepared.append((token, plan_version, phase_id, d2h + pfs))
        return True


class SplitGuardTransitionTests(unittest.TestCase):
    def test_runtime_transition_prefix_is_d2h_causal_and_pfs_continuous(self) -> None:
        windows = (
            controller.WindowCredit(
                phase_id=0,
                signature="lead-in",
                kind=controller.WindowKind.COMPUTE,
                d2h_budget_bytes=99 * MIB,
                pfs_budget_bytes=0,
                d2h_spill_bytes=0,
                pfs_spill_bytes=0,
                max_pfs_inflight_bytes=64 * MIB,
            ),
            controller.WindowCredit(
                phase_id=1,
                signature="all-gather",
                kind=controller.WindowKind.COLLECTIVE,
                d2h_budget_bytes=99 * MIB,
                pfs_budget_bytes=99 * MIB,
                d2h_spill_bytes=0,
                pfs_spill_bytes=0,
                max_pfs_inflight_bytes=64 * MIB,
            ),
            controller.WindowCredit(
                phase_id=2,
                signature="compute-after-gather",
                kind=controller.WindowKind.COMPUTE,
                d2h_budget_bytes=99 * MIB,
                pfs_budget_bytes=99 * MIB,
                d2h_spill_bytes=0,
                pfs_spill_bytes=0,
                max_pfs_inflight_bytes=64 * MIB,
            ),
        )
        engine = _Engine()
        backend = SimpleNamespace(
            # The split lane opens a finite, continuous PFS prefix.  It must
            # not be tied to compute/collective kind or mint an event-sized
            # allowance at every boundary.
            current_rank_plan=SimpleNamespace(
                windows=windows,
                planned_pfs_bytes=48 * MIB,
            ),
            split_guard_mode=True,
            scheduled=True,
            v4=controller,
            config=SimpleNamespace(
                d2h_quantum_bytes=MIB,
                max_pfs_inflight_bytes=64 * MIB,
                watchdog_timeout_ns=1_000_000_000,
            ),
            event_expected_pfs_bytes=48 * MIB,
            event_expected_state_bytes=2 * MIB,
            current_installable_phases={0, 1, 2},
            control_lock=threading.RLock(),
            next_runtime_plan_version=7,
            next_runtime_phase_id=100,
            current_runtime_plan_version=0,
            current_plan_step=7,
            checkpoint_step=0,
            current_phase_slots={},
            current_terminal_logical_phase_id=-1,
            current_terminal_credit_closed=False,
            event_runtime_slots={},
            event_installed_runtime_phases=set(),
            event_terminal_runtime_phases=set(),
            ckpt_engine=engine,
            _force_drain=lambda reason: (_ for _ in ()).throw(AssertionError(reason)),
        )

        from eval.sota_4node.train import TempoV4Backend

        self.assertTrue(TempoV4Backend._prepare_plan_transitions(backend))
        slots = backend.current_phase_slots
        self.assertEqual(slots[0]["d2h_active_budget_bytes"], 0)
        self.assertEqual(slots[0]["pfs_active_budget_bytes"], 4 * MIB)
        self.assertEqual(slots[0]["pfs_cumulative_ceiling_bytes"], 4 * MIB)
        # A collective boundary opens no new GPU-facing D2H allowance. Any
        # request already issued in the preceding compute interval is the
        # only residual; the cumulative compute ceiling may contain multiple
        # one-MiB physical requests.
        self.assertEqual(slots[1]["d2h_active_budget_bytes"], 0)
        # PFS is continuous and may advance while the collective executes;
        # only the GPU-facing D2H grant is restricted to compute slots.
        self.assertEqual(slots[1]["pfs_active_budget_bytes"], 4 * MIB)
        self.assertEqual(
            slots[1]["pfs_cumulative_ceiling_bytes"],
            slots[0]["pfs_cumulative_ceiling_bytes"] + 4 * MIB,
        )
        self.assertEqual(slots[2]["d2h_active_budget_bytes"], 2 * MIB)
        self.assertEqual(slots[2]["pfs_active_budget_bytes"], 4 * MIB)
        self.assertEqual(
            slots[2]["pfs_cumulative_ceiling_bytes"],
            slots[1]["pfs_cumulative_ceiling_bytes"] + 4 * MIB,
        )
        terminal = slots[3]
        self.assertEqual(terminal["d2h_active_budget_bytes"], 0)
        # Terminal CLOSE is bookkeeping only.  The ordinary phases have
        # already opened the bounded prefix; a large event-wide PFS release
        # here would recreate the late-admission deadline bug.
        self.assertEqual(terminal["pfs_active_budget_bytes"], 0)
        self.assertEqual(terminal["terminal_pfs_release_bytes"], 0)
        self.assertEqual(len(engine.prepared), 4)

    def test_noninstallable_compute_window_gets_one_residual_d2h_quantum(self) -> None:
        """A noninstallable compute slot may carry only one qD residual."""

        windows = tuple(
            controller.WindowCredit(
                phase_id=phase,
                signature=f"phase-{phase}",
                kind=(
                    controller.WindowKind.COMPUTE
                    if phase % 2 == 0
                    else controller.WindowKind.COLLECTIVE
                ),
                d2h_budget_bytes=8 * MIB,
                pfs_budget_bytes=4 * MIB,
                d2h_spill_bytes=0,
                pfs_spill_bytes=0,
                max_pfs_inflight_bytes=16 * MIB,
            )
            for phase in range(5)
        )
        engine = _Engine()
        backend = SimpleNamespace(
            current_rank_plan=SimpleNamespace(windows=windows),
            split_guard_mode=True,
            scheduled=True,
            v4=controller,
            config=SimpleNamespace(
                d2h_quantum_bytes=MIB,
                max_pfs_inflight_bytes=16 * MIB,
                watchdog_timeout_ns=1_000_000_000,
            ),
            event_expected_pfs_bytes=16 * MIB,
            event_expected_state_bytes=2 * MIB,
            # Phase 2 is a compute-shaped but group-noninstallable interval;
            # it may carry one residual qD request. Phase 4 is the next
            # group-safe compute interval and receives the remaining byte.
            current_installable_phases={0, 1, 3, 4},
            control_lock=threading.RLock(),
            next_runtime_plan_version=1,
            next_runtime_phase_id=1,
            current_runtime_plan_version=0,
            current_plan_step=7,
            checkpoint_step=0,
            current_phase_slots={},
            current_terminal_logical_phase_id=-1,
            current_terminal_credit_closed=False,
            event_runtime_slots={},
            event_installed_runtime_phases=set(),
            event_terminal_runtime_phases=set(),
            split_guard_d2h_cumulative_ceiling=0,
            split_guard_pfs_cumulative_ceiling=0,
            ckpt_engine=engine,
            _force_drain=lambda reason: (_ for _ in ()).throw(AssertionError(reason)),
        )

        from eval.sota_4node.train import TempoV4Backend

        self.assertTrue(TempoV4Backend._prepare_plan_transitions(backend))
        self.assertEqual(
            backend.current_phase_slots[2]["d2h_active_budget_bytes"],
            MIB,
        )
        self.assertEqual(
            backend.current_phase_slots[4]["d2h_active_budget_bytes"],
            MIB,
        )

    def test_published_pfs_lease_opens_once_and_terminal_is_zero(self) -> None:
        """Published storage work must not be deferred to terminal CLOSE."""

        windows = tuple(
            controller.WindowCredit(
                phase_id=phase,
                signature=f"phase-{phase}",
                kind=(
                    controller.WindowKind.COMPUTE
                    if phase != 1
                    else controller.WindowKind.COLLECTIVE
                ),
                d2h_budget_bytes=0,
                pfs_budget_bytes=0,
                d2h_spill_bytes=0,
                pfs_spill_bytes=0,
                max_pfs_inflight_bytes=16 * MIB,
            )
            for phase in range(3)
        )
        engine = _Engine()
        backend = SimpleNamespace(
            current_rank_plan=SimpleNamespace(windows=windows),
            split_guard_mode=True,
            scheduled=True,
            config=SimpleNamespace(
                d2h_quantum_bytes=MIB,
                max_pfs_inflight_bytes=16 * MIB,
                watchdog_timeout_ns=1_000_000_000,
            ),
            event_expected_pfs_bytes=48 * MIB,
            event_expected_state_bytes=MIB,
            event_logical_layout={
                "logical_file_extent_bytes": 48 * MIB,
                "fs_block_alignment_bytes": 4096,
            },
            current_event_host_ready_bytes=0,
            current_group_host_ready_bytes=0,
            current_group_host_ready_valid=True,
            current_installable_phases={0, 1, 2},
            current_plan_step=2,
            checkpoint_step=0,
            control_lock=threading.RLock(),
            next_runtime_plan_version=1,
            next_runtime_phase_id=1,
            current_runtime_plan_version=0,
            current_phase_slots={},
            current_terminal_logical_phase_id=-1,
            current_terminal_credit_closed=False,
            event_runtime_slots={},
            event_installed_runtime_phases=set(),
            event_terminal_runtime_phases=set(),
            split_guard_pfs_cumulative_ceiling=0,
            split_guard_d2h_cumulative_ceiling=0,
            ckpt_engine=engine,
            _force_drain=lambda reason: (_ for _ in ()).throw(AssertionError(reason)),
        )

        from eval.sota_4node.train import TempoV4Backend

        self.assertTrue(TempoV4Backend._prepare_plan_transitions(backend))
        ordinary = [
            slot
            for slot in backend.current_phase_slots.values()
            if not slot["terminal_close"]
        ]
        positive = [
            int(slot["pfs_active_budget_bytes"])
            for slot in ordinary
            if int(slot["pfs_active_budget_bytes"])
        ]
        self.assertEqual(positive, [48 * MIB])
        terminal = backend.current_phase_slots[
            backend.current_terminal_logical_phase_id
        ]
        self.assertEqual(int(terminal["pfs_active_budget_bytes"]), 0)
        self.assertEqual(int(terminal["terminal_pfs_release_bytes"]), 0)

    def test_production_2n_plus_1_shape_preserves_bounds(self) -> None:
        """A 26-collective profile cannot mint larger boundary residuals."""

        windows = tuple(
            controller.WindowCredit(
                phase_id=phase,
                signature=f"phase-{phase}",
                kind=(
                    controller.WindowKind.COMPUTE
                    if phase % 2 == 0
                    else controller.WindowKind.COLLECTIVE
                ),
                d2h_budget_bytes=8 * MIB,
                pfs_budget_bytes=8 * MIB,
                d2h_spill_bytes=0,
                pfs_spill_bytes=0,
                max_pfs_inflight_bytes=16 * MIB,
            )
            for phase in range(53)
        )
        engine = _Engine()
        backend = SimpleNamespace(
            current_rank_plan=SimpleNamespace(windows=windows),
            split_guard_mode=True,
            scheduled=True,
            v4=controller,
            config=SimpleNamespace(
                d2h_quantum_bytes=MIB,
                max_pfs_inflight_bytes=16 * MIB,
                watchdog_timeout_ns=1_000_000_000,
            ),
            # Exact archived rank-local extents: 402,705,672 B D2H and
            # 403,480,576 B logical PFS extent.
            event_expected_pfs_bytes=403_480_576,
            event_expected_state_bytes=402_705_672,
            current_installable_phases=set(range(53)),
            control_lock=threading.RLock(),
            next_runtime_plan_version=1,
            next_runtime_phase_id=1,
            current_runtime_plan_version=0,
            current_plan_step=7,
            checkpoint_step=0,
            current_phase_slots={},
            current_terminal_logical_phase_id=-1,
            current_terminal_credit_closed=False,
            event_runtime_slots={},
            event_installed_runtime_phases=set(),
            event_terminal_runtime_phases=set(),
            ckpt_engine=engine,
            _force_drain=lambda reason: (_ for _ in ()).throw(AssertionError(reason)),
        )

        from eval.sota_4node.train import TempoV4Backend

        self.assertTrue(TempoV4Backend._prepare_plan_transitions(backend))
        slots = backend.current_phase_slots
        ordinary = [slots[index] for index in range(53)]
        self.assertEqual(len(engine.prepared), 54)  # 53 phases + terminal CLOSE
        self.assertLessEqual(
            ordinary[-1]["pfs_cumulative_ceiling_bytes"], 403_480_576
        )
        self.assertEqual(
            sum(int(slot["d2h_active_budget_bytes"]) for slot in ordinary),
            402_705_672,
        )
        terminal = slots[max(slots)]
        self.assertEqual(int(terminal["terminal_pfs_release_bytes"]), 0)
        self.assertEqual(int(terminal["pfs_active_budget_bytes"]), 0)
        self.assertTrue(
            all(
                0 <= int(slot["pfs_active_budget_bytes"]) <= 4 * MIB
                for slot in ordinary
            )
        )
        self.assertTrue(
            all(
                0 <= int(slot["d2h_active_budget_bytes"]) <= 402_705_672
                for slot in ordinary
            )
        )
        self.assertTrue(
            all(
                int(slot["d2h_active_budget_bytes"]) == 0
                or int(slot["logical_phase_id"]) % 2 == 0
                for slot in ordinary
            )
        )
        self.assertEqual(
            [
                int(slot["pfs_cumulative_ceiling_bytes"])
                for slot in ordinary
            ],
            sorted(int(slot["pfs_cumulative_ceiling_bytes"]) for slot in ordinary),
        )

        # A later plan must open only the still-unissued event extent rather
        # than replaying the first prefix and exceeding the checkpoint size.
        backend.current_phase_slots = {}
        backend.current_terminal_logical_phase_id = -1
        backend.current_terminal_credit_closed = False
        self.assertTrue(TempoV4Backend._prepare_plan_transitions(backend))
        second = [
            backend.current_phase_slots[index] for index in range(53)
        ]
        self.assertEqual(
            second[-1]["pfs_cumulative_ceiling_bytes"],
            403_480_576,
        )
        self.assertEqual(
            second[-1]["d2h_cumulative_ceiling_bytes"],
            0,
        )
        self.assertEqual(
            backend.split_guard_d2h_cumulative_ceiling,
            402_705_672,
        )
        self.assertTrue(
            all(
                int(slot["d2h_active_budget_bytes"]) == 0
                for slot in second
                if not bool(slot.get("terminal_close", False))
            )
        )
        self.assertLessEqual(
            second[-1]["pfs_cumulative_ceiling_bytes"],
            backend.event_expected_pfs_bytes,
        )


if __name__ == "__main__":
    unittest.main()
