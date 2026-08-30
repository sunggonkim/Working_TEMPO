from __future__ import annotations

from dataclasses import replace
import unittest

from eval.sota_4node import run_inference_interconnect_2node as base
from eval.sota_4node.run_inference_epoch_2node import (
    CANONICAL_QUANTA,
    EPOCH_BLOCK_MODES,
    EPOCH_LATIN_ROWS,
    EPOCH_MODES,
    install_epoch_scheme,
    schedule_entries_for_plan,
)
from tempo.inference_epoch import (
    EpochProfile,
    WidthPoint,
    compile_epoch,
    make_epoch_artifact,
)


def _profile() -> EpochProfile:
    return EpochProfile(
        total_quanta=16,
        deadline_tokens=10,
        token_slack_ns=(1,) * 4 + (3,) * 6 + (0,) * 6,
        width_points=(
            WidthPoint(0, 0),
            WidthPoint(1, 1),
            WidthPoint(2, 3),
            WidthPoint(4, 9),
        ),
        max_width=2,
        protect_prefix_tokens=4,
        protect_prefix_max_width=1,
    )


class InferenceEpochRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = {
            "MODE_ORDER": base.MODE_ORDER,
            "LATIN_ROWS": base.LATIN_ROWS,
            "BLOCK_MODES": base.BLOCK_MODES,
            "COALESCED_AOT_MODES": base.COALESCED_AOT_MODES,
            "AOT_PAIR_CONCURRENCY_BY_MODE": base.AOT_PAIR_CONCURRENCY_BY_MODE,
            "schedule_entries": base.schedule_entries,
            "schedule_summary": base.schedule_summary,
            "aggregate_rank_records": base.aggregate_rank_records,
        }

    def tearDown(self) -> None:
        for name, value in self.original.items():
            setattr(base, name, value)

    def test_four_mode_latin_screen_is_balanced(self) -> None:
        self.assertEqual(len(EPOCH_BLOCK_MODES), 16)
        for row in EPOCH_LATIN_ROWS:
            self.assertEqual(set(row), set(EPOCH_MODES))
        for mode in EPOCH_MODES:
            self.assertEqual(EPOCH_BLOCK_MODES.count(mode), 4)
            positions = [
                index % len(EPOCH_MODES)
                for index, value in enumerate(EPOCH_BLOCK_MODES)
                if value == mode
            ]
            self.assertEqual(set(positions), set(range(4)))

    def test_modes_move_identical_quanta_without_hot_path_control(self) -> None:
        plan = compile_epoch(_profile())
        requests = 2
        expected = len(CANONICAL_QUANTA) * requests
        self.assertNotIn("tempo", EPOCH_MODES)
        self.assertEqual(
            len(schedule_entries_for_plan(plan, "greedy_coalesced", 0, requests_per_block=requests)),
            expected,
        )
        for mode in EPOCH_MODES[1:]:
            entries = [
                schedule_entries_for_plan(
                    plan,
                    mode,
                    token,
                    requests_per_block=requests,
                )
                for token in range(16)
            ]
            self.assertEqual(sum(map(len, entries)), expected)
        self.assertTrue(
            all(
                not schedule_entries_for_plan(
                    plan, "fg_only", token, requests_per_block=requests
                )
                for token in range(16)
            )
        )

    def test_compiled_calendar_maps_canonical_pair_chunks(self) -> None:
        plan = compile_epoch(_profile())
        self.assertEqual(
            schedule_entries_for_plan(plan, "tempo_epoch", 0),
            ((0, 0, 0),),
        )
        self.assertEqual(
            schedule_entries_for_plan(plan, "tempo_epoch", 4),
            ((0, 0, 1), (0, 1, 1)),
        )
        self.assertEqual(
            schedule_entries_for_plan(plan, "tempo_epoch", 10),
            (),
        )

    def test_install_reconfigures_only_the_compact_contract(self) -> None:
        profile = _profile()
        plan = compile_epoch(profile)
        artifact = make_epoch_artifact(profile, plan)
        install_epoch_scheme(
            profile,
            plan,
            artifact,
            artifact_path="results/test-plan.json",
        )
        self.assertEqual(base.MODE_ORDER, EPOCH_MODES)
        self.assertEqual(base.BLOCK_MODES, EPOCH_BLOCK_MODES)
        self.assertEqual(base.COALESCED_AOT_MODES, frozenset(EPOCH_MODES[1:]))
        self.assertEqual(
            base.schedule_summary("tempo_epoch", requests_per_block=2)["chunks"],
            32,
        )
        self.assertEqual(
            base.schedule_summary("greedy_coalesced", requests_per_block=2)[
                "max_active_pairs"
            ],
            4,
        )
        self.assertEqual(
            base.schedule_summary("static_serial", requests_per_block=2)[
                "max_active_pairs"
            ],
            1,
        )

    def test_infeasible_plan_is_refused_before_distributed_runtime(self) -> None:
        profile = replace(_profile(), deadline_tokens=4)
        plan = compile_epoch(profile)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            install_epoch_scheme(
                profile,
                plan,
                make_epoch_artifact(profile, plan),
                artifact_path="results/infeasible.json",
            )


if __name__ == "__main__":
    unittest.main()
