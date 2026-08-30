from __future__ import annotations

import unittest

from eval.sota_4node import run_lmcache_epoch_2node as base
from eval.sota_4node import run_lmcache_microburst_2node as microburst
from tempo.inference_epoch import EpochProfile, WidthPoint, compile_epoch


class LMCacheMicroburstGeometryTest(unittest.TestCase):
    def test_geometry_has_sixteen_chunks_and_sixty_four_quanta(self) -> None:
        self.assertEqual(microburst.MICROBURST_CHUNKS_PER_REQUEST, 16)
        self.assertEqual(len(microburst.MICROBURST_QUANTA), 64)
        self.assertEqual(microburst.MICROBURST_QUANTA[:4], ((0, 0), (1, 0), (2, 0), (3, 0)))
        self.assertEqual(microburst.MICROBURST_QUANTA[-1], (3, 15))

    def test_capacity_matched_plan_is_exact_protected_ramp(self) -> None:
        plan = compile_epoch(
            EpochProfile(
                total_quanta=64,
                deadline_tokens=34,
                token_slack_ns=(1,) * 4 + (3,) * 60,
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
        )
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.completion_token_exclusive, 34)
        self.assertEqual(plan.width_by_token, (1,) * 4 + (2,) * 30 + (0,) * 30)

    def test_install_changes_only_runtime_geometry_and_parser(self) -> None:
        saved = (
            base.MIB,
            base.CHUNKS_PER_REQUEST,
            base.CANONICAL_QUANTA,
            base._parse_args,
        )
        try:
            microburst.install_microburst_geometry()
            self.assertEqual(base.MIB, 1024)
            self.assertEqual(base.CHUNKS_PER_REQUEST, 16)
            self.assertEqual(base.CANONICAL_QUANTA, microburst.MICROBURST_QUANTA)
            self.assertIs(base._parse_args, microburst._parse_args)
        finally:
            base.MIB, base.CHUNKS_PER_REQUEST, base.CANONICAL_QUANTA, base._parse_args = saved


if __name__ == "__main__":
    unittest.main()
