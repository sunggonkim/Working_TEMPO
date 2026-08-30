from __future__ import annotations

import unittest

from eval.sota_4node.compile_inference_epoch_plan import (
    milliseconds_to_ns,
    parse_repeated_milliseconds,
    parse_width_penalties,
)
from tempo.inference_epoch import WidthPoint


class CompileInferenceEpochPlanTests(unittest.TestCase):
    def test_exact_decimal_millisecond_conversion(self) -> None:
        self.assertEqual(milliseconds_to_ns("0"), 0)
        self.assertEqual(milliseconds_to_ns("1.25"), 1_250_000)
        self.assertEqual(milliseconds_to_ns("0.000001"), 1)
        with self.assertRaisesRegex(ValueError, "whole nanoseconds"):
            milliseconds_to_ns("0.0000001")
        with self.assertRaisesRegex(ValueError, "finite"):
            milliseconds_to_ns("NaN")

    def test_repeated_slack_syntax(self) -> None:
        self.assertEqual(
            parse_repeated_milliseconds("1x4,3x6,0x6"),
            (1_000_000,) * 4 + (3_000_000,) * 6 + (0,) * 6,
        )
        self.assertEqual(
            parse_repeated_milliseconds("0.5,1"),
            (500_000, 1_000_000),
        )
        with self.assertRaisesRegex(ValueError, "repeat"):
            parse_repeated_milliseconds("1x0")

    def test_width_curve_parser(self) -> None:
        self.assertEqual(
            parse_width_penalties("0:0,1:1,2:3,4:9"),
            (
                WidthPoint(0, 0),
                WidthPoint(1, 1_000_000),
                WidthPoint(2, 3_000_000),
                WidthPoint(4, 9_000_000),
            ),
        )
        with self.assertRaisesRegex(ValueError, "canonical"):
            parse_width_penalties("00:0,1:1")


if __name__ == "__main__":
    unittest.main()
