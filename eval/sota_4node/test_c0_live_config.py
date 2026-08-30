from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.sota_4node.train import _v4_open_c0_rate_bps, parse_args


C0_RATE_GBPS = "5.936536675"
C0_RATE_BPS = 5_936_536_675


class C0LiveConfigTests(unittest.TestCase):
    @staticmethod
    def _argv(policy: str, *extra: str) -> list[str]:
        return [
            "train.py",
            "--policy",
            policy,
            "--output-dir",
            "/tmp/tempo-c0-output",
            "--checkpoint-dir",
            "/tmp/tempo-c0-checkpoints",
            *extra,
        ]

    def test_v4_open_accepts_frozen_c0_rate(self) -> None:
        with patch.object(
            sys,
            "argv",
            self._argv(
                "v4_open",
                "--tempo-v4-d2h-floor-gbps",
                C0_RATE_GBPS,
                "--v4-open-c0",
                "--tempo-v4-telemetry",
                "off",
            ),
        ):
            args = parse_args()
        self.assertEqual(_v4_open_c0_rate_bps(args), C0_RATE_BPS)

    def test_non_open_policy_rejects_c0_rate(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                self._argv(
                    "tempo_v4",
                    "--tempo-v4-d2h-floor-gbps",
                    C0_RATE_GBPS,
                    "--v4-open-c0",
                ),
            ),
            patch("sys.stderr", new=io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parse_args()

    def test_native_patch_paces_before_completion(self) -> None:
        source = (
            Path(__file__).with_name("datastates_tempo_v4.patch").read_text(
                encoding="utf-8"
            )
        )
        branch = source[source.index("if (credit_controller_ && src->tempo_credit_controlled)") :]
        admit = branch.index("admit_d2h_up_to")
        wait = branch.index("wait_until", admit)
        copy = branch.index("cudaMemcpyAsync", wait)
        advance = branch.index("pacing_next_issue_ += interval", copy)
        complete = branch.index("complete_d2h(granted)", advance)
        self.assertLess(admit, wait)
        self.assertLess(wait, copy)
        self.assertLess(copy, advance)
        self.assertLess(advance, complete)
        self.assertIn(
            "static_cast<double>(granted) / pacing_rate",
            branch[copy:complete],
        )

    def test_runner_is_one_node_two_cases_without_submission(self) -> None:
        source = Path(__file__).with_name("run_c0_1node.slurm").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --nodes=1", source)
        self.assertIn("#SBATCH --ntasks-per-node=4", source)
        self.assertIn("#SBATCH --gpus-per-node=4", source)
        self.assertIn("#SBATCH --no-requeue", source)
        self.assertEqual(source.count("run_case open 0"), 1)
        self.assertEqual(source.count("run_case c0 1"), 1)
        self.assertEqual(source.count("--v4-open-c0"), 1)
        self.assertLess(
            source.index("run_case c0 1"),
            source.index("run_case open 0"),
        )
        self.assertIn("--layers 2", source)
        self.assertNotIn("--layers 8", source)
        self.assertIn('event["commit_validated"]', source)
        self.assertIn('if expected:\n        assert all(event["deadline_met"]', source)
        self.assertNotIn("sbatch ", source)
        self.assertNotIn("salloc ", source)
        self.assertNotIn("find ", source)
        self.assertNotIn("retry", source.lower())


if __name__ == "__main__":
    unittest.main()
