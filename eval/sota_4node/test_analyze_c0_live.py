from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from eval.sota_4node.analyze_c0_live import analyze_c0_live, main


class AnalyzeC0LiveTests(unittest.TestCase):
    def _write_arm(
        self,
        root: Path,
        name: str,
        *,
        c0_enabled: bool,
        collective_tails: tuple[float, float],
        skews_ns: tuple[int, int],
        window_step_tails: tuple[float, float],
        deadlines: tuple[bool, bool] = (True, True),
    ) -> None:
        arm = root / name
        arm.mkdir()
        for rank in range(4):
            collective_path = arm / f"collectives_rank{rank}.csv"
            with collective_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "rank",
                        "sequence",
                        "step",
                        "phase_index",
                        "phase_signature",
                        "checkpoint_active_at_ready",
                        "finalize_at_ready",
                        "ready_corrected_ns",
                        "gpu_ms",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "rank": rank,
                        "sequence": -1,
                        "step": -1,
                        "phase_index": 0,
                        "phase_signature": "diagnostic",
                        "checkpoint_active_at_ready": 1,
                        "ready_corrected_ns": rank * 10_000,
                        "gpu_ms": 10_000,
                    }
                )
                for sequence, (tail, skew) in enumerate(
                    zip(collective_tails, skews_ns), start=1
                ):
                    writer.writerow(
                        {
                            "rank": rank,
                            "sequence": sequence,
                            "step": 1,
                            "phase_index": 0,
                            "phase_signature": "repeated-phase",
                            "checkpoint_active_at_ready": 1,
                            "ready_corrected_ns": sequence * 1_000_000 + (skew if rank == 3 else 0),
                            "gpu_ms": tail if rank == 3 else tail - 1,
                        }
                    )
                writer.writerow(
                    {
                        "rank": rank,
                        "sequence": 98,
                        "step": 1,
                        "phase_index": 0,
                        "phase_signature": "finalize-no-data",
                        "checkpoint_active_at_ready": 1,
                        "finalize_at_ready": 1,
                        "ready_corrected_ns": rank * 1_000_000,
                        "gpu_ms": 99_999,
                    }
                )
                if rank < 3:
                    writer.writerow(
                        {
                            "rank": rank,
                            "sequence": 99,
                            "step": 99,
                            "phase_index": 0,
                            "phase_signature": "incomplete",
                            "checkpoint_active_at_ready": 1,
                            "ready_corrected_ns": 0,
                            "gpu_ms": 99_999,
                        }
                    )

            step_path = arm / f"steps_rank{rank}.csv"
            with step_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("rank", "step", "checkpoint_window", "step_ms"),
                )
                writer.writeheader()
                for step, tail in enumerate(window_step_tails, start=1):
                    writer.writerow(
                        {
                            "rank": rank,
                            "step": step,
                            "checkpoint_window": 1,
                            "step_ms": tail if rank == 3 else tail - 1,
                        }
                    )

            (arm / f"checkpoint_events_rank{rank}.json").write_text(
                json.dumps(
                    [
                        {"durable_ms": 100 + rank, "deadline_met": deadlines[0]},
                        {"durable_ms": 120 + rank, "deadline_met": deadlines[1]},
                    ]
                ),
                encoding="utf-8",
            )
            (arm / f"summary_rank{rank}.json").write_text(
                json.dumps(
                    {
                        "policy": "v4_open",
                        "rank": rank,
                        "world_size": 4,
                        "c0_enabled": c0_enabled,
                        "c0_d2h_rate_bps": 5_936_536_675 if c0_enabled else None,
                    }
                ),
                encoding="utf-8",
            )

    def test_promising_screen_uses_only_complete_active_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_arm(
                root,
                "open",
                c0_enabled=False,
                collective_tails=(10.0, 20.0),
                skews_ns=(100, 200),
                window_step_tails=(30.0, 40.0),
            )
            self._write_arm(
                root,
                "c0",
                c0_enabled=True,
                collective_tails=(9.0, 18.0),
                skews_ns=(105, 210),
                window_step_tails=(28.0, 37.0),
            )

            result = analyze_c0_live(root)

        self.assertEqual(result["decision"]["verdict"], "promising")
        self.assertFalse(result["evidence_scope"]["causal_claim"])
        self.assertFalse(result["evidence_scope"]["promotion_decision"])
        self.assertEqual(
            result["arms"]["open"]["active_complete_collective_groups"], 2
        )
        self.assertAlmostEqual(
            result["arms"]["open"]["collective_slowest_rank_gpu_ms_p99"],
            19.9,
        )
        self.assertAlmostEqual(
            result["arms"]["open"]["corrected_arrival_skew_ns_p99"],
            199.0,
        )
        self.assertEqual(result["arms"]["c0"]["max_durable_ms"], 123.0)

    def test_missed_deadline_is_kill_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_arm(
                root,
                "open",
                c0_enabled=False,
                collective_tails=(10.0, 20.0),
                skews_ns=(100, 200),
                window_step_tails=(30.0, 40.0),
            )
            self._write_arm(
                root,
                "c0",
                c0_enabled=True,
                collective_tails=(8.0, 16.0),
                skews_ns=(100, 200),
                window_step_tails=(28.0, 35.0),
                deadlines=(True, False),
            )

            result = analyze_c0_live(root)

        self.assertEqual(result["decision"]["verdict"], "kill/no-go")
        self.assertFalse(result["gates"]["all_c0_deadlines_met"])

    def test_cli_accepts_positional_root_and_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_arm(
                root,
                "open",
                c0_enabled=False,
                collective_tails=(10.0, 20.0),
                skews_ns=(100, 200),
                window_step_tails=(30.0, 40.0),
            )
            self._write_arm(
                root,
                "c0",
                c0_enabled=True,
                collective_tails=(9.0, 18.0),
                skews_ns=(105, 210),
                window_step_tails=(28.0, 37.0),
            )
            output = root / "decision.json"
            with redirect_stdout(StringIO()):
                exit_code = main([str(root), "--output", str(output)])
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(written["decision"]["verdict"], "promising")


if __name__ == "__main__":
    unittest.main()
