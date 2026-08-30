from pathlib import Path
import unittest

from eval.sota_4node import run_tempo_pd_same_server_bursty_client_v322 as bursty


class BurstyTraceTests(unittest.TestCase):
    def test_exact_paired_burst_schedule(self):
        rows = bursty._rows(Path(
            "results/tempo_pd_cross_geometry_input_v216/workloads/validation.jsonl"),
            "measured")
        self.assertEqual(len(rows), 48)
        for item in range(24):
            offsets = {row["arrival_offset_ms"] for row in rows
                       if row["request_id"].endswith(f"item-{item:02d}")}
            self.assertEqual(len(offsets), 1)
        starts = sorted({row["arrival_offset_ms"] for row in rows})
        gaps = [right - left for left, right in zip(starts, starts[1:])]
        self.assertEqual(gaps.count(14.0), 18)
        self.assertEqual(gaps.count(220.0), 5)


if __name__ == "__main__":
    unittest.main()
