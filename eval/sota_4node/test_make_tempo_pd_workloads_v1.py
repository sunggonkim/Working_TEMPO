from __future__ import annotations

import unittest

from eval.sota_4node.make_tempo_pd_workloads_v1 import build_workloads


class MakeTempoPDWorkloadsTests(unittest.TestCase):
    def test_distinct_prompts_have_equal_bucket_and_balanced_suffixes(self) -> None:
        calibration, validation, buckets = build_workloads(
            lambda text: list(text.encode("utf-8")),
            repetitions=(2,), samples_per_bucket=4, output_tokens=8,
        )
        self.assertEqual(len(calibration), 4)
        self.assertNotEqual(calibration[0]["prompt"], validation[0]["prompt"])
        self.assertEqual(len(calibration[0]["prompt"]), len(validation[0]["prompt"]))
        self.assertEqual(buckets[0]["samples_per_route"], 4)
        self.assertEqual([row["request_id"].rsplit("r", 1)[1] for row in calibration],
                         ["0", "1", "2", "3"])


if __name__ == "__main__":
    unittest.main()
