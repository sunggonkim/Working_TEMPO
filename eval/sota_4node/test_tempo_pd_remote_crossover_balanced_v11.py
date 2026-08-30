from __future__ import annotations

from pathlib import Path
import unittest

from eval.sota_4node.make_tempo_pd_workloads_balanced_v11 import (
    LATIN_BUCKET_ORDER, _balanced,
)


class BalancedTests(unittest.TestCase):
    def test_latin_order_is_exact_and_bijective(self):
        rows = [
            {"request_id": f"val-b{bucket}-s{sample}-r{bucket * 3 + sample}"}
            for bucket in range(3) for sample in range(3)
        ]
        result = _balanced(rows)
        observed = tuple(int(str(row["request_id"]).split("-b")[1][0]) for row in result)
        self.assertEqual(observed, LATIN_BUCKET_ORDER)
        self.assertEqual({row["request_id"] for row in result}, {row["request_id"] for row in rows})

    def test_wrapper_and_launcher_are_bounded(self):
        root = Path(__file__).resolve().parent
        wrapper = (root / "vllm_lmcache_remote_crossover_balanced_node_v11.py").read_text()
        self.assertIn("context_safe._prepare_workloads = balanced.prepare", wrapper)
        launcher = (root / "run_tempo_pd_remote_crossover_balanced_v11_in_allocation.sh").read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertNotIn("salloc", launcher)


if __name__ == "__main__":
    unittest.main()
