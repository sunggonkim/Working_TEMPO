from __future__ import annotations

import json
from pathlib import Path
import unittest

from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as hybrid


class HybridBoostStaticTests(unittest.TestCase):
    def test_contract_exact(self) -> None:
        path = Path("eval/sota_4node/real_tp16_hybrid_boost_v5.json")
        self.assertEqual(json.loads(path.read_text()), hybrid._expected_contract())

    def test_latin_balance(self) -> None:
        self.assertEqual(len(hybrid.BLOCKS), 9)
        for prompt in range(3):
            modes = [mode for value, mode in hybrid.BLOCKS if value == prompt]
            self.assertEqual(set(modes), set(hybrid.MODES))
        for mode in hybrid.MODES:
            self.assertEqual(sum(value == mode for _, value in hybrid.BLOCKS), 3)

    def test_exact_geometry_and_cap(self) -> None:
        self.assertEqual(hybrid.BYTES_PER_SOURCE, 16 << 20)
        self.assertEqual(hybrid.GLOBAL_BYTES, 128 << 20)
        self.assertEqual(hybrid.SOURCE_COUNT, 8)
        self.assertEqual(hybrid.BOOST_WAIT_CAP_MS, 35.0)

    def test_launcher_has_one_bounded_srun_and_no_plan_cli(self) -> None:
        text = Path(
            "eval/sota_4node/run_vllm_lmcache_tp16_hybrid_boost_in_allocation.sh"
        ).read_text()
        self.assertEqual(text.count("srun --exact"), 1)
        self.assertIn("timeout --foreground", text)
        self.assertIn("${#TEMPO_JOB_HOSTS[@]}", text)
        self.assertNotIn('    --plan "${PLAN_PATH}"', text)
        self.assertIn("--time=00:29:30", text)


if __name__ == "__main__":
    unittest.main()
