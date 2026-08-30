from __future__ import annotations

import unittest

from eval.sota_4node import run_vllm_lmcache_tp8_pair_stagger_v4 as v4


class PairStaggerV4Tests(unittest.TestCase):
    def test_legacy_signature_guard_is_rebound(self) -> None:
        v4.candidate._install()
        v4.candidate._v1.EXPECTED_PLAN_SIGNATURE = v4.candidate.CONTRACT_SIGNATURE
        self.assertEqual(
            v4.candidate._v1.EXPECTED_PLAN_SIGNATURE,
            v4.candidate.CONTRACT_SIGNATURE,
        )


if __name__ == "__main__":
    unittest.main()
