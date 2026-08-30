from __future__ import annotations

import json
from pathlib import Path
import unittest

from eval.sota_4node import run_vllm_lmcache_tp8_pair_stagger_v3 as candidate


class PairStaggerV3Tests(unittest.TestCase):
    def test_mapping_exact_coverage_and_round_robin(self) -> None:
        candidate.validate_stagger_schedule()
        for pair in range(4):
            tokens = [
                token
                for token in range(64)
                if candidate.pair_stagger_object_indices(
                    "tempo_group2", token, pair_index=pair
                )
            ]
            self.assertEqual(tokens, [1 + 2 * pair + 8 * group for group in range(8)])
            indices = [
                index
                for token in tokens
                for index in candidate.pair_stagger_object_indices(
                    "tempo_group2", token, pair_index=pair
                )
            ]
            self.assertEqual(sorted(indices), list(range(32)))

    def test_request_start_hook_is_not_a_duplicate(self) -> None:
        candidate._runtime_shift = True
        candidate._runtime_token_zero_calls = 0
        try:
            self.assertEqual(
                candidate._runtime_schedule("tempo_group2", 0, pair_index=0), ()
            )
            self.assertEqual(
                candidate._runtime_schedule("tempo_group2", 0, pair_index=0),
                (0, 16, 1, 17),
            )
        finally:
            candidate._runtime_shift = False

    def test_contract_is_exact(self) -> None:
        path = Path("eval/sota_4node/real_tp8_pair_stagger_v1.json")
        payload, signature = candidate.load_stagger_contract(path)
        self.assertEqual(signature, candidate.CONTRACT_SIGNATURE)
        self.assertFalse(payload["provenance"]["global_single_flight"])
        self.assertTrue(payload["provenance"]["physical_transfer_overlap_possible"])

    def test_changed_mapping_fails_closed(self) -> None:
        path = Path("eval/sota_4node/real_tp8_pair_stagger_v1.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schedule"]["mapping"][0]["active_pair"] = 3
        temporary = Path("/tmp/tempo_pair_stagger_bad_contract.json")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        try:
            with self.assertRaisesRegex(ValueError, "contents changed"):
                candidate.load_stagger_contract(temporary)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
