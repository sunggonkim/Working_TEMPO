from __future__ import annotations

import unittest

from eval.sota_4node import run_tempo_pd_same_server_warm_reuse_client_v131 as client


class WarmReuseContractTest(unittest.TestCase):
    def test_balanced_arm_labels_map_to_stable_offsets(self) -> None:
        for arm, expected in (
            ("fixed_local", 100), ("tempo", 200), ("lmcache_remote", 300)
        ):
            value = {"same_server_balanced_contract": {"arm": arm}}
            client._patch_contract(value)
            contract = value["same_server_balanced_contract"]
            self.assertEqual(contract["nonce_offset"], expected)
            self.assertTrue(contract["cache_keys_reused_within_arm"])
            self.assertFalse(contract["cache_keys_disjoint_across_all_blocks"])

    def test_unknown_arm_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "arm mismatch"):
            client._patch_contract({"same_server_balanced_contract": {"arm": "bad"}})


if __name__ == "__main__":
    unittest.main()
