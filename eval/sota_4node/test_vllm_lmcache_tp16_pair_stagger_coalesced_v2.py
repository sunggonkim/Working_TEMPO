from pathlib import Path
import unittest

from eval.sota_4node import run_vllm_lmcache_tp16_pair_stagger_coalesced_v2 as candidate


class TP16CoalescedV2Tests(unittest.TestCase):
    def test_analyzer_contract(self):
        payload, contract_id = candidate.load_contract(
            Path("eval/sota_4node/real_tp16_pair_stagger_coalesced_v2.json")
        )
        self.assertEqual(contract_id, candidate.CONTRACT_ID)
        self.assertEqual(payload["topology"]["nodes"], 4)
        self.assertEqual(payload["topology"]["world_size"], 16)
        self.assertEqual(payload["schedule"]["active_sources"], list(range(8)))
        self.assertEqual(payload["schedule"]["active_pairs"], list(range(8)))
        self.assertEqual(payload["schedule"]["global_bytes"], 134_217_728)
        self.assertEqual(payload["schedule"]["source_calls_global"], 8)
        self.assertTrue(payload["campaign"]["single_allocation_required"])
        self.assertFalse(payload["provenance"]["promotion_valid"])

    def test_decorated_result_has_exact_block_accounting(self):
        blocks = []
        for mode in ("fg_only", "lmcache_greedy", "tempo_coalesced") * 3:
            background = mode != "fg_only"
            value = 134_217_728 if background else 0
            blocks.append(
                {
                    "mode": mode,
                    "expected_background_bytes": value,
                    "background_completed_bytes": value,
                    "receiver_verified_bytes": value,
                    "schedule_start_adherence_met": True,
                    "absolute_service_deadline_met": True,
                    "post_foreground_drain_ms": 0.0,
                    "start_lag_cap_met": True,
                }
            )
        result = {
            "config": {"campaign_index": 0},
            "blocks": blocks,
            "candidate_schedule_adherence_met": True,
            "candidate_absolute_deadline_met": True,
            "candidate_no_post_foreground_drain_met": True,
            "candidate_start_lag_cap_met": True,
            "coalesced_contract": {},
        }
        decorated = candidate._decorate_result(result, "allocation-7")
        self.assertEqual(decorated["allocation_id"], "allocation-7")
        self.assertEqual(decorated["nodes"], 4)
        self.assertEqual(decorated["world_size"], 16)
        self.assertFalse(decorated["promotion_valid"])
        for block in decorated["blocks"]:
            expected_calls = 0 if block["mode"] == "fg_only" else 8
            self.assertEqual(block["background_source_calls"], expected_calls)
        self.assertEqual(decorated["coalesced_contract"]["active_sources"], list(range(8)))


if __name__ == "__main__":
    unittest.main()
