from pathlib import Path
import unittest

from eval.sota_4node import run_vllm_lmcache_tp16_pair_stagger_coalesced_v1 as candidate


class TP16CoalescedTests(unittest.TestCase):
    def test_contract_and_transfer_geometry(self):
        payload, contract_id = candidate.load_contract(
            Path("eval/sota_4node/real_tp16_pair_stagger_coalesced_v1.json")
        )
        self.assertEqual(contract_id, candidate.CONTRACT_ID)
        self.assertEqual(payload["topology"]["nodes"], 4)
        self.assertEqual(payload["topology"]["pairing"], [[rank, rank + 8] for rank in range(8)])
        self.assertEqual(payload["schedule"]["global_bytes"], 134_217_728)
        self.assertEqual(payload["schedule"]["source_calls_global"], 8)
        self.assertFalse(payload["provenance"]["promotion_valid"])
        self.assertFalse(payload["provenance"]["global_single_flight"])

    def test_exact_one_coalesced_call_per_source(self):
        candidate.validate_schedule()
        for pair, token in enumerate(candidate.SCHEDULED_TOKENS):
            active = [
                scheduled
                for scheduled in range(candidate.TOKENS)
                if candidate.coalesced_indices(
                    "tempo_coalesced", scheduled, pair_index=pair
                )
            ]
            self.assertEqual(active, [token])
            self.assertEqual(
                candidate.coalesced_indices(
                    "tempo_coalesced", token, pair_index=pair
                ),
                tuple(range(32)),
            )

    def test_campaign_rotation_keeps_three_replicates(self):
        for campaign_index, first_mode in enumerate(candidate.MODES):
            specs = candidate.campaign_block_specs(campaign_index)
            sequence = [mode for _, _, mode in specs]
            self.assertEqual(sequence[0], first_mode)
            self.assertEqual(len(sequence), 9)
            self.assertEqual(
                {mode: sequence.count(mode) for mode in candidate.MODES},
                {mode: 3 for mode in candidate.MODES},
            )
        with self.assertRaisesRegex(ValueError, "campaign_index"):
            candidate.campaign_latin_rows(3)

    def test_runtime_boundary_shift_admits_tokens_one_through_eight(self):
        old_shift = candidate._runtime_shift
        old_calls = candidate._runtime_token_zero_calls
        try:
            candidate._runtime_shift = True
            candidate._runtime_token_zero_calls = 0
            self.assertEqual(candidate._runtime_schedule("tempo_group2", 0, pair_index=0), ())
            self.assertEqual(
                candidate._runtime_schedule("tempo_group2", 0, pair_index=0),
                tuple(range(32)),
            )
            for trigger_index in range(1, 8):
                self.assertEqual(
                    candidate._runtime_schedule(
                        "tempo_group2", trigger_index, pair_index=trigger_index
                    ),
                    tuple(range(32)),
                )
        finally:
            candidate._runtime_shift = old_shift
            candidate._runtime_token_zero_calls = old_calls


if __name__ == "__main__":
    unittest.main()
