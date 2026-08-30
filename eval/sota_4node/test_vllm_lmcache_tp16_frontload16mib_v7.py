from copy import deepcopy
from pathlib import Path
import unittest

from eval.sota_4node import run_vllm_lmcache_tp16_frontload16mib_v7 as candidate


class _DescCountOnly:
    def __init__(self, count):
        self.count = count

    def descCount(self):
        return self.count


class _ExistingLen:
    def __len__(self):
        return 11


def _decorated_source_block(mode, token):
    return {
        "mode": mode,
        "transfer_records": [
            {
                "object_indices": list(range(32)),
                "physical_transfer_descriptors": 1,
                "physical_transfer_bytes": 16 << 20,
                "official_batched_write_objects": 1,
                "scheduled_token": token,
                "triggered_after_token_index": token - 1,
            }
        ],
        "physical_transfer_calls": 1,
        "physical_transfer_descriptors": 1,
    }


def _aggregate_fixture():
    blocks = [
        {
            "mode": "tempo_coalesced",
            "background_completed_bytes": 128 << 20,
            "receiver_verified_bytes": 128 << 20,
        }
        for _ in range(3)
    ]
    result = {
        "blocks": blocks,
        "overall_correctness_met": True,
        "candidate_no_post_foreground_drain_met": False,
        "candidate_absolute_deadline_met": True,
        "candidate_schedule_adherence_met": True,
        "config": {},
        "background": {},
        "coalesced_contract": {},
        "frozen_group2": {},
    }
    records = []
    for rank in range(16):
        physical = 1 if rank < 8 else 0
        records.append(
            {
                "rank": rank,
                "blocks": [
                    {
                        "mode": "tempo_coalesced",
                        "physical_transfer_calls": physical,
                        "physical_transfer_descriptors": physical,
                    }
                    for _ in range(3)
                ],
            }
        )
    return result, records


class TP16Frontload16MiBV7Tests(unittest.TestCase):
    def test_contract_freezes_two_waves_and_one_descriptor(self):
        payload, contract_id = candidate.load_contract(
            Path("eval/sota_4node/real_tp16_frontload16mib_v7.json")
        )
        self.assertEqual(contract_id, candidate.CONTRACT_ID)
        schedule = payload["schedule"]
        self.assertEqual(schedule["scheduled_tokens"], [1, 2])
        self.assertEqual(
            schedule["source_scheduled_tokens"], [1, 1, 1, 1, 2, 2, 2, 2]
        )
        self.assertEqual(schedule["logical_object_indices_per_source_call"], list(range(32)))
        self.assertEqual(schedule["physical_descriptors_global"], 8)
        self.assertEqual(payload["descriptor_geometry"]["physical_bytes_global"], 128 << 20)

    def test_candidate_frontloads_full_batch_but_greedy_stays_at_start(self):
        for pair in range(8):
            expected = 1 if pair < 4 else 2
            self.assertEqual(candidate.source_scheduled_token(pair), expected)
            self.assertEqual(
                candidate.frontload_indices(
                    "tempo_coalesced", expected, pair_index=pair
                ),
                tuple(range(32)),
            )
            self.assertEqual(
                candidate.frontload_indices("lmcache_greedy", 0, pair_index=pair),
                tuple(range(32)),
            )
            self.assertEqual(
                candidate.frontload_indices(
                    "tempo_coalesced", 2 if expected == 1 else 1, pair_index=pair
                ),
                (),
            )
            self.assertEqual(
                candidate.frontload_indices("lmcache_greedy", 1, pair_index=pair),
                (),
            )

    def test_schedule_validator_keeps_three_by_three_campaign(self):
        candidate.validate_schedule()

    def test_desc_count_compatibility_is_narrow_and_idempotent(self):
        descriptor_type = candidate.install_nixl_descriptor_count_compatibility(
            _DescCountOnly
        )
        self.assertIs(descriptor_type, _DescCountOnly)
        self.assertEqual(len(_DescCountOnly(8)), 8)
        self.assertIs(
            candidate.install_nixl_descriptor_count_compatibility(_DescCountOnly),
            _DescCountOnly,
        )
        self.assertEqual(len(_ExistingLen()), 11)
        self.assertIs(
            candidate.install_nixl_descriptor_count_compatibility(_ExistingLen),
            _ExistingLen,
        )

    def test_rank_block_fails_closed_on_token_and_physical_descriptor(self):
        source = _decorated_source_block("tempo_coalesced", 1)
        decorated = candidate._decorate_frontload_block(
            source, rank=0, requested_mode="tempo_coalesced"
        )
        self.assertTrue(decorated["frontload_schedule_exact"])
        self.assertTrue(decorated["frontload_full_logical_batch"])

        wrong_token = _decorated_source_block("tempo_coalesced", 2)
        with self.assertRaisesRegex(RuntimeError, "trigger token"):
            candidate._decorate_frontload_block(
                wrong_token, rank=0, requested_mode="tempo_coalesced"
            )

        wrong_descriptor = _decorated_source_block("tempo_coalesced", 1)
        wrong_descriptor["transfer_records"][0]["physical_transfer_descriptors"] = 2
        with self.assertRaisesRegex(RuntimeError, "descriptor count"):
            candidate._decorate_frontload_block(
                wrong_descriptor, rank=0, requested_mode="tempo_coalesced"
            )

    def test_aggregate_requires_exactly_eight_physical_descriptors(self):
        result, records = _aggregate_fixture()
        decorated = candidate._decorate_frontload_result(result, records)
        for block in decorated["blocks"]:
            self.assertEqual(block["physical_nixl_calls_global"], 8)
            self.assertEqual(block["physical_transfer_descriptors_global"], 8)
        self.assertEqual(
            decorated["frontload_schedule"]["source_scheduled_tokens"],
            [1, 1, 1, 1, 2, 2, 2, 2],
        )
        self.assertTrue(decorated["candidate_gates"]["schedule_tokens_exact"])
        self.assertFalse(
            decorated["candidate_gates"]["no_post_foreground_drain_met"]
        )

        bad_result, bad_records = _aggregate_fixture()
        bad_records = deepcopy(bad_records)
        bad_records[0]["blocks"][0]["physical_transfer_descriptors"] = 2
        with self.assertRaisesRegex(ValueError, "global physical"):
            candidate._decorate_frontload_result(bad_result, bad_records)


if __name__ == "__main__":
    unittest.main()
