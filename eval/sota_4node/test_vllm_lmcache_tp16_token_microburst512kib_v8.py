from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from eval.sota_4node import run_vllm_lmcache_tp16_token_microburst512kib_v8 as candidate


class _DescriptorList:
    def descCount(self):
        return 32

    def __len__(self):
        raise AssertionError("descCount() is the only permitted count API")


class _OfficialChannel:
    def __init__(self, *args, **kwargs):
        self.init_kwargs = dict(kwargs)
        self.nixl_wrapper = SimpleNamespace(xfer_descs=_DescriptorList())
        self.calls = []

    def batched_write(self, objects, transfer_spec=None):
        self.calls.append(
            {
                "objects": list(objects),
                "remote_indexes": np.asarray(
                    transfer_spec["remote_indexes"], dtype=np.uint64
                ).tolist(),
            }
        )
        return len(objects)


def _logical_objects():
    physical = [object() for _ in range(32)]
    return [
        SimpleNamespace(
            _tempo_logical_index=index,
            _tempo_quantum_index=index,
            _tempo_quantum_transfer_object=physical[index],
        )
        for index in range(32)
    ]


class TP16TokenMicroburst512KiBV8Tests(unittest.TestCase):
    def setUp(self):
        candidate._install_profile()

    def test_contract_and_schedule_are_exact(self):
        payload, contract_id = candidate.load_contract(
            Path(
                "eval/sota_4node/real_tp16_token_microburst512kib_v8.json"
            )
        )
        self.assertEqual(contract_id, candidate.CONTRACT_ID)
        self.assertEqual(payload, candidate._expected_contract())
        self.assertEqual(payload["schedule"]["source_calls_global"], 256)
        self.assertEqual(payload["schedule"]["global_bytes"], 128 << 20)
        self.assertEqual(
            payload["descriptor_geometry"]["nixl_transfer_descriptor_bytes"],
            512 << 10,
        )
        self.assertEqual(
            payload["descriptor_geometry"]["registered_descriptors_per_rank"],
            32,
        )

    def test_each_source_covers_every_descriptor_once(self):
        candidate.validate_schedule()
        for pair in range(8):
            active = [
                token
                for token in range(candidate.v1.TOKENS)
                if candidate.microburst_indices(
                    "tempo_coalesced", token, pair_index=pair
                )
            ]
            self.assertEqual(tuple(active), candidate.source_scheduled_tokens(pair))
            batches = [
                candidate.microburst_indices(
                    "tempo_coalesced", token, pair_index=pair
                )
                for token in active
            ]
            self.assertEqual(batches, [(index,) for index in range(32)])
        self.assertEqual(candidate.source_scheduled_tokens(0)[-1], 63)
        self.assertEqual(candidate.source_scheduled_tokens(7)[-1], 62)

    def test_official_path_uses_one_512kib_descriptor_per_call(self):
        channel_type = candidate.v6._quantum2mib_channel_class(_OfficialChannel)
        channel = channel_type(
            role="sender",
            buffer_ptr=8 * (512 << 10),
            buffer_size=16 << 20,
            align_bytes=2 << 20,
        )
        objects = _logical_objects()
        for index, item in enumerate(objects):
            completed = channel.batched_write(
                objects=[item],
                transfer_spec={
                    "receiver_id": "rank-8",
                    "remote_indexes": np.asarray([index], dtype=np.uint64),
                },
            )
            self.assertEqual(completed, 1)
        self.assertEqual(channel.init_kwargs["align_bytes"], 512 << 10)
        self.assertEqual(channel.tempo_registered_descriptor_count, 32)
        self.assertEqual(len(channel.calls), 32)
        self.assertEqual(
            [call["remote_indexes"] for call in channel.calls],
            [[index] for index in range(32)],
        )
        self.assertTrue(all(len(call["objects"]) == 1 for call in channel.calls))
        self.assertEqual(channel.tempo_physical_write_calls, 32)


if __name__ == "__main__":
    unittest.main()
