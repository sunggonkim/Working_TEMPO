from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from eval.sota_4node import run_vllm_lmcache_tp16_pair_stagger_coalesced_v4 as candidate


class _FakeTensor:
    def __init__(self, address, length):
        self._address = address
        self._length = length

    def data_ptr(self):
        return self._address

    def numel(self):
        return self._length

    def __getitem__(self, item):
        if not isinstance(item, slice) or item.step not in (None, 1):
            raise AssertionError("test tensor only supports contiguous slices")
        start = 0 if item.start is None else item.start
        stop = self._length if item.stop is None else item.stop
        return _FakeTensor(self._address + start, stop - start)


class _FakeTorch:
    uint8 = object()

    @staticmethod
    def Size(values):
        return tuple(values)

    @staticmethod
    def zeros(length, *, dtype, device):
        if dtype is not _FakeTorch.uint8 or device != "cuda":
            raise AssertionError("unexpected allocation contract")
        return _FakeTensor(12_345, length)


class _FakeMetadata:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeMemoryObj:
    def __init__(self, *, raw_data, metadata, parent_allocator):
        self.raw_data = raw_data
        self.meta = metadata
        self.parent_allocator = parent_allocator


class _FakeMemoryFormat:
    BINARY = object()


class _FakeOfficialChannel:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = dict(kwargs)
        count = kwargs["buffer_size"] // kwargs["align_bytes"]
        self.nixl_wrapper = SimpleNamespace(xfer_descs=[object()] * count)
        self.underlying_objects = None
        self.underlying_spec = None

    def batched_write(self, objects, transfer_spec=None):
        self.underlying_objects = list(objects)
        self.underlying_spec = dict(transfer_spec)
        return len(objects)


class TP16SingleDescriptorV4Tests(unittest.TestCase):
    def test_contract_freezes_one_descriptor_and_exact_bytes(self):
        payload, contract_id = candidate.load_contract(
            Path("eval/sota_4node/real_tp16_pair_stagger_coalesced_v4.json")
        )
        self.assertEqual(contract_id, candidate.CONTRACT_ID)
        geometry = payload["descriptor_geometry"]
        self.assertEqual(geometry["nixl_transfer_descriptors_per_rank"], 1)
        self.assertEqual(geometry["official_batched_write_objects"], 1)
        self.assertEqual(geometry["nixl_transfer_descriptor_bytes"], 16 << 20)
        self.assertEqual(geometry["physical_bytes_global"], 128 << 20)
        self.assertEqual(payload["schedule"]["logical_chunks_per_source_call"], 32)

    def test_memory_keeps_chunk_verification_views_and_whole_object(self):
        backing, buffer, objects, index_by_address = (
            candidate._make_single_descriptor_memory(
                _FakeTorch,
                _FakeMemoryObj,
                _FakeMetadata,
                _FakeMemoryFormat,
                requests=2,
                chunk_bytes=512 << 10,
            )
        )
        self.assertEqual(backing.numel(), (32 << 20) - 1)
        self.assertEqual(buffer.numel(), 16 << 20)
        self.assertEqual(buffer.data_ptr() % (16 << 20), 0)
        self.assertEqual(len(objects), 32)
        whole = objects[0]._tempo_whole_transfer_object
        self.assertTrue(
            all(item._tempo_whole_transfer_object is whole for item in objects)
        )
        self.assertEqual(whole.meta.address, buffer.data_ptr())
        self.assertEqual(whole.meta.phy_size, 16 << 20)
        self.assertEqual(whole.meta.shape, (16 << 20,))
        self.assertEqual(index_by_address[whole.meta.address], 0)
        self.assertEqual(objects[-1].meta.phy_size, 512 << 10)

    def test_official_call_contains_one_whole_object_and_remote_zero(self):
        channel_type = candidate._single_descriptor_channel_class(
            _FakeOfficialChannel
        )
        channel = channel_type(
            role="sender",
            buffer_ptr=3 * (16 << 20),
            buffer_size=16 << 20,
            align_bytes=512 << 10,
        )
        whole = object()
        logical = [SimpleNamespace(_tempo_whole_transfer_object=whole) for _ in range(32)]
        completed = channel.batched_write(
            objects=logical,
            transfer_spec={
                "receiver_id": "rank-8",
                "remote_indexes": np.arange(32, dtype=np.uint64),
            },
        )
        self.assertEqual(channel.init_kwargs["align_bytes"], 16 << 20)
        self.assertEqual(channel.tempo_nixl_transfer_descriptor_count, 1)
        self.assertEqual(completed, 32)
        self.assertEqual(channel.underlying_objects, [whole])
        self.assertEqual(
            channel.underlying_spec["remote_indexes"].tolist(), [0]
        )

    def test_rank_block_records_physical_descriptor_fail_closed(self):
        channel = SimpleNamespace(
            tempo_nixl_transfer_descriptor_count=1,
            tempo_last_logical_object_count=32,
            tempo_last_remote_descriptor_indexes=[0],
        )
        block = {
            "transfer_records": [
                {
                    "completed_objects": 32,
                    "object_indices": list(range(32)),
                }
            ]
        }
        result = candidate._decorate_single_descriptor_block(
            block,
            channel=channel,
            rank=0,
            requested_mode="lmcache_greedy",
        )
        record = result["transfer_records"][0]
        self.assertEqual(record["physical_transfer_descriptors"], 1)
        self.assertEqual(record["official_batched_write_objects"], 1)
        self.assertEqual(record["physical_transfer_bytes"], 16 << 20)
        self.assertTrue(record["contiguous_single_descriptor"])
        self.assertEqual(result["physical_transfer_descriptors"], 1)

        block["transfer_records"][0]["completed_objects"] = 31
        with self.assertRaisesRegex(RuntimeError, "completion count"):
            candidate._decorate_single_descriptor_block(
                block,
                channel=channel,
                rank=0,
                requested_mode="lmcache_greedy",
            )

    def test_aggregate_result_declares_global_descriptor_totals(self):
        result = {
            "config": {},
            "background": {},
            "coalesced_contract": {},
            "frozen_group2": {},
            "blocks": [
                {
                    "mode": "lmcache_greedy",
                    "background_completed_bytes": 128 << 20,
                    "receiver_verified_bytes": 128 << 20,
                },
                {
                    "mode": "fg_only",
                    "background_completed_bytes": 0,
                    "receiver_verified_bytes": 0,
                },
            ],
        }
        decorated = candidate._decorate_result(result)
        self.assertEqual(
            decorated["descriptor_geometry"][
                "physical_source_descriptors_global"
            ],
            8,
        )
        self.assertEqual(
            decorated["blocks"][0]["physical_source_descriptors_global"], 8
        )
        self.assertEqual(
            decorated["blocks"][0]["physical_background_bytes"], 128 << 20
        )
        self.assertEqual(
            decorated["blocks"][1]["physical_source_descriptors_global"], 0
        )


if __name__ == "__main__":
    unittest.main()
