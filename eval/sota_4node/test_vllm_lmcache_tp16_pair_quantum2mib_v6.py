from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from eval.sota_4node import run_vllm_lmcache_tp16_pair_quantum2mib_v6 as candidate


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
            raise AssertionError("test tensor supports only contiguous slices")
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


class _FakeDescriptorList:
    def __init__(self, count):
        self._count = count

    def descCount(self):
        return self._count

    def __len__(self):
        raise AssertionError("runner must use descCount(), not len()")


class _FakeOfficialChannel:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = dict(kwargs)
        count = kwargs["buffer_size"] // kwargs["align_bytes"]
        self.nixl_wrapper = SimpleNamespace(xfer_descs=_FakeDescriptorList(count))
        self.official_calls = []

    def batched_write(self, objects, transfer_spec=None):
        self.official_calls.append(
            {
                "objects": list(objects),
                "remote_indexes": np.asarray(
                    transfer_spec["remote_indexes"], dtype=np.uint64
                ).tolist(),
            }
        )
        return len(objects)


class TP16Quantum2MiBV6Tests(unittest.TestCase):
    def test_contract_freezes_two_waves_and_clock_safety(self):
        payload, contract_id = candidate.load_contract(
            Path("eval/sota_4node/real_tp16_pair_quantum2mib_v6.json")
        )
        self.assertEqual(contract_id, candidate.CONTRACT_ID)
        self.assertEqual(payload["schedule"]["source_calls_global"], 64)
        self.assertEqual(payload["schedule"]["global_bytes"], 128 << 20)
        self.assertEqual(
            payload["descriptor_geometry"]["registered_descriptors_per_rank"], 8
        )
        self.assertEqual(
            payload["descriptor_geometry"]["nixl_transfer_descriptor_bytes"],
            2 << 20,
        )
        self.assertEqual(payload["schedule"]["source_scheduled_tokens"][0][-1], 50)
        self.assertEqual(payload["schedule"]["source_scheduled_tokens"][7][-1], 51)
        self.assertFalse(payload["timing"]["cross_host_raw_monotonic_subtraction"])

    def test_mapping_has_eight_aligned_quanta_and_exact_coverage(self):
        candidate.validate_schedule()
        for pair in range(8):
            expected_tokens = (
                candidate.WAVE0_TOKENS if pair < 4 else candidate.WAVE1_TOKENS
            )
            active = [
                token
                for token in range(candidate.v1.TOKENS)
                if candidate.quantum_indices(
                    "tempo_coalesced", token, pair_index=pair
                )
            ]
            self.assertEqual(tuple(active), expected_tokens)
            batches = [
                candidate.quantum_indices(
                    "tempo_coalesced", token, pair_index=pair
                )
                for token in active
            ]
            self.assertTrue(all(len(batch) == 4 for batch in batches))
            self.assertEqual(
                tuple(index for batch in batches for index in batch), tuple(range(32))
            )
        self.assertEqual(
            candidate.quantum_indices("lmcache_greedy", 0, pair_index=0),
            tuple(range(32)),
        )
        self.assertEqual(
            candidate.quantum_indices("lmcache_greedy", 1, pair_index=0), ()
        )

    def test_memory_factory_attaches_four_logical_views_per_quantum(self):
        backing, buffer, objects, index_by_address = candidate._make_quantum2mib_memory(
            _FakeTorch,
            _FakeMemoryObj,
            _FakeMetadata,
            _FakeMemoryFormat,
            requests=2,
            chunk_bytes=512 << 10,
        )
        self.assertEqual(backing.numel(), (18 << 20) - 1)
        self.assertEqual(buffer.numel(), 16 << 20)
        self.assertEqual(buffer.data_ptr() % (2 << 20), 0)
        self.assertEqual(len(objects), 32)
        for quantum_index in range(8):
            group = objects[4 * quantum_index : 4 * quantum_index + 4]
            whole = group[0]._tempo_quantum_transfer_object
            self.assertTrue(
                all(item._tempo_quantum_transfer_object is whole for item in group)
            )
            self.assertTrue(
                all(item._tempo_quantum_index == quantum_index for item in group)
            )
            self.assertEqual(whole.meta.phy_size, 2 << 20)
            self.assertEqual(whole.meta.address, buffer.data_ptr() + quantum_index * (2 << 20))
            self.assertEqual(index_by_address[whole.meta.address], quantum_index)

    @staticmethod
    def _logical_objects():
        quantum_objects = [object() for _ in range(8)]
        return [
            SimpleNamespace(
                _tempo_logical_index=index,
                _tempo_quantum_index=index // 4,
                _tempo_quantum_transfer_object=quantum_objects[index // 4],
            )
            for index in range(32)
        ]

    def test_official_calls_use_desc_count_and_one_descriptor_each(self):
        channel_type = candidate._quantum2mib_channel_class(_FakeOfficialChannel)
        channel = channel_type(
            role="sender",
            buffer_ptr=4 * (2 << 20),
            buffer_size=16 << 20,
            align_bytes=512 << 10,
        )
        objects = self._logical_objects()
        completed = channel.batched_write(
            objects=objects,
            transfer_spec={
                "receiver_id": "rank-8",
                "remote_indexes": np.arange(32, dtype=np.uint64),
            },
        )
        self.assertEqual(channel.init_kwargs["align_bytes"], 2 << 20)
        self.assertEqual(channel.tempo_registered_descriptor_count, 8)
        self.assertEqual(completed, 32)
        self.assertEqual(len(channel.official_calls), 8)
        self.assertEqual(
            [item["remote_indexes"] for item in channel.official_calls],
            [[index] for index in range(8)],
        )
        self.assertTrue(all(len(item["objects"]) == 1 for item in channel.official_calls))
        event = channel.tempo_physical_write_events[-1]
        self.assertEqual(event["physical_calls"], 8)
        self.assertEqual(event["completed_physical_descriptors"], 8)

        channel.official_calls.clear()
        completed = channel.batched_write(
            objects=objects[8:12],
            transfer_spec={
                "receiver_id": "rank-8",
                "remote_indexes": np.arange(8, 12, dtype=np.uint64),
            },
        )
        self.assertEqual(completed, 4)
        self.assertEqual(len(channel.official_calls), 1)
        self.assertEqual(channel.official_calls[0]["remote_indexes"], [2])

    def test_rank_block_preserves_queue_and_physical_instrumentation(self):
        channel_type = candidate._quantum2mib_channel_class(_FakeOfficialChannel)
        channel = channel_type(
            role="sender",
            buffer_ptr=4 * (2 << 20),
            buffer_size=16 << 20,
            align_bytes=512 << 10,
        )
        objects = self._logical_objects()
        channel.batched_write(
            objects=objects,
            transfer_spec={
                "receiver_id": "rank-8",
                "remote_indexes": np.arange(32, dtype=np.uint64),
            },
        )
        block = {
            "background_completed_bytes": 16 << 20,
            "expected_source_bytes": 16 << 20,
            "peak_pending_batches": 1,
            "transfer_records": [
                {
                    "object_indices": list(range(32)),
                    "trigger_ns": 10,
                    "finished_ns": 20,
                }
            ],
        }
        decorated = candidate._decorate_quantum_block(
            block,
            channel=channel,
            rank=0,
            requested_mode="lmcache_greedy",
            write_events=channel.tempo_physical_write_events,
        )
        self.assertEqual(decorated["registered_nixl_descriptors"], 8)
        self.assertEqual(decorated["physical_nixl_calls"], 8)
        self.assertEqual(decorated["physical_transfer_descriptors"], 8)
        self.assertEqual(decorated["physical_transfer_bytes"], 16 << 20)
        self.assertFalse(decorated["head_of_line_queue_detected"])

    def test_aggregate_exposes_64_calls_descriptors_and_peak_pending(self):
        aggregate_blocks = [
            {
                "mode": "fg_only",
                "background_completed_bytes": 0,
                "receiver_verified_bytes": 0,
            },
            {
                "mode": "tempo_coalesced",
                "background_completed_bytes": 128 << 20,
                "receiver_verified_bytes": 128 << 20,
            },
        ]
        records = []
        for rank in range(16):
            is_source = rank < 8
            records.append(
                {
                    "rank": rank,
                    "blocks": [
                        {
                            "registered_nixl_descriptors": 8,
                            "physical_nixl_calls": 0,
                            "physical_transfer_descriptors": 0,
                            "peak_pending_batches": 0,
                            "head_of_line_queue_detected": False,
                            "all_calls_completed_before_next_trigger": True,
                        },
                        {
                            "registered_nixl_descriptors": 8,
                            "physical_nixl_calls": 8 if is_source else 0,
                            "physical_transfer_descriptors": 8 if is_source else 0,
                            "peak_pending_batches": rank + 1 if is_source else 0,
                            "head_of_line_queue_detected": rank == 7,
                            "all_calls_completed_before_next_trigger": rank != 7,
                        },
                    ],
                }
            )
        result = {
            "config": {},
            "background": {},
            "coalesced_contract": {},
            "frozen_group2": {},
            "blocks": aggregate_blocks,
        }
        decorated = candidate._decorate_quantum_result(result, records)
        foreground, tempo = decorated["blocks"]
        self.assertEqual(foreground["physical_nixl_calls_global"], 0)
        self.assertEqual(tempo["physical_nixl_calls_global"], 64)
        self.assertEqual(tempo["physical_transfer_descriptors_global"], 64)
        self.assertEqual(tempo["physical_background_bytes"], 128 << 20)
        self.assertEqual(tempo["peak_pending_batches"], 8)
        self.assertTrue(tempo["head_of_line_queue_detected"])
        self.assertFalse(tempo["all_calls_completed_before_next_trigger"])


if __name__ == "__main__":
    unittest.main()
