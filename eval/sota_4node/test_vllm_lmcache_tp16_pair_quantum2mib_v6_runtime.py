from types import SimpleNamespace
import unittest

import numpy as np

from eval.sota_4node import run_vllm_lmcache_tp16_pair_quantum2mib_v6 as candidate


class _DescriptorList:
    def descCount(self):
        return 8

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
    quanta = [object() for _ in range(8)]
    return [
        SimpleNamespace(
            _tempo_logical_index=index,
            _tempo_quantum_index=index // 4,
            _tempo_quantum_transfer_object=quanta[index // 4],
        )
        for index in range(32)
    ]


def _channel():
    channel_type = candidate._quantum2mib_channel_class(_OfficialChannel)
    return channel_type(
        role="sender",
        buffer_ptr=4 * (2 << 20),
        buffer_size=16 << 20,
        align_bytes=512 << 10,
    )


class TP16Quantum2MiBV6RuntimeTests(unittest.TestCase):
    def test_warmup_objects_zero_and_thirty_one_use_two_physical_quanta(self):
        channel = _channel()
        objects = _logical_objects()
        completed = channel.batched_write(
            objects=[objects[0], objects[31]],
            transfer_spec={
                "receiver_id": "rank-8",
                "remote_indexes": np.asarray([0, 31], dtype=np.uint64),
            },
        )
        self.assertEqual(completed, 2)
        self.assertEqual([call["remote_indexes"] for call in channel.calls], [[0], [7]])
        event = channel.tempo_physical_write_events[-1]
        self.assertEqual(event["logical_indices"], [0, 31])
        self.assertEqual(event["quantum_indices"], [0, 7])
        self.assertEqual(event["physical_calls"], 2)
        self.assertEqual(event["completed_physical_descriptors"], 2)

    def test_candidate_rank_block_has_eight_batches_and_eight_physical_calls(self):
        channel = _channel()
        objects = _logical_objects()
        transfer_records = []
        for group in range(8):
            indices = list(range(4 * group, 4 * group + 4))
            channel.batched_write(
                objects=[objects[index] for index in indices],
                transfer_spec={
                    "receiver_id": "rank-8",
                    "remote_indexes": np.asarray(indices, dtype=np.uint64),
                },
            )
            trigger_ns = group * 100
            transfer_records.append(
                {
                    "object_indices": indices,
                    "completed_objects": 4,
                    "trigger_ns": trigger_ns,
                    "finished_ns": trigger_ns + 50,
                }
            )
        block = {
            "background_completed_bytes": 16 << 20,
            "expected_source_bytes": 16 << 20,
            "peak_pending_batches": 1,
            "schedule_start_adherence_met": True,
            "absolute_service_deadline_met": True,
            "post_foreground_drain_ms": 0.0,
            "transfer_records": transfer_records,
        }
        decorated = candidate._decorate_quantum_block(
            block,
            channel=channel,
            rank=3,
            requested_mode="tempo_coalesced",
            write_events=channel.tempo_physical_write_events,
        )
        self.assertEqual(decorated["logical_background_batch_calls"], 8)
        self.assertEqual(len(decorated["transfer_records"]), 8)
        self.assertEqual(decorated["physical_nixl_calls"], 8)
        self.assertEqual(decorated["physical_transfer_descriptors"], 8)
        self.assertTrue(
            all(record["physical_nixl_calls"] == 1 for record in transfer_records)
        )
        self.assertTrue(decorated["schedule_start_adherence_met"])
        self.assertTrue(decorated["absolute_service_deadline_met"])
        self.assertEqual(decorated["post_foreground_drain_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
