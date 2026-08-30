from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("run_lmcache_nixl_contention_2node.py")
SPEC = importlib.util.spec_from_file_location("lmcache_nixl_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class LMCacheNixlContentionTests(unittest.TestCase):
    def test_official_api_is_named_directly(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("third_party\" / \"lmcache", text)
        self.assertIn("from lmcache.v1.transfer_channel.nixl_channel import NixlChannel", text)
        self.assertIn("channel.lazy_init_peer_connection(", text)
        self.assertIn("channel.batched_write(", text)
        self.assertIn('backends=["UCX"]', text)
        self.assertNotIn("proxy", text.split("def main()", 1)[1].lower())

    def test_aggregate_requires_and_accounts_for_all_bytes(self) -> None:
        config = {
            "requests": 2,
            "kv_bytes": 1024,
            "token_iters": 2,
            "blocks": 1,
            "foreground_bytes": 4096,
            "port_base": 29940,
        }
        records = []
        for rank in range(8):
            records.append(
                {
                    "rank": rank,
                    "config": config,
                    "blocks": [
                        {
                            "token_latency_ms": [rank + 1.0, rank + 2.0],
                            "foreground_correct": True,
                            "source_completed_bytes": 2048 if rank < 4 else 0,
                            "receiver_verified_bytes": 2048 if rank >= 4 else 0,
                            "transfer_elapsed_ms": 3.0 if rank < 4 else 0.0,
                            "post_foreground_drain_ms": 0.5 if rank < 4 else 0.0,
                            "transfer_error": None,
                        }
                    ],
                }
            )
        result = RUNNER.aggregate_rank_records(records)
        block = result["blocks"][0]
        self.assertEqual(block["source_completed_bytes"], 8192)
        self.assertEqual(block["receiver_verified_bytes"], 8192)
        self.assertEqual(block["global_token_tail_p99_ms"], 9.0)
        self.assertTrue(result["overall_correctness_met"])

    def test_receiver_incast_uses_disjoint_descriptors_and_one_receiver(self) -> None:
        self.assertEqual(
            RUNNER._remote_descriptor_indices(3, RUNNER.TRAFFIC_INCAST, 8),
            list(range(24, 32)),
        )
        self.assertEqual(
            RUNNER._receiver_source_pairs(4, RUNNER.TRAFFIC_INCAST),
            (0, 1, 2, 3),
        )
        self.assertEqual(
            RUNNER._memory_object_count(4, RUNNER.TRAFFIC_INCAST, 8), 32
        )
        self.assertEqual(
            RUNNER._memory_object_count(5, RUNNER.TRAFFIC_INCAST, 8), 0
        )

        config = {
            "requests": 2,
            "kv_bytes": 1024,
            "token_iters": 1,
            "blocks": 1,
            "foreground_bytes": 4096,
            "port_base": 29940,
            "traffic_pattern": RUNNER.TRAFFIC_INCAST,
            "background_mode": "nixl_ucx",
        }
        records = []
        for rank in range(8):
            records.append({
                "rank": rank,
                "config": config,
                "blocks": [{
                    "token_latency_ms": [1.0],
                    "foreground_correct": True,
                    "source_completed_bytes": 2048 if rank < 4 else 0,
                    "receiver_verified_bytes": 8192 if rank == 4 else 0,
                    "transfer_elapsed_ms": 2.0 if rank < 4 else 0.0,
                    "post_foreground_drain_ms": 0.0,
                    "transfer_error": None,
                }],
            })
        result = RUNNER.aggregate_rank_records(records)
        self.assertEqual(result["pairing"], [[0, 4], [1, 4], [2, 4], [3, 4]])
        self.assertEqual(result["blocks"][0]["receiver_verified_bytes"], 8192)
        self.assertTrue(result["overall_correctness_met"])

    def test_live_observer_snapshot_is_rank_aggregated_and_fail_closed(self) -> None:
        config = {
            "requests": 1,
            "kv_bytes": 1024,
            "token_iters": 2,
            "blocks": 1,
            "foreground_bytes": 4096,
            "port_base": 29940,
            "background_mode": "nixl_ucx",
        }
        block = {
            "token_latency_ms": [2.0, 4.0],
            "foreground_correct": True,
            "transfer_elapsed_ms": 7.0,
            "transfer_error": None,
        }
        snapshot = RUNNER._observer_snapshot(
            rank_blocks=[dict(block) for _ in range(8)],
            config=config,
            hosts=["nid00001"] * 4 + ["nid00002"] * 4,
            sequence=3,
            producer_state="active",
        )
        self.assertEqual(snapshot.sequence, 3)
        self.assertEqual(snapshot.rank_count, 8)
        self.assertEqual(snapshot.nccl_collective_p99_ms, 4.0)
        self.assertEqual(snapshot.lmcache_transfer_p99_ms, 7.0)
        self.assertTrue(snapshot.correctness_met)

    def test_active_horizon_uses_slowest_rank_receipt(self) -> None:
        config = {
            "requests": 1,
            "kv_bytes": 1024,
            "token_iters": 1,
            "blocks": 1,
            "foreground_bytes": 4096,
            "port_base": 29940,
            "background_mode": "nccl_only",
            "minimum_active_duration_s": 30.0,
        }
        records = []
        for rank in range(8):
            records.append({
                "rank": rank,
                "config": config,
                "active_loop_elapsed_ms": 30_100.0 - rank * 10.0,
                "blocks": [{
                    "token_latency_ms": [1.0],
                    "foreground_correct": True,
                    "source_completed_bytes": 0,
                    "receiver_verified_bytes": 0,
                    "transfer_elapsed_ms": 0.0,
                    "post_foreground_drain_ms": 0.0,
                    "transfer_error": None,
                }],
            })
        result = RUNNER.aggregate_rank_records(records)
        self.assertEqual(result["active_loop"]["rank_min_elapsed_ms"], 30_030.0)
        self.assertTrue(result["active_loop"]["horizon_met"])

        records[-1]["active_loop_elapsed_ms"] = 29_999.0
        result = RUNNER.aggregate_rank_records(records)
        self.assertFalse(result["active_loop"]["horizon_met"])

    def test_nixl_descriptor_uses_physical_data_pointer(self) -> None:
        text = (SCRIPT.parents[2] / "third_party/lmcache/lmcache/v1/transfer_channel/nixl_channel.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("address = int(mem_obj.data_ptr)", text)
        self.assertIn("mem_obj.get_physical_size()", text)
        self.assertNotIn("address = int(mem_obj.meta.address)", text)

    def test_collective_timeout_is_bounded_and_recorded(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("timedelta(seconds=args.process_group_timeout_s)", text)
        self.assertIn('"process_group_timeout_s": args.process_group_timeout_s', text)
        self.assertIn("process-group-timeout-s must be in [5, 3600]", text)
        self.assertIn("minimum-active-duration-s", text)
        self.assertIn("maximum-blocks", text)

    def test_nixl_failure_is_synchronized_before_the_next_barrier(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("local_block_failure", text)
        self.assertIn("dist.all_reduce(block_failure, op=dist.ReduceOp.MAX)", text)
        self.assertIn("synchronized co-job block failure", text)
        self.assertNotIn(
            'raise RuntimeError(\n                f"NIXL transfer timeout',
            text,
        )

    def test_nccl_only_incast_control_does_not_verify_absent_descriptors(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        verification = text.split("verified_bytes = 0", 1)[1].split(
            "dist.barrier()", 1
        )[0]
        self.assertIn("if background_transfer:", verification)


if __name__ == "__main__":
    unittest.main()
