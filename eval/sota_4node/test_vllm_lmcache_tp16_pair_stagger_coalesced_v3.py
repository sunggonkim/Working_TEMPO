from pathlib import Path
import unittest

from eval.sota_4node import run_vllm_lmcache_tp16_pair_stagger_coalesced_v3 as candidate


class _FakeDist:
    def __init__(self) -> None:
        self.broadcasts = 0

    def broadcast_object_list(self, values, *, src):
        self.broadcasts += 1


class TP16CoalescedV3ClockTests(unittest.TestCase):
    def test_contract_declares_no_cross_host_monotonic_subtraction(self):
        payload, contract_id = candidate.load_contract(
            Path("eval/sota_4node/real_tp16_pair_stagger_coalesced_v3.json")
        )
        self.assertEqual(contract_id, candidate.CONTRACT_ID)
        self.assertFalse(payload["timing"]["cross_host_raw_monotonic_subtraction"])
        self.assertEqual(
            payload["timing"]["control_trigger_timestamp"],
            "local_receipt_immediately_after_gloo_broadcast",
        )

    def test_large_host_epoch_offset_cancels_in_local_receipt_intervals(self):
        rank_zero_origin = 728_535_595_616_391
        local_intervals = []
        for local_epoch in (728_535_596_000_000, 923_139_164_000_000):
            ticks = iter((local_epoch, local_epoch + 80_000_000))
            proxy = candidate._RankLocalControlClock(
                _FakeDist(), now_ns=lambda: next(ticks)
            )
            started = [{"kind": "started", "value": rank_zero_origin}]
            proxy.broadcast_object_list(started, src=0)
            token = [
                {
                    "kind": "token",
                    "value": {
                        "token_id": 7,
                        "arrival_ns": rank_zero_origin + 80_000_000,
                    },
                }
            ]
            proxy.broadcast_object_list(token, src=0)
            local_intervals.append(
                token[0]["value"]["arrival_ns"] - started[0]["value"]
            )
        self.assertEqual(local_intervals, [80_000_000, 80_000_000])

    def test_rank_zero_sse_metrics_are_restored_from_original_client_clock(self):
        start = 728_535_595_616_391
        block = {
            "client": {
                "request_started_ns": start,
                "token_arrival_ns": [start + 80_000_000, start + 100_000_000],
                "finished_ns": start + 1_100_000_000,
                "ttft_ms": -1.0,
                "request_e2e_ms": -1.0,
            },
            "transfer_records": [
                {
                    "trigger_ns": 923_139_164_000_000,
                    "enqueue_ns": 923_139_164_500_000,
                    "started_ns": 923_139_164_550_000,
                    "finished_ns": 923_139_200_000_000,
                    "control_delivery_lag_ms": 0.5,
                }
            ],
        }
        result = candidate._decorate_clock_safe_block(
            block, request_timeout_s=180.0
        )
        self.assertEqual(result["client"]["ttft_ms"], 80.0)
        self.assertEqual(result["client"]["request_e2e_ms"], 1100.0)
        self.assertEqual(
            result["transfer_records"][0]["clock_domain"],
            "rank_local_perf_counter_ns",
        )

    def test_mixed_raw_clock_timeline_fails_closed(self):
        rank_zero_trigger = 728_535_595_616_391
        remote_enqueue = 923_139_164_135_566
        block = {
            "client": None,
            "transfer_records": [
                {
                    "trigger_ns": rank_zero_trigger,
                    "enqueue_ns": remote_enqueue,
                    "started_ns": remote_enqueue + 40_000,
                    "finished_ns": remote_enqueue + 1_000_000,
                    "control_delivery_lag_ms": (
                        remote_enqueue - rank_zero_trigger
                    )
                    / 1_000_000.0,
                }
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "mixed clock domains"):
            candidate._decorate_clock_safe_block(block, request_timeout_s=180.0)


if __name__ == "__main__":
    unittest.main()
