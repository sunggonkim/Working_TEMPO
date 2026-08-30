import unittest

from tempo.domain_evidence import CounterSupport
from tempo.pd_endpoint_evidence import (
    EndpointDuration,
    KVTransferFeedback,
    KVTransferFeedbackLedger,
    PDEndpointEvidenceStore,
    PDEndpointIdentity,
    PDEndpointRole,
    PDEndpointSnapshot,
    TransferStatus,
    endpoint_metric_names,
    endpoint_metrics,
)


def snapshot(
    identity: PDEndpointIdentity,
    *,
    sequence: int,
    endpoint_ns: int,
    running: int = 0,
    waiting: int = 0,
) -> PDEndpointSnapshot:
    values: dict[str, int | float] = {
        "running_requests": running,
        "waiting_requests": waiting,
        "kv_cache_usage_fraction": 0.25,
        "kv_transfer_bytes_inflight": 0,
        "kv_transfer_ops_inflight": 0,
    }
    if identity.role is PDEndpointRole.PREFILL:
        values.update({
            "active_prefill_tokens": 512,
            "prefill_token_ms_inflight": 1024,
            "prefill_service_p50_ns": 10_000_000,
            "prefill_service_p90_ns": 20_000_000,
        })
    else:
        values.update({
            "active_decode_tokens": 128,
            "active_local_prefill_tokens": 0,
            "local_prefill_token_ms_inflight": 0,
            "decode_step_p90_ns": 25_000_000,
            "kv_install_p90_ns": 5_000_000,
        })
    return PDEndpointSnapshot(
        identity=identity,
        sequence=sequence,
        endpoint_monotonic_ns=endpoint_ns,
        source="endpoint_push_fixture",
        metrics=endpoint_metrics(identity.role, supported=values),
    )


class PDEndpointEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prefill = PDEndpointIdentity(
            "node-p0", PDEndpointRole.PREFILL, 0)
        self.decoder = PDEndpointIdentity(
            "node-d0", PDEndpointRole.DECODER, 0)

    def test_metric_inventory_requires_explicit_missing_support(self):
        names = endpoint_metric_names(PDEndpointRole.PREFILL)
        supported = {name: 0 for name in names}
        supported["kv_cache_usage_fraction"] = 0.0
        supported.pop("prefill_service_p90_ns")
        metrics = endpoint_metrics(
            PDEndpointRole.PREFILL,
            supported=supported,
            unavailable={
                "prefill_service_p90_ns": CounterSupport.NOT_COLLECTED,
            },
        )
        value = next(
            metric for metric in metrics
            if metric.name == "prefill_service_p90_ns"
        )
        self.assertIsNone(value.value)
        self.assertIs(value.support, CounterSupport.NOT_COLLECTED)

        with self.assertRaises(ValueError):
            endpoint_metrics(
                PDEndpointRole.PREFILL,
                supported={"running_requests": 0},
            )

    def test_store_uses_router_receipt_only_for_cross_host_freshness(self):
        store = PDEndpointEvidenceStore((self.prefill, self.decoder))
        # Endpoint clocks deliberately have unrelated epochs.  This is valid:
        # the store never subtracts one endpoint clock from the other.
        store.observe(
            snapshot(
                self.prefill, sequence=1,
                endpoint_ns=9_000_000_000_000,
            ),
            router_received_monotonic_ns=1_000,
        )
        store.observe(
            snapshot(self.decoder, sequence=1, endpoint_ns=17),
            router_received_monotonic_ns=1_010,
        )
        view = store.pair_view(
            0, now_monotonic_ns=1_100, max_age_ns=200)
        self.assertEqual(view.age_ns(PDEndpointRole.PREFILL), 100)
        self.assertEqual(view.age_ns(PDEndpointRole.DECODER), 90)
        self.assertEqual(
            view.prefill.snapshot.metric("active_prefill_tokens").value,
            512,
        )

    def test_store_fails_closed_on_missing_stale_or_regressed_snapshot(self):
        store = PDEndpointEvidenceStore((self.prefill, self.decoder))
        store.observe(
            snapshot(self.prefill, sequence=1, endpoint_ns=100),
            router_received_monotonic_ns=1_000,
        )
        with self.assertRaisesRegex(ValueError, "missing decoder"):
            store.pair_view(
                0, now_monotonic_ns=1_050, max_age_ns=100)

        store.observe(
            snapshot(self.decoder, sequence=1, endpoint_ns=200),
            router_received_monotonic_ns=1_010,
        )
        with self.assertRaisesRegex(ValueError, "stale prefill"):
            store.pair_view(
                0, now_monotonic_ns=1_201, max_age_ns=200)
        with self.assertRaisesRegex(ValueError, "sequence"):
            store.observe(
                snapshot(self.prefill, sequence=1, endpoint_ns=101),
                router_received_monotonic_ns=1_020,
            )
        with self.assertRaisesRegex(ValueError, "endpoint-local"):
            store.observe(
                snapshot(self.prefill, sequence=2, endpoint_ns=99),
                router_received_monotonic_ns=1_020,
            )

    def test_transfer_feedback_keeps_endpoint_durations_separate(self):
        feedback = KVTransferFeedback(
            request_id="request-1",
            pair_index=0,
            source_endpoint_id=self.prefill.endpoint_id,
            destination_endpoint_id=self.decoder.endpoint_id,
            potential_kv_bytes=64 * 1024 * 1024,
            completed_kv_bytes=64 * 1024 * 1024,
            semantic_operations=8,
            status=TransferStatus.SUCCESS,
            durations=(
                EndpointDuration(
                    "producer_submit_to_complete",
                    self.prefill.endpoint_id,
                    12_000_000,
                    "lmcache_producer_completion",
                ),
                EndpointDuration(
                    "consumer_wait_to_install",
                    self.decoder.endpoint_id,
                    8_000_000,
                    "vllm_consumer_install",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "cannot be summed"):
            feedback.total_duration_ns()

        ledger = KVTransferFeedbackLedger((self.prefill, self.decoder))
        ledger.observe(feedback)
        self.assertIs(ledger.get("request-1"), feedback)
        with self.assertRaisesRegex(ValueError, "already observed"):
            ledger.observe(feedback)

    def test_failed_transfer_requires_error_and_success_requires_bytes(self):
        common = {
            "request_id": "request-2",
            "pair_index": 0,
            "source_endpoint_id": self.prefill.endpoint_id,
            "destination_endpoint_id": self.decoder.endpoint_id,
            "potential_kv_bytes": 1024,
            "semantic_operations": 1,
            "durations": (),
        }
        with self.assertRaisesRegex(ValueError, "completed byte"):
            KVTransferFeedback(
                **common,
                completed_kv_bytes=None,
                status=TransferStatus.SUCCESS,
            )
        with self.assertRaisesRegex(ValueError, "needs an error"):
            KVTransferFeedback(
                **common,
                completed_kv_bytes=0,
                status=TransferStatus.TIMEOUT,
            )


if __name__ == "__main__":
    unittest.main()
