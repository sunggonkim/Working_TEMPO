from __future__ import annotations

import unittest

from eval.sota_4node import analyze_tempo_pd_kv_only_attribution as analyzer


class KVOnlyAttributionAnalyzerTest(unittest.TestCase):
    def test_empty_optional_tenant_is_explicit_not_fabricated(self):
        summary = analyzer._latencies([], allow_empty=True)
        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["e2e_median_ms"])
        with self.assertRaisesRegex(ValueError, "latency group is empty"):
            analyzer._latencies([])

    def test_weighted_endpoint_mean_uses_success_count(self):
        cumulative = {
            "pair0-prefill": {
                "delta": {"vllm:request_success_total": 1},
                "derived": {"mean_inference_time_seconds": 1.0},
            },
            "pair1-prefill": {
                "delta": {"vllm:request_success_total": 3},
                "derived": {"mean_inference_time_seconds": 3.0},
            },
            "pair0-decoder": {
                "delta": {"vllm:request_success_total": 99},
                "derived": {"mean_inference_time_seconds": 99.0},
            },
        }
        self.assertEqual(
            analyzer._weighted_endpoint_mean(
                cumulative,
                role_suffix="-prefill",
                metric="mean_inference_time_seconds",
            ),
            2.5,
        )
        totals = analyzer._role_delta_totals(
            {
                "pair0-prefill": {"delta": {
                    "vllm:prompt_tokens_total": 100,
                    "vllm:prompt_tokens_cached_total": 99,
                }},
                "pair1-prefill": {"delta": {
                    "vllm:prompt_tokens_total": 200,
                    "vllm:prompt_tokens_cached_total": 198,
                }},
                "pair0-decoder": {"delta": {
                    "vllm:prompt_tokens_total": 999,
                    "vllm:prompt_tokens_cached_total": 0,
                }},
            },
            role_suffix="-prefill",
        )
        self.assertEqual(totals["vllm:prompt_tokens_total"], 300)
        self.assertEqual(totals["vllm:prompt_tokens_cached_total"], 297)

    def test_cassini_invalid_is_missing_not_zero(self):
        def probe(valid, value, reason=None):
            return {"cassini": {
                "valid": valid,
                "invalid_reason": reason,
                "window_ms": 100.0 if valid else None,
                "signals": {
                    "rx_pause_fraction_max": value,
                    "tx_pause_fraction_max": 0,
                    "receive_overflow_fraction_max": 0,
                    "ecn_fraction_max": 0,
                    "resource_nacks": 0,
                    "retries": 0,
                    "timeouts": 0,
                    "host_posted_cycles_per_packet_max": 5,
                    "host_nonposted_cycles_per_packet_max": 0,
                },
            }}

        result = analyzer._cassini_summary({
            "midpoint": {"pair0-decoder": probe(True, 0.25)},
            "after": {"pair0-decoder": probe(
                False, None, "counter_window_stale")},
        })
        self.assertEqual(result["samples_valid"], 1)
        self.assertEqual(result["fraction_max"]["rx_pause_fraction_max"], 0.25)
        self.assertEqual(
            result["invalid_samples"][0]["invalid_reason"],
            "counter_window_stale",
        )
        self.assertTrue(result["invalid_is_missing_not_zero"])


if __name__ == "__main__":
    unittest.main()
