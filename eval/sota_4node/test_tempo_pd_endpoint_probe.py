from __future__ import annotations

import unittest
from urllib.error import URLError
from unittest import mock

from eval.sota_4node import tempo_pd_endpoint_probe as probe
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from tempo.pd_endpoint_evidence import PDEndpointIdentity, PDEndpointRole


_METRICS = b"""# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="tempo-qwen25-7b-pd-perf",engine="0"} 3
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="tempo-qwen25-7b-pd-perf",engine="0"} 2
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="tempo-qwen25-7b-pd-perf",engine="0"} 0.25
"""


def _cumulative() -> dict[str, object]:
    values = {}
    for name in probe.VLLM_CUMULATIVE_METRICS:
        values[name] = 0 if name.endswith(("_total", "_count")) else 0.0
    return {
        "schema": probe.VLLM_CUMULATIVE_SCHEMA,
        "source": "vllm_prometheus_on_demand",
        "model_name": "tempo-qwen25-7b-pd-perf",
        "engine_indices": [0],
        "values": values,
    }


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return _METRICS


class EndpointProbeTest(unittest.TestCase):
    def test_cumulative_parser_accepts_real_prometheus_counter_names(self) -> None:
        registry = CollectorRegistry()
        labels = ["model_name", "engine"]
        label_values = ("tempo-qwen25-7b-pd-perf", "0")
        for name in sorted(
            item for item in probe.VLLM_CUMULATIVE_METRICS
            if item.endswith("_total")
        ):
            metric = Counter(
                name.removesuffix("_total"), name, labels, registry=registry)
            metric.labels(*label_values).inc(3)
        prefixes = sorted({
            name.removesuffix("_sum")
            for name in probe.VLLM_CUMULATIVE_METRICS
            if name.endswith("_sum")
        })
        for prefix in prefixes:
            metric = Histogram(prefix, prefix, labels, registry=registry)
            metric.labels(*label_values).observe(0.25)
        parsed = probe.parse_vllm_endpoint_cumulative(
            generate_latest(registry).decode("utf-8"),
            served_model_name="tempo-qwen25-7b-pd-perf",
        )
        probe.validate_vllm_endpoint_cumulative(parsed)
        self.assertEqual(parsed["values"]["vllm:prompt_tokens_total"], 3)
        self.assertEqual(
            parsed["values"]["vllm:request_prefill_time_seconds_count"], 1)

    def test_snapshot_keeps_role_inventory_and_local_clock_owner(self) -> None:
        cassini = mock.Mock()
        cassini.sample.return_value = {
            "endpoint_id": "pair0-decoder",
            "role": "decoder",
            "pair_index": 0,
            "valid": True,
            "sequence": 1,
        }
        with (
            mock.patch.object(probe, "CassiniEndpointSampler",
                              return_value=cassini),
            mock.patch.object(probe, "urlopen", return_value=_Response()),
            mock.patch.object(probe, "parse_vllm_endpoint_cumulative",
                              return_value=_cumulative()),
        ):
            state = probe.EndpointProbe(
                identity=PDEndpointIdentity(
                    "pair0-decoder", PDEndpointRole.DECODER, 0),
                vllm_metrics_url="http://decoder:14000",
                served_model_name="tempo-qwen25-7b-pd-perf",
                metrics_timeout_s=3.0,
            )
            result = state.snapshot()

        self.assertEqual(result["schema"], probe.SCHEMA)
        probe.validate_vllm_endpoint_cumulative(result["vllm_cumulative"])
        endpoint = result["endpoint"]
        self.assertEqual(endpoint["endpoint_id"], "pair0-decoder")
        self.assertEqual(endpoint["source"], "vllm_prometheus_on_demand")
        self.assertEqual(endpoint["metrics"]["running_requests"], {
            "support": "supported", "value": 3,
        })
        self.assertEqual(endpoint["metrics"]["waiting_requests"]["value"], 2)
        self.assertEqual(
            endpoint["metrics"]["active_decode_tokens"]["support"],
            "not_collected",
        )
        cassini.sample.assert_called_once_with(force=True)

    def test_metrics_retry_is_local_bounded_and_cassini_samples_once(self) -> None:
        cassini = mock.Mock()
        cassini.sample.return_value = {
            "endpoint_id": "pair0-decoder",
            "role": "decoder",
            "pair_index": 0,
            "valid": True,
            "sequence": 1,
        }
        with (
            mock.patch.object(probe, "CassiniEndpointSampler",
                              return_value=cassini),
            mock.patch.object(
                probe, "urlopen",
                side_effect=[URLError("transient busy"), _Response()],
            ) as fetch,
            mock.patch.object(probe.time, "sleep") as sleep,
            mock.patch.object(probe, "parse_vllm_endpoint_cumulative",
                              return_value=_cumulative()),
        ):
            state = probe.EndpointProbe(
                identity=PDEndpointIdentity(
                    "pair0-decoder", PDEndpointRole.DECODER, 0),
                vllm_metrics_url="http://decoder:14000",
                served_model_name="tempo-qwen25-7b-pd-perf",
                metrics_timeout_s=1.0,
                metrics_attempts=2,
            )
            result = state.snapshot()

        self.assertEqual(fetch.call_count, 2)
        sleep.assert_called_once_with(0.05)
        cassini.sample.assert_called_once_with(force=True)
        self.assertEqual(result["vllm_metrics_fetch"]["attempts_used"], 2)
        self.assertEqual(
            result["vllm_metrics_fetch"]["transient_errors"][0]["error"],
            "URLError",
        )

    def test_cumulative_validator_rejects_missing_metric(self) -> None:
        value = _cumulative()
        value["values"].pop("vllm:prompt_tokens_total")
        with self.assertRaisesRegex(ValueError, "inventory"):
            probe.validate_vllm_endpoint_cumulative(value)


if __name__ == "__main__":
    unittest.main()
