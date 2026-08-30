from __future__ import annotations

import io
import json
import unittest

from eval.sota_4node import run_tempo_pd_stream_metrics_v3 as client


def _event(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


class _Clock:
    def __init__(self):
        self.value = 100

    def __call__(self):
        self.value += 10
        return self.value


class TempoPDStreamMetricsV3Tests(unittest.TestCase):
    def test_multi_token_sse_event_preserves_cardinality_and_timestamp(self) -> None:
        stream = io.BytesIO(b"".join((
            _event({"id": "x", "model": "m", "choices": [{
                "text": "ABC", "finish_reason": None,
                "logprobs": {"tokens": ["A", "B", "C"]},
            }]}),
            _event({"id": "x", "model": "m", "choices": [{
                "text": "D", "finish_reason": "length",
                "logprobs": {"tokens": ["D"]},
            }]}),
            _event({"id": "x", "model": "m", "choices": [], "usage": {
                "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
            }}),
            b"data: [DONE]\n\n",
        )))
        record = client._stream_record(
            stream, dispatch_ns=100, run_start_ns=100, expected_tokens=4,
            route="decoder_local_recompute_or_cache", clock_ns=_Clock(),
        )
        self.assertEqual(record["output_token_values"], ["A", "B", "C", "D"])
        self.assertEqual(record["sse_token_group_sizes"], [3, 1])
        self.assertEqual(record["token_arrival_offsets_ns"][:3], [10, 10, 10])
        self.assertEqual(record["sse_coalesced_token_event_count"], 1)
        self.assertEqual(record["contract_violations"], [])


if __name__ == "__main__":
    unittest.main()
