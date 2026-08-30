from __future__ import annotations

import io
import json
import unittest

from eval.sota_4node import run_tempo_pd_stream_metrics_v2 as client


def _event(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


class _Clock:
    def __init__(self):
        self.value = 100

    def __call__(self):
        self.value += 10
        return self.value


class TempoPDStreamMetricsV2Tests(unittest.TestCase):
    def test_length_choice_token_is_counted(self) -> None:
        stream = io.BytesIO(b"".join((
            _event({"id": "x", "model": "m", "choices": [{
                "text": "A", "finish_reason": None,
                "logprobs": {"tokens": ["A"]},
            }]}),
            _event({"id": "x", "model": "m", "choices": [{
                "text": "B", "finish_reason": "length",
                "logprobs": {"tokens": ["B"]},
            }]}),
            _event({"id": "x", "model": "m", "choices": [], "usage": {
                "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
            }}),
            b"data: [DONE]\n\n",
        )))
        record = client._stream_record(
            stream, dispatch_ns=100, run_start_ns=100, expected_tokens=2,
            route="decoder_local_recompute_or_cache", clock_ns=_Clock(),
        )
        self.assertEqual(record["output_token_values"], ["A", "B"])
        self.assertEqual(record["finish_reason"], "length")
        self.assertEqual(record["contract_violations"], [])

    def test_finish_only_choice_does_not_add_or_reject_token(self) -> None:
        stream = io.BytesIO(b"".join((
            _event({"id": "x", "model": "m", "choices": [{
                "text": "A", "finish_reason": None,
                "logprobs": {"tokens": ["A"]},
            }]}),
            _event({"id": "x", "model": "m", "choices": [{
                "text": "", "finish_reason": "length", "logprobs": None,
            }]}),
            _event({"id": "x", "model": "m", "choices": [], "usage": {
                "prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11,
            }}),
            b"data: [DONE]\n\n",
        )))
        record = client._stream_record(
            stream, dispatch_ns=100, run_start_ns=100, expected_tokens=1,
            route="decoder_local_recompute_or_cache", clock_ns=_Clock(),
        )
        self.assertEqual(record["output_token_values"], ["A"])
        self.assertEqual(record["contract_violations"], [])


if __name__ == "__main__":
    unittest.main()
