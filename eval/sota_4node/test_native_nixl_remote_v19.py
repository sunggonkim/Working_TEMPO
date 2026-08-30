from __future__ import annotations

import io
import json
from pathlib import Path
import unittest

from eval.sota_4node import run_tempo_pd_stream_metrics_native_v18 as client


def _event(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


class _Clock:
    def __init__(self):
        self.value = 100

    def __call__(self):
        self.value += 10
        return self.value


class NativeNixlV19Tests(unittest.TestCase):
    def test_native_remote_requires_exact_logprob_stream(self) -> None:
        stream = io.BytesIO(b"".join((
            _event({"id": "decoder", "model": "m", "choices": [{
                "text": "AB", "finish_reason": "length",
                "logprobs": {"tokens": ["A", "B"]},
            }]}),
            _event({"id": "decoder", "model": "m", "choices": [], "usage": {
                "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
            }}),
            b"data: [DONE]\n\n",
        )))
        record = client._stream_record(
            stream, dispatch_ns=100, run_start_ns=100, expected_tokens=2,
            route="remote_prefill_live_kv", clock_ns=_Clock(),
        )
        self.assertEqual(record["contract_violations"], [])
        self.assertEqual(record["output_token_values"], ["A", "B"])
        self.assertNotIn("official_lmcache_proxy_single_prefill_token",
                         record["output_token_proofs"])

    def test_launcher_is_one_bounded_step_and_uses_v19(self) -> None:
        root = Path(__file__).resolve().parent
        launcher = (root / "run_native_nixl_remote_v19_in_allocation.sh").read_text()
        entry = (root / "native_nixl_remote_node_entry_v19.sh").read_text()
        node = (root / "vllm_native_nixl_remote_node_v19.py").read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertNotIn("salloc", launcher)
        self.assertIn("vllm_native_nixl_remote_node_v19", entry)
        self.assertIn("run_tempo_pd_stream_metrics_native_v18", node)


if __name__ == "__main__":
    unittest.main()
