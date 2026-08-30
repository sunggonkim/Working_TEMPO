from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval.sota_4node import run_vllm_stream_metrics as client
from eval.sota_4node import vllm_stream_metrics_api as sidecar


def _event(value: dict[str, object]) -> bytes:
    return b"data: " + json.dumps(value).encode("utf-8") + b"\n\n"


class _Response(io.BytesIO):
    def getcode(self) -> int:
        return 200

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class _Clock:
    def __init__(self, *values: int):
        self.values = iter(values)

    def __call__(self) -> int:
        return next(self.values)


def _stream(*, multiple_tokens: bool = False) -> bytes:
    first_tokens = ["A", "B"] if multiple_tokens else ["A"]
    values = [
        {"id": "cmpl-1", "model": "local", "choices": [{
            "index": 0, "text": "".join(first_tokens), "finish_reason": None,
            "logprobs": {"tokens": first_tokens},
        }]},
    ]
    if not multiple_tokens:
        values.append({"id": "cmpl-1", "model": "local", "choices": [{
            "index": 0, "text": "B", "finish_reason": None,
            "logprobs": {"tokens": ["B"]},
        }]})
    values.extend([
        {"id": "cmpl-1", "model": "local", "choices": [{
            "index": 0, "text": "", "finish_reason": "length", "logprobs": None,
        }]},
        {"id": "cmpl-1", "model": "local", "choices": [],
         "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
    ])
    return b"".join(_event(value) for value in values) + b"data: [DONE]\n\n"


class VllmStreamClientTest(unittest.TestCase):
    def test_exact_stream_records_one_timestamp_per_token(self):
        captured: dict[str, object] = {}

        def opener(req: object, *, timeout: float) -> _Response:
            captured["body"] = json.loads(req.data)
            captured["timeout"] = timeout
            return _Response(_stream())

        record = client.execute_request(
            client.WorkItem(0, "r0", "hello", 2, 0),
            endpoint="http://127.0.0.1:8000/v1/completions",
            served_model_name="local",
            run_start_ns=1_000,
            timeout_s=5.0,
            seed=7,
            api_key=None,
            opener=opener,
            clock_ns=_Clock(1_000, 1_000, 1_010, 1_030, 1_040, 1_050, 1_060, 1_070),
        )
        self.assertTrue(record["valid"])
        self.assertEqual(record["token_arrival_offsets_ns"], [10, 30])
        self.assertEqual(record["output_tokens"], ["A", "B"])
        self.assertEqual(captured["body"]["ignore_eos"], True)
        self.assertEqual(captured["body"]["logprobs"], 1)
        self.assertEqual(captured["body"]["stream_options"], {"include_usage": True})

    def test_multiple_tokens_in_one_event_fails_closed(self):
        record = client.execute_request(
            client.WorkItem(0, "r0", "hello", 2, 0),
            endpoint="http://127.0.0.1:8000/v1/completions",
            served_model_name="local",
            run_start_ns=1_000,
            timeout_s=5.0,
            seed=7,
            api_key=None,
            opener=lambda *_args, **_kwargs: _Response(_stream(multiple_tokens=True)),
            clock_ns=_Clock(1_000, 1_000, 1_010, 1_020, 1_030, 1_040, 1_050),
        )
        self.assertFalse(record["valid"])
        self.assertIn("multiple_tokens_in_one_sse_event", record["contract_violations"])

    def test_workload_rejects_unknown_payload_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "workload.jsonl"
            path.write_text(json.dumps({"request_id": "r", "prompt": "p", "extra_body": {}}))
            with self.assertRaises(client.ContractError):
                client.load_workload(path, default_max_tokens=2, request_rate=None)

    def test_importable_sidecar_returns_artifact(self):
        record = {
            "request_index": 0, "request_id": "r", "prompt_sha256": "0" * 64,
            "prompt_utf8_bytes": 1, "requested_max_tokens": 2,
            "scheduled_dispatch_offset_ns": 0, "valid": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary).resolve()
            (model / "config.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(client, "run_workload", return_value=(10, 50, [record])) as call:
                artifact = sidecar.run_workload(
                    "http://127.0.0.1:8000", model,
                    [{"request_id": "r", "prompt": "p", "max_tokens": 2}],
                    mode="fg_only", max_workers=4,
                )
        self.assertTrue(artifact["validation"]["all_requests_valid"])
        self.assertEqual(artifact["run"]["client_window_ns"], 40)
        self.assertEqual(call.call_args.kwargs["max_workers"], 4)
        self.assertEqual(call.call_args.args[0][0].max_tokens, 2)


if __name__ == "__main__":
    unittest.main()
