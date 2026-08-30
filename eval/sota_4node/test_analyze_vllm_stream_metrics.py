from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node import analyze_vllm_stream_metrics as analyzer
from eval.sota_4node.run_vllm_stream_metrics import SCHEMA as RAW_SCHEMA


def _record(
    request_id: str,
    dispatch_ns: int,
    arrivals_ns: list[int],
    tokens: list[str],
) -> dict[str, object]:
    text = "".join(tokens)
    prompt = f"prompt-{request_id}"
    return {
        "request_index": int(request_id[1:]),
        "request_id": request_id,
        "prompt_sha256": analyzer._digest_bytes(prompt.encode("utf-8")),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "requested_max_tokens": len(tokens),
        "scheduled_dispatch_offset_ns": dispatch_ns,
        "http_status": 200,
        "dispatch_offset_ns": dispatch_ns,
        "token_arrival_offsets_ns": arrivals_ns,
        "stream_end_offset_ns": arrivals_ns[-1] + 5_000_000,
        "output_tokens": tokens,
        "output_token_sha256": analyzer._digest_json(tokens),
        "output_text": text,
        "output_text_sha256": analyzer._digest_bytes(text.encode("utf-8")),
        "finish_reason": "length",
        "usage": {"prompt_tokens": 4, "completion_tokens": len(tokens),
                  "total_tokens": 4 + len(tokens)},
        "done_seen": True,
        "response_id": f"completion-{request_id}",
        "response_model": "local-model",
        "contract_violations": [],
        "error": None,
        "valid": True,
    }


def _artifact(mode: str = "fg_only") -> dict[str, object]:
    records = [
        _record("r0", 0, [100_000_000, 120_000_000, 140_000_000], ["a", "b", "c"]),
        _record("r1", 10_000_000, [60_000_000, 90_000_000, 110_000_000], ["x", "y", "z"]),
    ]
    return {
        "schema_version": RAW_SCHEMA,
        "evidence_state": "native_vllm_client_stream",
        "run": {"run_id": "fixture", "mode": mode, "endpoint": "http://localhost:8000/v1/completions",
                "started_at_utc": "2026-08-14T00:00:00+00:00",
                "completed_at_utc": "2026-08-14T00:00:01+00:00", "client_window_ns": 150_000_000},
        "model": {"source": "explicit_local_directory", "local_path": "/workspace/model",
                  "served_model_name": "local-model", "config_sha256": "a" * 64,
                  "offline_server_assumption": True,
                  "offline_server_assumption_verified_by_client": False},
        "endpoint_contract": {"api": "OpenAI-compatible POST /v1/completions",
                              "stream": True, "logprobs": 1,
                              "stream_options_include_usage": True, "ignore_eos": True,
                              "retry_count": 0, "api_key_present": False},
        "clock": {"name": "time.perf_counter_ns", "scope": "single client process",
                  "timestamp_semantics": "complete SSE data event observed by client"},
        "workload": {"schema_version": "tempo-vllm-stream-workload-jsonl-1",
                     "sha256": "b" * 64, "request_count": 2, "max_workers": 2,
                     "request_rate_per_s": None, "default_max_tokens": 3, "seed": 1},
        "requests": records,
        "validation": {"all_requests_valid": True, "valid_requests": 2,
                       "invalid_requests": 0, "performance_claim_allowed": True},
        "limitations": [],
    }


def _write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class AnalyzeVllmStreamMetricsTest(unittest.TestCase):
    def test_exact_metrics_and_full_window_goodput(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write(Path(temporary) / "raw.json", _artifact())
            report = analyzer.analyze(
                [("fg", path)], ttft_slo_ms=100.0, tpot_slo_ms=25.0,
                itl_slo_ms=30.0, e2e_slo_ms=140.0,
            )
        metric = report["runs"][0]["performance"]
        self.assertAlmostEqual(metric["measurement_window_s"], 0.14)
        self.assertAlmostEqual(metric["output_token_throughput_per_s"], 6 / 0.14)
        self.assertAlmostEqual(metric["slo_goodput"]["request_goodput_per_s"], 2 / 0.14)
        self.assertEqual(metric["request_metrics"][0]["itl_ms"], [20.0, 20.0])
        self.assertEqual(metric["request_metrics"][1]["tpot_ms"], 25.0)
        self.assertIsNone(report["correctness"]["cross_run_output_equivalence"])

    def test_two_modes_require_identical_outputs(self):
        first = _artifact("fg_only")
        second = copy.deepcopy(_artifact("tempo"))
        second_record = second["requests"][0]
        second_record["output_tokens"] = ["q", "b", "c"]
        second_record["output_text"] = "qbc"
        second_record["output_token_sha256"] = analyzer._digest_json(second_record["output_tokens"])
        second_record["output_text_sha256"] = analyzer._digest_bytes(b"qbc")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = _write(root / "first.json", first)
            second_path = _write(root / "second.json", second)
            report = analyzer.analyze(
                [("fg", first_path), ("tempo", second_path)],
                ttft_slo_ms=100.0, tpot_slo_ms=25.0,
            )
        self.assertFalse(report["correctness"]["cross_run_output_equivalence"])
        self.assertFalse(report["correctness"]["correctness_met"])
        self.assertFalse(report["comparison_claim_allowed"])

    def test_invalid_request_suppresses_performance_aggregate(self):
        value = _artifact()
        value["requests"][0]["contract_violations"] = ["done_event_missing"]
        value["requests"][0]["valid"] = False
        value["validation"] = {"all_requests_valid": False, "valid_requests": 1,
                               "invalid_requests": 1, "performance_claim_allowed": False}
        with tempfile.TemporaryDirectory() as temporary:
            path = _write(Path(temporary) / "raw.json", value)
            report = analyzer.analyze(
                [("bad", path)], ttft_slo_ms=100.0, tpot_slo_ms=25.0,
            )
        self.assertFalse(report["runs"][0]["evidence_valid"])
        self.assertIsNone(report["runs"][0]["performance"])

    def test_tampered_output_digest_is_rejected(self):
        value = _artifact()
        value["requests"][0]["output_text_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = _write(Path(temporary) / "raw.json", value)
            with self.assertRaises(analyzer.AnalysisError):
                analyzer.analyze([( "bad", path)], ttft_slo_ms=100.0, tpot_slo_ms=25.0)


if __name__ == "__main__":
    unittest.main()
