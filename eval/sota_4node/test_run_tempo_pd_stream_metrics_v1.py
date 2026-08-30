from __future__ import annotations

import io
import json
import unittest

from eval.sota_4node import run_tempo_pd_stream_metrics_v1 as client
from eval.sota_4node import run_vllm_stream_metrics as base


def _event(value: dict[str, object]) -> bytes:
    return b"data: " + json.dumps(value).encode() + b"\n\n"


class _Headers(dict):
    def get(self, key: str, default=None):
        for name, value in self.items():
            if name.lower() == key.lower():
                return value
        return default


class _Response(io.BytesIO):
    def __init__(self, value: bytes, route: str):
        super().__init__(value)
        self.headers = _Headers({
            "X-Tempo-PD-Schema": client.ROUTER_SCHEMA,
            "X-Tempo-PD-Request-Id": "r0",
            "X-Tempo-PD-Mode": "tempo_auto",
            "X-Tempo-PD-Route": route,
            "X-Tempo-PD-Reason": "test",
            "X-Tempo-PD-Workload": "f" * 64,
            "X-Tempo-PD-Profile": "none",
            "X-Tempo-PD-Manifest": "none",
        })

    def getcode(self) -> int:
        return 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class _Clock:
    def __init__(self, *values: int):
        self.values = iter(values)

    def __call__(self) -> int:
        return next(self.values)


def _stream(remote: bool) -> bytes:
    first = {
        "id": "prefill" if remote else "decode",
        "model": "served",
        "choices": [{
            "index": 0, "text": "A", "finish_reason": None,
            "logprobs": None if remote else {"tokens": ["A"]},
        }],
    }
    second = {
        "id": "decode", "model": "served",
        "choices": [{"index": 0, "text": "B", "finish_reason": None,
                     "logprobs": {"tokens": ["B"]}}],
    }
    finish = {"id": "decode", "model": "served",
              "choices": [{"index": 0, "text": "", "finish_reason": "length"}]}
    completion = 1 if remote else 2
    usage = {"id": "decode", "model": "served", "choices": [],
             "usage": {"prompt_tokens": 10, "completion_tokens": completion,
                       "total_tokens": 10 + completion}}
    return b"".join(_event(value) for value in (first, second, finish, usage)) + b"data: [DONE]\n\n"


class TempoPDStreamMetricsTests(unittest.TestCase):
    def test_http_error_classification_separates_reject_and_sensor_timeout(self):
        queue = client.error.HTTPError(
            "http://router/v1/completions", 503, "Service Unavailable", {},
            io.BytesIO(b'{"detail":"global admission queue timed out"}'),
        )
        telemetry = client.error.HTTPError(
            "http://router/v1/completions", 503, "Service Unavailable", {},
            io.BytesIO(b'{"detail":"global telemetry refresh timed out"}'),
        )
        self.assertEqual(
            client._classify_http_error(queue),
            "global_admission_queue_timeout",
        )
        self.assertEqual(
            client._classify_http_error(telemetry),
            "global_telemetry_refresh_timeout",
        )

        structured_reject = client.error.HTTPError(
            "http://router/v1/completions", 503, "Service Unavailable", {},
            io.BytesIO(
                b'{"detail":{"code":"tempo_go_global_reject",'
                b'"reason":"global_admission_queue_timeout"}}'
            ),
        )
        failed_refresh = client.error.HTTPError(
            "http://router/v1/completions", 503, "Service Unavailable", {},
            io.BytesIO(b'{"detail":"global telemetry refresh failed"}'),
        )
        self.assertEqual(
            client._classify_http_error(structured_reject),
            "global_admission_queue_timeout",
        )
        self.assertEqual(
            client._classify_http_error(failed_refresh),
            "global_telemetry_refresh_failed",
        )

        service_lane = client.error.HTTPError(
            "http://router/v1/completions", 503, "Service Unavailable", {},
            io.BytesIO(
                b'{"detail":{"code":"tempo_go_service_lane_reservation_timeout",'
                b'"reason":"endpoint_bounded_queue_lease_timeout"}}'
            ),
        )
        self.assertEqual(
            client._classify_http_error(service_lane),
            "endpoint_bounded_queue_lease_timeout",
        )

    def _execute(self, remote: bool):
        route = "remote_prefill_live_kv" if remote else "decoder_local_recompute_or_cache"
        return client.execute_request(
            base.WorkItem(0, "r0", "prompt", 2, 0),
            endpoint="http://router/v1/completions",
            served_model_name="served",
            run_start_ns=1_000,
            timeout_s=5.0,
            seed=1,
            api_key=None,
            opener=lambda *_args, **_kwargs: _Response(_stream(remote), route),
            clock_ns=_Clock(1_000, 1_000, 1_010, 1_030, 1_040, 1_050, 1_060, 1_070),
        )

    def test_local_requires_logprob_identity(self) -> None:
        record = self._execute(False)
        self.assertTrue(record["valid"])
        self.assertEqual(record["output_token_proofs"],
                         ["vllm_logprobs_exactly_one"] * 2)

    def test_remote_first_token_has_explicit_proxy_proof(self) -> None:
        record = self._execute(True)
        self.assertTrue(record["valid"])
        self.assertEqual(record["output_text"], "AB")
        self.assertEqual(record["output_token_proofs"][0],
                         "official_lmcache_proxy_single_prefill_token")
        self.assertEqual(record["token_arrival_offsets_ns"], [10, 30])

    def test_missing_proxy_proof_fails_closed(self) -> None:
        response = _Response(_stream(False), "remote_prefill_live_kv")
        record = client.execute_request(
            base.WorkItem(0, "r0", "prompt", 2, 0),
            endpoint="http://router/v1/completions", served_model_name="served",
            run_start_ns=1_000, timeout_s=5.0, seed=1, api_key=None,
            opener=lambda *_args, **_kwargs: response,
            clock_ns=_Clock(1_000, 1_000, 1_010, 1_030, 1_040, 1_050, 1_060, 1_070),
        )
        self.assertFalse(record["valid"])
        self.assertIn("proxy_first_token_proof_missing", record["contract_violations"])

    def test_503_is_valid_only_with_explicit_global_reject_receipt(self) -> None:
        record = {
            "request_id": "r0",
            "valid": True,
            "terminal_reject_candidate": True,
            "contract_violations": [],
            "error": None,
        }
        self.assertTrue(client._apply_decision_receipts(
            [record],
            [{
                "request_id": "r0",
                "phase": "rejected",
                "error": None,
                "tempo_go_rejected": True,
                "global_decision_kind": "reject",
                "global_decision_reason": "global_admission_queue_timeout",
            }],
        ))
        self.assertTrue(record["valid"])
        self.assertEqual(record["terminal_kind"], "global_reject")

        unreceipted = dict(record)
        unreceipted.update({
            "terminal_reject_candidate": True,
            "valid": True,
            "contract_violations": [],
            "error": None,
        })
        self.assertFalse(client._apply_decision_receipts(
            [unreceipted], []))
        self.assertFalse(unreceipted["valid"])
        self.assertEqual(
            unreceipted["contract_violations"],
            ["unreceipted_terminal_reject"],
        )

    def test_telemetry_503_is_valid_only_with_explicit_global_reject_receipt(self) -> None:
        record = {
            "request_id": "telemetry-r0",
            "valid": True,
            "terminal_reject_candidate": True,
            "contract_violations": [],
            "error": None,
            "terminal_error_kind": "global_telemetry_refresh_failed",
        }
        self.assertTrue(client._apply_decision_receipts(
            [record],
            [{
                "request_id": "telemetry-r0",
                "phase": "rejected",
                "error": None,
                "tempo_go_rejected": True,
                "global_decision_kind": "reject",
                "global_decision_reason": "global_telemetry_refresh_failed",
            }],
        ))
        self.assertTrue(record["valid"])
        self.assertEqual(record["terminal_kind"], "global_reject")

    def test_service_lane_503_is_valid_only_with_explicit_failure_receipt(self) -> None:
        record = {
            "request_id": "service-lane-r0",
            "valid": True,
            "terminal_reject_candidate": False,
            "terminal_service_lane_failure_candidate": True,
            "contract_violations": [],
            "error": None,
            "terminal_error_kind": "endpoint_bounded_queue_lease_timeout",
        }
        self.assertTrue(client._apply_decision_receipts(
            [record],
            [{
                "request_id": "service-lane-r0",
                "phase": "failed",
                "error": "endpoint_bounded_queue_lease_timeout",
                "frontend_tempo_go_failure_scope": "service_lane",
                "frontend_tempo_go_failure_kind": (
                    "endpoint_bounded_queue_lease_timeout"),
                "frontend_tempo_go_reservation_failure": {
                    "failure_kind": "endpoint_bounded_queue_lease_timeout",
                    "schema": "tempo-go-service-lane-reservation-v1",
                },
            }],
        ))
        self.assertTrue(record["valid"])
        self.assertEqual(record["terminal_kind"], "service_lane_failure")

        unreceipted = dict(record)
        unreceipted.update({
            "terminal_service_lane_failure_candidate": True,
            "valid": True,
            "contract_violations": [],
            "error": None,
        })
        self.assertFalse(client._apply_decision_receipts([unreceipted], []))
        self.assertFalse(unreceipted["valid"])
        self.assertEqual(
            unreceipted["contract_violations"],
            ["unreceipted_terminal_service_lane_failure"],
        )

    def test_route_5xx_is_valid_only_with_global_failure_receipt(self) -> None:
        record = {
            "request_id": "route-failure-r0",
            "valid": True,
            "terminal_route_failure_candidate": True,
            "contract_violations": [],
            "error": None,
            "terminal_error_kind": "http_error",
        }
        decision = {
            "request_id": "route-failure-r0",
            "phase": "failed",
            "error": "upstream_http_status_502",
            "frontend_tempo_go_failure_scope": "route",
            "frontend_tempo_go_failure_kind": "upstream_http_status_502",
            "frontend_tempo_go_failure": {
                "schema": "tempo-go-global-failure-v1",
                "request_id": "route-failure-r0",
                "failure_kind": "upstream_http_status_502",
                "terminal_phase": "failed",
            },
        }
        self.assertTrue(client._apply_decision_receipts([record], [decision]))
        self.assertTrue(record["valid"])
        self.assertEqual(record["terminal_kind"], "route_failure")

        unreceipted = dict(record)
        unreceipted.update({
            "terminal_route_failure_candidate": True,
            "valid": True,
            "contract_violations": [],
            "error": None,
        })
        self.assertFalse(client._apply_decision_receipts([unreceipted], []))
        self.assertFalse(unreceipted["valid"])
        self.assertEqual(
            unreceipted["contract_violations"],
            ["unreceipted_terminal_route_failure"],
        )


if __name__ == "__main__":
    unittest.main()
