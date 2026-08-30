from __future__ import annotations

import copy
import unittest

from eval.sota_4node import analyze_tempo_pd_performance_v1 as analyzer
from eval.sota_4node import build_tempo_pd_profile_manifest_v1 as builder
from eval.sota_4node.run_tempo_pd_stream_metrics_v1 import SCHEMA as RAW_SCHEMA
from tempo.pd_admission import PDRequestContext, PDRoute, PDWorkloadClass


WORKLOAD = PDWorkloadClass(
    model_id="qwen", model_revision="sha", topology_id="2x-tp4",
    remote_backend="lmcache-ucx",
    prompt_bucket="c1:prompt_tokens:10",
    output_bucket="c1:output_tokens:2",
    decoder_load_bucket="c1:decoder_load:streams:7",
    kv_bytes_bucket="c1:kv_bytes:1000",
)


def _raw(mode: str, route: str, e2e_ms: float) -> dict[str, object]:
    requests = []
    decisions = []
    reason = (
        "fixed_local_baseline" if mode == "fixed_local"
        else "fixed_official_lmcache_remote_baseline" if mode == "lmcache_always_remote"
        else "remote_benefit_lower_bound_below_margin"
    )
    for index in range(3):
        request_id = f"r{index}"
        dispatch = index * 10_000_000
        first = dispatch + 20_000_000
        last = dispatch + round(e2e_ms * 1_000_000)
        text = f"output-{index}"
        router = {
            "schema": "tempo-live-pd-router-1", "request_id": request_id,
            "mode": mode, "route": route, "reason": reason,
            "workload_fingerprint": WORKLOAD.fingerprint,
            "profile_id": "none", "manifest_id": "none",
        }
        requests.append({
            "request_index": index, "request_id": request_id,
            "prompt_sha256": f"{index}" * 64, "prompt_utf8_bytes": 10,
            "requested_max_tokens": 2, "scheduled_dispatch_offset_ns": dispatch,
            "router": router, "http_status": 200, "dispatch_offset_ns": dispatch,
            "token_arrival_offsets_ns": [first, last],
            "stream_end_offset_ns": last + 1_000_000,
            "output_token_values": ["a", "b"],
            "output_token_proofs": ["vllm_logprobs_exactly_one"] * 2,
            "output_text": text,
            "output_text_sha256": analyzer.__import__("hashlib").sha256(text.encode()).hexdigest()
            if False else __import__("hashlib").sha256(text.encode()).hexdigest(),
            "finish_reason": "length",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            "done_seen": True, "response_ids": ["id"], "response_models": ["served"],
            "contract_violations": [], "error": None, "valid": True,
        })
        decisions.append({
            "request_id": request_id, "mode": mode, "route": route,
            "reason": reason, "workload": WORKLOAD.canonical_dict(),
            "workload_fingerprint": WORKLOAD.fingerprint,
            "profile_id": None, "manifest_id": None, "policy_epoch": None,
            "remote_advantage_lower_bound_ms": None,
            "prompt_tokens": 10, "potential_kv_bytes": 1000,
            "decided_ns": index + 1, "phase": "complete",
            "finished_ns": index + 2, "error": None,
        })
    return {
        "schema": RAW_SCHEMA,
        "evidence": "actual_vllm_pd_router_client_stream",
        "run": {"run_id": mode, "mode": mode, "endpoint": "http://router/v1/completions",
                "started_at_utc": "2026-08-15T00:00:00Z",
                "completed_at_utc": "2026-08-15T00:00:01Z", "client_window_ns": 1},
        "model": {"local_path": "/model", "served_model_name": "served",
                  "config_sha256": "a" * 64},
        "workload": {"schema": "tempo-vllm-stream-workload-jsonl-1",
                     "explicit_path": "/workload", "sha256": "b" * 64,
                     "request_count": 3, "max_workers": 1,
                     "request_rate_per_s": None, "seed": 1},
        "requests": requests,
        "router_decisions": decisions,
        "validation": {"all_streams_valid": True, "router_decisions_exact": True,
                       "performance_claim_allowed": True},
        "metric_contract": {},
    }


class TempoPDPerformanceTests(unittest.TestCase):
    def test_build_manifest_freezes_conservative_bounds(self) -> None:
        local = _raw("fixed_local", PDRoute.DECODER_LOCAL.value, 60.0)
        remote = _raw("lmcache_always_remote", PDRoute.REMOTE_PREFILL.value, 80.0)
        manifest, report = builder.build_manifest(
            local, remote,
            classifier_version="c1", policy_epoch=7,
            minimum_samples_per_route=3, remote_advantage_margin_ms=5.0,
        )
        profile = manifest.profiles[0]
        self.assertEqual(profile.local_latency_lower_bound_ms, 60.0)
        self.assertEqual(profile.remote_latency_upper_bound_ms, 80.0)
        self.assertEqual(report["groups"][0]["remote_advantage_lower_bound_ms"], -20.0)
        decision = manifest.build_policy(allow_screen_profiles=True).decide(
            PDRequestContext("v", WORKLOAD, 7)
        )
        self.assertEqual(decision.route, PDRoute.DECODER_LOCAL)

    def test_three_mode_analysis_reports_goodput_and_paired_win(self) -> None:
        local = _raw("fixed_local", PDRoute.DECODER_LOCAL.value, 60.0)
        remote = _raw("lmcache_always_remote", PDRoute.REMOTE_PREFILL.value, 80.0)
        tempo = _raw("tempo_auto", PDRoute.DECODER_LOCAL.value, 61.0)
        report = analyzer.analyze(
            [("local", local), ("lmcache", remote), ("tempo", tempo)],
            ttft_slo_ms=30.0, tpot_slo_ms=50.0, e2e_slo_ms=70.0,
        )
        self.assertTrue(report["comparison_claim_allowed"])
        comparison = report["comparisons"]["tempo_vs_official_lmcache_remote"]
        self.assertEqual(comparison["e2e_win_count"], 3)
        self.assertAlmostEqual(comparison["e2e_delta_median_ms"], -19.0)
        self.assertGreater(comparison["request_goodput_delta_per_s"], 0.0)
        self.assertTrue(report["route_evidence"]["local_branch_observed"])
        self.assertFalse(report["route_evidence"]["remote_branch_observed"])

    def test_output_mismatch_suppresses_comparison(self) -> None:
        local = _raw("fixed_local", PDRoute.DECODER_LOCAL.value, 60.0)
        remote = _raw("lmcache_always_remote", PDRoute.REMOTE_PREFILL.value, 80.0)
        tempo = _raw("tempo_auto", PDRoute.DECODER_LOCAL.value, 61.0)
        tempo = copy.deepcopy(tempo)
        tempo["requests"][0]["output_text_sha256"] = "0" * 64
        report = analyzer.analyze(
            [("local", local), ("remote", remote), ("tempo", tempo)],
            ttft_slo_ms=30.0, tpot_slo_ms=50.0, e2e_slo_ms=70.0,
        )
        self.assertFalse(report["comparison_claim_allowed"])
        self.assertIsNone(report["comparisons"]["tempo_vs_official_lmcache_remote"])


if __name__ == "__main__":
    unittest.main()
