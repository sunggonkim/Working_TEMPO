from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from eval.sota_4node import run_tempo_pd_c4_fixed_phase_client as c4
from tempo.pd_cache_state_protocol import build_cache_preparation_plan
from tempo.pd_contention_workload import CacheState, ForegroundArm, Tenant


class _Tokenizer:
    def __init__(self):
        self.values = {}
        self.unknown = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in self.values:
            return list(self.values[text])
        if text not in self.unknown:
            base = 50000 + 2 * len(self.unknown)
            self.unknown[text] = (base, base + 1)
        return list(self.unknown[text])

    def decode(
        self, token_ids, *, skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        value = "tokens:" + ",".join(str(item) for item in token_ids)
        self.values[value] = tuple(token_ids)
        return value


def _manifest():
    return {
        "background_rates_per_s": {
            "decoder_hot": 1.0,
            "cold_remote_hot": 1.0,
            "kv_remote_hot": 1.0,
        },
        "foreground_rate_per_s": 1.0,
        "phase_duration_ms": 1000.0,
    }


def _endpoint_sample(stage: str, sequence: int) -> dict[str, object]:
    identities = (
        "pair0-prefill", "pair0-decoder",
        "pair1-prefill", "pair1-decoder",
    )
    return {
        "schema": c4.fixed.ENDPOINT_EVIDENCE_SCHEMA,
        "stage": stage,
        "snapshots": [{
            "client_received_monotonic_ns": 1_000_000 + sequence,
            "probe": {
                "endpoint": {
                    "endpoint_id": endpoint_id,
                    "sequence": sequence,
                },
                "cassini": {
                    "sequence": sequence,
                    "valid": True,
                },
            },
        } for endpoint_id in identities],
    }


class C4FixedPhaseClientTest(unittest.TestCase):
    def test_artifact_binding_detects_later_block_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "block.raw.json"
            path.write_text('{"value": 1}\n', encoding="utf-8")
            binding = c4._artifact_binding(path)
            self.assertEqual(
                binding,
                {"path": str(path.resolve()), "sha256": c4._sha256(path)},
            )
            path.write_text('{"value": 2}\n', encoding="utf-8")
            self.assertNotEqual(binding["sha256"], c4._sha256(path))

    def test_invalid_start_marker_terminates_measured_child(self):
        class Child:
            def __init__(self):
                self.pid = 123
                self.running = True
                self.terminated = False

            def poll(self):
                return None if self.running else -15

            def terminate(self):
                self.terminated = True
                self.running = False

            def wait(self, timeout):
                del timeout
                return -15

            def kill(self):
                self.running = False

        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "measurement-start.json"
            child = Child()

            def launch(_command, *, env):
                self.assertEqual(
                    env[c4.protocol_client.START_MARKER_ENV], str(marker))
                marker.write_text("{}\n", encoding="utf-8")
                return child

            with (
                patch.object(c4.subprocess, "Popen", side_effect=launch),
                patch.object(
                    c4, "_capture_c4_endpoint_evidence",
                    return_value=_endpoint_sample("before_process_start", 0),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "marker is invalid"):
                    c4._run_with_endpoint_evidence(
                        ["child"],
                        args=SimpleNamespace(
                            endpoint_evidence_url=["a", "b", "c", "d"],
                            timeout_s=1.0,
                            phase_duration_ms=1000.0,
                        ),
                        env={},
                        start_marker=marker.resolve(),
                        first_arrival_offset_ms=250.0,
                    )
            self.assertTrue(child.terminated)

    def test_c4_stage_names_are_adapted_to_shared_capture_contract(self):
        calls = []

        def capture(_urls, *, stage, require_valid_delta):
            calls.append((stage, require_valid_delta))
            return _endpoint_sample(stage, len(calls))

        cases = (
            ("before_process_start", "before", False),
            ("measurement_start", "before", True),
            ("phase_midpoint", "midpoint", True),
            ("phase_boundary", "after", True),
        )
        with patch.object(
            c4.fixed, "_capture_endpoint_evidence", side_effect=capture,
        ):
            for c4_stage, shared_stage, require_valid in cases:
                sample = c4._capture_c4_endpoint_evidence(
                    ["a", "b", "c", "d"],
                    stage=c4_stage,
                    require_valid_delta=require_valid,
                )
                self.assertEqual(sample["stage"], c4_stage)
                self.assertEqual(calls[-1], (shared_stage, require_valid))

    def test_paired_gate_requires_complete_phase_geometry_cells(self):
        tokenizer = _Tokenizer()
        templates = {
            length: tuple(range(length)) for length in (512, 2048, 4094)
        }
        manifest = {
            "background_rates_per_s": {
                "decoder_hot": 1.0,
                "cold_remote_hot": 1.0,
                "kv_remote_hot": 1.0,
            },
            "foreground_rate_per_s": 2.0,
            "phase_duration_ms": 6000.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocks = []
            for sequence, (arm, replicate) in enumerate(c4.BLOCK_ORDER):
                block = c4._materialize_block(
                    sequence=sequence, arm=arm, replicate=replicate,
                    manifest=manifest,
                    factory=c4._PromptFactory(tokenizer, templates),
                )
                requests = []
                for request_id, metadata in block["request_index"].items():
                    dispatch = int(metadata["arrival_offset_ms"] * 1_000_000)
                    route_penalty = 20_000_000 if arm.value == "local" else 30_000_000
                    output_tokens = int(metadata["output_tokens"])
                    arrivals = [
                        dispatch + route_penalty + index * 1_000_000
                        for index in range(output_tokens)
                    ]
                    requests.append({
                        "request_id": request_id,
                        "valid": True,
                        "requested_max_tokens": output_tokens,
                        "dispatch_offset_ns": dispatch,
                        "token_arrival_offsets_ns": arrivals,
                        "stream_end_offset_ns": arrivals[-1] + 1_000_000,
                        "output_text_sha256": "a" * 64,
                    })
                raw_path = root / f"block-{sequence}.json"
                raw_path.write_text(json.dumps({
                    "requests": requests,
                    "c4_fixed_phase_contract": {
                        "request_index": block["request_index"],
                        "all_requests_valid": True,
                        "completion_cache_evidence_exact": True,
                        "workload_start_marker_exact": True,
                        "phase_aligned_endpoint_evidence": True,
                    },
                }), encoding="utf-8")
                blocks.append({**block, "raw_path": str(raw_path)})
            gate = c4._paired_gate(blocks)
            self.assertTrue(gate["phase_geometry_cells_complete"])
            self.assertTrue(gate["phase_aligned_endpoint_evidence"])
            self.assertEqual(len(gate["phase_service_rows"]), 36)
            self.assertEqual(len(gate["phase_route_summaries"]), 6)
            self.assertEqual(gate["paired_output_count"], 144)
            self.assertTrue(all(
                row["paired_samples"] == 4
                for row in gate["phase_service_rows"]
            ))

            changed = json.loads(
                Path(blocks[0]["raw_path"]).read_text(encoding="utf-8"))
            changed["c4_fixed_phase_contract"][
                "phase_aligned_endpoint_evidence"] = False
            Path(blocks[0]["raw_path"]).write_text(
                json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "phase-aligned"):
                c4._paired_gate(blocks)

    def test_phase_endpoint_evidence_is_workload_aligned_and_nonmixing(self):
        with tempfile.TemporaryDirectory() as directory:
            marker_path = Path(directory) / "measurement-start.json"
            marker_value = {
                "schema": c4.protocol_client.START_MARKER_SCHEMA,
                "clock": "client time.perf_counter_ns",
                "run_start_ns": 1000,
                "publisher_pid": 123,
            }
            marker_path.write_text(
                json.dumps(marker_value, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            boundaries = []
            start = _endpoint_sample("measurement_start", 1)
            start.update({
                "boundary_index": 0,
                "completed_phase": None,
                "begins_phase": c4.manifest_builder.PHASES[0].value,
            })
            boundaries.append(start)
            midpoints = []
            for index, phase in enumerate(c4.manifest_builder.PHASES):
                midpoint = _endpoint_sample("phase_midpoint", 2 + 2 * index)
                midpoint.update({"phase_index": index, "phase": phase.value})
                midpoints.append(midpoint)
                boundary = _endpoint_sample(
                    "phase_boundary", 3 + 2 * index)
                boundary.update({
                    "boundary_index": index + 1,
                    "completed_phase": phase.value,
                    "begins_phase": (
                        c4.manifest_builder.PHASES[index + 1].value
                        if index + 1 < len(c4.manifest_builder.PHASES)
                        else None
                    ),
                })
                boundaries.append(boundary)
            evidence = {
                "schema": c4.ENDPOINT_EVIDENCE_SCHEMA,
                "sampling_policy": c4.ENDPOINT_SAMPLING_POLICY,
                "cross_endpoint_clock_subtraction_allowed": False,
                "measurement_clock_alignment": (
                    "same_frontend_host_child_time_perf_counter_ns_marker"),
                "before_process_start": _endpoint_sample(
                    "before_process_start", 0),
                "measurement_start_marker": {
                    **marker_value,
                    "path": str(marker_path),
                    "sha256": c4._sha256(marker_path),
                    "parent_observed_child_pid": 123,
                    "parent_observed_offset_ns": 10,
                },
                "first_arrival_offset_ns": 250_000_000,
                "measurement_start_capture_completed_offset_ns": 20_000_000,
                "phase_boundaries": boundaries,
                "phase_midpoints": midpoints,
            }
            c4._validate_c4_endpoint_evidence(evidence)
            evidence["phase_midpoints"][1]["phase"] = "c0_cool"
            with self.assertRaisesRegex(ValueError, "midpoint"):
                c4._validate_c4_endpoint_evidence(evidence)

    def test_paired_blocks_share_semantics_and_prompts_with_arm_isolation(self):
        tokenizer = _Tokenizer()
        templates = {
            length: tuple(range(length)) for length in (512, 2048, 4094)
        }
        factory = c4._PromptFactory(tokenizer, templates)
        local = c4._materialize_block(
            sequence=0, arm=ForegroundArm.LOCAL, replicate=0,
            manifest=_manifest(), factory=factory)
        remote = c4._materialize_block(
            sequence=1, arm=ForegroundArm.REMOTE, replicate=0,
            manifest=_manifest(), factory=factory)
        self.assertEqual(local["schedule_sha256"], remote["schedule_sha256"])

        def foreground(block):
            return {
                value["pair_key"]: value["prompt_token_sha256"]
                for value in block["request_index"].values()
                if value["tenant"] == Tenant.FOREGROUND.value
            }

        self.assertEqual(foreground(local), foreground(remote))
        self.assertTrue(all(
            sum(marker in item.request_id for marker in (
                "-cache-miss-measured-",
                "-cache-p-only-measured-",
                "-cache-d-only-measured-",
                "-cache-both-measured-",
            )) == 1
            for item in (*local["items"], *remote["items"])
        ))
        plan = build_cache_preparation_plan(
            (*local["items"], *remote["items"]))
        self.assertGreater(len(plan.source_probe_rows), 0)
        self.assertGreater(len(plan.decoder_prepare_rows), 0)

    def test_source_and_decoder_artifact_validation_is_completion_backed(self):
        tokenizer = _Tokenizer()
        templates = {
            length: tuple(range(length)) for length in (512, 2048, 4094)
        }
        block = c4._materialize_block(
            sequence=0, arm=ForegroundArm.LOCAL, replicate=0,
            manifest=_manifest(),
            factory=c4._PromptFactory(tokenizer, templates))
        plan = build_cache_preparation_plan(block["items"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.json"
            source_requests = []
            source_decisions = []
            for row in plan.source_probe_rows:
                request_id = row["request_id"]
                prompt_tokens = c4._plan_row_prompt_tokens(plan, row)
                source_requests.append({
                    "request_id": request_id,
                    "valid": True,
                    "p_only_cache_seed": {
                        "request_id": request_id.replace(
                            "-warm-", "-warm-seed-o2-", 1),
                        "valid": True,
                        "route": c4._REMOTE_ROUTE,
                    },
                })
                source_decisions.append({
                    "request_id": request_id,
                    "route": c4._REMOTE_ROUTE,
                    "completion_cache_residency": "prefill_only",
                    "lmcache_source_full_hit_observed": True,
                    "decoder_prefix_cached_tokens": 0,
                    "decoder_total_cached_tokens": prompt_tokens,
                    "decoder_external_cached_tokens": prompt_tokens,
                    "decoder_prefix_usage_prompt_tokens": prompt_tokens + 1,
                    "decoder_prefix_cache_evidence_source": (
                        c4.DECODER_CACHE_EVIDENCE_SOURCE),
                    "frontend_pair_policy": "item_modulo_v1",
                    "frontend_pair_index": (
                        c4._plan_row_items(plan, row)[0].terminal_item % 2),
                })
            source_path.write_text(json.dumps({
                "requests": source_requests,
                "router_decisions": source_decisions,
            }))
            evidence = c4.validate_source_preparation(source_path, plan)
            self.assertTrue(
                evidence["all_seed_misses_and_probe_full_hits_exact"])

            decoder_path = root / "decoder.json"
            decoder_requests = []
            decoder_decisions = []
            expected = c4._decoder_expected(plan)
            for row in plan.decoder_prepare_rows:
                request_id = row["request_id"]
                prompt_tokens = c4._plan_row_prompt_tokens(plan, row)
                expected_hit = c4.full_prefix_hit_tokens(prompt_tokens)
                decoder_requests.append({"request_id": request_id, "valid": True})
                if "-cache-d-seed-" in request_id:
                    decoder_decisions.append({
                        "request_id": request_id,
                        "route": c4._LOCAL_ROUTE,
                        "decoder_prefix_cached_tokens": 0,
                        "decoder_total_cached_tokens": 0,
                        "decoder_external_cached_tokens": 0,
                        "decoder_prefix_usage_prompt_tokens": prompt_tokens,
                        "decoder_prefix_cache_evidence_source": (
                            c4.DECODER_CACHE_EVIDENCE_SOURCE),
                        "decoder_prefix_read_skipped": True,
                        "frontend_pair_index": 0,
                        "frontend_pair_policy": "item_modulo_v1",
                        "frontend_pair_decode_affinity_required": False,
                        "frontend_pair_decode_affinity_hit": False,
                    })
                else:
                    decoder_decisions.append({
                        "request_id": request_id,
                        "route": c4._LOCAL_ROUTE,
                        "decoder_prefix_full_hit_observed": True,
                        "decoder_prefix_cached_tokens": expected_hit,
                        "decoder_total_cached_tokens": expected_hit,
                        "decoder_external_cached_tokens": 0,
                        "decoder_prefix_usage_prompt_tokens": prompt_tokens,
                        "decoder_prefix_cache_evidence_source": (
                            c4.DECODER_CACHE_EVIDENCE_SOURCE),
                        "decoder_prefix_read_skipped": False,
                        "frontend_pair_index": 0,
                        "frontend_pair_policy": "item_modulo_v1",
                        "frontend_pair_decode_affinity_required": False,
                        "frontend_pair_decode_affinity_hit": False,
                        "completion_cache_residency": c4._DECISION_STATE[
                            expected[request_id]],
                    })
            decoder_path.write_text(json.dumps({
                "requests": decoder_requests,
                "router_decisions": decoder_decisions,
            }))
            evidence = c4.validate_decoder_preparation(decoder_path, plan)
            self.assertTrue(
                evidence["all_seed_misses_and_probe_full_hits_exact"])
            self.assertTrue(
                evidence[
                    "same_decoder_pair_enforced_by_terminal_item_modulo"])

    def test_measured_decision_requires_exact_state_specific_hits(self):
        for arm, route in (
            ("local", c4._LOCAL_ROUTE),
            ("remote", c4._REMOTE_ROUTE),
        ):
            for state in CacheState:
                prompt_tokens = 512
                skipped = state in {CacheState.MISS, CacheState.P_ONLY}
                usage_prompt = prompt_tokens + int(arm == "remote")
                local_cached = (
                    0 if skipped
                    else c4.full_prefix_hit_tokens(prompt_tokens)
                )
                total_cached = (
                    prompt_tokens if arm == "remote" else local_cached)
                metadata = {
                    "arm": arm,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": 16,
                    "cache_state": state.value,
                    "terminal_item": 0,
                }
                decision = {
                    "route": route,
                    "request_cache_contract": state.value,
                    "decision_cache_residency": c4._DECISION_STATE[state],
                    "decoder_prefix_read_skipped": skipped,
                    "decoder_prefix_cached_tokens": local_cached,
                    "decoder_total_cached_tokens": total_cached,
                    "decoder_external_cached_tokens": (
                        total_cached - local_cached),
                    "decoder_prefix_usage_prompt_tokens": usage_prompt,
                    "decoder_prefix_expected_full_hit_tokens": (
                        c4.full_prefix_hit_tokens(prompt_tokens)),
                    "decoder_prefix_full_hit_observed": not skipped,
                    "decoder_prefix_cache_evidence_source": (
                        c4.DECODER_CACHE_EVIDENCE_SOURCE),
                    "frontend_pair_policy": "item_modulo_v1",
                    "frontend_pair_index": 0,
                    "lmcache_source_cached_tokens": (
                        prompt_tokens
                        if state in {CacheState.P_ONLY, CacheState.BOTH}
                        else 0
                    ),
                }
                with self.subTest(arm=arm, state=state):
                    c4._validate_measured_decision(decision, metadata)

    def test_remote_block_aligned_prompt_requires_preparation_proven_prefix(self):
        prompt_tokens = 512
        metadata = {
            "arm": "remote",
            "prompt_tokens": prompt_tokens,
            "output_tokens": 128,
            "cache_state": CacheState.BOTH.value,
            "terminal_item": 0,
        }
        prepared_hit = c4.full_prefix_hit_tokens(prompt_tokens)
        decision = {
            "route": c4._REMOTE_ROUTE,
            "request_cache_contract": CacheState.BOTH.value,
            "decision_cache_residency": c4._DECISION_STATE[CacheState.BOTH],
            "decoder_prefix_read_skipped": False,
            "decoder_prefix_cached_tokens": prepared_hit,
            "decoder_total_cached_tokens": prompt_tokens,
            "decoder_external_cached_tokens": prompt_tokens - prepared_hit,
            "decoder_prefix_usage_prompt_tokens": prompt_tokens + 1,
            "decoder_prefix_expected_full_hit_tokens": prepared_hit,
            "decoder_prefix_full_hit_observed": True,
            "decoder_prefix_cache_evidence_source": (
                c4.DECODER_CACHE_EVIDENCE_SOURCE),
            "frontend_pair_policy": "item_modulo_v1",
            "frontend_pair_index": 0,
            "lmcache_source_cached_tokens": prompt_tokens,
        }
        c4._validate_measured_decision(decision, metadata)
        decision["decoder_prefix_cached_tokens"] = c4.full_prefix_hit_tokens(
            prompt_tokens + 1)
        decision["decoder_external_cached_tokens"] = 0
        decision["decoder_prefix_expected_full_hit_tokens"] = (
            decision["decoder_prefix_cached_tokens"])
        with self.assertRaisesRegex(ValueError, "cache-source evidence"):
            c4._validate_measured_decision(decision, metadata)


if __name__ == "__main__":
    unittest.main()
