from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from eval.sota_4node import analyze_tempo_pd_c4_fixed_phase as analyzer
from eval.sota_4node import run_tempo_pd_c4_fixed_phase_client as client
from eval.sota_4node import tempo_pd_endpoint_probe as endpoint_probe
from eval.sota_4node import verify_tempo_pd_c4_implementation as implementation
from tempo.cassini_endpoint import SCHEMA as CASSINI_SCHEMA
from tempo.domain_evidence import CounterSupport
from tempo.pd_contention_workload import CacheState, Tenant
from tempo.pd_endpoint_evidence import (
    PDEndpointIdentity,
    PDEndpointSnapshot,
    endpoint_metric_names,
    endpoint_metrics,
)


MANIFEST = (
    analyzer.REPO_ROOT / "eval/sota_4node/tempo_pd_c4_phase_manifest_v2.json"
)
IMPLEMENTATION = (
    analyzer.REPO_ROOT
    / "eval/sota_4node/tempo_pd_c4_implementation_contract_v1.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest_path(entry: dict[str, object]) -> Path:
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = analyzer.REPO_ROOT / path
    return path.resolve()


def _decision(request_id: str, metadata: dict[str, object]) -> dict[str, object]:
    state = CacheState(metadata["cache_state"])
    route = (
        client._LOCAL_ROUTE
        if metadata["arm"] == "local" else client._REMOTE_ROUTE
    )
    prompt_tokens = int(metadata["prompt_tokens"])
    skipped = state in {CacheState.MISS, CacheState.P_ONLY}
    local_cached = (
        0 if skipped else client.full_prefix_hit_tokens(prompt_tokens)
    )
    total_cached = prompt_tokens if route == client._REMOTE_ROUTE else local_cached
    value = {
        "request_id": request_id,
        "phase": "complete",
        "error": None,
        "route": route,
        "request_cache_contract": state.value,
        "decision_cache_residency": client._DECISION_STATE[state],
        "frontend_pair_policy": "item_modulo_v1",
        "frontend_pair_index": int(metadata["terminal_item"]) % 2,
        "decoder_prefix_read_skipped": skipped,
        "decoder_prefix_cached_tokens": local_cached,
        "decoder_total_cached_tokens": total_cached,
        "decoder_external_cached_tokens": total_cached - local_cached,
        "decoder_prefix_usage_prompt_tokens": (
            prompt_tokens + int(route == client._REMOTE_ROUTE)),
        "decoder_prefix_expected_full_hit_tokens": (
            client.full_prefix_hit_tokens(prompt_tokens)),
        "decoder_prefix_full_hit_observed": not skipped,
        "decoder_prefix_cache_evidence_source": (
            client.DECODER_CACHE_EVIDENCE_SOURCE),
    }
    if route == client._REMOTE_ROUTE:
        value["lmcache_source_cached_tokens"] = (
            prompt_tokens
            if state in {CacheState.P_ONLY, CacheState.BOTH} else 0
        )
    return value


def _cumulative(ordinal: int, endpoint_offset: int) -> dict[str, object]:
    values = {}
    base = endpoint_offset * 10_000 + ordinal * 100
    for name in endpoint_probe.VLLM_CUMULATIVE_METRICS:
        value = base + 10
        values[name] = (
            int(value)
            if name.endswith("_total") or name.endswith("_count")
            else float(value)
        )
    return {
        "schema": endpoint_probe.VLLM_CUMULATIVE_SCHEMA,
        "source": "vllm_prometheus_on_demand",
        "model_name": "synthetic-qwen",
        "engine_indices": [0, 1, 2, 3],
        "values": dict(sorted(values.items())),
    }


def _cassini(
    *, endpoint_id: str, role: str, pair_index: int,
    sequence: int, sampled_ns: int,
) -> dict[str, object]:
    return {
        "schema": CASSINI_SCHEMA,
        "endpoint_id": endpoint_id,
        "role": role,
        "pair_index": pair_index,
        "source": "cassini_sysfs_endpoint_delta",
        "nic_count": 4,
        "support": {
            "ecn_fraction": CounterSupport.NOT_SUPPORTED.value,
            "host_nonposted_cycles_per_packet": (
                CounterSupport.NOT_SUPPORTED.value),
            "host_posted_cycles_per_packet": CounterSupport.SUPPORTED.value,
            "packet_counts": CounterSupport.NOT_SUPPORTED.value,
            "receive_overflow_fraction": CounterSupport.NOT_SUPPORTED.value,
            "rx_pause_fraction": CounterSupport.SUPPORTED.value,
            "transport_fault_counts": CounterSupport.NOT_SUPPORTED.value,
            "tx_pause_fraction": CounterSupport.SUPPORTED.value,
        },
        "valid": True,
        "invalid_reason": None,
        "sequence": sequence,
        "sampled_ns": sampled_ns,
        "read_ms": 1.0,
        "cache_age_ms": 0.0,
        "window_ms": 4000.0,
        "signals": {
            "ecn_fraction_max": None,
            "ecn_fraction_mean": None,
            "host_nonposted_cycles_per_packet_max": None,
            "host_posted_cycles_per_packet_max": 1.0,
            "receive_overflow_fraction_max": None,
            "receive_overflow_fraction_mean": None,
            "resource_nacks": None,
            "retries": None,
            "rx_packets": None,
            "rx_pause_fraction_max": 0.0,
            "rx_pause_fraction_mean": 0.0,
            "timeouts": None,
            "tx_packets": None,
            "tx_pause_fraction_max": 0.0,
            "tx_pause_fraction_mean": 0.0,
        },
    }


def _sample(stage: str, ordinal: int, received_ns: int) -> dict[str, object]:
    snapshots = []
    for endpoint_offset, (endpoint_id, role, pair_index) in enumerate(
        analyzer._EXPECTED_ENDPOINTS
    ):
        identity = PDEndpointIdentity(
            endpoint_id=endpoint_id, role=role, pair_index=pair_index)
        supported = {
            "running_requests": ordinal + endpoint_offset,
            "waiting_requests": endpoint_offset,
            "kv_cache_usage_fraction": 0.25,
        }
        unavailable = {
            name: CounterSupport.NOT_COLLECTED
            for name in endpoint_metric_names(role) if name not in supported
        }
        endpoint = PDEndpointSnapshot(
            identity=identity,
            sequence=ordinal + 1,
            endpoint_monotonic_ns=received_ns + endpoint_offset,
            source="vllm_prometheus_on_demand",
            metrics=endpoint_metrics(
                role, supported=supported, unavailable=unavailable),
        ).as_dict()
        probe = {
            "schema": endpoint_probe.SCHEMA,
            "endpoint": endpoint,
            "vllm_cumulative": _cumulative(ordinal, endpoint_offset),
            "vllm_metrics_fetch": {
                "attempts_configured": 1,
                "attempts_used": 1,
                "timeout_s_per_attempt": 3.0,
                "retry_backoff_s": 0.05,
                "transient_errors": [],
                "elapsed_ns": 1_000_000,
            },
            "cassini": _cassini(
                endpoint_id=endpoint_id,
                role=role.value,
                pair_index=pair_index,
                sequence=ordinal + 1,
                sampled_ns=received_ns + endpoint_offset,
            ),
        }
        snapshots.append({
            "source_url": f"http://endpoint-{endpoint_offset}:9000",
            "client_fetch_started_monotonic_ns": received_ns - 1_000_000,
            "client_received_monotonic_ns": received_ns,
            "probe": probe,
        })
    return {
        "schema": client.fixed.ENDPOINT_EVIDENCE_SCHEMA,
        "stage": stage,
        "snapshots": sorted(
            snapshots,
            key=lambda row: row["probe"]["endpoint"]["endpoint_id"],
        ),
    }


def _endpoint_evidence(
    root: Path, *, block_sequence: int,
    request_index: dict[str, dict[str, object]], phase_duration_ms: float,
) -> dict[str, object]:
    run_start_ns = 10_000_000_000 + block_sequence * 100_000_000_000
    marker = root / f"block-{block_sequence}-measurement-start.json"
    marker_value = {
        "schema": client.protocol_client.START_MARKER_SCHEMA,
        "clock": "client time.perf_counter_ns",
        "run_start_ns": run_start_ns,
        "publisher_pid": 12345 + block_sequence,
    }
    _write(marker, marker_value)
    boundaries = []
    start = _sample("measurement_start", 1, run_start_ns + 10_000_000)
    start.update({
        "boundary_index": 0,
        "completed_phase": None,
        "begins_phase": analyzer.manifest_builder.PHASES[0].value,
    })
    boundaries.append(start)
    midpoints = []
    for index, phase in enumerate(analyzer.manifest_builder.PHASES):
        midpoint_received = run_start_ns + int(
            (index + 0.5) * phase_duration_ms * 1_000_000) + 10_000_000
        midpoint = _sample("phase_midpoint", 2 + 2 * index, midpoint_received)
        midpoint.update({"phase_index": index, "phase": phase.value})
        midpoints.append(midpoint)
        boundary_received = run_start_ns + int(
            (index + 1.0) * phase_duration_ms * 1_000_000) + 10_000_000
        boundary = _sample("phase_boundary", 3 + 2 * index, boundary_received)
        boundary.update({
            "boundary_index": index + 1,
            "completed_phase": phase.value,
            "begins_phase": (
                analyzer.manifest_builder.PHASES[index + 1].value
                if index + 1 < len(analyzer.manifest_builder.PHASES)
                else None
            ),
        })
        boundaries.append(boundary)
    first_arrival = min(
        int(float(value["arrival_offset_ms"]) * 1_000_000)
        for value in request_index.values()
    )
    return {
        "schema": client.ENDPOINT_EVIDENCE_SCHEMA,
        "sampling_policy": client.ENDPOINT_SAMPLING_POLICY,
        "cross_endpoint_clock_subtraction_allowed": False,
        "measurement_clock_alignment": (
            "same_frontend_host_child_time_perf_counter_ns_marker"),
        "before_process_start": _sample(
            "before_process_start", 0, run_start_ns - 10_000_000),
        "measurement_start_marker": {
            **marker_value,
            "path": str(marker.resolve()),
            "sha256": _sha(marker),
            "parent_observed_child_pid": marker_value["publisher_pid"],
            "parent_observed_offset_ns": 1_000_000,
        },
        "first_arrival_offset_ns": first_arrival,
        "measurement_start_capture_completed_offset_ns": 10_000_000,
        "phase_boundaries": boundaries,
        "phase_midpoints": midpoints,
    }


class _Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.run_root = root / "run"
        self.client_root = self.run_root / "tempo_pd_c4_fixed_phase"
        self.client_raw = self.client_root / "raw.json"
        self.result = self.run_root / "result.json"
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.implementation = json.loads(
            IMPLEMENTATION.read_text(encoding="utf-8"))
        self.block_paths: dict[str, Path] = {}
        self._build()

    def _build_block(
        self, *, key: str, sequence: int, arm, replicate: int,
    ) -> tuple[Path, dict[str, object]]:
        expected, schedule_sha = analyzer._expected_request_index(
            self.manifest, sequence=sequence, arm=arm, replicate=replicate)
        request_index = {}
        requests = []
        decisions = []
        for request_id, base_metadata in expected.items():
            metadata = {
                **base_metadata,
                "prompt_token_sha256": hashlib.sha256(
                    request_id.encode("utf-8")).hexdigest(),
            }
            request_index[request_id] = metadata
            dispatch = int(float(metadata["arrival_offset_ms"]) * 1_000_000)
            first = dispatch + (
                20_000_000 if metadata["arm"] == "local" else 30_000_000)
            token_arrivals = [
                first + index * 1_000_000
                for index in range(int(metadata["output_tokens"]))
            ]
            requests.append({
                "request_id": request_id,
                "valid": True,
                "requested_max_tokens": metadata["output_tokens"],
                "dispatch_offset_ns": dispatch,
                "token_arrival_offsets_ns": token_arrivals,
                "stream_end_offset_ns": token_arrivals[-1] + 1_000_000,
                "output_text_sha256": "a" * 64,
            })
            decisions.append(_decision(request_id, metadata))
        contract = {
            "schema": client.BLOCK_SCHEMA,
            "sequence": sequence,
            "foreground_arm": arm.value,
            "replicate": replicate,
            "semantic_schedule_sha256": schedule_sha,
            "request_index": request_index,
            "all_requests_valid": True,
            "decision_cache_states_exact": True,
            "completion_cache_evidence_exact": True,
            "workload_start_marker_exact": True,
            "phase_aligned_endpoint_evidence": True,
            "preparation_outside_measurement": True,
            "actual_inference_background_only": True,
            "cross_endpoint_clock_subtraction_allowed": False,
        }
        raw = {
            "schema": analyzer.STREAM_SCHEMA,
            "requests": requests,
            "router_decisions": decisions,
            "validation": {
                "all_streams_valid": True,
                "router_decisions_exact": True,
                "performance_claim_allowed": True,
            },
            "c4_fixed_phase_contract": contract,
            "endpoint_evidence": _endpoint_evidence(
                self.client_root,
                block_sequence=sequence,
                request_index=request_index,
                phase_duration_ms=float(self.manifest["phase_duration_ms"]),
            ),
        }
        path = _write(self.client_root / "c4_fixed_phase" / f"{key}.raw.json", raw)
        return path, contract

    def _build(self) -> None:
        artifacts = {}
        contracts = {}
        paired_blocks = []
        for sequence, (key, arm, replicate) in enumerate(
            analyzer._EXPECTED_BLOCKS
        ):
            path, contract = self._build_block(
                key=key, sequence=sequence, arm=arm, replicate=replicate)
            self.block_paths[key] = path
            artifacts[key] = {"path": str(path.resolve()), "sha256": _sha(path)}
            contracts[key] = contract
            paired_blocks.append({
                "sequence": sequence,
                "arm": arm,
                "replicate": replicate,
                "schedule_sha256": contract["semantic_schedule_sha256"],
                "raw_path": str(path.resolve()),
            })
        gate = client._paired_gate(paired_blocks)
        plan = _write(self.client_root / "cache_preparation_plan.json", {
            "schema": "synthetic-cache-plan"})
        evidence = _write(self.client_root / "cache_runtime_evidence.json", {
            "schema": "synthetic-cache-runtime-evidence"})
        parent = {
            "schema": client.SCHEMA,
            "run_id": "synthetic-c4",
            "manifest": str(MANIFEST.resolve()),
            "manifest_sha256": _sha(MANIFEST),
            "manifest_fingerprint_sha256": self.manifest["fingerprint_sha256"],
            "cache_plan": str(plan.resolve()),
            "cache_plan_sha256": _sha(plan),
            "cache_runtime_evidence": str(evidence.resolve()),
            "cache_runtime_evidence_sha256": _sha(evidence),
            "block_order": [
                {"arm": arm.value, "replicate": replicate}
                for _key, arm, replicate in analyzer._EXPECTED_BLOCKS
            ],
            "artifacts": artifacts,
            "contracts": contracts,
            "gate": gate,
            "performance_claim_allowed": False,
            "controller_tuning_allowed": True,
        }
        _write(self.client_raw, parent)
        source = _manifest_path(self.manifest["source_workload"])
        profile = _manifest_path(self.manifest["elastic_profile"])
        node = {
            "schema": analyzer.NODE_SCHEMA,
            "raw": str(self.client_raw.resolve()),
            "raw_sha256": _sha(self.client_raw),
            "phase_manifest": str(MANIFEST.resolve()),
            "phase_manifest_sha256": _sha(MANIFEST),
            "phase_manifest_fingerprint_sha256": self.manifest[
                "fingerprint_sha256"],
            "implementation_contract": str(IMPLEMENTATION.resolve()),
            "implementation_contract_sha256": _sha(IMPLEMENTATION),
            "implementation_fingerprint_sha256": self.implementation[
                "fingerprint_sha256"],
            "implementation_file_count": len(self.implementation["files"]),
            "implementation_git_heads": self.implementation["git_heads"],
            "implementation_environment_versions": self.implementation[
                "environment_versions"],
            "fixed_runtime_environment": self.manifest[
                "fixed_runtime_environment"],
            "transport_environment": {},
            "elastic_profile": str(profile),
            "elastic_profile_sha256": _sha(profile),
            "source_workload": str(source),
            "source_workload_sha256": _sha(source),
            "slurm_job_id": "synthetic-unit-test",
            "startup_readiness_timeout_s": 3600.0,
            "block_count": 4,
            "paired_output_count": gate["paired_output_count"],
            "phase_service_row_count": len(gate["phase_service_rows"]),
            "phase_route_summary_count": len(gate["phase_route_summaries"]),
            "cache_state_protocol_completion_backed": True,
            "decoder_cache_source_breakdown_exact": True,
            "phase_aligned_endpoint_evidence": True,
            "decoder_residency_basis": (
                "exact_local_preparation_hit_on_original_P_token_prompt"),
            "characterization_gate_pass": True,
            "controller_tuning_allowed": True,
            "performance_claim_allowed": False,
            "physical_switch_bottleneck_claim_allowed": False,
            "unchanged_pd_data_plane": True,
            "transport": "LMCacheConnectorV1:UCX",
        }
        _write(self.result, node)

    def rebind_client_raw(self) -> str:
        node = json.loads(self.result.read_text(encoding="utf-8"))
        node["raw_sha256"] = _sha(self.client_raw)
        _write(self.result, node)
        return _sha(self.result)

    def rebind_block(self, key: str) -> str:
        parent = json.loads(self.client_raw.read_text(encoding="utf-8"))
        parent["artifacts"][key]["sha256"] = _sha(self.block_paths[key])
        _write(self.client_raw, parent)
        return self.rebind_client_raw()


class AnalyzeC4FixedPhaseTest(unittest.TestCase):
    def test_valid_four_block_evidence_emits_exact_phase_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            result = analyzer.analyze(
                fixture.result, expected_result_sha256=_sha(fixture.result))
            self.assertTrue(result["authorizes_profile_fit"])
            self.assertFalse(result["performance_claim_allowed"])
            self.assertFalse(result["physical_switch_bottleneck_claim_allowed"])
            self.assertEqual(len(result["endpoint_phase_rows"]), 96)
            self.assertEqual(len(result["request_phase_tenant_rows"]), 96)
            self.assertEqual(
                len(result["foreground_paired_samples"]),
                result["fixed_gate"]["paired_output_count"],
            )
            self.assertEqual(len(result["fixed_gate"]["phase_service_rows"]), 36)
            self.assertEqual(
                result["fingerprint_sha256"],
                analyzer._analysis_fingerprint(result),
            )

    def test_block_hash_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            expected = _sha(fixture.result)
            path = fixture.block_paths["00_local_r0"]
            path.write_text(path.read_text(encoding="utf-8") + "\n")
            with self.assertRaisesRegex(ValueError, "block 00_local_r0 digest"):
                analyzer.analyze(
                    fixture.result, expected_result_sha256=expected)

    def test_endpoint_sequence_regression_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            key = "00_local_r0"
            path = fixture.block_paths[key]
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["endpoint_evidence"]["phase_midpoints"][0]["snapshots"][0][
                "probe"]["endpoint"]["sequence"] = 2
            _write(path, raw)
            expected = fixture.rebind_block(key)
            with self.assertRaisesRegex(ValueError, "sequence did not increase"):
                analyzer.analyze(
                    fixture.result, expected_result_sha256=expected)

    def test_cumulative_vllm_regression_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            key = "00_local_r0"
            path = fixture.block_paths[key]
            raw = json.loads(path.read_text(encoding="utf-8"))
            boundaries = raw["endpoint_evidence"]["phase_boundaries"]
            metric = "vllm:prompt_tokens_total"
            initial = boundaries[0]["snapshots"][0]["probe"][
                "vllm_cumulative"]["values"][metric]
            boundaries[1]["snapshots"][0]["probe"][
                "vllm_cumulative"]["values"][metric] = initial - 1
            _write(path, raw)
            expected = fixture.rebind_block(key)
            with self.assertRaisesRegex(ValueError, "cumulative vLLM metric regressed"):
                analyzer.analyze(
                    fixture.result, expected_result_sha256=expected)

    def test_missing_phase_geometry_cell_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            parent = json.loads(fixture.client_raw.read_text(encoding="utf-8"))
            parent["gate"]["phase_service_rows"].pop()
            _write(fixture.client_raw, parent)
            expected = fixture.rebind_client_raw()
            with self.assertRaisesRegex(ValueError, "phase service cell inventory"):
                analyzer.analyze(
                    fixture.result, expected_result_sha256=expected)


if __name__ == "__main__":
    unittest.main()
