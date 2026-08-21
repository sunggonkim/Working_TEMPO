#!/usr/bin/env python3
"""Build the C4 phase-trace manifest only from passed parent evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from tempo.pd_contention_workload import (
    CacheState,
    ContentionState,
    KV_REMOTE_HOT_GEOMETRY,
    VALIDATION_FOREGROUND_GEOMETRIES,
)
from tempo.pd_decoder_cache_evidence import EVIDENCE_SOURCE


SCHEMA = "tempo-pd-c4-phase-manifest-v2"
CACHE_STATE_PROTOCOL_SCHEMA = "tempo-pd-cache-state-protocol-v1"
ENDPOINT_EVIDENCE_CONTRACT_SCHEMA = (
    "tempo-pd-c4-endpoint-evidence-contract-v1")
C3_GATE_SCHEMA = "tempo-pd-c3-coupled-abba-gate-v1"
C3_MANIFEST_SCHEMA = "tempo-pd-c3-coupled-abba-manifest-v2"
C3_RESULT_SCHEMA = "tempo-pd-kv-only-attribution-node-v1"
C3_CHARACTERIZATION_SCHEMA = "tempo-pd-kv-only-characterization-v3"
COLD_RESULT_SCHEMA = "tempo-pd-contention-node-result-v7"
COLD_CHARACTERIZATION_SCHEMA = "tempo-pd-endpoint-characterization-v1"
P_ONLY_RESULT_SCHEMA = "tempo-pd-kv-only-attribution-node-v1"
P_ONLY_CHARACTERIZATION_SCHEMA = "tempo-pd-kv-only-characterization-v2"
FROZEN_COLD_MANIFEST_SCHEMA = "tempo-pd-contention-frozen-manifest-v1"
PHASES = (
    ContentionState.C0,
    ContentionState.C1,
    ContentionState.C2,
    ContentionState.C2_KV,
    ContentionState.C3,
    ContentionState.RECOVERY,
)
C4_FIXED_RUNTIME_ENVIRONMENT = {
    "TEMPO_PD_C4_FIXED_APPROVED": "YES",
    "TEMPO_PD_C4_PHASE_DURATION_MS": "8000",
    "TEMPO_PD_C4_COOLDOWN_S": "2",
    "TEMPO_PD_BENCHMARK_COLD_MEASURED": "0",
    "TEMPO_PD_BENCHMARK_RESET_DECODER_APC": "1",
    "TEMPO_VLLM_DECODER_PREFIX_CACHING": "1",
    "TEMPO_PD_FRONTEND_PAIR_POLICY": (
        "tempo-min-outstanding-decode-tokens-v1"),
    "TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY": "1",
    "TEMPO_PD_DECODER_REUSE_ITEMS": "all",
    "TEMPO_PD_FORWARD_TOKEN_IDS": "0",
    "TEMPO_PD_PROXY_KV_CONTROL_OVERLAP": "0",
    "TEMPO_PD_REMOTE_DECODE_PLACEMENT": "paired",
    "TEMPO_PD_PROXY_TOKENIZER_PLACEMENT": "round_robin",
    "TEMPO_LMCACHE_NIXL_BACKEND": "UCX",
    "TEMPO_LMCACHE_LOCAL_CPU_GB": "16",
    "TEMPO_LMCACHE_PD_BUFFER_BYTES": "2147483648",
    "TEMPO_ELASTIC_PD_PROFILE_SCOPE": "screen_only",
    "TEMPO_PD_ENDPOINT_FEEDBACK_MODE": "disabled",
    "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK": "0",
    "TEMPO_PD_PRESSURE_MODE": "disabled",
    "TEMPO_VLLM_LOAD_SNAPSHOT_MODE": "disabled",
    "TEMPO_VLLM_MAX_NUM_SEQS": "16",
    "TEMPO_VLLM_ASYNC_SCHEDULING": "0",
    "TEMPO_VLLM_DECODER_MAX_NUM_BATCHED_TOKENS": "32768",
    "TEMPO_VLLM_SCHEDULING_POLICY": "fcfs",
    "TEMPO_PD_REMOTE_CATCHUP_PRIORITY": "0",
    "TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY": "0",
    "TEMPO_PD_LONG_REMOTE_CATCHUP_PRIORITY": "0",
    "TEMPO_PD_LONG_REMOTE_CATCHUP_MIN_PROMPT_TOKENS": "0",
    "TEMPO_PD_MEDIAN_GUARD_PRIORITY": "0",
    "TEMPO_PD_MEDIUM_REMOTE_CATCHUP_PRIORITY": "0",
    "TEMPO_PD_REMOTE_CATCHUP_MIN_OUTPUT_TOKENS": "256",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, *, schema: str, name: str) -> dict[str, object]:
    path = path.resolve()
    _require(path.is_file(), f"{name} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} is not an object")
    _require(value.get("schema") == schema, f"{name} schema mismatch")
    return value


def _repo_path(repo_root: Path, path: Path, *, name: str) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError as exc:
        raise ValueError(f"{name} is outside the repository") from exc


def _artifact(repo_root: Path, path: Path, *, name: str) -> dict[str, str]:
    path = path.resolve()
    _require(path.is_file(), f"{name} is missing")
    return {
        "path": _repo_path(repo_root, path, name=name),
        "sha256": _sha256(path),
    }


def _declared_artifact(
    repo_root: Path,
    raw_path: object,
    raw_sha256: object,
    *,
    name: str,
) -> Path:
    _require(isinstance(raw_path, str) and raw_path,
             f"{name} path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    _require(path.is_file(), f"{name} is missing")
    _require(_sha256(path) == raw_sha256, f"{name} digest mismatch")
    _repo_path(repo_root, path, name=name)
    return path


def manifest_fingerprint(value: Mapping[str, object]) -> str:
    _require(isinstance(value, Mapping), "C4 manifest must be a mapping")
    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    *,
    repo_root: Path,
    c3_gate_path: Path,
    cold_result_path: Path,
    cold_characterization_path: Path,
    p_only_result_path: Path,
    p_only_characterization_path: Path,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    _require(repo_root.is_dir(), "repository root is missing")
    c3_gate_path = c3_gate_path.resolve()
    cold_result_path = cold_result_path.resolve()
    cold_characterization_path = cold_characterization_path.resolve()
    p_only_result_path = p_only_result_path.resolve()
    p_only_characterization_path = p_only_characterization_path.resolve()

    c3_gate = _load(
        c3_gate_path, schema=C3_GATE_SCHEMA, name="C3 ABBA gate")
    _require(c3_gate.get("c3_coupled_characterization_valid") is True,
             "C3 ABBA characterization did not pass")
    _require(c3_gate.get("authorizes_c4_phase_trace") is True,
             "C3 ABBA gate does not authorize C4")
    _require(c3_gate.get("performance_claim_allowed") is False,
             "C3 ABBA gate improperly permits a performance claim")
    _require(c3_gate.get("physical_switch_bottleneck_claim_allowed") is False,
             "C3 ABBA gate improperly permits a switch claim")
    c3_manifest_path = _declared_artifact(
        repo_root, c3_gate.get("manifest"), c3_gate.get("manifest_sha256"),
        name="C3 ABBA manifest")
    c3_manifest = _load(
        c3_manifest_path, schema=C3_MANIFEST_SCHEMA,
        name="C3 ABBA manifest")
    c3_result_path = _declared_artifact(
        repo_root, c3_gate.get("result"), c3_gate.get("result_sha256"),
        name="C3 ABBA result")
    c3_characterization_path = _declared_artifact(
        repo_root,
        c3_gate.get("characterization"),
        c3_gate.get("characterization_sha256"),
        name="C3 ABBA characterization",
    )
    c3_result = _load(
        c3_result_path, schema=C3_RESULT_SCHEMA, name="C3 ABBA result")
    c3_characterization = _load(
        c3_characterization_path, schema=C3_CHARACTERIZATION_SCHEMA,
        name="C3 ABBA characterization")
    _require(c3_result.get("performance_claim_allowed") is False,
             "C3 ABBA result improperly permits a performance claim")
    _require(c3_result.get(
        "physical_switch_bottleneck_claim_allowed") is False,
        "C3 ABBA result improperly permits a switch claim")
    _require(c3_characterization.get(
        "all_measured_requests_valid") is True,
        "C3 ABBA characterization has invalid requests")
    _require(c3_gate.get(
        "remote_control_replicate_direction_correct") == [True, True]
        and float(c3_gate.get("remote_control_median_gain")) >= 0.05,
        "C3 ABBA remote-control gate differs")
    _require(c3_gate.get(
        "local_overload_replicate_direction_correct") == [True, True]
        and float(c3_gate.get("local_overload_median_gain")) >= 0.05,
        "C3 ABBA local-overload gate differs")
    _require(c3_manifest.get("performance_claim_allowed") is False,
             "C3 ABBA manifest improperly permits a performance claim")
    _require(c3_manifest.get("arm_order_policy") == "paired_abba",
             "C3 ABBA order policy differs")
    _require(c3_manifest.get("within_rate_block_order") == [
        "local", "remote", "remote", "local"],
        "C3 ABBA within-rate order differs")
    rates = tuple(float(value) for value in c3_manifest.get(
        "p_only_rates_per_s", []))
    _require(rates == (0.0, 4.0, 8.0, 12.0),
             "C3 ABBA P_ONLY rate ladder differs")

    cold_result = _load(
        cold_result_path, schema=COLD_RESULT_SCHEMA,
        name="cold C1/C2 result")
    cold_characterization = _load(
        cold_characterization_path, schema=COLD_CHARACTERIZATION_SCHEMA,
        name="cold C1/C2 characterization")
    _require(cold_result.get("controller_tuning_allowed") is True,
             "cold C1/C2 result does not permit workload calibration")
    _require(cold_result.get("performance_claim_allowed") is False,
             "cold C1/C2 result improperly permits a performance claim")
    cold_gate = cold_result.get("crossover_gate")
    _require(isinstance(cold_gate, dict)
             and cold_gate.get("workload_valid_for_controller_tuning") is True,
             "cold C1/C2 crossover gate did not pass")
    _require(cold_characterization.get("crossover_gate") == cold_gate,
             "cold result/characterization crossover gates differ")
    cold_raw = Path(str(cold_result.get("raw"))).resolve()
    _require(Path(str(cold_characterization.get("source"))).resolve() == cold_raw,
             "cold characterization source differs")
    _require(cold_raw.is_file(), "cold C1/C2 raw artifact is missing")
    cold_manifest_path = _declared_artifact(
        repo_root,
        cold_result.get("frozen_workload_manifest"),
        cold_result.get("frozen_workload_manifest_sha256"),
        name="cold frozen manifest",
    )
    cold_manifest = _load(
        cold_manifest_path, schema=FROZEN_COLD_MANIFEST_SCHEMA,
        name="cold frozen manifest")

    p_only_result = _load(
        p_only_result_path, schema=P_ONLY_RESULT_SCHEMA,
        name="P_ONLY result")
    p_only_characterization = _load(
        p_only_characterization_path,
        schema=P_ONLY_CHARACTERIZATION_SCHEMA,
        name="P_ONLY characterization",
    )
    _require(p_only_result.get("performance_claim_allowed") is False,
             "P_ONLY result improperly permits a performance claim")
    _require(p_only_result.get(
        "physical_switch_bottleneck_claim_allowed") is False,
        "P_ONLY result improperly permits a switch claim")
    _require(p_only_result.get("stopped_after_first_invalid_block") is None,
             "P_ONLY campaign stopped on an invalid block")
    _require(p_only_characterization.get(
        "all_measured_requests_valid") is True,
        "P_ONLY characterization has invalid requests")
    _require(float(p_only_characterization.get(
        "first_rate_with_2x_remote_foreground_median")) == 12.0,
        "P_ONLY 2x service knee differs")
    _require(float(p_only_characterization.get(
        "first_rate_with_over_10pct_remote_drain")) == 12.0,
        "P_ONLY drain knee differs")
    p_only_raw = Path(str(p_only_result.get("raw"))).resolve()
    _require(Path(str(p_only_characterization.get("source"))).resolve()
             == p_only_raw, "P_ONLY characterization source differs")
    _require(p_only_raw.is_file(), "P_ONLY raw artifact is missing")
    p_only_invariants = p_only_characterization.get("invariants")
    _require(isinstance(p_only_invariants, dict)
             and p_only_invariants.get("background_full_source_hits_exact") is True
             and p_only_invariants.get("preseed_outside_measurement_window") is True
             and p_only_invariants.get("decoder_prefix_caching") is False
             and p_only_invariants.get("synthetic_network_background") is False,
             "P_ONLY cache/measurement invariants failed")

    source_workload = Path(str(c3_manifest["source_workload"]["path"]))
    if not source_workload.is_absolute():
        source_workload = repo_root / source_workload
    source_workload = source_workload.resolve()
    profile = Path(str(c3_manifest["profile"]["path"]))
    if not profile.is_absolute():
        profile = repo_root / profile
    profile = profile.resolve()
    _require(_sha256(source_workload)
             == c3_manifest["source_workload"]["sha256"],
             "C3 source workload digest differs")
    _require(_sha256(profile) == c3_manifest["profile"]["sha256"],
             "C3 profile digest differs")
    _require(Path(str(cold_result.get("source_workload"))).resolve()
             == source_workload,
             "cold and C3 source workloads differ")
    _require(Path(str(p_only_result.get("source_workload"))).resolve()
             == source_workload,
             "P_ONLY and C3 source workloads differ")
    _require(Path(str(cold_result.get("profile"))).resolve() == profile,
             "cold and C3 profiles differ")
    _require(Path(str(p_only_result.get("profile"))).resolve() == profile,
             "P_ONLY and C3 profiles differ")
    _require(p_only_result.get("profile_sha256") == _sha256(profile),
             "P_ONLY profile digest differs")
    _require(p_only_result.get("source_workload_sha256")
             == _sha256(source_workload),
             "P_ONLY source workload digest differs")

    cold_load = cold_manifest.get("load")
    _require(isinstance(cold_load, dict), "cold load selection is missing")
    decoder_rate = float(cold_load.get("decoder_offered_rate_per_s"))
    cold_remote_rate = float(cold_load.get("remote_offered_rate_per_s"))
    _require(decoder_rate == float(c3_manifest["decoder_hot_rate_per_s"]),
             "cold/C3 decoder-hot rates differ")
    _require(float(c3_manifest["foreground_rate_per_s"])
             == float(cold_manifest["foreground_rate_per_s"]),
             "cold/C3 foreground rates differ")
    _require(c3_manifest.get("transport") == "LMCacheConnectorV1:UCX",
             "C3 transport differs")

    value: dict[str, object] = {
        "schema": SCHEMA,
        "purpose": "C4 phase-changing workload characterization",
        "performance_claim_allowed": False,
        "controller_tuning_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "authorizes_profile_fit_only_after_c4_gate": True,
        "phase_order": [state.value for state in PHASES],
        "phase_duration_ms": float(c3_manifest["phase_duration_ms"]),
        "traffic_shape": "stable",
        "replicates": 2,
        "fixed_arm_order": ["local", "remote", "remote", "local"],
        "fixed_runtime_environment": dict(sorted(
            C4_FIXED_RUNTIME_ENVIRONMENT.items())),
        "foreground_rate_per_s": float(c3_manifest["foreground_rate_per_s"]),
        "background_rates_per_s": {
            "decoder_hot": decoder_rate,
            "cold_remote_hot": cold_remote_rate,
            "kv_remote_hot": 12.0,
        },
        "background_contract": {
            "actual_inference_only": True,
            "synthetic_network_background": False,
            "route_pinned": True,
            "passive_endpoint_feedback_marked": True,
            "kv_remote_cache_state": CacheState.P_ONLY.value,
            "kv_remote_prompt_tokens": KV_REMOTE_HOT_GEOMETRY.prompt_tokens,
            "kv_remote_output_tokens": KV_REMOTE_HOT_GEOMETRY.output_tokens,
            "kv_remote_preseed_outside_measurement": True,
            "kv_remote_full_source_hit_required": True,
            "zero_producer_compute_claim_allowed": False,
        },
        "endpoint_evidence_contract": {
            "schema": ENDPOINT_EVIDENCE_CONTRACT_SCHEMA,
            "measurement_start_marker_required": True,
            "publisher_pid_matches_measured_child": True,
            "measurement_clock": (
                "same_frontend_host_child_time_perf_counter_ns"),
            "sampling_policy": (
                "workload_start_boundary_midpoint_and_end_boundary"),
            "phase_boundary_samples": len(PHASES) + 1,
            "phase_midpoint_samples": len(PHASES),
            "cassini_phase_windows": (
                "two_nonoverlapping_half_phase_endpoint_deltas"),
            "vllm_phase_windows": (
                "boundary_to_boundary_cumulative_deltas"),
            "cross_host_clock_subtraction_allowed": False,
        },
        "foreground_geometries": [
            {
                "prompt_tokens": item.prompt_tokens,
                "output_tokens": item.output_tokens,
                "cache_state": item.cache_state.value,
            }
            for item in VALIDATION_FOREGROUND_GEOMETRIES
        ],
        "cache_state_protocol": {
            "schema": CACHE_STATE_PROTOCOL_SCHEMA,
            "scope": "per_arm_exact_prompt_namespace",
            "physical_namespace": (
                "stable_arm_and_exact_prompt_token_sha256_cache_salt"),
            "fixed_arm_pair_placement": "terminal_item_modulo_two_pairs",
            "logical_namespace_includes_output_tokens": False,
            "decoder_block_size_tokens": 16,
            "decoder_evidence_source": EVIDENCE_SOURCE,
            "decoder_usage_breakdown_required": True,
            "stock_cached_tokens_without_source_breakdown_allowed": False,
            "preparation_order": [
                "remote_seed_and_full_hit_probe_for_p_only_and_both",
                "quiescent_all_decoder_apc_reset_preserving_external_lmcache",
                "local_miss_seed_and_full_hit_probe_for_d_only_and_both",
                "measured_trace",
            ],
            "state_contracts": {
                "miss": {
                    "preparation_requests": 0,
                    "catalog_at_commit": "unseen_namespace_mapped_to_miss",
                    "decoder_apc_read": "skipped",
                    "completion_required": "exact_zero_cache_hit",
                },
                "p_only": {
                    "remote_seed_source_cached_tokens": 0,
                    "remote_probe_source_cached_tokens": "prompt_tokens",
                    "decoder_apc_reset_after_source_probe": True,
                    "decoder_apc_read_during_measurement": "skipped",
                    "catalog_at_commit": "completed_remote_probe_p_only",
                },
                "d_only": {
                    "producer_namespace_preparation": False,
                    "local_seed_decoder_cached_tokens": 0,
                    "local_probe_decoder_cached_tokens": (
                        "floor((prompt_tokens-1)/16)*16"),
                    "same_decoder_pair_required": True,
                    "catalog_at_commit": "completed_local_probe_d_only",
                },
                "both": {
                    "remote_seed_source_cached_tokens": 0,
                    "remote_probe_source_cached_tokens": "prompt_tokens",
                    "decoder_reset_preserves_external_lmcache": True,
                    "local_seed_decoder_cached_tokens": 0,
                    "local_probe_decoder_cached_tokens": (
                        "floor((prompt_tokens-1)/16)*16"),
                    "same_decoder_pair_required": True,
                    "catalog_at_commit": "completed_remote_and_local_probes_both",
                },
            },
            "measured_decoder_route_contracts": {
                "decoder_local_chunked_prefill": {
                    "usage_prompt_tokens": "P",
                    "local_cached_tokens_by_state": {
                        "miss": 0,
                        "p_only": 0,
                        "d_only": "floor((P-1)/16)*16",
                        "both": "floor((P-1)/16)*16",
                    },
                    "external_cached_tokens": 0,
                    "total_cached_tokens": "local_cached_tokens",
                },
                "official_lmcache_remote_prefill": {
                    "usage_prompt_tokens": "P+1",
                    "decoder_residency_basis": (
                        "exact_local_preparation_hit_on_original_P_token_prompt"),
                    "local_cached_tokens_by_state": {
                        "miss": 0,
                        "p_only": 0,
                        "d_only": "floor((P-1)/16)*16",
                        "both": "floor((P-1)/16)*16",
                    },
                    "external_cached_tokens": "P-local_cached_tokens",
                    "total_cached_tokens": "P",
                    "source_cached_tokens_by_state": {
                        "miss": 0,
                        "p_only": "P",
                        "d_only": 0,
                        "both": "P",
                    },
                },
            },
            "request_id_labels_without_completion_evidence_allowed": False,
            "partial_decoder_hits_allowed": False,
            "measurement_includes_preparation_requests": False,
        },
        "source_workload": _artifact(
            repo_root, source_workload, name="source workload"),
        "elastic_profile": _artifact(
            repo_root, profile, name="elastic profile"),
        "parent_evidence": {
            "cold_c1_c2_result": _artifact(
                repo_root, cold_result_path, name="cold C1/C2 result"),
            "cold_c1_c2_characterization": _artifact(
                repo_root, cold_characterization_path,
                name="cold C1/C2 characterization"),
            "p_only_result": _artifact(
                repo_root, p_only_result_path, name="P_ONLY result"),
            "p_only_characterization": _artifact(
                repo_root, p_only_characterization_path,
                name="P_ONLY characterization"),
            "c3_abba_gate": _artifact(
                repo_root, c3_gate_path, name="C3 ABBA gate"),
            "c3_abba_manifest": _artifact(
                repo_root, c3_manifest_path, name="C3 ABBA manifest"),
            "c3_abba_result": _artifact(
                repo_root, c3_result_path, name="C3 ABBA result"),
            "c3_abba_characterization": _artifact(
                repo_root, c3_characterization_path,
                name="C3 ABBA characterization"),
        },
        "transport": "LMCacheConnectorV1:UCX",
        "route_commit": "request_start_one_way",
        "cross_host_clock_subtraction_allowed": False,
    }
    value["fingerprint_sha256"] = manifest_fingerprint(value)
    return value


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--c3-gate", type=Path, required=True)
    parser.add_argument("--cold-result", type=Path, required=True)
    parser.add_argument("--cold-characterization", type=Path, required=True)
    parser.add_argument("--p-only-result", type=Path, required=True)
    parser.add_argument("--p-only-characterization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    output = args.output.resolve()
    _require(not output.exists(), "refusing to overwrite C4 manifest")
    value = build_manifest(
        repo_root=args.repo_root,
        c3_gate_path=args.c3_gate,
        cold_result_path=args.cold_result,
        cold_characterization_path=args.cold_characterization,
        p_only_result_path=args.p_only_result,
        p_only_characterization_path=args.p_only_characterization,
    )
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(output),
        "fingerprint_sha256": value["fingerprint_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
