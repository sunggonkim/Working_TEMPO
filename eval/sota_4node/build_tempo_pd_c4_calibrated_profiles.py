#!/usr/bin/env python3
"""Build C4-calibrated Elastic and endpoint profiles from frozen C0 pairs.

No controller multiplier is searched.  The formulas in this module are fixed
before the C4 run and consume only the C0 paired foreground samples from the
hash-bound C4 analysis.  Outputs remain screen/calibration-only until an
offline replay and an independent live validation authorize stronger use.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping

from eval.sota_4node import analyze_tempo_pd_c4_fixed_phase as analyzer
from eval.sota_4node import build_tempo_pd_c4_phase_manifest as manifest_builder
from tempo.pd_contention_workload import (
    CacheState,
    VALIDATION_FOREGROUND_GEOMETRIES,
)
from tempo.pd_elastic_controller_v443 import CacheResidency
from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_elastic_profile_v444 import SCHEMA as ELASTIC_SCHEMA
from tempo.pd_endpoint_profile import (
    SCHEMA as ENDPOINT_SCHEMA,
    endpoint_service_profile_fingerprint,
    load_endpoint_service_profile,
)


SCHEMA = "tempo-pd-c4-calibrated-profile-receipt-v1"
LIVE_MANIFEST_SCHEMA = "tempo-pd-c4-adaptive-screen-manifest-v2"
FORMULA_ID = "tempo-pd-c4-c0-paired-topology-window-v1"
DEFAULT_E2E_SLO_MS = 16_000.0
TTFT_SLO_MS = 3_000.0
TPOT_SLO_MS = 250.0
PD_PAIR_COUNT = 2
MAX_NUM_SEQS_PER_ENDPOINT = 16
ROUTE_MARGIN_MS = 5.0

_PAIR_SAMPLE_KEYS = frozenset({
    "pair_key", "replicate", "phase", "arrival_offset_ms", "prompt_tokens",
    "output_tokens", "cache_state", "ordinal", "local_block_key", "remote_block_key",
    "local_request_id", "remote_request_id", "output_text_sha256", "local",
    "remote", "remote_minus_local",
})
_SERVICE_METRIC_KEYS = frozenset({"ttft_ms", "e2e_ms", "tpot_ms"})
_STATE_TO_RESIDENCY = {
    CacheState.MISS: CacheResidency.MISS,
    CacheState.P_ONLY: CacheResidency.P_ONLY,
    CacheState.D_ONLY: CacheResidency.D_ONLY,
    CacheState.BOTH: CacheResidency.BOTH,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object, *, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be lowercase SHA-256",
    )
    return value


def _load_object(path: Path, *, name: str) -> dict[str, object]:
    _require(path.is_file(), f"{name} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read {name}") from exc
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _profile_fingerprint(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _receipt_fingerprint(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    return _profile_fingerprint(payload)


def _validate_analysis(
    path: Path, *, expected_sha256: str,
) -> dict[str, object]:
    expected_sha256 = _canonical_sha(
        expected_sha256, name="C4 analysis SHA-256")
    _require(_sha256(path) == expected_sha256, "C4 analysis digest differs")
    value = _load_object(path, name="C4 fixed-phase analysis")
    _require(
        value.get("schema") == analyzer.SCHEMA
        and value.get("fingerprint_sha256") == analyzer._analysis_fingerprint(value),
        "C4 analysis schema or fingerprint differs",
    )
    _require(
        value.get("authorizes_profile_fit") is True
        and value.get("profile_fit_scope") == "calibration_only"
        and value.get("authorizes_controller_parameter_search") is False
        and value.get("authorizes_live_validation") is False
        and value.get("performance_claim_allowed") is False
        and value.get("physical_switch_bottleneck_claim_allowed") is False,
        "C4 analysis does not authorize formula-only calibration",
    )
    source = value.get("source_node_result")
    _require(isinstance(source, dict) and set(source) == {"path", "sha256"},
             "C4 analysis node-result binding differs")
    source_path = Path(str(source["path"])).resolve()
    reproduced = analyzer.analyze(
        source_path, expected_result_sha256=source["sha256"])
    _require(reproduced == value,
             "C4 analysis does not reproduce from its bound node result")
    return value


def _validate_live_manifest(
    path: Path, *, expected_sha256: str,
    analysis_path: Path, analysis_sha256: str,
    analysis_fingerprint: str, phase_duration_ms: float,
) -> dict[str, object]:
    expected_sha256 = _canonical_sha(
        expected_sha256, name="adaptive workload manifest SHA-256")
    _require(_sha256(path) == expected_sha256,
             "adaptive workload manifest digest differs")
    value = _load_object(path, name="adaptive workload manifest")
    manifest_payload = dict(value)
    declared_fingerprint = manifest_payload.pop("fingerprint_sha256", None)
    binding = value.get("calibration_analysis")
    measurement = value.get("measurement")
    _require(
        value.get("schema") == LIVE_MANIFEST_SCHEMA
        and declared_fingerprint == _profile_fingerprint(manifest_payload)
        and isinstance(binding, dict)
        and set(binding) == {"path", "sha256", "fingerprint_sha256"}
        and Path(str(binding["path"])).resolve() == analysis_path
        and binding["sha256"] == analysis_sha256
        and binding["fingerprint_sha256"] == analysis_fingerprint
        and value.get("profile_fit_formula") == FORMULA_ID
        and value.get("phase_order")
        == [phase.value for phase in manifest_builder.PHASES]
        and float(value.get("phase_duration_ms")) == phase_duration_ms
        and value.get("transport") == "LMCacheConnectorV1:UCX"
        and value.get("unchanged_pd_data_plane") is True
        and value.get("controller_tuning_allowed") is False
        and value.get("performance_claim_allowed") is False
        and value.get("physical_switch_bottleneck_claim_allowed") is False,
        "adaptive workload manifest calibration contract differs",
    )
    _require(
        isinstance(measurement, dict)
        and float(measurement.get("e2e_slo_ms")) == DEFAULT_E2E_SLO_MS
        and float(measurement.get("ttft_slo_ms")) == TTFT_SLO_MS
        and float(measurement.get("tpot_slo_ms")) == TPOT_SLO_MS,
        "adaptive workload manifest SLO contract differs",
    )
    return value


def _metric_value(sample: Mapping[str, object], route: str, name: str) -> float:
    metrics = sample.get(route)
    _require(isinstance(metrics, Mapping) and set(metrics) == _SERVICE_METRIC_KEYS,
             f"C4 {route} service metric inventory differs")
    value = metrics.get(name)
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0,
        f"C4 {route} {name} is invalid",
    )
    return float(value)


def _paired_c0_groups(
    analysis: Mapping[str, object],
) -> dict[tuple[int, int, CacheState], list[Mapping[str, object]]]:
    samples = analysis.get("foreground_paired_samples")
    _require(isinstance(samples, list), "C4 paired samples are missing")
    groups: dict[
        tuple[int, int, CacheState], list[Mapping[str, object]]
    ] = defaultdict(list)
    pair_keys = set()
    for sample in samples:
        _require(isinstance(sample, dict) and set(sample) == _PAIR_SAMPLE_KEYS,
                 "C4 paired sample inventory differs")
        pair_key = sample.get("pair_key")
        _require(type(pair_key) is str and pair_key not in pair_keys,
                 "C4 paired sample key is missing or duplicated")
        pair_keys.add(pair_key)
        _canonical_sha(sample.get("output_text_sha256"),
                       name=f"{pair_key}.output_text_sha256")
        for route in ("local", "remote"):
            for metric in _SERVICE_METRIC_KEYS:
                _metric_value(sample, route, metric)
        if sample.get("phase") != "c0_cool":
            continue
        try:
            state = CacheState(sample["cache_state"])
        except (TypeError, ValueError) as exc:
            raise ValueError("C4 paired sample cache state is invalid") from exc
        key = (int(sample["prompt_tokens"]), int(sample["output_tokens"]), state)
        groups[key].append(sample)
    expected = {
        (row.prompt_tokens, row.output_tokens, row.cache_state)
        for row in VALIDATION_FOREGROUND_GEOMETRIES
    }
    _require(set(groups) == expected,
             "C4 C0 paired geometry/state inventory differs")
    _require(all(len(values) >= 4 for values in groups.values()),
             "each C4 C0 geometry/state requires four paired samples")
    return dict(groups)


def _route_concurrency(
    *, foreground_rate_per_s: float, max_ttft_ms: float,
) -> int:
    # Little's-law occupancy plus one burst slot per physical P/D pair.
    occupancy = math.ceil(foreground_rate_per_s * max_ttft_ms / 1000.0)
    return min(
        PD_PAIR_COUNT * MAX_NUM_SEQS_PER_ENDPOINT,
        max(PD_PAIR_COUNT, occupancy + PD_PAIR_COUNT),
    )


def _uncertainty_ms(samples: list[Mapping[str, object]]) -> float:
    gaps = [
        _metric_value(sample, "local", "e2e_ms")
        - _metric_value(sample, "remote", "e2e_ms")
        for sample in samples
    ]
    center = statistics.median(gaps)
    return max(1.0, max(abs(value - center) for value in gaps))


def build_profiles(
    *, analysis_path: Path, expected_analysis_sha256: str,
    workload_manifest_path: Path, expected_workload_manifest_sha256: str,
    elastic_profile_id: str, endpoint_profile_id: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    analysis_path = analysis_path.resolve()
    workload_manifest_path = workload_manifest_path.resolve()
    _require(type(elastic_profile_id) is str and elastic_profile_id.strip(),
             "Elastic profile ID must be nonempty")
    _require(type(endpoint_profile_id) is str and endpoint_profile_id.strip(),
             "endpoint profile ID must be nonempty")
    analysis = _validate_analysis(
        analysis_path, expected_sha256=expected_analysis_sha256)
    source_phase = analysis.get("phase_manifest")
    _require(isinstance(source_phase, dict)
             and set(source_phase) == {"path", "sha256", "fingerprint_sha256"},
             "C4 analysis phase-manifest binding differs")
    source_phase_path = Path(str(source_phase["path"])).resolve()
    _require(source_phase_path.is_file()
             and _sha256(source_phase_path) == source_phase["sha256"],
             "C4 source phase manifest digest differs")
    source_phase_value = _load_object(
        source_phase_path, name="C4 source phase manifest")
    phase_duration_ms = float(source_phase_value["phase_duration_ms"])
    live_manifest = _validate_live_manifest(
        workload_manifest_path,
        expected_sha256=expected_workload_manifest_sha256,
        analysis_path=analysis_path,
        analysis_sha256=expected_analysis_sha256,
        analysis_fingerprint=analysis["fingerprint_sha256"],
        phase_duration_ms=phase_duration_ms,
    )
    del live_manifest

    source_elastic = analysis.get("elastic_profile")
    _require(isinstance(source_elastic, dict)
             and set(source_elastic) == {"path", "sha256"},
             "C4 source Elastic-profile binding differs")
    source_elastic_path = Path(str(source_elastic["path"])).resolve()
    _require(source_elastic_path.is_file()
             and _sha256(source_elastic_path) == source_elastic["sha256"],
             "C4 source Elastic-profile digest differs")
    identity_profile = load_elastic_profile(source_elastic_path)
    identity = identity_profile.identity
    groups = _paired_c0_groups(analysis)

    elastic_rows = []
    endpoint_rows = []
    maximum_local_ttft = 0.0
    maximum_remote_ttft = 0.0
    for prompt_tokens, output_tokens, state in sorted(
        groups, key=lambda key: (key[0], key[1], key[2].value)
    ):
        samples = groups[(prompt_tokens, output_tokens, state)]
        local_ttft = [_metric_value(row, "local", "ttft_ms") for row in samples]
        remote_ttft = [_metric_value(row, "remote", "ttft_ms") for row in samples]
        local_e2e = [_metric_value(row, "local", "e2e_ms") for row in samples]
        remote_e2e = [_metric_value(row, "remote", "e2e_ms") for row in samples]
        local_tpot = [_metric_value(row, "local", "tpot_ms") for row in samples]
        remote_tpot = [_metric_value(row, "remote", "tpot_ms") for row in samples]
        _require(
            max(local_ttft + remote_ttft) <= TTFT_SLO_MS
            and max(local_e2e + remote_e2e) <= DEFAULT_E2E_SLO_MS
            and max(local_tpot + remote_tpot) <= TPOT_SLO_MS,
            f"C4 C0 route is not idle-SLO-safe: "
            f"{prompt_tokens}/{output_tokens}/{state.value}",
        )
        local_ttft_prior = float(statistics.median(local_ttft))
        remote_ttft_prior = float(statistics.median(remote_ttft))
        maximum_local_ttft = max(maximum_local_ttft, local_ttft_prior)
        maximum_remote_ttft = max(maximum_remote_ttft, remote_ttft_prior)
        elastic_rows.append({
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "local_upper_bound_ms": max(local_e2e),
            "remote_upper_bound_ms": max(remote_e2e),
            "uncertainty_ms": _uncertainty_ms(samples),
            "local_tbt_safe": True,
            "remote_evidence_valid": True,
            "local_compute_cost_us": math.ceil(max(local_ttft) * 1000.0),
            "remote_kv_bytes": prompt_tokens * identity.kv_bytes_per_token,
            "samples_local": len(samples),
            "samples_remote": len(samples),
            "outputs_equivalent": True,
            "remote_transfer_failures": 0,
        })
        endpoint_rows.append({
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "cache_residency": _STATE_TO_RESIDENCY[state].value,
            "local_ttft_prior_ms": local_ttft_prior,
            "remote_ttft_prior_ms": remote_ttft_prior,
            "local_token_ms": math.ceil(prompt_tokens * local_ttft_prior),
            "remote_prefill_token_ms": math.ceil(
                prompt_tokens * remote_ttft_prior),
            "samples_local": len(samples),
            "samples_remote": len(samples),
            "outputs_equivalent": True,
            "evidence_valid": True,
        })

    foreground_rate = float(source_phase_value["foreground_rate_per_s"])
    local_concurrency = _route_concurrency(
        foreground_rate_per_s=foreground_rate,
        max_ttft_ms=maximum_local_ttft)
    remote_concurrency = _route_concurrency(
        foreground_rate_per_s=foreground_rate,
        max_ttft_ms=maximum_remote_ttft)
    max_local_compute = max(row["local_compute_cost_us"] for row in elastic_rows)
    max_remote_kv = max(row["remote_kv_bytes"] for row in elastic_rows)
    elastic: dict[str, object] = {
        "schema": ELASTIC_SCHEMA,
        "profile_id": elastic_profile_id,
        "deployment_scope": "screen_only",
        "identity": {
            "model_id": identity.model_id,
            "model_revision": identity.model_revision,
            "topology_id": identity.topology_id,
            "remote_backend": identity.remote_backend,
            "classifier_version": identity.classifier_version,
            "kv_bytes_per_token": identity.kv_bytes_per_token,
        },
        "controller": {
            "local_compute_budget_us": local_concurrency * max_local_compute,
            "remote_kv_budget_bytes": remote_concurrency * max_remote_kv,
            "arrival_window": 2 * PD_PAIR_COUNT,
            "enter_high_gap_ns": 39_000_000,
            "exit_high_gap_ns": 78_000_000,
            "exit_consecutive_windows": 2,
            "route_margin_ms": ROUTE_MARGIN_MS,
            "spill_regression_budget_ms": ROUTE_MARGIN_MS,
        },
        "rows": elastic_rows,
    }
    elastic_fingerprint = _profile_fingerprint(elastic)

    max_local_work = max(row["local_token_ms"] for row in endpoint_rows)
    max_remote_prefill_work = max(
        row["remote_prefill_token_ms"] for row in endpoint_rows)
    phase_ns = round(phase_duration_ms * 1_000_000)
    history = min(64, max(8, math.ceil(
        foreground_rate * phase_duration_ms / 1000.0)))
    endpoint: dict[str, object] = {
        "schema": ENDPOINT_SCHEMA,
        "profile_id": endpoint_profile_id,
        "elastic_profile_fingerprint_sha256": elastic_fingerprint,
        "workload_manifest_sha256": expected_workload_manifest_sha256,
        "deployment_scope": "calibration_only",
        "default_e2e_deadline_ms": DEFAULT_E2E_SLO_MS,
        "controller": {
            "local_token_ms_window": local_concurrency * max_local_work,
            "remote_prefill_token_ms_window": (
                remote_concurrency * max_remote_prefill_work),
            "remote_kv_bytes_window": remote_concurrency * max_remote_kv,
            "remote_semantic_ops_window": remote_concurrency,
            "feedback_history": history,
            "feedback_quantile": 0.9,
            "minimum_feedback": PD_PAIR_COUNT,
            "route_margin_ms": ROUTE_MARGIN_MS,
            "feedback_fresh_ns": phase_ns,
            "probe_after_ns": phase_ns // 2,
            "denied_probe_after_ns": phase_ns,
        },
        "rows": endpoint_rows,
    }
    endpoint["fingerprint_sha256"] = endpoint_service_profile_fingerprint(endpoint)

    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "formula_id": FORMULA_ID,
        "source_analysis": {
            "path": str(analysis_path),
            "sha256": expected_analysis_sha256,
            "fingerprint_sha256": analysis["fingerprint_sha256"],
        },
        "workload_manifest": {
            "path": str(workload_manifest_path),
            "sha256": expected_workload_manifest_sha256,
        },
        "source_identity_profile": {
            "path": str(source_elastic_path),
            "sha256": source_elastic["sha256"],
        },
        "formula_contract": {
            "calibration_phase": "c0_cool",
            "latency_upper_bound": "max_paired_c0_e2e_ms",
            "ttft_prior": "median_paired_c0_ttft_ms",
            "route_gap_uncertainty": (
                "max_1ms_and_max_absolute_deviation_from_median_paired_gap"),
            "work_weight": "prompt_tokens_times_median_idle_ttft_ms",
            "route_concurrency": (
                "min_32_max_2_ceil_foreground_rate_times_max_idle_ttft_plus_2"),
            "physical_pd_pairs": PD_PAIR_COUNT,
            "max_num_seqs_per_endpoint": MAX_NUM_SEQS_PER_ENDPOINT,
            "controller_parameter_search": False,
        },
        "slo_contract": {
            "e2e_slo_ms": DEFAULT_E2E_SLO_MS,
            "ttft_slo_ms": TTFT_SLO_MS,
            "tpot_slo_ms": TPOT_SLO_MS,
        },
        "derived_route_concurrency": {
            "local": local_concurrency,
            "remote": remote_concurrency,
        },
        "elastic_profile": {
            "profile_id": elastic_profile_id,
            "fingerprint_sha256": elastic_fingerprint,
            "rows": len(elastic_rows),
        },
        "endpoint_profile": {
            "profile_id": endpoint_profile_id,
            "fingerprint_sha256": endpoint["fingerprint_sha256"],
            "rows": len(endpoint_rows),
        },
        "all_six_geometry_state_rows_exact": True,
        "includes_4094_256_d_only": any(
            row["prompt_tokens"] == 4094
            and row["output_tokens"] == 256
            and row["cache_residency"] == CacheResidency.D_ONLY.value
            for row in endpoint_rows),
        "remote_admission_for_d_only_or_both_allowed": False,
        "calibration_only": True,
        "offline_replay_required": True,
        "independent_validation_required": True,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
    }
    _require(receipt["includes_4094_256_d_only"] is True,
             "C4 calibrated profiles omit 4094/256 D_ONLY")
    receipt["fingerprint_sha256"] = _receipt_fingerprint(receipt)
    return elastic, endpoint, receipt


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--expected-analysis-sha256", required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--expected-workload-manifest-sha256", required=True)
    parser.add_argument("--elastic-profile-id", required=True)
    parser.add_argument("--endpoint-profile-id", required=True)
    parser.add_argument("--elastic-output", type=Path, required=True)
    parser.add_argument("--endpoint-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    outputs = (args.elastic_output, args.endpoint_output, args.receipt_output)
    _require(len({path.resolve() for path in outputs}) == 3,
             "profile outputs must be distinct")
    _require(all(not path.exists() for path in outputs),
             "refusing to overwrite calibrated profile output")
    elastic, endpoint, receipt = build_profiles(
        analysis_path=args.analysis,
        expected_analysis_sha256=args.expected_analysis_sha256,
        workload_manifest_path=args.workload_manifest,
        expected_workload_manifest_sha256=(
            args.expected_workload_manifest_sha256),
        elastic_profile_id=args.elastic_profile_id,
        endpoint_profile_id=args.endpoint_profile_id,
    )
    for path, value in zip(outputs, (elastic, endpoint, receipt), strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    loaded_elastic = load_elastic_profile(args.elastic_output.resolve())
    loaded_endpoint = load_endpoint_service_profile(args.endpoint_output.resolve())
    _require(
        loaded_elastic.fingerprint_sha256
        == endpoint["elastic_profile_fingerprint_sha256"]
        and loaded_endpoint.fingerprint_sha256
        == endpoint["fingerprint_sha256"],
        "published calibrated profiles failed strict round-trip",
    )
    print(json.dumps({
        "schema": SCHEMA,
        "elastic_fingerprint_sha256": loaded_elastic.fingerprint_sha256,
        "endpoint_fingerprint_sha256": loaded_endpoint.fingerprint_sha256,
        "receipt_fingerprint_sha256": receipt["fingerprint_sha256"],
        "rows": len(loaded_elastic.rows),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
