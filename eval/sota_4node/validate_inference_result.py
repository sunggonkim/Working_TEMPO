#!/usr/bin/env python3
"""Fail-closed validator for an observed inference KV-flow result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from tempo.causal_gate import InferenceModeRecord, evaluate_inference_matrix
    from tempo.domain_counters import CounterSnapshot, validate_counter_series
    from tempo.domain_evidence import CounterSupport, DomainEvidence, PathStatus
    from tempo.resource_domain import EvidenceLevel, ResourceDomain, allowed_counter_scopes
    from tempo.foreground_path import validate_foreground_path
    from tempo.observation_window import validate_observation_windows
    from eval.sota_4node.inference_kv_runner import build_kv_matrix
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tempo.causal_gate import InferenceModeRecord, evaluate_inference_matrix
    from tempo.domain_counters import CounterSnapshot, validate_counter_series
    from tempo.domain_evidence import CounterSupport, DomainEvidence, PathStatus
    from tempo.resource_domain import EvidenceLevel, ResourceDomain, allowed_counter_scopes
    from tempo.foreground_path import validate_foreground_path
    from tempo.observation_window import validate_observation_windows
    from eval.sota_4node.inference_kv_runner import build_kv_matrix


MODES = ("fg_only", "open_combined", "d2h_only", "remote_fabric", "persistent_tier", "combined")
_RESULT_KEYS = {
    "schema_version",
    "evidence_state",
    "world_size",
    "nodes",
    "source_bundle_sha256",
    "backend",
    "endpoint",
    "kv_bytes_per_request",
    "deadline_ns",
    "offered_load_requests",
    "operation",
    "admission_contract",
    "foreground_path",
    "modes",
}
_ADMISSION_KEYS = {
    "controller",
    "flow_adapter",
    "shared_domain_intersection",
    "completion_owner",
}
_MODE_KEYS = {"metrics", "correctness", "integrity", "route_evidence", "observation_windows"}
_METRIC_KEYS = {
    "observation_id",
    "domain",
    "foreground_domains",
    "shared_domains",
    "ttft_p99_ns",
    "itl_p99_ns",
    "slo_goodput_milli",
    "deadline_met",
    "correctness_met",
    "samples",
    "max_domain_exposure_ns",
    "domain_exposure_ns",
}
_CORRECTNESS_KEYS = {
    "native_version_identity",
    "output_token_equivalence",
    "stale_version_rejection",
    "prefetch_before_use",
    "exact_completion_bytes",
}
_INTEGRITY_KEYS = {
    "published_versions",
    "stale_rejections",
    "output_token_mismatches",
    "prefetch_before_use_violations",
    "admitted_bytes",
    "completed_bytes",
}
_ROUTE_KEYS = {
    "observation_id",
    "mode",
    "domain",
    "scope",
    "scope_id",
    "intervention_id",
    "overlapping_bytes",
    "overlap_ns",
    "tail_delta_ns",
    "evidence",
    "counter_support",
    "path_status",
    "uncertainty_ns",
    "counter_samples",
    "counter_series",
    "source",
    "path_evidence",
    "counter_family",
}


def _hex(value: object, name: str) -> None:
    if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")


def _intervention_domains(mode: str) -> tuple[ResourceDomain, ...]:
    if mode == "d2h_only":
        return (
            ResourceDomain.GPU_LOCAL,
            ResourceDomain.PCIE_HOST,
            ResourceDomain.HOST_NUMA,
        )
    if mode == "remote_fabric":
        return (ResourceDomain.NIC_FABRIC, ResourceDomain.SLINGSHOT_FABRIC)
    if mode == "persistent_tier":
        return (
            ResourceDomain.NIC_FABRIC,
            ResourceDomain.SLINGSHOT_FABRIC,
            ResourceDomain.PERSISTENT_ENDPOINT,
        )
    return ()


def _parse_domain_exposure(raw: object, mode: str) -> dict[ResourceDomain, int]:
    if type(raw) is not dict:
        raise ValueError(f"{mode}: domain_exposure_ns must be an object")
    if any(type(name) is not str for name in raw):
        raise ValueError(f"{mode}: domain_exposure_ns keys must be strings")
    if list(raw) != sorted(raw):
        raise ValueError(f"{mode}: domain_exposure_ns keys must be sorted")
    parsed: dict[ResourceDomain, int] = {}
    for name, value in raw.items():
        try:
            domain = ResourceDomain(name)
        except ValueError as exc:
            raise ValueError(f"{mode}: domain_exposure_ns contains an unknown domain") from exc
        if type(value) is not int or value < 0:
            raise ValueError(f"{mode}: domain_exposure_ns values must be non-negative ints")
        parsed[domain] = value
    return parsed


def _metric(mode: str, raw: object, domain: ResourceDomain | None) -> InferenceModeRecord:
    if type(raw) is not dict or set(raw) != _METRIC_KEYS:
        raise ValueError(f"{mode}: metric keys are not exact")
    if type(raw["observation_id"]) is not str or not raw["observation_id"]:
        raise ValueError(f"{mode}: observation_id is invalid")
    exposure = _parse_domain_exposure(raw["domain_exposure_ns"], mode)
    record = InferenceModeRecord(
        mode=mode,
        domain=domain,
        ttft_p99_ns=raw["ttft_p99_ns"],
        itl_p99_ns=raw["itl_p99_ns"],
        slo_goodput_milli=raw["slo_goodput_milli"],
        deadline_met=raw["deadline_met"],
        correctness_met=raw["correctness_met"],
        samples=raw["samples"],
        max_domain_exposure_ns=raw["max_domain_exposure_ns"],
        domain_exposure_ns=exposure,
    )
    return record


def _parse_domain_list(raw: object, name: str, *, allow_empty: bool) -> tuple[ResourceDomain, ...]:
    if type(raw) is not list or (not allow_empty and not raw):
        raise ValueError(f"{name} must be a {'possibly empty ' if allow_empty else ''}list")
    if any(type(item) is not str for item in raw):
        raise ValueError(f"{name} must contain domain strings")
    try:
        domains = tuple(ResourceDomain(item) for item in raw)
    except ValueError as exc:
        raise ValueError(f"{name} contains an unknown resource domain") from exc
    if len(set(domains)) != len(domains):
        raise ValueError(f"{name} must not contain duplicate domains")
    canonical = tuple(sorted(domains, key=lambda item: item.value))
    if domains != canonical:
        raise ValueError(f"{name} must be sorted by domain value")
    return domains


def _route_evidence(
    raw: object, mode: str, *, expected_observation_id: str | None = None
) -> list[DomainEvidence]:
    if type(raw) is not list:
        raise ValueError(f"{mode}: route_evidence must be a list")
    records: list[DomainEvidence] = []
    for item in raw:
        if type(item) is not dict or set(item) != _ROUTE_KEYS:
            raise ValueError(f"{mode}: route evidence keys are not exact")
        if item["mode"] != mode:
            raise ValueError(f"{mode}: route evidence mode mismatch")
        try:
            route_domain = ResourceDomain(item["domain"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{mode}: route evidence domain is invalid") from exc
        if (
            type(item["observation_id"]) is not str
            or not item["observation_id"]
        ):
            raise ValueError(f"{mode}: route evidence observation_id is invalid")
        if expected_observation_id is not None and item["observation_id"] != expected_observation_id:
            raise ValueError(f"{mode}: route evidence observation_id does not match metrics")
        if (
            type(item["scope"]) is not str
            or not item["scope"]
            or item["scope"] not in allowed_counter_scopes(route_domain)
            or type(item["scope_id"]) is not str
            or not item["scope_id"]
            or item["intervention_id"] != mode
        ):
            raise ValueError(f"{mode}: route counter scope/intervention binding is invalid")
        # ``combined`` is a full-flow consistency replicate, not an isolated
        # intervention.  Treating it as interventional would let a route that
        # contains several domains rescue a failed single-domain screen.
        expected_evidence = "observational" if mode in {"open_combined", "combined"} else "interventional"
        if item["evidence"] != expected_evidence:
            raise ValueError(f"{mode}: route evidence must be {expected_evidence}")
        if type(item["counter_samples"]) is not int or item["counter_samples"] < 2:
            raise ValueError(f"{mode}: at least two route counter samples are required")
        raw_series = item["counter_series"]
        if type(raw_series) is not list or len(raw_series) != item["counter_samples"]:
            raise ValueError(f"{mode}: route counter series/count mismatch")
        snapshots: list[CounterSnapshot] = []
        for sample in raw_series:
            if type(sample) is not dict or set(sample) != {
                "observation_id", "domain", "sample_id", "source", "timestamp_ns", "cumulative_bytes",
                "cumulative_busy_ns", "support",
            }:
                raise ValueError(f"{mode}: route counter snapshot keys are not exact")
            for field in ("sample_id", "source"):
                if type(sample[field]) is not str or not sample[field]:
                    raise ValueError(f"{mode}: route counter {field} must be a non-empty string")
            try:
                snapshots.append(
                    CounterSnapshot(
                        domain=ResourceDomain(sample["domain"]),
                        sample_id=sample["sample_id"],
                        source=sample["source"],
                        timestamp_ns=sample["timestamp_ns"],
                        cumulative_bytes=sample["cumulative_bytes"],
                        cumulative_busy_ns=sample["cumulative_busy_ns"],
                        support=CounterSupport(sample["support"]),
                    )
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"{mode}: invalid route counter snapshot: {exc}") from exc
        if any(snapshot.domain.value != item["domain"] for snapshot in snapshots):
            raise ValueError(f"{mode}: route counter domain mismatch")
        if any(sample["observation_id"] != item["observation_id"] for sample in raw_series):
            raise ValueError(f"{mode}: route counter observation_id mismatch")
        if any(snapshot.source != item["source"] for snapshot in snapshots):
            raise ValueError(f"{mode}: route counter source is not bound to evidence source")
        if any(snapshot.support is not CounterSupport.SUPPORTED for snapshot in snapshots):
            raise ValueError(f"{mode}: route counter series is not supported")
        validate_counter_series(snapshots)
        try:
            record = DomainEvidence(
                domain=ResourceDomain(item["domain"]),
                mode=mode,
                foreground_kind="inference_request",
                auxiliary_kind="kv_flow",
                overlapping_bytes=item["overlapping_bytes"],
                overlap_ns=item["overlap_ns"],
                tail_delta_ns=item["tail_delta_ns"],
                evidence=(
                    EvidenceLevel.OBSERVATIONAL
                    if mode in {"open_combined", "combined"}
                    else EvidenceLevel.INTERVENTIONAL
                ),
                counter_support=CounterSupport(item["counter_support"]),
                path_status=PathStatus(item["path_status"]),
                uncertainty_ns=item["uncertainty_ns"],
                source=item["source"],
                path_evidence=item["path_evidence"],
                counter_family=item["counter_family"],
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"{mode}: invalid route evidence: {exc}") from exc
        if record.path_status is not PathStatus.OBSERVED or record.counter_support is not CounterSupport.SUPPORTED:
            raise ValueError(f"{mode}: route evidence must be observed/supported")
        records.append(record)
    return records


def validate_inference_result(result: dict[str, object]) -> dict[str, object]:
    if type(result) is not dict or set(result) != _RESULT_KEYS:
        raise ValueError("inference result keys are not exact")
    if result["schema_version"] != "tempo-rd-inference-result-5":
        raise ValueError("unsupported inference result schema")
    if result["evidence_state"] != "live_observed":
        raise ValueError("inference result must be live_observed")
    if result["world_size"] != 1 or result["nodes"] != 1:
        raise ValueError("initial inference result must be one GPU and one node")
    _hex(result["source_bundle_sha256"], "source_bundle_sha256")
    for key in ("kv_bytes_per_request", "deadline_ns", "offered_load_requests"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"{key} must be a positive int")
    if result["operation"] != "prefetch":
        raise ValueError("inference result operation must be prefetch")
    admission = result["admission_contract"]
    if type(admission) is not dict or set(admission) != _ADMISSION_KEYS:
        raise ValueError("inference admission contract keys are not exact")
    expected_admission = {
        "controller": "DomainAdmissionController",
        "flow_adapter": "KVFlowLedger.admit_via_domain_controller",
        "shared_domain_intersection": "explicit_foreground_route_intersection",
        "completion_owner": "KVFlowLedger.complete",
    }
    if admission != expected_admission:
        raise ValueError("inference result must use the shared domain controller contract")
    if type(result["backend"]) is not dict or set(result["backend"]) != {"name", "version", "executable_sha256"}:
        raise ValueError("backend provenance keys are not exact")
    for key in ("name", "version"):
        if type(result["backend"][key]) is not str or not result["backend"][key]:
            raise ValueError(f"backend {key} must be a non-empty string")
    _hex(result["backend"]["executable_sha256"], "backend.executable_sha256")
    if type(result["endpoint"]) is not str or not result["endpoint"]:
        raise ValueError("endpoint must be a non-empty string")
    foreground_path = validate_foreground_path(result["foreground_path"])

    raw_modes = result["modes"]
    if type(raw_modes) is not dict or set(raw_modes) != set(MODES):
        raise ValueError("inference result mode set is not exact")
    expected_runs = {run.mode: run for run in build_kv_matrix()}
    expected_endpoints = {
        run.endpoint for run in expected_runs.values() if run.mode != "fg_only"
    }
    if len(expected_endpoints) != 1 or result["endpoint"] != next(iter(expected_endpoints)):
        raise ValueError(
            "inference endpoint must match the common auxiliary matched-open endpoint"
        )
    metric_records: list[InferenceModeRecord] = []
    metric_domains: dict[str, dict[str, object]] = {}
    for mode in MODES:
        raw_mode = raw_modes[mode]
        if type(raw_mode) is not dict or set(raw_mode) != _MODE_KEYS:
            raise ValueError(f"{mode}: result keys are not exact")
        raw_metrics = raw_mode["metrics"]
        expected_observation_id = (
            raw_metrics.get("observation_id") if type(raw_metrics) is dict else None
        )
        route = _route_evidence(
            raw_mode["route_evidence"],
            mode,
            expected_observation_id=expected_observation_id,
        )
        expected_route = tuple(expected_runs[mode].route)
        joined_windows = validate_observation_windows(
            raw_mode["observation_windows"],
            expected_mode=mode,
            expected_observation_id=expected_observation_id,
            require_auxiliary=mode != "fg_only",
        )
        if not any(window.uncertainty_safe for window in joined_windows):
            raise ValueError(f"{mode}: observation overlap does not exceed uncertainty")
        observed_route = tuple(item.domain.value for item in route)
        if observed_route != expected_route:
            raise ValueError(f"{mode}: ordered route domains do not match the frozen KV matrix")
        if mode == "fg_only" and route:
            raise ValueError("fg_only must not contain route evidence")

        correctness = raw_mode["correctness"]
        if type(correctness) is not dict or set(correctness) != _CORRECTNESS_KEYS:
            raise ValueError(f"{mode}: correctness keys are not exact")
        if any(type(correctness[key]) is not bool for key in _CORRECTNESS_KEYS):
            raise ValueError(f"{mode}: correctness values must be strict bools")
        if not all(correctness.values()):
            raise ValueError(f"{mode}: KV correctness contract failed")

        integrity = raw_mode["integrity"]
        if type(integrity) is not dict or set(integrity) != _INTEGRITY_KEYS:
            raise ValueError(f"{mode}: integrity keys are not exact")
        if any(type(integrity[key]) is not int or integrity[key] < 0 for key in _INTEGRITY_KEYS):
            raise ValueError(f"{mode}: integrity values must be non-negative ints")
        expected_bytes = 0 if mode == "fg_only" else result["kv_bytes_per_request"] * result["offered_load_requests"]
        if integrity["published_versions"] <= 0:
            raise ValueError(f"{mode}: no published KV versions were recorded")
        if integrity["stale_rejections"] <= 0:
            raise ValueError(f"{mode}: stale-version rejection was not exercised")
        if integrity["admitted_bytes"] != expected_bytes or integrity["completed_bytes"] != expected_bytes:
            raise ValueError(f"{mode}: admitted/completed KV bytes are not exact")
        if integrity["output_token_mismatches"] != 0 or integrity["prefetch_before_use_violations"] != 0:
            raise ValueError(f"{mode}: output or prefetch correctness violations")

        raw_domain = raw_mode["metrics"].get("domain")
        domain = None
        if mode in {"fg_only", "open_combined", "combined"}:
            if raw_domain is not None:
                raise ValueError(f"{mode}: non-intervention metric domain must be null")
        else:
            try:
                domain = ResourceDomain(raw_domain)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{mode}: metric domain must name an explicit intervention domain") from exc
            if domain not in _intervention_domains(mode):
                raise ValueError(f"{mode}: metric domain is not on the route")
        if domain is not None:
            matched = [item for item in route if item.domain is domain]
            if not matched or not any(item.tail_delta_ns > item.uncertainty_ns for item in matched):
                raise ValueError(
                    f"{mode}: intervention tail delta does not exceed uncertainty for metric domain"
                )
        if mode != "fg_only":
            for item in route:
                if item.overlapping_bytes != result["kv_bytes_per_request"]:
                    raise ValueError(
                        f"{mode}: route bytes do not cover the offered KV workload"
                    )
        metric = _metric(mode, raw_mode["metrics"], domain)
        observed_domain_exposure: dict[ResourceDomain, int] = {}
        for item in route:
            observed_domain_exposure[item.domain] = max(
                observed_domain_exposure.get(item.domain, 0), item.overlap_ns
            )
        observed_max_exposure = max(observed_domain_exposure.values(), default=0)
        if metric.max_domain_exposure_ns != observed_max_exposure:
            raise ValueError(
                f"{mode}: max_domain_exposure_ns is not derived from route evidence"
            )
        if dict(metric.domain_exposure_ns or {}) != observed_domain_exposure:
            raise ValueError(
                f"{mode}: domain_exposure_ns is not derived from route evidence"
            )
        # A KV attribution matrix is incomplete if an auxiliary route missed
        # its prefetch deadline or output/version correctness contract.  Do
        # not let a strong isolated latency number or the combined replicate
        # hide that failed endpoint contract.
        if mode != "fg_only" and (
            metric.deadline_met is not True or metric.correctness_met is not True
        ):
            raise ValueError(
                f"{mode}: live KV mode must satisfy deadline and correctness"
            )
        foreground = _parse_domain_list(
            raw_mode["metrics"]["foreground_domains"],
            f"{mode}: foreground_domains",
            allow_empty=False,
        )
        shared = _parse_domain_list(
            raw_mode["metrics"]["shared_domains"],
            f"{mode}: shared_domains",
            allow_empty=True,
        )
        auxiliary = tuple(ResourceDomain(item) for item in expected_route)
        expected_shared = tuple(
            sorted((item for item in auxiliary if item in foreground), key=lambda item: item.value)
        )
        if shared != expected_shared:
            raise ValueError(
                f"{mode}: shared_domains must equal foreground/auxiliary route intersection"
            )
        metric_domains[mode] = {
            "foreground_domains": [item.value for item in foreground],
            "shared_domains": [item.value for item in shared],
            "max_domain_exposure_ns": metric.max_domain_exposure_ns,
            "domain_exposure_ns": {
                item.value: value
                for item, value in sorted(
                    (metric.domain_exposure_ns or {}).items(),
                    key=lambda pair: pair[0].value,
                )
            },
        }
        metric_records.append(metric)

    # TTFT/ITL comparisons are only matched if the foreground serving route
    # is identical. A mode that changes the foreground footprint can appear
    # faster simply by moving the request to a different resource path.
    foreground_reference = tuple(foreground_path["domains"])
    for mode, info in metric_domains.items():
        if tuple(info["foreground_domains"]) != foreground_reference:
            raise ValueError(
                f"{mode}: foreground_domains do not match fg_only matched workload"
            )

    promotion = evaluate_inference_matrix(metric_records)
    return {
        "schema_version": "tempo-rd-inference-evaluation-1",
        "status": "pass",
        "evidence_ready": True,
        "promote_static_policy": promotion.promote_static_policy,
        "eligible_domains": sorted(domain.value for domain in promotion.eligible_domains),
        "headroom": promotion.headroom,
        "placebo_clean": promotion.placebo_clean,
        "reasons": list(promotion.reasons),
        "metric_domains": metric_domains,
        "live_external_execution": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_inference_result(json.loads(args.result.read_text(encoding="utf-8")))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
