#!/usr/bin/env python3
"""Fail-closed validator for an observed two-node TEMPO-RD fabric result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from tempo.causal_gate import CausalModeRecord, evaluate_causal_matrix
    from tempo.domain_counters import CounterSnapshot, validate_counter_series
    from tempo.domain_evidence import CounterSupport, DomainEvidence, PathStatus
    from tempo.resource_domain import EvidenceLevel, ResourceDomain, allowed_counter_scopes, domain_contract
    from tempo.observation_window import validate_observation_windows
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tempo.causal_gate import CausalModeRecord, evaluate_causal_matrix
    from tempo.domain_counters import CounterSnapshot, validate_counter_series
    from tempo.domain_evidence import CounterSupport, DomainEvidence, PathStatus
    from tempo.resource_domain import EvidenceLevel, ResourceDomain, allowed_counter_scopes, domain_contract
    from tempo.observation_window import validate_observation_windows

from eval.sota_4node.validate_g1_result import validate_g1_result


G2_MODES = (
    "fg_only",
    "open_combined",
    "causal_domain_static_cap",
    "unrelated_domain_placebo",
    "combined",
)
COLLECTIVE_SLICES = ("intra_node", "inter_node")
FABRIC_SPLITS = ("gdr_gpu_originated", "host_originated", "pfs_endpoint")
FULL_PATH = (
    ResourceDomain.GPU_LOCAL,
    ResourceDomain.PCIE_HOST,
    ResourceDomain.HOST_NUMA,
    ResourceDomain.NIC_FABRIC,
    ResourceDomain.SLINGSHOT_FABRIC,
    ResourceDomain.PERSISTENT_ENDPOINT,
)
_RESULT_KEYS = {
    "schema_version",
    "evidence_state",
    "world_size",
    "nodes",
    "source_bundle_sha256",
    "g1_result",
    "promoted_domain",
    "placebo_domain",
    "state_bytes_per_rank",
    "deadline_ns",
    "checkpoint_steps",
    "collective_slices",
    "fabric_splits",
    "modes",
}
_METRIC_KEYS = {
    "observation_id",
    "foreground_domains",
    "shared_domains",
    "tail_p99_ns",
    "skew_p99_ns",
    "deadline_met",
    "correctness_met",
    "samples",
    "active_exposure_ns",
    "active_groups",
    "domain_exposure_ns",
}
_SLICE_METRIC_KEYS = {"tail_p99_ns", "skew_p99_ns", "samples"}
_FABRIC_KEYS = {
    "observation_id",
    "mode",
    "collective_slice",
    "traffic_origin",
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

FOREGROUND_PATH = tuple(
    sorted(
        (
            ResourceDomain.GPU_LOCAL,
            ResourceDomain.NVLINK_P2P,
            ResourceDomain.PCIE_HOST,
            ResourceDomain.HOST_NUMA,
            ResourceDomain.NIC_FABRIC,
            ResourceDomain.SLINGSHOT_FABRIC,
        ),
        key=lambda item: item.value,
    )
)


def _hex(value: object, name: str) -> None:
    if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")


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


def _parse_domain_exposure(raw: object, mode: str) -> dict[ResourceDomain, int]:
    if type(raw) is not dict or list(raw) != sorted(raw):
        raise ValueError(f"{mode}: domain_exposure_ns must be a sorted object")
    parsed: dict[ResourceDomain, int] = {}
    for name, value in raw.items():
        try:
            domain = ResourceDomain(name)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{mode}: domain_exposure_ns has an unknown domain") from exc
        if type(value) is not int or value < 0:
            raise ValueError(f"{mode}: domain_exposure_ns values must be non-negative ints")
        parsed[domain] = value
    return parsed


def _metric(
    mode: str,
    raw: object,
    domain: ResourceDomain | None,
    auxiliary_domains: set[ResourceDomain],
) -> tuple[CausalModeRecord, dict[str, object]]:
    if type(raw) is not dict or set(raw) != _METRIC_KEYS:
        raise ValueError(f"{mode}: metric keys are not exact")
    if type(raw["observation_id"]) is not str or not raw["observation_id"]:
        raise ValueError(f"{mode}: observation_id is invalid")
    record = CausalModeRecord(
        mode=mode,
        domain=domain,
        tail_p99_ns=raw["tail_p99_ns"],
        skew_p99_ns=raw["skew_p99_ns"],
        deadline_met=raw["deadline_met"],
        correctness_met=raw["correctness_met"],
        samples=raw["samples"],
        domain_exposure_ns=_parse_domain_exposure(raw["domain_exposure_ns"], mode),
    )
    for key in ("active_exposure_ns", "active_groups"):
        if type(raw[key]) is not int or raw[key] < 0:
            raise ValueError(f"{mode}: {key} must be a non-negative int")
    foreground = _parse_domain_list(
        raw["foreground_domains"], f"{mode}: foreground_domains", allow_empty=False
    )
    shared = _parse_domain_list(
        raw["shared_domains"], f"{mode}: shared_domains", allow_empty=True
    )
    expected_shared = tuple(
        sorted((item for item in auxiliary_domains if item in foreground), key=lambda item: item.value)
    )
    if shared != expected_shared:
        raise ValueError(
            f"{mode}: shared_domains must equal foreground/auxiliary route intersection"
        )
    return record, {
        "observation_id": raw["observation_id"],
        "foreground_domains": [item.value for item in foreground],
        "shared_domains": [item.value for item in shared],
        "active_exposure_ns": int(raw["active_exposure_ns"]),
        "active_groups": int(raw["active_groups"]),
        "domain_exposure_ns": {
            domain.value: value for domain, value in sorted(
                (record.domain_exposure_ns or {}).items(), key=lambda item: item[0].value
            )
        },
    }


def _fabric_record(
    raw: object, expected_mode: str, *, expected_observation_id: str | None = None
) -> DomainEvidence:
    if type(raw) is not dict or set(raw) != _FABRIC_KEYS:
        raise ValueError(f"{expected_mode}: fabric evidence keys are not exact")
    if raw["mode"] != expected_mode:
        raise ValueError(f"{expected_mode}: fabric evidence mode mismatch")
    if (
        type(raw["observation_id"]) is not str
        or not raw["observation_id"]
        or (
            expected_observation_id is not None
            and raw["observation_id"] != expected_observation_id
        )
    ):
        raise ValueError(f"{expected_mode}: observation_id binding is invalid")
    try:
        raw_domain = ResourceDomain(raw["domain"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{expected_mode}: fabric evidence domain is invalid") from exc
    if (
        type(raw["scope"]) is not str
        or not raw["scope"]
        or raw["scope"] not in allowed_counter_scopes(raw_domain)
        or type(raw["scope_id"]) is not str
        or not raw["scope_id"]
        or raw["intervention_id"] != expected_mode
    ):
        raise ValueError(f"{expected_mode}: fabric counter scope/intervention binding is invalid")
    expected_evidence = (
        "observational"
        if expected_mode in {"open_combined", "combined"}
        else "interventional"
    )
    if raw["evidence"] != expected_evidence:
        raise ValueError(f"{expected_mode}: fabric evidence must be {expected_evidence}")
    if raw["collective_slice"] not in COLLECTIVE_SLICES:
        raise ValueError(f"{expected_mode}: invalid collective slice")
    if raw["traffic_origin"] not in FABRIC_SPLITS:
        raise ValueError(f"{expected_mode}: invalid fabric split")
    if type(raw["counter_samples"]) is not int or raw["counter_samples"] < 2:
        raise ValueError(f"{expected_mode}: at least two fabric counter samples are required")
    raw_series = raw["counter_series"]
    if type(raw_series) is not list or len(raw_series) != raw["counter_samples"]:
        raise ValueError(f"{expected_mode}: counter series/count mismatch")
    snapshots: list[CounterSnapshot] = []
    for item in raw_series:
        if type(item) is not dict or set(item) != {
            "observation_id", "domain", "sample_id", "source", "timestamp_ns", "cumulative_bytes",
            "cumulative_busy_ns", "support",
        }:
            raise ValueError(f"{expected_mode}: fabric counter snapshot keys are not exact")
        try:
            snapshot = CounterSnapshot(
                domain=ResourceDomain(item["domain"]),
                sample_id=item["sample_id"],
                source=item["source"],
                timestamp_ns=item["timestamp_ns"],
                cumulative_bytes=item["cumulative_bytes"],
                cumulative_busy_ns=item["cumulative_busy_ns"],
                support=CounterSupport(item["support"]),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"{expected_mode}: invalid fabric counter snapshot: {exc}") from exc
        snapshots.append(snapshot)
    if any(item.domain.value != raw["domain"] for item in snapshots):
        raise ValueError(f"{expected_mode}: fabric counter domain mismatch")
    if any(item["observation_id"] != raw["observation_id"] for item in raw_series):
        raise ValueError(f"{expected_mode}: fabric counter observation_id mismatch")
    if any(item.source != raw["source"] for item in snapshots):
        raise ValueError(f"{expected_mode}: fabric counter source is not bound to evidence source")
    if any(item.support is not CounterSupport.SUPPORTED for item in snapshots):
        raise ValueError(f"{expected_mode}: fabric counter series is not supported")
    validate_counter_series(snapshots)
    try:
        record = DomainEvidence(
            domain=ResourceDomain(raw["domain"]),
            mode=expected_mode,
            foreground_kind="fsdp_collective",
            auxiliary_kind="checkpoint_flow",
            overlapping_bytes=raw["overlapping_bytes"],
            overlap_ns=raw["overlap_ns"],
            tail_delta_ns=raw["tail_delta_ns"],
            evidence=EvidenceLevel(raw["evidence"]),
            counter_support=CounterSupport(raw["counter_support"]),
            path_status=PathStatus(raw["path_status"]),
            uncertainty_ns=raw["uncertainty_ns"],
            source=raw["source"],
            path_evidence=raw["path_evidence"],
            counter_family=raw["counter_family"],
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"{expected_mode}: invalid fabric evidence: {exc}") from exc
    if record.path_status is not PathStatus.OBSERVED or record.counter_support is not CounterSupport.SUPPORTED:
        raise ValueError(f"{expected_mode}: fabric evidence must be observed/supported")
    contract = domain_contract(record.domain)
    if record.path_evidence != contract.path_evidence:
        raise ValueError(
            f"{expected_mode}: path evidence does not match the domain contract"
        )
    if record.counter_family != contract.counter_family:
        raise ValueError(
            f"{expected_mode}: counter family does not match the domain contract"
        )
    return record


def validate_g2_result(result: dict[str, object]) -> dict[str, object]:
    if type(result) is not dict or set(result) != _RESULT_KEYS:
        raise ValueError("G2 result keys are not exact")
    if result["schema_version"] != "tempo-rd-g2-result-5":
        raise ValueError("unsupported G2 result schema")
    if result["evidence_state"] != "live_observed":
        raise ValueError("G2 result must be live_observed")
    if result["world_size"] != 8 or result["nodes"] != 2:
        raise ValueError("G2 result must be two nodes and eight ranks")
    _hex(result["source_bundle_sha256"], "source_bundle_sha256")
    for key in ("state_bytes_per_rank", "deadline_ns"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"{key} must be a positive int")
    steps = result["checkpoint_steps"]
    if type(steps) is not list or not steps or any(type(step) is not int for step in steps):
        raise ValueError("checkpoint_steps must be a non-empty integer list")
    if steps != sorted(set(steps)):
        raise ValueError("checkpoint_steps must be sorted and unique")
    if result["collective_slices"] != list(COLLECTIVE_SLICES):
        raise ValueError("collective_slices are not exact")
    if result["fabric_splits"] != list(FABRIC_SPLITS):
        raise ValueError("fabric_splits are not exact")
    try:
        promoted = ResourceDomain(result["promoted_domain"])
        placebo = ResourceDomain(result["placebo_domain"])
    except (TypeError, ValueError) as exc:
        raise ValueError("promoted/placebo domain is invalid") from exc
    if promoted is placebo:
        raise ValueError("promoted and placebo domains must differ")

    g1_raw = result["g1_result"]
    g1_eval = validate_g1_result(g1_raw)
    # G2 is a promotion of the exact G1 experiment, not a new geometry or
    # deadline comparison.  Bind these inputs across the embedded result so a
    # caller cannot combine a valid G1 promotion with a different state size,
    # checkpoint schedule, or durability deadline.
    for key in ("state_bytes_per_rank", "deadline_ns", "checkpoint_steps"):
        if result[key] != g1_raw[key]:
            raise ValueError(f"G2 {key} does not match the embedded G1 result")
    if not g1_eval["promote_static_policy"] or promoted.value not in g1_eval["eligible_domains"]:
        raise ValueError("G2 requires a recomputed successful G1 promotion for the promoted domain")

    raw_modes = result["modes"]
    if type(raw_modes) is not dict or set(raw_modes) != set(G2_MODES):
        raise ValueError("G2 result mode set is not exact")
    expected_domains = {
        "fg_only": set(),
        "open_combined": set(FULL_PATH),
        "causal_domain_static_cap": {promoted},
        "unrelated_domain_placebo": {placebo},
        "combined": set(FULL_PATH),
    }
    metrics: list[CausalModeRecord] = []
    metric_domains: dict[str, dict[str, object]] = {}
    fabric_by_mode: dict[str, list[DomainEvidence]] = {}
    for mode in G2_MODES:
        raw_mode = raw_modes[mode]
        if type(raw_mode) is not dict or set(raw_mode) != {
            "metrics", "slice_metrics", "fabric_evidence", "observation_windows"
        }:
            raise ValueError(f"{mode}: result keys are not exact")
        domain = None
        if mode == "causal_domain_static_cap":
            domain = promoted
        elif mode == "unrelated_domain_placebo":
            domain = placebo
        # ``combined`` is the full-flow consistency replicate.  It is not an
        # intervention for the promoted domain; only the isolated static cap
        # can establish a G2 causal promotion.
        metric, domain_info = _metric(mode, raw_mode["metrics"], domain, expected_domains[mode])
        # A failed auxiliary mode makes the two-node attribution matrix
        # incomplete.  Do not let a good isolated metric or a self-attested
        # combined record hide a deadline/correctness failure.
        if mode != "fg_only" and (
            metric.deadline_met is not True or metric.correctness_met is not True
        ):
            raise ValueError(
                f"{mode}: live fabric mode must satisfy deadline and correctness"
            )
        metrics.append(metric)
        metric_domains[mode] = domain_info
        joined_windows = validate_observation_windows(
            raw_mode["observation_windows"],
            expected_mode=mode,
            expected_observation_id=domain_info["observation_id"],
            require_auxiliary=mode != "fg_only",
        )
        if not any(window.uncertainty_safe for window in joined_windows):
            raise ValueError(f"{mode}: observation overlap does not exceed uncertainty")
        metric_domains[mode]["observation_window_count"] = len(joined_windows)
        slices = raw_mode["slice_metrics"]
        if type(slices) is not dict or set(slices) != set(COLLECTIVE_SLICES):
            raise ValueError(f"{mode}: slice metric keys are not exact")
        for slice_name in COLLECTIVE_SLICES:
            item = slices[slice_name]
            if type(item) is not dict or set(item) != _SLICE_METRIC_KEYS:
                raise ValueError(f"{mode}/{slice_name}: slice metric keys are not exact")
            for key in _SLICE_METRIC_KEYS:
                if type(item[key]) is not int or item[key] < 0:
                    raise ValueError(f"{mode}/{slice_name}: slice metrics must be non-negative ints")
        raw_fabric = raw_mode["fabric_evidence"]
        if type(raw_fabric) is not list:
            raise ValueError(f"{mode}: fabric_evidence must be a list")
        evidence = [
            _fabric_record(
                item,
                mode,
                expected_observation_id=domain_info["observation_id"],
            )
            for item in raw_fabric
        ]
        if {item.domain for item in evidence} != expected_domains[mode]:
            raise ValueError(f"{mode}: fabric evidence domains are not exact")
        if mode == "fg_only" and evidence:
            raise ValueError("fg_only must not contain fabric evidence")
        expected_pairs = {
            (collective_slice, traffic_origin)
            for collective_slice in COLLECTIVE_SLICES
            for traffic_origin in FABRIC_SPLITS
        }
        pairs_by_domain: dict[ResourceDomain, set[tuple[str, str]]] = {}
        for raw, item in zip(raw_fabric, evidence):
            pairs_by_domain.setdefault(item.domain, set()).add(
                (raw["collective_slice"], raw["traffic_origin"])
            )
        if mode != "fg_only" and (
            {item.domain for item in evidence} != expected_domains[mode]
            or len(raw_fabric) != len(expected_domains[mode]) * len(expected_pairs)
            or any(pairs_by_domain.get(domain, set()) != expected_pairs for domain in expected_domains[mode])
        ):
            raise ValueError(f"{mode}: fabric slice/origin Cartesian coverage is incomplete")
        if mode == "causal_domain_static_cap" and not any(
            item.tail_delta_ns > item.uncertainty_ns for item in evidence
        ):
            raise ValueError(f"{mode}: intervention tail delta does not exceed uncertainty")
        observed_exposure: dict[ResourceDomain, int] = {}
        for item in evidence:
            observed_exposure[item.domain] = max(
                observed_exposure.get(item.domain, 0), item.overlap_ns
            )
        if dict(metric.domain_exposure_ns or {}) != observed_exposure:
            raise ValueError(f"{mode}: domain_exposure_ns is not derived from fabric evidence")
        fabric_by_mode[mode] = evidence

    # All G2 modes must execute the same foreground collective footprint. A
    # changed foreground route is a different experiment, not evidence that
    # the promoted fabric domain was orchestrated more safely.
    foreground_reference = tuple(g1_raw["foreground_path"]["domains"])
    for mode, info in metric_domains.items():
        if tuple(info["foreground_domains"]) != foreground_reference:
            raise ValueError(
                f"{mode}: foreground_domains do not match fg_only matched workload"
            )

    # Do not accept a nominal tail improvement that increases the duration or
    # number of foreground groups exposed to the auxiliary flow.  This is the
    # two-node form of the Pareto/no-bottleneck-shift condition; the previous
    # gate only bounded tail/skew regressions and could still promote a longer
    # active interval.
    open_aux = metric_domains["open_combined"]
    for mode in ("causal_domain_static_cap", "unrelated_domain_placebo", "combined"):
        candidate_aux = metric_domains[mode]
        if candidate_aux["active_exposure_ns"] > open_aux["active_exposure_ns"]:
            raise ValueError(f"{mode}: active exposure exceeds optimized-open")
        if candidate_aux["active_groups"] > open_aux["active_groups"]:
            raise ValueError(f"{mode}: active group count exceeds optimized-open")

    promotion = evaluate_causal_matrix(metrics)
    if placebo in promotion.eligible_domains:
        raise ValueError("unrelated-domain placebo improved over open")
    if promoted not in promotion.eligible_domains:
        raise ValueError("promoted domain does not improve both tail and skew over open")
    return {
        "schema_version": "tempo-rd-g2-evaluation-1",
        "status": "pass",
        "evidence_ready": True,
        "promote_static_policy": True,
        "eligible_domains": sorted(domain.value for domain in promotion.eligible_domains),
        "headroom": promotion.headroom,
        "placebo_clean": promotion.placebo_clean,
        "reasons": list(promotion.reasons),
        "fabric_modes": sorted(fabric_by_mode),
        "metric_domains": metric_domains,
        "live_external_execution": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_g2_result(json.loads(args.result.read_text(encoding="utf-8")))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
