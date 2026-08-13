#!/usr/bin/env python3
"""Fail-closed validator for an observed one-node TEMPO-RD G1 result.

The input is a result record, not the design manifest.  It must contain
observed path/counter evidence for every traversed domain and paired
foreground/open/intervention metrics.  Promotion is recomputed locally; a
serialized self-attested promotion field is never trusted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

try:
    from tempo.causal_gate import CausalModeRecord
    from tempo.domain_counters import CounterSnapshot, validate_counter_series
    from tempo.domain_evidence import CounterSupport, DomainEvidence, PathStatus
    from tempo.resource_domain import (
        EvidenceLevel,
        ResourceDomain,
        allowed_counter_scopes,
        domain_contract,
    )
    from tempo.foreground_path import validate_foreground_path
    from tempo.observation_window import validate_observation_windows
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tempo.causal_gate import CausalModeRecord
    from tempo.domain_counters import CounterSnapshot, validate_counter_series
    from tempo.domain_evidence import CounterSupport, DomainEvidence, PathStatus
    from tempo.resource_domain import (
        EvidenceLevel,
        ResourceDomain,
        allowed_counter_scopes,
        domain_contract,
    )
    from tempo.foreground_path import validate_foreground_path
    from tempo.observation_window import validate_observation_windows

from tempo.tier_attribution import evaluate_tier_attribution, mode_spec


G1_MODES = ("fg_only", "open_combined", "d2h_only", "persist_only", "combined")
_RESULT_KEYS = {
    "schema_version",
    "evidence_state",
    "world_size",
    "nodes",
    "source_bundle_sha256",
    "host_pressure_raw_digest",
    "state_bytes_per_rank",
    "logical_file_extent_bytes",
    "deadline_ns",
    "checkpoint_steps",
    "foreground_path",
    "modes",
    "placebo",
}
_METRIC_KEYS = {
    "observation_id",
    "domain",
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
_EVIDENCE_KEYS = {
    "observation_id",
    "domain",
    "mode",
    "foreground_kind",
    "auxiliary_kind",
    "overlapping_bytes",
    "overlap_ns",
    "tail_delta_ns",
    "evidence",
    "counter_support",
    "path_status",
    "uncertainty_ns",
    "source",
    "path_evidence",
    "counter_family",
    "scope",
    "scope_id",
    "intervention_id",
}
_COUNTER_KEYS = {
    "observation_id",
    "domain",
    "sample_id",
    "source",
    "timestamp_ns",
    "cumulative_bytes",
    "cumulative_busy_ns",
    "support",
    "scope",
    "scope_id",
    "intervention_id",
}


def _strict_hex(value: object, name: str) -> None:
    if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")


def _parse_evidence(raw: object) -> DomainEvidence:
    if type(raw) is not dict or set(raw) != _EVIDENCE_KEYS:
        raise ValueError("G1 evidence record keys are not exact")
    try:
        if type(raw["observation_id"]) is not str or not raw["observation_id"]:
            raise ValueError("observation_id is invalid")
        domain = ResourceDomain(raw["domain"])
        if (
            type(raw["scope"]) is not str
            or not raw["scope"]
            or raw["scope"] not in allowed_counter_scopes(domain)
            or type(raw["scope_id"]) is not str
            or not raw["scope_id"]
            or raw["intervention_id"] != raw["mode"]
        ):
            raise ValueError("scope/intervention binding is invalid")
        record = DomainEvidence(
            domain=domain,
            mode=raw["mode"],
            foreground_kind=raw["foreground_kind"],
            auxiliary_kind=raw["auxiliary_kind"],
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
        raise ValueError(f"invalid G1 evidence record: {exc}") from exc
    contract = domain_contract(record.domain)
    if record.path_evidence != contract.path_evidence:
        raise ValueError(
            f"{record.domain.value}: path evidence does not match the domain contract"
        )
    if record.counter_family != contract.counter_family:
        raise ValueError(
            f"{record.domain.value}: counter family does not match the domain contract"
        )
    return record


def _parse_counter(
    raw: object,
    *,
    expected_mode: str,
    expected_scope: str | None = None,
    expected_scope_id: str | None = None,
    expected_observation_id: str | None = None,
) -> CounterSnapshot:
    if type(raw) is not dict or set(raw) != _COUNTER_KEYS:
        raise ValueError("G1 counter snapshot keys are not exact")
    try:
        domain = ResourceDomain(raw["domain"])
        if (
            type(raw["observation_id"]) is not str
            or not raw["observation_id"]
            or
            type(raw["scope"]) is not str
            or not raw["scope"]
            or raw["scope"] not in allowed_counter_scopes(domain)
            or type(raw["scope_id"]) is not str
            or not raw["scope_id"]
            or raw["intervention_id"] != expected_mode
            or (expected_scope is not None and raw["scope"] != expected_scope)
            or (expected_scope_id is not None and raw["scope_id"] != expected_scope_id)
            or (expected_observation_id is not None and raw["observation_id"] != expected_observation_id)
        ):
            raise ValueError("scope/intervention binding is invalid")
        return CounterSnapshot(
            domain=domain,
            sample_id=raw["sample_id"],
            source=raw["source"],
            timestamp_ns=raw["timestamp_ns"],
            cumulative_bytes=raw["cumulative_bytes"],
            cumulative_busy_ns=raw["cumulative_busy_ns"],
            support=CounterSupport(raw["support"]),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid G1 counter snapshot: {exc}") from exc


def _intervention_domains(mode: str) -> tuple[ResourceDomain, ...]:
    if mode == "d2h_only":
        return (
            ResourceDomain.GPU_LOCAL,
            ResourceDomain.PCIE_HOST,
            ResourceDomain.HOST_NUMA,
        )
    if mode == "persist_only":
        return (
            ResourceDomain.NIC_FABRIC,
            ResourceDomain.SLINGSHOT_FABRIC,
            ResourceDomain.PERSISTENT_ENDPOINT,
        )
    return ()


def _parse_domain_list(raw: object, name: str, *, allow_empty: bool) -> tuple[ResourceDomain, ...]:
    """Parse a canonical, explicit resource-domain list.

    G1 must not infer a foreground footprint from the mode name.  The list is
    therefore part of every live metric record; ``shared_domains`` is checked
    separately against the declared auxiliary route below.
    """
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


def _parse_metric(mode: str, raw: object) -> tuple[CausalModeRecord, dict[str, object]]:
    if type(raw) is not dict or set(raw) != _METRIC_KEYS:
        raise ValueError(f"{mode}: metric keys are not exact")
    raw_domain = raw["domain"]
    if type(raw["observation_id"]) is not str or not raw["observation_id"]:
        raise ValueError(f"{mode}: observation_id is invalid")
    metric_domain: ResourceDomain | None
    if mode in {"fg_only", "open_combined", "combined", "host_pressure"}:
        if raw_domain is not None:
            raise ValueError(f"{mode}: baseline/placebo metric domain must be null")
        metric_domain = None
    else:
        try:
            metric_domain = ResourceDomain(raw_domain)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{mode}: metric domain must name an explicit intervention domain") from exc
        if metric_domain not in _intervention_domains(mode):
            raise ValueError(f"{mode}: metric domain is not on the isolated route")
    record = CausalModeRecord(
        mode=mode,
        # The intervention domain is supplied by measured path/counter
        # attribution.  Never infer PCIe (or any other hop) merely from the
        # name of a multi-hop isolated mode.
        domain=metric_domain,
        tail_p99_ns=raw["tail_p99_ns"],
        skew_p99_ns=raw["skew_p99_ns"],
        deadline_met=raw["deadline_met"],
        correctness_met=raw["correctness_met"],
        samples=raw["samples"],
        domain_exposure_ns=_parse_domain_exposure(raw["domain_exposure_ns"], mode),
    )
    for name in ("active_exposure_ns", "active_groups"):
        if type(raw[name]) is not int or raw[name] < 0:
            raise ValueError(f"{mode}: {name} must be a non-negative int")
    foreground_domains = _parse_domain_list(
        raw["foreground_domains"], f"{mode}: foreground_domains", allow_empty=False
    )
    auxiliary_domains = tuple(mode_spec(mode).auxiliary_domains)
    shared_domains = _parse_domain_list(
        raw["shared_domains"], f"{mode}: shared_domains", allow_empty=True
    )
    expected_shared = tuple(
        sorted(
            (domain for domain in auxiliary_domains if domain in foreground_domains),
            key=lambda item: item.value,
        )
    )
    if shared_domains != expected_shared:
        raise ValueError(
            f"{mode}: shared_domains must equal foreground/auxiliary route intersection"
        )
    return record, {
        "observation_id": raw["observation_id"],
        "domain": None if metric_domain is None else metric_domain.value,
        "foreground_domains": [domain.value for domain in foreground_domains],
        "shared_domains": [domain.value for domain in shared_domains],
        **{name: raw[name] for name in ("active_exposure_ns", "active_groups")},
        "domain_exposure_ns": {
            domain.value: value for domain, value in sorted(
                (record.domain_exposure_ns or {}).items(), key=lambda item: item[0].value
            )
        },
    }


def validate_g1_result(result: dict[str, object]) -> dict[str, object]:
    if type(result) is not dict or set(result) != _RESULT_KEYS:
        raise ValueError("G1 result keys are not exact")
    if result["schema_version"] != "tempo-rd-g1-result-5":
        raise ValueError("unsupported G1 result schema")
    if result["evidence_state"] != "live_observed":
        raise ValueError("G1 result must be live_observed, not design_only")
    if result["world_size"] != 4 or result["nodes"] != 1:
        raise ValueError("G1 result must be one node and four ranks")
    _strict_hex(result["source_bundle_sha256"], "source_bundle_sha256")
    _strict_hex(result["host_pressure_raw_digest"], "host_pressure_raw_digest")
    for key in ("state_bytes_per_rank", "logical_file_extent_bytes", "deadline_ns"):
        value = result[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive int")
    if result["logical_file_extent_bytes"] < result["state_bytes_per_rank"]:
        raise ValueError("logical file extent must cover state bytes")
    steps = result["checkpoint_steps"]
    if type(steps) is not list or not steps or any(type(step) is not int for step in steps):
        raise ValueError("checkpoint_steps must be a non-empty integer list")
    if steps != sorted(set(steps)):
        raise ValueError("checkpoint_steps must be sorted and unique")
    foreground_path = validate_foreground_path(result["foreground_path"])

    raw_modes = result["modes"]
    if type(raw_modes) is not dict or set(raw_modes) != set(G1_MODES):
        raise ValueError("G1 result mode set is not exact")
    evidence_by_mode: dict[str, list[DomainEvidence]] = {}
    metrics: list[CausalModeRecord] = []
    metric_aux: dict[str, dict[str, object]] = {}
    for mode in G1_MODES:
        raw_mode = raw_modes[mode]
        if type(raw_mode) is not dict or set(raw_mode) != {
            "metrics", "evidence", "counters", "observation_windows"
        }:
            raise ValueError(f"{mode}: result keys are not exact")
        metric, aux = _parse_metric(mode, raw_mode["metrics"])
        # A live causal matrix is not structurally complete if an auxiliary
        # flow failed its deadline or correctness contract.  The causal gate
        # may reject a valid intervention for lack of benefit, but it must
        # never promote a matrix whose combined path silently failed.
        if mode != "fg_only" and (
            metric.deadline_met is not True or metric.correctness_met is not True
        ):
            raise ValueError(
                f"{mode}: live tier mode must satisfy deadline and correctness"
            )
        metrics.append(metric)
        metric_aux[mode] = aux
        raw_evidence = raw_mode["evidence"]
        if type(raw_evidence) is not list:
            raise ValueError(f"{mode}: evidence must be a list")
        evidence = [_parse_evidence(item) for item in raw_evidence]
        evidence_scope = {
            item["domain"]: (item["scope"], item["scope_id"])
            for item in raw_evidence
        }
        observation_id = aux["observation_id"]
        joined_windows = validate_observation_windows(
            raw_mode["observation_windows"],
            expected_mode=mode,
            expected_observation_id=observation_id,
            require_auxiliary=mode != "fg_only",
        )
        if not any(window.uncertainty_safe for window in joined_windows):
            raise ValueError(f"{mode}: observation overlap does not exceed uncertainty")
        aux["observation_window_count"] = len(joined_windows)
        aux["observation_overlap_ns"] = sum(window.overlap_ns for window in joined_windows)
        if any(item["observation_id"] != observation_id for item in raw_evidence):
            raise ValueError(f"{mode}: evidence observation_id does not match metrics")
        if mode == "fg_only":
            if evidence:
                raise ValueError("fg_only must not claim auxiliary path evidence")
        else:
            for item in evidence:
                if item.path_status is not PathStatus.OBSERVED or item.counter_support is not CounterSupport.SUPPORTED:
                    raise ValueError(f"{mode}: path/counters must be observed and supported")
                if item.evidence is not EvidenceLevel.INTERVENTIONAL and mode not in {"open_combined", "combined"}:
                    raise ValueError(f"{mode}: intervention evidence must be interventional")
            expected = tuple(mode_spec(mode).auxiliary_domains)
            observed = tuple(item.domain for item in evidence)
            if observed != expected:
                raise ValueError(f"{mode}: evidence does not cover the exact ordered traversed path")
            metric_domain = metric.domain
            if metric_domain is not None:
                matched = [item for item in evidence if item.domain is metric_domain]
                if not matched or not any(item.tail_delta_ns > item.uncertainty_ns for item in matched):
                    raise ValueError(
                        f"{mode}: intervention tail delta does not exceed uncertainty for metric domain"
                    )
        observed_exposure: dict[ResourceDomain, int] = {}
        for item in evidence:
            observed_exposure[item.domain] = max(
                observed_exposure.get(item.domain, 0), item.overlap_ns
            )
        if dict(metric.domain_exposure_ns or {}) != observed_exposure:
            raise ValueError(f"{mode}: domain_exposure_ns is not derived from evidence")
        evidence_by_mode[mode] = evidence

        raw_counters = raw_mode["counters"]
        if type(raw_counters) is not dict:
            raise ValueError(f"{mode}: counters must be a domain mapping")
        expected_domains = {item.domain for item in evidence}
        if set(raw_counters) != {domain.value for domain in expected_domains}:
            raise ValueError(f"{mode}: counters do not match observed evidence domains")
        evidence_sources = {item.domain.value: item.source for item in evidence}
        for domain_name, raw_series in raw_counters.items():
            if type(raw_series) is not list or len(raw_series) < 2:
                raise ValueError(f"{mode}/{domain_name}: at least two counter snapshots are required")
            snapshots = [
                _parse_counter(
                    item,
                    expected_mode=mode,
                    expected_scope=evidence_scope[domain_name][0],
                    expected_scope_id=evidence_scope[domain_name][1],
                    expected_observation_id=observation_id,
                )
                for item in raw_series
            ]
            if any(item.domain.value != domain_name for item in snapshots):
                raise ValueError(f"{mode}/{domain_name}: counter domain mismatch")
            if any(item.support is not CounterSupport.SUPPORTED for item in snapshots):
                raise ValueError(f"{mode}/{domain_name}: live evidence requires supported counters")
            if any(item.source != evidence_sources[domain_name] for item in snapshots):
                raise ValueError(f"{mode}/{domain_name}: counter source is not bound to evidence source")
            validate_counter_series(snapshots)

    # The foreground workload is the matched control, not an intervention
    # knob. Every auxiliary mode and the host-pressure placebo must carry the
    # exact same declared foreground footprint as fg_only; otherwise changing
    # the workload route could masquerade as a domain-causal improvement.
    foreground_reference = tuple(foreground_path["domains"])
    for mode, aux in metric_aux.items():
        if tuple(aux["foreground_domains"]) != foreground_reference:
            raise ValueError(
                f"{mode}: foreground_domains do not match fg_only matched workload"
            )

    # A host-pressure placebo is mandatory for causal attribution.  It uses
    # the same observed/supported path contract as an intervention, but its
    # metric domain is deliberately null so it can only disqualify
    # attribution; it can never become an eligible controller domain.
    raw_placebo = result["placebo"]
    if type(raw_placebo) is not dict or set(raw_placebo) != {
        "metrics", "evidence", "counters", "observation_windows"
    }:
        raise ValueError("host_pressure placebo keys are not exact")
    placebo_metric, placebo_aux = _parse_metric("host_pressure", raw_placebo["metrics"])
    placebo_windows = validate_observation_windows(
        raw_placebo["observation_windows"],
        expected_mode="host_pressure",
        expected_observation_id=placebo_aux["observation_id"],
        require_auxiliary=True,
    )
    if not any(window.uncertainty_safe for window in placebo_windows):
        raise ValueError("host_pressure: observation overlap does not exceed uncertainty")
    placebo_aux["observation_window_count"] = len(placebo_windows)
    placebo_aux["observation_overlap_ns"] = sum(window.overlap_ns for window in placebo_windows)
    raw_placebo_evidence = raw_placebo["evidence"]
    if type(raw_placebo_evidence) is not list:
        raise ValueError("host_pressure: evidence must be a list")
    placebo_evidence = [_parse_evidence(item) for item in raw_placebo_evidence]
    placebo_scope = (
        raw_placebo_evidence[0]["scope"],
        raw_placebo_evidence[0]["scope_id"],
    ) if raw_placebo_evidence else (None, None)
    if any(item["observation_id"] != placebo_aux["observation_id"] for item in raw_placebo_evidence):
        raise ValueError("host_pressure: evidence observation_id does not match metrics")
    if tuple(item.domain for item in placebo_evidence) != (ResourceDomain.HOST_NUMA,):
        raise ValueError("host_pressure: evidence does not cover the exact placebo path")
    if any(
        item.evidence is not EvidenceLevel.INTERVENTIONAL
        or item.path_status is not PathStatus.OBSERVED
        or item.counter_support is not CounterSupport.SUPPORTED
        for item in placebo_evidence
    ):
        raise ValueError("host_pressure: placebo path/counters must be observed and supported")
    raw_placebo_counters = raw_placebo["counters"]
    if type(raw_placebo_counters) is not dict or set(raw_placebo_counters) != {ResourceDomain.HOST_NUMA.value}:
        raise ValueError("host_pressure: counters do not match the placebo path")
    raw_series = raw_placebo_counters[ResourceDomain.HOST_NUMA.value]
    if type(raw_series) is not list or len(raw_series) < 2:
        raise ValueError("host_pressure/host_numa: at least two counter snapshots are required")
    snapshots = [
        _parse_counter(
            item,
            expected_mode="host_pressure",
            expected_scope=placebo_scope[0],
            expected_scope_id=placebo_scope[1],
            expected_observation_id=placebo_aux["observation_id"],
        )
        for item in raw_series
    ]
    if any(item.domain is not ResourceDomain.HOST_NUMA for item in snapshots):
        raise ValueError("host_pressure/host_numa: counter domain mismatch")
    if any(item.support is not CounterSupport.SUPPORTED for item in snapshots):
        raise ValueError("host_pressure/host_numa: live evidence requires supported counters")
    if any(item.source != placebo_evidence[0].source for item in snapshots):
        raise ValueError("host_pressure/host_numa: counter source is not bound to evidence source")
    validate_counter_series(snapshots)
    evidence_by_mode["host_pressure"] = placebo_evidence
    metrics.append(placebo_metric)
    metric_aux["host_pressure"] = placebo_aux

    # Live mode evidence requires observed path and supported counters.  The
    # promotion is recomputed from the supplied metrics and cannot be faked by
    # adding a promotion field to the input.
    evaluation = evaluate_tier_attribution(evidence_by_mode, metrics, require_observed=True)
    # A lower tail is not a win if the intervention keeps the foreground
    # exposed for longer or creates more active groups.  The staged objective
    # treats this as a bottleneck shift, so compare every auxiliary mode with
    # the matched open lane before allowing domain promotion.  These fields
    # are carried outside CausalModeRecord because they are diagnostics for
    # the whole flow, not scalar tail inputs.
    open_aux = metric_aux["open_combined"]
    exposure_clean = True
    exposure_reasons: list[str] = []
    for mode in ("d2h_only", "persist_only", "combined"):
        candidate_aux = metric_aux[mode]
        if candidate_aux["active_exposure_ns"] > open_aux["active_exposure_ns"]:
            exposure_clean = False
            exposure_reasons.append(
                f"{mode}: active exposure exceeds optimized-open"
            )
        if candidate_aux["active_groups"] > open_aux["active_groups"]:
            exposure_clean = False
            exposure_reasons.append(
                f"{mode}: active group count exceeds optimized-open"
            )
    if not exposure_clean:
        evaluation = type(evaluation)(
            promotion=type(evaluation.promotion)(
                eligible_domains=frozenset(),
                headroom=evaluation.promotion.headroom,
                placebo_clean=evaluation.promotion.placebo_clean,
                reasons=tuple(evaluation.promotion.reasons) + tuple(exposure_reasons),
            ),
            evidence_ready=evaluation.evidence_ready,
            reasons=tuple(evaluation.reasons) + tuple(exposure_reasons),
        )
    return {
        "schema_version": "tempo-rd-g1-evaluation-1",
        "status": "pass",
        "evidence_ready": evaluation.evidence_ready,
        "promote_static_policy": bool(
            exposure_clean and evaluation.promote_static_policy
        ),
        "eligible_domains": sorted(domain.value for domain in evaluation.promotion.eligible_domains),
        "headroom": evaluation.promotion.headroom,
        "placebo_clean": evaluation.promotion.placebo_clean,
        "reasons": list(evaluation.reasons),
        "metric_auxiliary": metric_aux,
        "live_external_execution": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_g1_result(json.loads(args.result.read_text(encoding="utf-8")))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
