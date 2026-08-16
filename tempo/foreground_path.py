"""Strict observed foreground-route evidence for TEMPO-RD causal claims.

An auxiliary counter cannot establish that the foreground operation used the
same resource domain.  This contract is therefore kept separate from the
auxiliary-flow evidence: every declared foreground domain must have an
observed, scope-bound monotonic hardware counter series with positive traffic.
Topology labels and a copied ``foreground_domains`` list are not sufficient.
"""

from __future__ import annotations

from typing import Any

from tempo.domain_counters import CounterSnapshot, validate_counter_series
from tempo.domain_evidence import CounterSupport, PathStatus
from tempo.resource_domain import ResourceDomain, allowed_counter_scopes, domain_contract


FOREGROUND_PATH_KEYS = frozenset(
    {
        "domains",
        "path_status",
        "counter_support",
        "path_evidence",
        "counter_family",
        "counters",
    }
)
FOREGROUND_COUNTER_KEYS = frozenset(
    {
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
)


def validate_foreground_path(raw: object, *, intervention_id: str = "fg_only") -> dict[str, Any]:
    """Validate and normalize a live foreground path evidence object."""

    if type(raw) is not dict or set(raw) != FOREGROUND_PATH_KEYS:
        raise ValueError("foreground path evidence keys are not exact")
    domains_raw = raw["domains"]
    if type(domains_raw) is not list or not domains_raw:
        raise ValueError("foreground path domains must be a non-empty list")
    if any(type(item) is not str for item in domains_raw):
        raise ValueError("foreground path domains must contain strings")
    try:
        domains = tuple(ResourceDomain(item) for item in domains_raw)
    except ValueError as exc:
        raise ValueError("foreground path contains an unknown domain") from exc
    if len(set(domains)) != len(domains) or list(domains_raw) != sorted(domains_raw):
        raise ValueError("foreground path domains must be unique and sorted")
    domain_names = [domain.value for domain in domains]

    for field, expected in (
        ("path_status", PathStatus.OBSERVED.value),
        ("counter_support", CounterSupport.SUPPORTED.value),
    ):
        values = raw[field]
        if type(values) is not dict or set(values) != set(domain_names):
            raise ValueError(f"foreground path {field} keys are not exact")
        if any(type(values[name]) is not str or values[name] != expected for name in domain_names):
            raise ValueError(f"foreground path {field} must be observed/supported")

    for field in ("path_evidence", "counter_family"):
        values = raw[field]
        if type(values) is not dict or set(values) != set(domain_names):
            raise ValueError(f"foreground path {field} keys are not exact")
        for domain in domains:
            contract = domain_contract(domain)
            if values[domain.value] != getattr(contract, field):
                raise ValueError(f"foreground path {field} does not match {domain.value} contract")

    raw_counters = raw["counters"]
    if type(raw_counters) is not dict or list(raw_counters) != sorted(raw_counters):
        raise ValueError("foreground path counter keys are not exact/sorted")
    if set(raw_counters) != set(domain_names):
        raise ValueError("foreground path counters do not cover declared domains")

    normalized: dict[str, Any] = {
        "domains": domain_names,
        "path_status": dict(raw["path_status"]),
        "counter_support": dict(raw["counter_support"]),
        "path_evidence": dict(raw["path_evidence"]),
        "counter_family": dict(raw["counter_family"]),
        "counters": {},
    }
    for domain in domains:
        name = domain.value
        series_raw = raw_counters[name]
        if type(series_raw) is not list or len(series_raw) < 2:
            raise ValueError(f"foreground path {name} requires at least two samples")
        snapshots: list[CounterSnapshot] = []
        source: str | None = None
        scope: str | None = None
        scope_id: str | None = None
        for item in series_raw:
            if type(item) is not dict or set(item) != FOREGROUND_COUNTER_KEYS:
                raise ValueError(f"foreground path {name} counter keys are not exact")
            if item["domain"] != name or item["intervention_id"] != intervention_id:
                raise ValueError(f"foreground path {name} counter binding is invalid")
            if type(item["scope"]) is not str or item["scope"] not in allowed_counter_scopes(domain):
                raise ValueError(f"foreground path {name} counter scope is invalid")
            if type(item["scope_id"]) is not str or not item["scope_id"]:
                raise ValueError(f"foreground path {name} scope_id is invalid")
            if source is None:
                source = item["source"]
                scope = item["scope"]
                scope_id = item["scope_id"]
            elif (item["source"], item["scope"], item["scope_id"]) != (source, scope, scope_id):
                raise ValueError(f"foreground path {name} samples change source/scope")
            try:
                snapshot = CounterSnapshot(
                    domain=domain,
                    sample_id=item["sample_id"],
                    source=item["source"],
                    timestamp_ns=item["timestamp_ns"],
                    cumulative_bytes=item["cumulative_bytes"],
                    cumulative_busy_ns=item["cumulative_busy_ns"],
                    support=CounterSupport(item["support"]),
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"foreground path {name} counter sample is invalid") from exc
            if snapshot.support is not CounterSupport.SUPPORTED:
                raise ValueError(f"foreground path {name} counter is not supported")
            snapshots.append(snapshot)
        validate_counter_series(snapshots)
        if snapshots[-1].cumulative_bytes <= snapshots[0].cumulative_bytes:
            raise ValueError(f"foreground path {name} has no positive byte traffic")
        if snapshots[-1].cumulative_busy_ns <= snapshots[0].cumulative_busy_ns:
            raise ValueError(f"foreground path {name} has no positive busy interval")
        normalized["counters"][name] = [dict(item) for item in series_raw]
    return normalized
