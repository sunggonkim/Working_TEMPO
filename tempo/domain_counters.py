"""Fail-closed counter snapshots for resource-domain attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tempo.domain_evidence import CounterSupport
from tempo.resource_domain import ResourceDomain


@dataclass(frozen=True)
class CounterSnapshot:
    domain: ResourceDomain
    sample_id: str
    source: str
    timestamp_ns: int
    cumulative_bytes: int
    cumulative_busy_ns: int
    support: CounterSupport

    def __post_init__(self) -> None:
        if not isinstance(self.domain, ResourceDomain):
            raise TypeError("domain must be a ResourceDomain")
        if not self.sample_id or not self.source:
            raise ValueError("sample_id and source are required")
        for name, value in (
            ("timestamp_ns", self.timestamp_ns),
            ("cumulative_bytes", self.cumulative_bytes),
            ("cumulative_busy_ns", self.cumulative_busy_ns),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if not isinstance(self.support, CounterSupport):
            raise TypeError("support must be CounterSupport")
        if self.support is not CounterSupport.SUPPORTED and (
            self.cumulative_bytes != 0 or self.cumulative_busy_ns != 0
        ):
            raise ValueError("unsupported counter snapshots must have zero values")


@dataclass(frozen=True)
class CounterDelta:
    domain: ResourceDomain
    source: str
    interval_ns: int
    bytes: int
    busy_ns: int
    support: CounterSupport
    bytes_per_second: int | None


def counter_delta(previous: CounterSnapshot, current: CounterSnapshot) -> CounterDelta:
    if previous.domain is not current.domain or previous.source != current.source:
        raise ValueError("counter snapshots must share domain and source")
    if current.timestamp_ns <= previous.timestamp_ns:
        raise ValueError("counter timestamps must increase")
    interval_ns = current.timestamp_ns - previous.timestamp_ns
    if previous.support is not CounterSupport.SUPPORTED or current.support is not CounterSupport.SUPPORTED:
        support = (
            current.support
            if current.support is not CounterSupport.SUPPORTED
            else previous.support
        )
        return CounterDelta(previous.domain, previous.source, interval_ns, 0, 0, support, None)
    if current.cumulative_bytes < previous.cumulative_bytes:
        raise ValueError("counter bytes regressed")
    if current.cumulative_busy_ns < previous.cumulative_busy_ns:
        raise ValueError("counter busy time regressed")
    bytes_delta = current.cumulative_bytes - previous.cumulative_bytes
    busy_delta = current.cumulative_busy_ns - previous.cumulative_busy_ns
    rate = (bytes_delta * 1_000_000_000 // busy_delta) if busy_delta else None
    return CounterDelta(
        previous.domain,
        previous.source,
        interval_ns,
        bytes_delta,
        busy_delta,
        CounterSupport.SUPPORTED,
        rate,
    )


def validate_counter_series(snapshots: Iterable[CounterSnapshot]) -> tuple[CounterSnapshot, ...]:
    """Validate one domain/source series and return it in supplied order."""

    values = tuple(snapshots)
    seen: set[str] = set()
    for snapshot in values:
        if not isinstance(snapshot, CounterSnapshot):
            raise TypeError("snapshots must contain CounterSnapshot values")
        if snapshot.sample_id in seen:
            raise ValueError("duplicate counter sample_id")
        seen.add(snapshot.sample_id)
    for previous, current in zip(values, values[1:]):
        counter_delta(previous, current)
    return values
