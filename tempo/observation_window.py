"""Strict common-clock joins for foreground and auxiliary observations.

The resource-domain validators bind records to an observation id, but an id
alone does not prove that the foreground and auxiliary bytes were measured in
the same time interval.  This module provides the small, backend-independent
join primitive used by future G1/G2/KV collectors.  It deliberately does not
infer a route from a topology label and it never turns a non-overlap into a
causal observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable, Mapping

from tempo.resource_domain import ResourceDomain


@dataclass(frozen=True)
class ObservationInterval:
    """One interval in the shared corrected monotonic clock domain."""

    observation_id: str
    mode: str
    rank: int
    event_id: str
    clock_domain: str
    source_snapshot_id: str
    source: str
    start_ns: int
    end_ns: int
    role: str
    domain: ResourceDomain | None = None
    uncertainty_ns: int = 0

    def __post_init__(self) -> None:
        if type(self.observation_id) is not str or not self.observation_id:
            raise ValueError("observation_id must be a non-empty string")
        if type(self.mode) is not str or not self.mode:
            raise ValueError("mode must be a non-empty string")
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("rank must be a non-negative int")
        if type(self.event_id) is not str or not self.event_id:
            raise ValueError("event_id must be a non-empty string")
        if type(self.clock_domain) is not str or not self.clock_domain:
            raise ValueError("clock_domain must be a non-empty string")
        if type(self.source_snapshot_id) is not str or not self.source_snapshot_id:
            raise ValueError("source_snapshot_id must be a non-empty string")
        if type(self.source) is not str or not self.source:
            raise ValueError("source must be a non-empty string")
        if type(self.start_ns) is not int or type(self.end_ns) is not int:
            raise TypeError("interval bounds must be ints")
        if self.start_ns < 0 or self.end_ns <= self.start_ns:
            raise ValueError("interval must satisfy 0 <= start_ns < end_ns")
        if type(self.role) is not str or self.role not in {"foreground", "auxiliary", "counter"}:
            raise ValueError("role must be foreground, auxiliary, or counter")
        if self.role == "counter" and not isinstance(self.domain, ResourceDomain):
            raise ValueError("counter intervals require a resource domain")
        if self.role != "counter" and self.domain is not None:
            raise ValueError("foreground/auxiliary intervals must not name a counter domain")
        if type(self.uncertainty_ns) is not int or self.uncertainty_ns < 0:
            raise ValueError("uncertainty_ns must be a non-negative int")


@dataclass(frozen=True)
class JoinedObservationWindow:
    """The exact common interval across foreground, auxiliary, and counters."""

    observation_id: str
    mode: str
    rank: int
    event_id: str
    clock_domain: str
    source_snapshot_id: str
    start_ns: int
    end_ns: int
    overlap_ns: int
    uncertainty_ns: int
    counter_domains: tuple[ResourceDomain, ...]

    @property
    def uncertainty_safe(self) -> bool:
        """Whether the common overlap exceeds the declared uncertainty."""

        return self.overlap_ns > self.uncertainty_ns


WINDOW_KEYS = frozenset(
    {
        "observation_id",
        "mode",
        "rank",
        "event_id",
        "clock_domain",
        "source_snapshot_id",
        "source",
        "start_ns",
        "end_ns",
        "role",
        "domain",
        "uncertainty_ns",
    }
)

# The joined record is deliberately a different schema from the raw
# intervals.  It contains only the intersection that was actually proven by
# the supplied foreground/auxiliary/counter intervals; it never manufactures
# a duration from a logical stage timestamp or a topology label.
JOINED_WINDOW_KEYS = frozenset(
    {
        "observation_id",
        "mode",
        "rank",
        "event_id",
        "clock_domain",
        "source_snapshot_id",
        "start_ns",
        "end_ns",
        "overlap_ns",
        "uncertainty_ns",
        "uncertainty_safe",
        "counter_domains",
    }
)


def observation_window_contract() -> dict[str, object]:
    """Return the canonical manifest contract for interval collection."""

    return {
        "schema_version": "tempo-rd-observation-window-1",
        "clock_domain": "corrected_monotonic_ns",
        "interval_keys": sorted(WINDOW_KEYS),
        "joined_keys": sorted(JOINED_WINDOW_KEYS),
        "roles": ["auxiliary", "counter", "foreground"],
        "join_keys": [
            "observation_id",
            "mode",
            "rank",
            "event_id",
            "clock_domain",
            "source_snapshot_id",
        ],
        "uncertainty_rule": "common_overlap_ns_gt_uncertainty_ns",
        "counter_required": True,
        "auxiliary_required_for_non_foreground": True,
    }


def serialize_observation_interval(interval: ObservationInterval) -> dict[str, object]:
    """Return one raw interval in the exact JSON interchange shape.

    This is intentionally a serializer, not a coercing adapter: callers must
    construct an :class:`ObservationInterval` first, so malformed hardware or
    clock records fail before they can enter a result artifact.
    """

    if not isinstance(interval, ObservationInterval):
        raise TypeError("interval must be an ObservationInterval")
    return {
        "observation_id": interval.observation_id,
        "mode": interval.mode,
        "rank": interval.rank,
        "event_id": interval.event_id,
        "clock_domain": interval.clock_domain,
        "source_snapshot_id": interval.source_snapshot_id,
        "source": interval.source,
        "start_ns": interval.start_ns,
        "end_ns": interval.end_ns,
        "role": interval.role,
        "domain": None if interval.domain is None else interval.domain.value,
        "uncertainty_ns": interval.uncertainty_ns,
    }


def serialize_joined_observation_window(window: JoinedObservationWindow) -> dict[str, object]:
    """Return a deterministic, source-bound joined-window record."""

    if not isinstance(window, JoinedObservationWindow):
        raise TypeError("window must be a JoinedObservationWindow")
    return {
        "observation_id": window.observation_id,
        "mode": window.mode,
        "rank": window.rank,
        "event_id": window.event_id,
        "clock_domain": window.clock_domain,
        "source_snapshot_id": window.source_snapshot_id,
        "start_ns": window.start_ns,
        "end_ns": window.end_ns,
        "overlap_ns": window.overlap_ns,
        "uncertainty_ns": window.uncertainty_ns,
        "uncertainty_safe": window.uncertainty_safe,
        "counter_domains": [domain.value for domain in window.counter_domains],
    }


def canonicalize_observation_windows(
    raw: object,
    *,
    expected_mode: str,
    expected_observation_id: str,
    require_auxiliary: bool,
) -> list[dict[str, object]]:
    """Validate and materialize joined windows in deterministic order.

    The returned records are suitable for a sidecar or digest.  They are
    derived only from explicit intervals and therefore cannot be mistaken for
    a hardware counter collection.  A non-uncertainty-safe join is retained
    with ``uncertainty_safe=false`` so the analyzer can report it, while live
    causal promotion may reject it.
    """

    joined = validate_observation_windows(
        raw,
        expected_mode=expected_mode,
        expected_observation_id=expected_observation_id,
        require_auxiliary=require_auxiliary,
    )
    records = [serialize_joined_observation_window(window) for window in joined]
    records.sort(key=lambda item: (item["rank"], item["event_id"]))
    return records


def parse_observation_interval(raw: object) -> ObservationInterval:
    """Parse one source-bound JSON interval without coercing types."""

    if type(raw) is not dict or set(raw) != WINDOW_KEYS:
        raise ValueError("observation interval keys are not exact")
    domain_raw = raw["domain"]
    if domain_raw is not None:
        try:
            domain = ResourceDomain(domain_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("observation interval domain is invalid") from exc
    else:
        domain = None
    try:
        return ObservationInterval(
            observation_id=raw["observation_id"],
            mode=raw["mode"],
            rank=raw["rank"],
            event_id=raw["event_id"],
            clock_domain=raw["clock_domain"],
            source_snapshot_id=raw["source_snapshot_id"],
            source=raw["source"],
            start_ns=raw["start_ns"],
            end_ns=raw["end_ns"],
            role=raw["role"],
            domain=domain,
            uncertainty_ns=raw["uncertainty_ns"],
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid observation interval: {exc}") from exc


def validate_observation_windows(
    raw: object,
    *,
    expected_mode: str,
    expected_observation_id: str,
    require_auxiliary: bool,
) -> tuple[JoinedObservationWindow, ...]:
    """Validate and join JSON intervals grouped by rank and event.

    Each rank/event group must contain exactly one foreground interval, at least
    one measured counter interval, and (for auxiliary modes) exactly one
    auxiliary interval.  The returned windows are never synthesized from
    topology or aggregate metrics.
    """

    if type(raw) is not list or not raw:
        raise ValueError("observation_windows must be a non-empty list")
    intervals: list[ObservationInterval] = []
    for item in raw:
        interval = parse_observation_interval(item)
        if interval.mode != expected_mode:
            raise ValueError("observation interval mode does not match metrics")
        if interval.observation_id != expected_observation_id:
            raise ValueError("observation interval observation_id does not match metrics")
        intervals.append(interval)

    grouped: dict[tuple[int, str], list[ObservationInterval]] = defaultdict(list)
    for interval in intervals:
        grouped[(interval.rank, interval.event_id)].append(interval)
    joined: list[JoinedObservationWindow] = []
    for key in sorted(grouped):
        values = grouped[key]
        foreground = [item for item in values if item.role == "foreground"]
        auxiliary = [item for item in values if item.role == "auxiliary"]
        counters = [item for item in values if item.role == "counter"]
        if len(foreground) != 1:
            raise ValueError("each rank/event requires exactly one foreground interval")
        if require_auxiliary and len(auxiliary) != 1:
            raise ValueError("auxiliary mode requires exactly one auxiliary interval")
        if not require_auxiliary and auxiliary:
            raise ValueError("foreground-only mode must not contain auxiliary intervals")
        if not counters:
            raise ValueError("each rank/event requires a counter interval")
        try:
            joined.append(join_observation_window(foreground[0], auxiliary[0] if auxiliary else None, counters))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rank/event {key} observation join is invalid: {exc}") from exc
    if not joined:
        raise ValueError("observation_windows produced no joins")
    return tuple(joined)


def join_observation_window(
    foreground: ObservationInterval,
    auxiliary: ObservationInterval | None,
    counters: Iterable[ObservationInterval],
) -> JoinedObservationWindow:
    """Join intervals only when identity and clock provenance agree.

    ``auxiliary=None`` is reserved for foreground-only/placebo observations.
    Every join still requires at least one counter interval so a logical timing
    record cannot be promoted without a measured domain series.  Counter
    intervals may have different domain labels, but they must share the same
    event, source snapshot, and corrected clock domain.
    """

    if not isinstance(foreground, ObservationInterval) or foreground.role != "foreground":
        raise TypeError("foreground must be a foreground ObservationInterval")
    if auxiliary is not None and (
        not isinstance(auxiliary, ObservationInterval) or auxiliary.role != "auxiliary"
    ):
        raise TypeError("auxiliary must be an auxiliary ObservationInterval or None")
    counter_values = tuple(counters)
    if not counter_values or any(
        not isinstance(item, ObservationInterval) or item.role != "counter"
        for item in counter_values
    ):
        raise ValueError("at least one counter interval is required")

    intervals = (foreground,) + ((auxiliary,) if auxiliary is not None else ()) + counter_values
    identity = (
        foreground.observation_id,
        foreground.mode,
        foreground.rank,
        foreground.event_id,
        foreground.clock_domain,
        foreground.source_snapshot_id,
    )
    for item in intervals[1:]:
        current = (
            item.observation_id,
            item.mode,
            item.rank,
            item.event_id,
            item.clock_domain,
            item.source_snapshot_id,
        )
        if current != identity:
            raise ValueError("observation intervals do not share identity/clock provenance")

    start_ns = max(item.start_ns for item in intervals)
    end_ns = min(item.end_ns for item in intervals)
    if end_ns <= start_ns:
        raise ValueError("foreground and auxiliary/counter intervals do not overlap")
    domains = tuple(sorted({item.domain for item in counter_values}, key=lambda item: item.value))
    return JoinedObservationWindow(
        observation_id=foreground.observation_id,
        mode=foreground.mode,
        rank=foreground.rank,
        event_id=foreground.event_id,
        clock_domain=foreground.clock_domain,
        source_snapshot_id=foreground.source_snapshot_id,
        start_ns=start_ns,
        end_ns=end_ns,
        overlap_ns=end_ns - start_ns,
        uncertainty_ns=max(item.uncertainty_ns for item in intervals),
        counter_domains=domains,
    )
