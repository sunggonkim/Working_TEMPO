"""Fail-closed host-NUMA pressure placebo contract for TEMPO-RD.

The placebo is deliberately separate from checkpoint/KV admission.  It is a
foreground-only run with an independent host-memory pressure worker, used to
test whether a claimed auxiliary-flow effect is merely host pressure.  This
module contains only the immutable specification and counter-series checks;
it never allocates memory, launches a process, or infers NUMA placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


MIN_BUFFER_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class HostPressureSpec:
    """Exact runtime contract for one rank's placebo worker."""

    rank: int
    world_size: int
    numa_node: int
    buffer_bytes: int
    duration_ns: int
    sample_period_ns: int
    source: str = "proc_self_numa_maps_plus_touch_loop"

    def __post_init__(self) -> None:
        for name, value in (
            ("rank", self.rank),
            ("world_size", self.world_size),
            ("numa_node", self.numa_node),
            ("buffer_bytes", self.buffer_bytes),
            ("duration_ns", self.duration_ns),
            ("sample_period_ns", self.sample_period_ns),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.world_size != 4:
            raise ValueError("host-pressure placebo is fixed to the G1 four-rank screen")
        if self.rank >= self.world_size:
            raise ValueError("rank must be within world_size")
        if self.buffer_bytes < MIN_BUFFER_BYTES:
            raise ValueError("host-pressure buffer is below the frozen minimum")
        if self.duration_ns <= 0 or self.sample_period_ns <= 0:
            raise ValueError("duration and sample period must be positive")
        if self.sample_period_ns > self.duration_ns:
            raise ValueError("sample period cannot exceed duration")
        if type(self.source) is not str or not self.source:
            raise ValueError("source must be a non-empty string")


@dataclass(frozen=True)
class HostPressureSample:
    """One monotonic counter sample from the selected NUMA node."""

    sample_id: str
    timestamp_ns: int
    cumulative_touched_bytes: int
    cumulative_busy_ns: int
    numa_node_bytes: int

    def __post_init__(self) -> None:
        if type(self.sample_id) is not str or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        for name, value in (
            ("timestamp_ns", self.timestamp_ns),
            ("cumulative_touched_bytes", self.cumulative_touched_bytes),
            ("cumulative_busy_ns", self.cumulative_busy_ns),
            ("numa_node_bytes", self.numa_node_bytes),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")


def validate_host_pressure_series(
    spec: HostPressureSpec, samples: Iterable[HostPressureSample]
) -> tuple[HostPressureSample, ...]:
    """Validate a placebo series without treating it as causal evidence."""

    if not isinstance(spec, HostPressureSpec):
        raise TypeError("spec must be a HostPressureSpec")
    values = tuple(samples)
    if len(values) < 2:
        raise ValueError("host-pressure placebo requires at least two samples")
    seen: set[str] = set()
    previous: HostPressureSample | None = None
    for sample in values:
        if not isinstance(sample, HostPressureSample):
            raise TypeError("samples must contain HostPressureSample values")
        if sample.sample_id in seen:
            raise ValueError("duplicate host-pressure sample_id")
        seen.add(sample.sample_id)
        if previous is not None:
            if sample.timestamp_ns <= previous.timestamp_ns:
                raise ValueError("host-pressure timestamps must increase")
            for name in (
                "cumulative_touched_bytes",
                "cumulative_busy_ns",
                "numa_node_bytes",
            ):
                if getattr(sample, name) < getattr(previous, name):
                    raise ValueError(f"host-pressure {name} regressed")
        previous = sample
    if values[0].timestamp_ns < 0 or values[-1].timestamp_ns <= values[0].timestamp_ns:
        raise ValueError("host-pressure series has no positive interval")
    if values[-1].cumulative_touched_bytes < spec.buffer_bytes:
        raise ValueError("placebo did not touch the declared host buffer")
    if values[-1].cumulative_busy_ns <= 0:
        raise ValueError("placebo has no measured busy interval")
    return values


def host_pressure_route_is_placebo(route: tuple[object, ...]) -> bool:
    """Return true only for an empty auxiliary route.

    A host-pressure placebo must not be mislabeled as a checkpoint or KV route.
    The caller still needs an observed HOST_NUMA path/counter record.
    """

    return type(route) is tuple and len(route) == 0
