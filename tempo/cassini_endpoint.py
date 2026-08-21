"""Endpoint-scoped HPE Cassini evidence without a scalar pressure label.

The sampler reads a bounded list of explicit telemetry paths under four local
Cassini devices.  It keeps sender pause, receiver pause, host backpressure,
receive matching, ECN, and transport faults separate.  Optional hardware
counters are reported as unsupported or ambiguous rather than as zero.

No persistent watcher is created.  A vLLM endpoint or sidecar may call
``sample`` from its existing heartbeat and push the resulting snapshot to the
router asynchronously.
"""

from __future__ import annotations

import math
from pathlib import Path
import threading
import time
from typing import Any

from tempo.cassini_pressure import _counter_value
from tempo.domain_evidence import CounterSupport
from tempo.pd_endpoint_evidence import PDEndpointIdentity, PDEndpointRole


SCHEMA = "tempo-cassini-endpoint-v2"

_TRAFFIC_CLASSES = tuple(range(8))
_CORE_COUNTERS = (
    *(f"hni_rx_paused_{traffic_class}" for traffic_class in _TRAFFIC_CLASSES),
    *(f"hni_tx_paused_{traffic_class}" for traffic_class in _TRAFFIC_CLASSES),
    "parbs_tarb_pi_posted_blocked_cnt",
    "parbs_tarb_pi_posted_pkts",
)

_OPTIONAL_COUNTER_GROUPS: dict[str, tuple[str, ...]] = {
    "host_nonposted_cycles_per_packet": (
        "parbs_tarb_pi_non_posted_blocked_cnt",
        "parbs_tarb_pi_non_posted_pkts",
    ),
    "packet_counts": (
        *(f"hni_pkts_sent_by_tc_{traffic_class}"
          for traffic_class in _TRAFFIC_CLASSES),
        *(f"hni_pkts_recv_by_tc_{traffic_class}"
          for traffic_class in _TRAFFIC_CLASSES),
    ),
    "receive_overflow_fraction": (
        "lpe_net_match_priority_0",
        "lpe_net_match_overflow_0",
    ),
    "ecn_fraction": (
        *(f"ixe_tc_{kind}_{marked}_pkts_{traffic_class}"
          for kind in ("req", "rsp")
          for marked in ("ecn", "no_ecn")
          for traffic_class in _TRAFFIC_CLASSES),
    ),
    "transport_fault_counts": (
        "pct_no_tct_nacks",
        "pct_no_trs_nacks",
        "pct_no_mst_nacks",
        "pct_retry_srb_requests",
        "pct_sct_timeouts",
        "pct_spt_timeouts",
    ),
}

_SIGNAL_KEYS = frozenset({
    "rx_pause_fraction_max",
    "rx_pause_fraction_mean",
    "tx_pause_fraction_max",
    "tx_pause_fraction_mean",
    "host_posted_cycles_per_packet_max",
    "host_nonposted_cycles_per_packet_max",
    "tx_packets",
    "rx_packets",
    "receive_overflow_fraction_max",
    "receive_overflow_fraction_mean",
    "ecn_fraction_max",
    "ecn_fraction_mean",
    "resource_nacks",
    "retries",
    "timeouts",
})

_SUPPORT_KEYS = frozenset({
    "rx_pause_fraction",
    "tx_pause_fraction",
    "host_posted_cycles_per_packet",
    *_OPTIONAL_COUNTER_GROUPS,
})


class CassiniEndpointSampler:
    """Read explicit local Cassini counters for one P or D endpoint."""

    def __init__(
        self,
        identity: PDEndpointIdentity,
        root: str | Path = "/sys/class/cxi",
        *,
        nic_count: int = 4,
        min_interval_ms: float = 20.0,
        max_window_ms: float = 2000.0,
    ) -> None:
        if not isinstance(identity, PDEndpointIdentity):
            raise TypeError("identity must be PDEndpointIdentity")
        if type(nic_count) is not int or not 1 <= nic_count <= 8:
            raise ValueError("nic_count must be in [1, 8]")
        for name, value, low, high in (
            ("min_interval_ms", min_interval_ms, 0.0, 1000.0),
            ("max_window_ms", max_window_ms, 1.0, 10000.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not low <= float(value) <= high
            ):
                raise ValueError(f"{name} must be in [{low:g}, {high:g}]")
        if max_window_ms <= min_interval_ms:
            raise ValueError("max_window_ms must exceed min_interval_ms")

        self.identity = identity
        self.root = Path(root)
        self.nic_count = nic_count
        self.min_interval_ns = int(float(min_interval_ms) * 1_000_000)
        self.max_window_ns = int(float(max_window_ms) * 1_000_000)
        self._paths: dict[tuple[int, str], Path] = {}

        for nic in range(nic_count):
            telemetry = self.root / f"cxi{nic}" / "device" / "telemetry"
            for counter in _CORE_COUNTERS:
                path = telemetry / counter
                if not path.is_file():
                    raise FileNotFoundError(path)
                self._paths[(nic, counter)] = path

        self._support: dict[str, CounterSupport] = {
            "rx_pause_fraction": CounterSupport.SUPPORTED,
            "tx_pause_fraction": CounterSupport.SUPPORTED,
            "host_posted_cycles_per_packet": CounterSupport.SUPPORTED,
        }
        for group, counters in _OPTIONAL_COUNTER_GROUPS.items():
            candidates = [
                (
                    nic,
                    counter,
                    self.root / f"cxi{nic}" / "device" / "telemetry" / counter,
                )
                for nic in range(nic_count)
                for counter in counters
            ]
            present = [path.is_file() for _nic, _counter, path in candidates]
            if all(present):
                self._support[group] = CounterSupport.SUPPORTED
                for nic, counter, path in candidates:
                    self._paths[(nic, counter)] = path
            elif any(present):
                self._support[group] = CounterSupport.AMBIGUOUS
            else:
                self._support[group] = CounterSupport.NOT_SUPPORTED

        self._previous: dict[tuple[int, str], tuple[int, int]] | None = None
        self._last_result: dict[str, Any] | None = None
        self._last_read_ns: int | None = None
        self._sequence = 0
        self._lock = threading.Lock()

    def _read(self) -> dict[tuple[int, str], tuple[int, int]]:
        return {
            key: _counter_value(path.read_text(encoding="ascii"))
            for key, path in self._paths.items()
        }

    @staticmethod
    def _with_age(value: dict[str, Any], now_ns: int) -> dict[str, Any]:
        result = dict(value)
        result["signals"] = dict(value["signals"])
        result["support"] = dict(value["support"])
        sampled_ns = result.get("sampled_ns")
        result["cache_age_ms"] = (
            max(0.0, (now_ns - sampled_ns) / 1_000_000)
            if type(sampled_ns) is int else None
        )
        return result

    def _empty_signals(self) -> dict[str, None]:
        return {key: None for key in sorted(_SIGNAL_KEYS)}

    def _base(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "endpoint_id": self.identity.endpoint_id,
            "role": self.identity.role.value,
            "pair_index": self.identity.pair_index,
            "source": "cassini_sysfs_endpoint_delta",
            "nic_count": self.nic_count,
            "support": {
                name: self._support[name].value
                for name in sorted(self._support)
            },
        }

    def _invalid(
        self, *, sampled_ns: int, read_ms: float, reason: str,
    ) -> dict[str, Any]:
        return {
            **self._base(),
            "valid": False,
            "invalid_reason": reason,
            "sequence": self._sequence,
            "sampled_ns": sampled_ns,
            "read_ms": read_ms,
            "cache_age_ms": 0.0,
            "window_ms": None,
            "signals": self._empty_signals(),
        }

    def sample(self, *, force: bool = False) -> dict[str, Any]:
        if type(force) is not bool:
            raise TypeError("force must be bool")
        with self._lock:
            now_ns = time.perf_counter_ns()
            if (
                not force
                and self._last_result is not None
                and self._last_read_ns is not None
                and now_ns - self._last_read_ns < self.min_interval_ns
            ):
                return self._with_age(self._last_result, now_ns)

            started_ns = now_ns
            try:
                current = self._read()
            except (OSError, TypeError, ValueError) as exc:
                sampled_ns = time.perf_counter_ns()
                self._sequence += 1
                result = self._invalid(
                    sampled_ns=sampled_ns,
                    read_ms=(sampled_ns - started_ns) / 1_000_000,
                    reason=f"counter_read_error:{type(exc).__name__}",
                )
                self._last_read_ns = sampled_ns
                self._last_result = result
                return self._with_age(result, sampled_ns)

            sampled_ns = time.perf_counter_ns()
            self._sequence += 1
            previous = self._previous
            self._previous = current
            self._last_read_ns = sampled_ns
            read_ms = (sampled_ns - started_ns) / 1_000_000
            if previous is None:
                result = self._invalid(
                    sampled_ns=sampled_ns,
                    read_ms=read_ms,
                    reason="counter_baseline_initialized",
                )
                self._last_result = result
                return self._with_age(result, sampled_ns)

            deltas: dict[tuple[int, str], tuple[int, int]] = {}
            for key, (value, timestamp_ns) in current.items():
                prior_value, prior_timestamp_ns = previous[key]
                delta = value - prior_value
                window_ns = timestamp_ns - prior_timestamp_ns
                if delta < 0 or window_ns <= 0:
                    result = self._invalid(
                        sampled_ns=sampled_ns,
                        read_ms=read_ms,
                        reason="counter_regressed_or_timestamp_not_monotonic",
                    )
                    self._last_result = result
                    return self._with_age(result, sampled_ns)
                if window_ns > self.max_window_ns:
                    result = self._invalid(
                        sampled_ns=sampled_ns,
                        read_ms=read_ms,
                        reason="counter_window_stale",
                    )
                    self._last_result = result
                    return self._with_age(result, sampled_ns)
                deltas[key] = (delta, window_ns)

            def delta(nic: int, counter: str) -> int:
                return deltas[(nic, counter)][0]

            windows = [window for _value, window in deltas.values()]
            rx_pause: list[float] = []
            tx_pause: list[float] = []
            posted: list[float] = []
            for nic in range(self.nic_count):
                for direction, target in (("rx", rx_pause), ("tx", tx_pause)):
                    for traffic_class in _TRAFFIC_CLASSES:
                        value, window_ns = deltas[
                            (nic, f"hni_{direction}_paused_{traffic_class}")
                        ]
                        fraction = value / window_ns
                        if not 0.0 <= fraction <= 1.05:
                            result = self._invalid(
                                sampled_ns=sampled_ns,
                                read_ms=read_ms,
                                reason="pause_fraction_outside_hardware_range",
                            )
                            self._last_result = result
                            return self._with_age(result, sampled_ns)
                        target.append(min(1.0, fraction))
                blocked = delta(nic, "parbs_tarb_pi_posted_blocked_cnt")
                packets = delta(nic, "parbs_tarb_pi_posted_pkts")
                posted.append(blocked / packets if packets else 0.0)

            signals: dict[str, int | float | None] = self._empty_signals()
            signals.update({
                "rx_pause_fraction_max": max(rx_pause),
                "rx_pause_fraction_mean": sum(rx_pause) / len(rx_pause),
                "tx_pause_fraction_max": max(tx_pause),
                "tx_pause_fraction_mean": sum(tx_pause) / len(tx_pause),
                "host_posted_cycles_per_packet_max": max(posted),
            })

            if self._support["host_nonposted_cycles_per_packet"] is CounterSupport.SUPPORTED:
                ratios = []
                for nic in range(self.nic_count):
                    blocked = delta(
                        nic, "parbs_tarb_pi_non_posted_blocked_cnt")
                    packets = delta(nic, "parbs_tarb_pi_non_posted_pkts")
                    ratios.append(blocked / packets if packets else 0.0)
                signals["host_nonposted_cycles_per_packet_max"] = max(ratios)

            if self._support["packet_counts"] is CounterSupport.SUPPORTED:
                signals["tx_packets"] = sum(
                    delta(nic, f"hni_pkts_sent_by_tc_{traffic_class}")
                    for nic in range(self.nic_count)
                    for traffic_class in _TRAFFIC_CLASSES
                )
                signals["rx_packets"] = sum(
                    delta(nic, f"hni_pkts_recv_by_tc_{traffic_class}")
                    for nic in range(self.nic_count)
                    for traffic_class in _TRAFFIC_CLASSES
                )

            if self._support["receive_overflow_fraction"] is CounterSupport.SUPPORTED:
                fractions = []
                for nic in range(self.nic_count):
                    priority = delta(nic, "lpe_net_match_priority_0")
                    overflow = delta(nic, "lpe_net_match_overflow_0")
                    total = priority + overflow
                    fractions.append(overflow / total if total else 0.0)
                signals["receive_overflow_fraction_max"] = max(fractions)
                signals["receive_overflow_fraction_mean"] = (
                    sum(fractions) / len(fractions)
                )

            if self._support["ecn_fraction"] is CounterSupport.SUPPORTED:
                fractions = []
                for nic in range(self.nic_count):
                    marked = sum(
                        delta(nic, f"ixe_tc_{kind}_ecn_pkts_{traffic_class}")
                        for kind in ("req", "rsp")
                        for traffic_class in _TRAFFIC_CLASSES
                    )
                    unmarked = sum(
                        delta(nic, f"ixe_tc_{kind}_no_ecn_pkts_{traffic_class}")
                        for kind in ("req", "rsp")
                        for traffic_class in _TRAFFIC_CLASSES
                    )
                    total = marked + unmarked
                    fractions.append(marked / total if total else 0.0)
                signals["ecn_fraction_max"] = max(fractions)
                signals["ecn_fraction_mean"] = sum(fractions) / len(fractions)

            if self._support["transport_fault_counts"] is CounterSupport.SUPPORTED:
                signals["resource_nacks"] = sum(
                    delta(nic, counter)
                    for nic in range(self.nic_count)
                    for counter in (
                        "pct_no_tct_nacks",
                        "pct_no_trs_nacks",
                        "pct_no_mst_nacks",
                    )
                )
                signals["retries"] = sum(
                    delta(nic, "pct_retry_srb_requests")
                    for nic in range(self.nic_count)
                )
                signals["timeouts"] = sum(
                    delta(nic, counter)
                    for nic in range(self.nic_count)
                    for counter in ("pct_sct_timeouts", "pct_spt_timeouts")
                )

            result = {
                **self._base(),
                "valid": True,
                "invalid_reason": None,
                "sequence": self._sequence,
                "sampled_ns": sampled_ns,
                "read_ms": read_ms,
                "cache_age_ms": 0.0,
                "window_ms": max(windows) / 1_000_000,
                "signals": signals,
            }
            validate_cassini_endpoint_sample(result)
            self._last_result = result
            return self._with_age(result, sampled_ns)


def validate_cassini_endpoint_sample(raw: object) -> None:
    """Validate the exact JSON-compatible endpoint sample shape."""

    if type(raw) is not dict:
        raise TypeError("Cassini endpoint sample must be a dict")
    expected_keys = {
        "schema", "endpoint_id", "role", "pair_index", "source",
        "nic_count", "support", "valid", "invalid_reason", "sequence",
        "sampled_ns", "read_ms", "cache_age_ms", "window_ms", "signals",
    }
    if set(raw) != expected_keys:
        raise ValueError("Cassini endpoint sample keys are not exact")
    if raw["schema"] != SCHEMA:
        raise ValueError("Cassini endpoint schema is not canonical")
    if type(raw["endpoint_id"]) is not str or not raw["endpoint_id"].strip():
        raise ValueError("Cassini endpoint_id must be nonempty")
    try:
        PDEndpointRole(raw["role"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Cassini endpoint role is invalid") from exc
    if type(raw["pair_index"]) is not int or raw["pair_index"] < 0:
        raise ValueError("Cassini pair_index must be a non-negative int")
    if raw["source"] != "cassini_sysfs_endpoint_delta":
        raise ValueError("Cassini endpoint source is not canonical")
    if type(raw["nic_count"]) is not int or not 1 <= raw["nic_count"] <= 8:
        raise ValueError("Cassini nic_count must be in [1, 8]")
    if type(raw["valid"]) is not bool:
        raise TypeError("Cassini valid flag must be bool")
    if type(raw["sequence"]) is not int or raw["sequence"] < 1:
        raise ValueError("Cassini sequence must be a positive int")
    if type(raw["sampled_ns"]) is not int or raw["sampled_ns"] < 0:
        raise ValueError("Cassini sampled_ns must be a non-negative int")
    for name in ("read_ms", "cache_age_ms"):
        value = raw[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"Cassini {name} must be finite and non-negative")
    support = raw["support"]
    signals = raw["signals"]
    if type(support) is not dict or set(support) != _SUPPORT_KEYS:
        raise ValueError("Cassini support inventory is not exact")
    if type(signals) is not dict or set(signals) != _SIGNAL_KEYS:
        raise ValueError("Cassini signal inventory is not exact")
    try:
        parsed_support = {
            name: CounterSupport(value) for name, value in support.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("Cassini support value is invalid") from exc
    for core_group in (
        "rx_pause_fraction",
        "tx_pause_fraction",
        "host_posted_cycles_per_packet",
    ):
        if parsed_support[core_group] is not CounterSupport.SUPPORTED:
            raise ValueError(f"core Cassini group {core_group} must be supported")
    if raw["valid"] is not True:
        if type(raw["invalid_reason"]) is not str or not raw[
            "invalid_reason"
        ].strip():
            raise ValueError("invalid Cassini sample requires invalid_reason")
        if raw["window_ms"] is not None:
            raise ValueError("invalid Cassini sample cannot expose window_ms")
        if any(value is not None for value in signals.values()):
            raise ValueError("invalid Cassini samples cannot expose signal values")
        return
    if raw["invalid_reason"] is not None:
        raise ValueError("valid Cassini sample cannot have invalid_reason")
    window_ms = raw["window_ms"]
    if (
        isinstance(window_ms, bool)
        or not isinstance(window_ms, (int, float))
        or not math.isfinite(float(window_ms))
        or float(window_ms) <= 0.0
    ):
        raise ValueError("valid Cassini sample requires positive window_ms")
    required_values = (
        "rx_pause_fraction_max", "rx_pause_fraction_mean",
        "tx_pause_fraction_max", "tx_pause_fraction_mean",
        "host_posted_cycles_per_packet_max",
    )
    if any(signals[name] is None for name in required_values):
        raise ValueError("valid Cassini sample lacks a core signal")
    group_signals = {
        "host_nonposted_cycles_per_packet": (
            "host_nonposted_cycles_per_packet_max",
        ),
        "packet_counts": ("tx_packets", "rx_packets"),
        "receive_overflow_fraction": (
            "receive_overflow_fraction_max",
            "receive_overflow_fraction_mean",
        ),
        "ecn_fraction": ("ecn_fraction_max", "ecn_fraction_mean"),
        "transport_fault_counts": ("resource_nacks", "retries", "timeouts"),
    }
    for group, names in group_signals.items():
        present = [signals[name] is not None for name in names]
        if parsed_support[group] is CounterSupport.SUPPORTED:
            if not all(present):
                raise ValueError(f"supported Cassini group {group} lacks values")
        elif any(present):
            raise ValueError(f"unavailable Cassini group {group} has values")

    fraction_signals = {
        "rx_pause_fraction_max",
        "rx_pause_fraction_mean",
        "tx_pause_fraction_max",
        "tx_pause_fraction_mean",
        "receive_overflow_fraction_max",
        "receive_overflow_fraction_mean",
        "ecn_fraction_max",
        "ecn_fraction_mean",
    }
    integer_signals = {
        "tx_packets", "rx_packets", "resource_nacks", "retries", "timeouts",
    }
    for name, value in signals.items():
        if value is None:
            continue
        if name in integer_signals:
            if type(value) is not int or value < 0:
                raise ValueError(f"Cassini {name} must be a non-negative int")
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"Cassini {name} must be finite and non-negative")
        if name in fraction_signals and float(value) > 1.0:
            raise ValueError(f"Cassini {name} must be in [0, 1]")


__all__ = [
    "CassiniEndpointSampler",
    "SCHEMA",
    "validate_cassini_endpoint_sample",
]
