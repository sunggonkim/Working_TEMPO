"""Bounded, on-demand HPE Cassini pressure sampling for Perlmutter."""

from __future__ import annotations

import math
from pathlib import Path
import re
import threading
import time
from typing import Any


SCHEMA = "tempo-cassini-pressure-v1"
_COUNTER_VALUE = re.compile(r"^([0-9]+)@([0-9]+)\.([0-9]{1,9})$")
_COUNTERS = (
    "hni_rx_paused_0",
    "hni_rx_paused_1",
    "hni_tx_paused_0",
    "hni_tx_paused_1",
    "parbs_tarb_pi_posted_blocked_cnt",
    "parbs_tarb_pi_posted_pkts",
)


def _counter_value(text: str) -> tuple[int, int]:
    if not isinstance(text, str):
        raise TypeError("Cassini counter payload must be text")
    match = _COUNTER_VALUE.fullmatch(text.strip())
    if match is None:
        raise ValueError("Cassini counter payload must be value@timestamp")
    value = int(match.group(1))
    fraction = match.group(3).ljust(9, "0")
    timestamp_ns = int(match.group(2)) * 1_000_000_000 + int(fraction)
    return value, timestamp_ns


class CassiniPressureSampler:
    """Read four explicit local Cassini NICs without a persistent watcher."""

    def __init__(
        self,
        root: str | Path = "/sys/class/cxi",
        *,
        nic_count: int = 4,
        min_interval_ms: float = 20.0,
        max_window_ms: float = 2000.0,
    ) -> None:
        if type(nic_count) is not int or nic_count < 1 or nic_count > 8:
            raise ValueError("nic_count must be in [1, 8]")
        for name, value, low, high in (
            ("min_interval_ms", min_interval_ms, 0.0, 1000.0),
            ("max_window_ms", max_window_ms, 1.0, 10000.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < low
                or float(value) > high
            ):
                raise ValueError(f"{name} must be in [{low:g}, {high:g}]")
        if max_window_ms <= min_interval_ms:
            raise ValueError("max_window_ms must exceed min_interval_ms")
        self.root = Path(root)
        self.nic_count = nic_count
        self.min_interval_ns = int(float(min_interval_ms) * 1_000_000)
        self.max_window_ns = int(float(max_window_ms) * 1_000_000)
        self._paths: dict[tuple[int, str], Path] = {}
        for nic in range(nic_count):
            telemetry = self.root / f"cxi{nic}" / "device" / "telemetry"
            for counter in _COUNTERS:
                path = telemetry / counter
                if not path.is_file():
                    raise FileNotFoundError(path)
                self._paths[(nic, counter)] = path
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

    def _invalid(
        self, *, sampled_ns: int, read_ms: float, reason: str,
    ) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "source": "cassini_sysfs_counter_delta",
            "valid": False,
            "invalid_reason": reason,
            "sequence": self._sequence,
            "sampled_ns": sampled_ns,
            "read_ms": read_ms,
            "cache_age_ms": 0.0,
            "window_ms": None,
            "nic_count": self.nic_count,
            "rx_pause_fraction_max": None,
            "rx_pause_fraction_mean": None,
            "tx_pause_fraction_max": None,
            "tx_pause_fraction_mean": None,
            "host_blocked_cycles_per_packet_max": None,
        }

    @staticmethod
    def _with_age(value: dict[str, Any], now_ns: int) -> dict[str, Any]:
        result = dict(value)
        sampled_ns = result.get("sampled_ns")
        result["cache_age_ms"] = (
            max(0.0, (now_ns - sampled_ns) / 1_000_000)
            if type(sampled_ns) is int else None
        )
        return result

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
                return dict(result)
            sampled_ns = time.perf_counter_ns()
            self._sequence += 1
            previous = self._previous
            self._previous = current
            self._last_read_ns = sampled_ns
            read_ms = (sampled_ns - started_ns) / 1_000_000
            if previous is None:
                result = self._invalid(
                    sampled_ns=sampled_ns, read_ms=read_ms,
                    reason="counter_baseline_initialized",
                )
                self._last_result = result
                return dict(result)

            deltas: dict[tuple[int, str], tuple[int, int]] = {}
            for key, (value, timestamp_ns) in current.items():
                prior_value, prior_timestamp_ns = previous[key]
                delta = value - prior_value
                window_ns = timestamp_ns - prior_timestamp_ns
                if delta < 0 or window_ns <= 0:
                    result = self._invalid(
                        sampled_ns=sampled_ns, read_ms=read_ms,
                        reason="counter_regressed_or_timestamp_not_monotonic",
                    )
                    self._last_result = result
                    return dict(result)
                if window_ns > self.max_window_ns:
                    result = self._invalid(
                        sampled_ns=sampled_ns, read_ms=read_ms,
                        reason="counter_window_stale",
                    )
                    self._last_result = result
                    return dict(result)
                deltas[key] = (delta, window_ns)

            rx_pause = []
            tx_pause = []
            host_blocked = []
            windows = []
            for nic in range(self.nic_count):
                for direction, target in (("rx", rx_pause), ("tx", tx_pause)):
                    for traffic_class in (0, 1):
                        delta, window_ns = deltas[
                            (nic, f"hni_{direction}_paused_{traffic_class}")]
                        fraction = delta / window_ns
                        if fraction < 0.0 or fraction > 1.05:
                            result = self._invalid(
                                sampled_ns=sampled_ns, read_ms=read_ms,
                                reason="pause_fraction_outside_hardware_range",
                            )
                            self._last_result = result
                            return dict(result)
                        target.append(min(1.0, fraction))
                        windows.append(window_ns)
                blocked, blocked_window = deltas[
                    (nic, "parbs_tarb_pi_posted_blocked_cnt")]
                packets, packet_window = deltas[
                    (nic, "parbs_tarb_pi_posted_pkts")]
                windows.extend((blocked_window, packet_window))
                host_blocked.append(blocked / packets if packets else 0.0)

            result = {
                "schema": SCHEMA,
                "source": "cassini_sysfs_counter_delta",
                "valid": True,
                "invalid_reason": None,
                "sequence": self._sequence,
                "sampled_ns": sampled_ns,
                "read_ms": read_ms,
                "cache_age_ms": 0.0,
                "window_ms": max(windows) / 1_000_000,
                "nic_count": self.nic_count,
                "rx_pause_fraction_max": max(rx_pause),
                "rx_pause_fraction_mean": sum(rx_pause) / len(rx_pause),
                "tx_pause_fraction_max": max(tx_pause),
                "tx_pause_fraction_mean": sum(tx_pause) / len(tx_pause),
                "host_blocked_cycles_per_packet_max": max(host_blocked),
            }
            self._last_result = result
            return dict(result)


__all__ = ["CassiniPressureSampler", "SCHEMA", "_counter_value"]
