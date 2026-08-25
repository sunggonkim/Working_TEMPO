"""Request-triggered telemetry refresh for TEMPO-GO.

Perlmutter safety rules prohibit persistent polling and background agents.
This component therefore creates no long-lived task: an admission request
uses the cached atomic batch while it is fresh, or becomes the one bounded
single-flight caller that concurrently fetches the frontend ledger and all
pair endpoints.  A failed or timed-out refresh never installs partial state;
the coordinator may separately authorize a bounded, tenant-scoped use of the
last complete snapshot.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
import time
from typing import Any

from tempo.pd_global_telemetry import (
    GlobalTelemetryAdapter,
    GlobalTelemetryBatch,
)


SnapshotFetcher = Callable[[], Awaitable[Mapping[str, Any]]]


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


class RequestTriggeredTelemetryAgent:
    """Bounded single-flight refresh with fail-closed stale handling."""

    def __init__(
        self,
        adapter: GlobalTelemetryAdapter,
        *,
        frontend_fetcher: SnapshotFetcher,
        endpoint_fetchers: Sequence[SnapshotFetcher],
        freshness_ns: int,
        refresh_timeout_ns: int,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not isinstance(adapter, GlobalTelemetryAdapter):
            raise TypeError("adapter must be GlobalTelemetryAdapter")
        if not callable(frontend_fetcher):
            raise TypeError("frontend_fetcher must be callable")
        if (
            not isinstance(endpoint_fetchers, Sequence)
            or isinstance(endpoint_fetchers, (str, bytes, bytearray))
            or len(endpoint_fetchers) != len(adapter.contracts)
            or any(not callable(item) for item in endpoint_fetchers)
        ):
            raise TypeError("one endpoint fetcher is required per pair")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self.adapter = adapter
        self.frontend_fetcher = frontend_fetcher
        self.endpoint_fetchers = tuple(endpoint_fetchers)
        self.freshness_ns = _positive_int("freshness_ns", freshness_ns)
        self.refresh_timeout_ns = _positive_int(
            "refresh_timeout_ns", refresh_timeout_ns)
        self.clock_ns = clock_ns
        self._batch: GlobalTelemetryBatch | None = None
        self._lock = asyncio.Lock()
        self._requests = 0
        self._cache_hits = 0
        self._refreshes = 0
        self._failures = 0
        self._timeouts = 0
        self._quarantines = 0
        self._forced_refresh_coalesces = 0
        self._last_quarantined_pairs: tuple[int, ...] = ()
        self._last_error: str | None = None
        # Leave causal-assembly headroom inside the profile's collection
        # span.  Endpoint fetches that exceed this bounded share are returned
        # as per-pair exceptions and quarantined below; they must not race an
        # equal-deadline outer timeout and turn one slow endpoint into an
        # allocation-wide refresh failure.
        self._per_fetch_timeout_ns = min(
            self.refresh_timeout_ns,
            max(1_000_000, self.adapter.maximum_collection_span_ns // 2),
        )

    def _fresh(self, now_ns: int) -> bool:
        return (
            self._batch is not None
            and self._batch.sampled_ns <= now_ns
            and now_ns - self._batch.sampled_ns <= self.freshness_ns
        )

    async def get(self, *, force: bool = False) -> GlobalTelemetryBatch:
        """Return fresh telemetry; never return stale state after a failure."""

        if type(force) is not bool:
            raise TypeError("force must be bool")
        self._requests += 1
        force_after_sequence = (
            self._batch.sequence if force and self._batch is not None else 0
        )
        now_ns = self.clock_ns()
        if not force and self._fresh(now_ns):
            self._cache_hits += 1
            assert self._batch is not None
            return self._batch
        async with self._lock:
            now_ns = self.clock_ns()
            if not force and self._fresh(now_ns):
                self._cache_hits += 1
                assert self._batch is not None
                return self._batch
            if (
                force
                and self._batch is not None
                and self._batch.sequence > force_after_sequence
                and self._fresh(now_ns)
            ):
                # Concurrent queue-timeout callers require a batch newer than
                # the one each observed at entry, not one physical scrape per
                # waiter.  Reuse the single-flight caller's new batch.
                self._cache_hits += 1
                self._forced_refresh_coalesces += 1
                return self._batch
            started_ns = self.clock_ns()
            async def bounded(fetcher: SnapshotFetcher):
                return await asyncio.wait_for(
                    fetcher(),
                    self._per_fetch_timeout_ns / 1_000_000_000,
                )

            values = await asyncio.gather(
                bounded(self.frontend_fetcher),
                *(bounded(fetcher) for fetcher in self.endpoint_fetchers),
                return_exceptions=True,
            )
            finished_ns = self.clock_ns()
            frontend = values[0]
            if isinstance(frontend, asyncio.TimeoutError):
                self._failures += 1
                self._timeouts += 1
                self._last_error = "telemetry_refresh_timeout"
                raise RuntimeError("global telemetry refresh timed out")
            if isinstance(frontend, BaseException):
                self._failures += 1
                self._last_error = f"telemetry_fetch:{type(frontend).__name__}"
                raise RuntimeError("global telemetry refresh failed") from frontend
            endpoints = {}
            quarantined: dict[int, str] = {}
            for index in range(len(self.endpoint_fetchers)):
                value = values[index + 1]
                if isinstance(value, BaseException):
                    quarantined[index] = (
                        f"endpoint_fetch:{type(value).__name__}")
                else:
                    endpoints[index] = value
            if not endpoints:
                # Do not replace the last complete allocation snapshot with
                # an all-pairs quarantine merely because the shared HTTP
                # control plane missed one bounded scrape under data-plane
                # overload.  Raising here keeps the previous batch immutable;
                # the coordinator can use it only when the requesting tenant
                # has an explicit stale-grace policy.  A partial failure still
                # quarantines only the failed pair and preserves survivor
                # routing below.
                all_timed_out = all(
                    isinstance(values[index + 1], asyncio.TimeoutError)
                    for index in range(len(self.endpoint_fetchers))
                )
                self._failures += 1
                if all_timed_out:
                    self._timeouts += 1
                    self._last_error = "telemetry_all_endpoints_timeout"
                    raise RuntimeError("global telemetry refresh timed out")
                self._last_error = "telemetry_all_endpoints_failed"
                raise RuntimeError("global telemetry refresh failed")
            try:
                batch = self.adapter.assemble(
                    frontend,
                    endpoints,
                    collection_started_ns=started_ns,
                    collection_finished_ns=finished_ns,
                    quarantined_pairs=quarantined,
                )
            except Exception as exc:
                self._failures += 1
                self._last_error = f"telemetry_validation:{type(exc).__name__}"
                raise RuntimeError("global telemetry validation failed") from exc
            self._batch = batch
            self._refreshes += 1
            self._last_quarantined_pairs = tuple(sorted(quarantined))
            self._quarantines += len(quarantined)
            self._last_error = None
            return batch

    def status(self) -> dict[str, object]:
        batch = self._batch
        return {
            "mode": "request_triggered_bounded_single_flight",
            "background_polling": False,
            "requests": self._requests,
            "cache_hits": self._cache_hits,
            "refreshes": self._refreshes,
            "failures": self._failures,
            "timeouts": self._timeouts,
            "quarantines": self._quarantines,
            "forced_refresh_coalesces": self._forced_refresh_coalesces,
            "last_quarantined_pairs": list(self._last_quarantined_pairs),
            "last_error": self._last_error,
            "last_sequence": batch.sequence if batch is not None else None,
            "last_sampled_ns": batch.sampled_ns if batch is not None else None,
            "freshness_ns": self.freshness_ns,
            "refresh_timeout_ns": self.refresh_timeout_ns,
            "per_fetch_timeout_ns": self._per_fetch_timeout_ns,
        }


__all__ = ["RequestTriggeredTelemetryAgent", "SnapshotFetcher"]
