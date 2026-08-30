"""Minimal D2H admission controller for the C0 research prototype."""

from __future__ import annotations

from dataclasses import dataclass


_NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class C0Config:
    """The two C0 knobs: an in-flight byte cap and an optional rate cap."""

    max_inflight_bytes: int
    rate_bytes_per_second: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_inflight_bytes, bool) or self.max_inflight_bytes <= 0:
            raise ValueError("max_inflight_bytes must be positive")
        rate = self.rate_bytes_per_second
        if rate is not None and (isinstance(rate, bool) or rate <= 0):
            raise ValueError("rate_bytes_per_second must be positive when set")


@dataclass(frozen=True)
class C0Decision:
    admitted: bool
    granted_bytes: int
    reason: str


@dataclass(frozen=True)
class C0Snapshot:
    available_rate_tokens: int
    last_update_ns: int | None
    admitted_bytes: int
    admitted_requests: int
    rejected_requests: int


class C0Admission:
    """All-or-nothing D2H admission with no phase or deadline prediction.

    The caller owns completion accounting and passes the current physical
    inflight_bytes on every request. When rate limiting is enabled, the
    bucket burst is fixed to max_inflight_bytes so C0 keeps two knobs.
    """

    def __init__(self, config: C0Config) -> None:
        self.config = config
        self._capacity_units = config.max_inflight_bytes * _NS_PER_SECOND
        self._token_units = self._capacity_units
        self._last_update_ns: int | None = None
        self._admitted_bytes = 0
        self._admitted_requests = 0
        self._rejected_requests = 0

    def reset(self, now_ns: int | None = None) -> None:
        if now_ns is not None and now_ns < 0:
            raise ValueError("now_ns must be nonnegative")
        self._token_units = self._capacity_units
        self._last_update_ns = now_ns
        self._admitted_bytes = 0
        self._admitted_requests = 0
        self._rejected_requests = 0

    def try_admit(
        self,
        *,
        now_ns: int,
        request_bytes: int,
        inflight_bytes: int,
    ) -> C0Decision:
        if now_ns < 0:
            raise ValueError("now_ns must be nonnegative")
        if request_bytes <= 0:
            raise ValueError("request_bytes must be positive")
        if inflight_bytes < 0:
            raise ValueError("inflight_bytes must be nonnegative")
        if inflight_bytes > self.config.max_inflight_bytes:
            raise ValueError("inflight_bytes exceeds the configured cap")

        self._refill(now_ns)
        if request_bytes > self.config.max_inflight_bytes - inflight_bytes:
            return self._reject("inflight_cap")

        required_units = request_bytes * _NS_PER_SECOND
        if (
            self.config.rate_bytes_per_second is not None
            and required_units > self._token_units
        ):
            return self._reject("rate_cap")

        if self.config.rate_bytes_per_second is not None:
            self._token_units -= required_units
        self._admitted_bytes += request_bytes
        self._admitted_requests += 1
        return C0Decision(True, request_bytes, "admitted")

    def snapshot(self) -> C0Snapshot:
        tokens = (
            self.config.max_inflight_bytes
            if self.config.rate_bytes_per_second is None
            else self._token_units // _NS_PER_SECOND
        )
        return C0Snapshot(
            available_rate_tokens=tokens,
            last_update_ns=self._last_update_ns,
            admitted_bytes=self._admitted_bytes,
            admitted_requests=self._admitted_requests,
            rejected_requests=self._rejected_requests,
        )

    def _refill(self, now_ns: int) -> None:
        if self._last_update_ns is None:
            self._last_update_ns = now_ns
            return
        if now_ns < self._last_update_ns:
            raise ValueError("now_ns must be monotonic")
        rate = self.config.rate_bytes_per_second
        if rate is not None:
            elapsed_ns = now_ns - self._last_update_ns
            self._token_units = min(
                self._capacity_units,
                self._token_units + elapsed_ns * rate,
            )
        self._last_update_ns = now_ns

    def _reject(self, reason: str) -> C0Decision:
        self._rejected_requests += 1
        return C0Decision(False, 0, reason)
