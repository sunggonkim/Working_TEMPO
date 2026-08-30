"""Pair-local arrival-regime admission for prefill/decode routing.

The controller observes only pair-local monotonic arrival times and local
ownership.  It calibrates once, then freezes a regime for the admission epoch:

* low arrival pressure: decoder-local;
* mid arrival pressure: decoder-local;
* high arrival pressure: decoder-local up to a credit cap, then remote spill.

Transport and request execution remain outside this module.  Call ``release``
when a routed request completes so high-regime credits represent live work.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum


class ArrivalRegime(str, Enum):
    CALIBRATING = "calibrating"
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class AdmissionRoute(str, Enum):
    LOCAL = "decoder_local"
    REMOTE = "remote_prefill"


@dataclass(frozen=True)
class RegimeDecision:
    request_id: str
    route: AdmissionRoute
    regime: ArrivalRegime
    reason: str
    observed_mean_pair_interval_ns: float | None
    local_inflight_before: int


class PairArrivalRegimeController:
    """Deterministic, thread-safe-under-caller-lock pair admission state."""

    def __init__(
        self,
        *,
        high_pair_interval_ns: int = 74_000_000,
        mid_pair_interval_ns: int = 110_000_000,
        calibration_requests: int = 4,
        high_local_inflight_cap: int = 8,
    ) -> None:
        if not (0 < high_pair_interval_ns < mid_pair_interval_ns):
            raise ValueError("arrival thresholds must be positive and ordered")
        if calibration_requests < 2:
            raise ValueError("calibration_requests must be at least two")
        if high_local_inflight_cap < 1:
            raise ValueError("high_local_inflight_cap must be positive")
        self.high_pair_interval_ns = high_pair_interval_ns
        self.mid_pair_interval_ns = mid_pair_interval_ns
        self.calibration_requests = calibration_requests
        self.high_local_inflight_cap = high_local_inflight_cap
        self._intervals: deque[int] = deque(maxlen=calibration_requests - 1)
        self._last_arrival_ns: int | None = None
        self._mean_interval_ns: float | None = None
        self._regime: ArrivalRegime | None = None
        self._owned: dict[str, AdmissionRoute] = {}

    @property
    def regime(self) -> ArrivalRegime:
        return self._regime or ArrivalRegime.CALIBRATING

    @property
    def local_inflight(self) -> int:
        return sum(route is AdmissionRoute.LOCAL for route in self._owned.values())

    def decide(
        self, request_id: str, now_ns: int, *, force_local: bool = False
    ) -> RegimeDecision:
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be nonempty")
        if request_id in self._owned:
            raise ValueError("duplicate request_id")
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a nonnegative int")
        if self._last_arrival_ns is not None:
            interval_ns = now_ns - self._last_arrival_ns
            if interval_ns <= 0:
                raise ValueError("arrival clock must advance")
            self._intervals.append(interval_ns)
        self._last_arrival_ns = now_ns
        if self._regime is None and len(self._intervals) == self.calibration_requests - 1:
            self._mean_interval_ns = sum(self._intervals) / len(self._intervals)
            if self._mean_interval_ns <= self.high_pair_interval_ns:
                self._regime = ArrivalRegime.HIGH
            elif self._mean_interval_ns <= self.mid_pair_interval_ns:
                self._regime = ArrivalRegime.MID
            else:
                self._regime = ArrivalRegime.LOW

        regime = self.regime
        local_before = self.local_inflight
        if type(force_local) is not bool:
            raise ValueError("force_local must be a bool")
        if force_local:
            route = AdmissionRoute.LOCAL
            reason = "workload_guard_local"
        elif regime is ArrivalRegime.HIGH and local_before >= self.high_local_inflight_cap:
            route = AdmissionRoute.REMOTE
            reason = "high_arrival_local_credit_exhausted"
        else:
            route = AdmissionRoute.LOCAL
            reason = f"{regime.value}_arrival_local"
        self._owned[request_id] = route
        return RegimeDecision(
            request_id=request_id,
            route=route,
            regime=regime,
            reason=reason,
            observed_mean_pair_interval_ns=self._mean_interval_ns,
            local_inflight_before=local_before,
        )

    def release(self, request_id: str) -> None:
        if request_id not in self._owned:
            raise KeyError(request_id)
        del self._owned[request_id]
